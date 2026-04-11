"""
CLI 入口: 数据准备 → 训练 → 评估 → 解释 → 对比
用法:
  python main.py prepare
  python main.py train
  python main.py evaluate
  python main.py interpret --sample_idx 0 --fold 0
  python main.py compare
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

from config import Config


def cmd_prepare(cfg):
    from data_prepare import scan_dataset, convert_all
    metadata = scan_dataset(cfg.dataset_root)
    print("\n产品分布:")
    print(metadata["code"].value_counts().to_string())
    convert_all(metadata, cfg.prepared_dir, cfg)


def cmd_train(cfg):
    """单阶段统一训练 (留出类 + leave-one-batch-out)。"""
    from train import train_all_folds
    from evaluate import evaluate_all_settings

    fold_results, split_info = train_all_folds(cfg)
    evaluate_all_settings(fold_results, split_info, cfg)


def cmd_evaluate(cfg):
    """加载已保存模型, 运行 Setting A/B/C 评估。"""
    from dataset import GCMSDataset, unified_splits
    from models import GCMSConsistencyNet
    from evaluate import evaluate_all_settings
    from register import PrototypeStore
    from torch.utils.data import DataLoader

    from config import get_device
    device = get_device()
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")

    # 读取保存的切分信息
    split_file = Path(cfg.output_dir) / "split_info.json"
    if not split_file.exists():
        print("未找到 split_info.json, 请先运行 train")
        return
    with open(split_file) as f:
        saved_split = json.load(f)

    # 重新构建 unified_splits
    split_info = unified_splits(
        metadata_csv, product_col=product_col,
        num_open_classes=cfg.num_open_test_classes,
        seed=cfg.seed)

    fold_results = []
    for fold in split_info["folds"]:
        fold_dir = Path(cfg.output_dir) / f"fold_{fold['fold_idx']}"
        if not (fold_dir / "model.pt").exists():
            print(f"Fold {fold['fold_idx']} 模型不存在, 跳过")
            continue

        with open(fold_dir / "product_classes.json") as f:
            classes = json.load(f)

        ds_train = GCMSDataset(metadata_csv, product_col=product_col,
                               augmentation=None, indices=fold["train_idx"])
        ds_val = GCMSDataset(metadata_csv, product_col=product_col,
                             augmentation=None, indices=fold["val_idx"])

        # 统一编码器: 合并 train/val 所有可能的标签值
        from sklearn.preprocessing import LabelEncoder
        all_batch_vals = sorted(
            set(ds_train.df["batch_idx"].unique())
            | set(ds_val.df["batch_idx"].unique())
        )
        all_product_vals = sorted(
            set(ds_train.df[product_col].unique())
            | set(ds_val.df[product_col].unique())
        )
        shared_batch_enc = LabelEncoder().fit(all_batch_vals)
        shared_product_enc = LabelEncoder().fit(all_product_vals)
        for ds in (ds_train, ds_val):
            ds.product_enc = shared_product_enc
            ds.batch_enc = shared_batch_enc
            ds.df["product_label"] = shared_product_enc.transform(
                ds.df[product_col])
            ds.df["batch_label"] = shared_batch_enc.transform(
                ds.df["batch_idx"])
            ds.num_products = len(shared_product_enc.classes_)
            ds.num_batches = len(shared_batch_enc.classes_)

        loader_val = DataLoader(ds_val, batch_size=cfg.batch_size,
                                shuffle=False)

        model = GCMSConsistencyNet(
            ds_train.num_batches, cfg
        ).to(device)
        model.load_state_dict(torch.load(
            fold_dir / "model.pt", map_location=device, weights_only=True))

        proto_store = PrototypeStore()
        proto_dir = fold_dir / "prototypes"
        if proto_dir.exists():
            proto_store.load(proto_dir)

        fold_results.append({
            "fold": fold["fold_idx"],
            "test_batch": fold["test_batch"],
            "model": model,
            "proto_store": proto_store,
            "ds_train": ds_train,
            "ds_val": ds_val,
            "loader_val": loader_val,
            "train_idx": fold["train_idx"],
        })

    if fold_results:
        evaluate_all_settings(fold_results, split_info, cfg)


def cmd_interpret(cfg, fold_idx=0, sample_idx=0):
    """对指定样本做 Grad-CAM 解释 (基于嵌入距离)。"""
    from dataset import GCMSDataset, unified_splits
    from models import GCMSConsistencyNet
    from interpret import GradCAM, find_top_regions, plot_interpretation
    from register import PrototypeStore

    from config import get_device
    device = get_device()
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")

    split_info = unified_splits(
        metadata_csv, product_col=product_col,
        num_open_classes=cfg.num_open_test_classes, seed=cfg.seed)
    fold = split_info["folds"][fold_idx]

    ds_val = GCMSDataset(metadata_csv, product_col=product_col,
                         augmentation=None, indices=fold["val_idx"])

    fold_dir = Path(cfg.output_dir) / f"fold_{fold_idx}"
    with open(fold_dir / "product_classes.json") as f:
        classes = json.load(f)

    model = GCMSConsistencyNet(
        len(ds_val.batch_enc.classes_), cfg
    ).to(device)
    model.load_state_dict(torch.load(fold_dir / "model.pt",
                                     map_location=device,
                                     weights_only=True))

    proto_store = PrototypeStore()
    proto_dir = fold_dir / "prototypes"
    if proto_dir.exists():
        proto_store.load(proto_dir)

    sample = ds_val[sample_idx]
    x = sample["input"].unsqueeze(0).to(device)

    z = model.encode(x)
    pred_result = proto_store.predict(z) if proto_store.num_classes > 0 else None

    # Grad-CAM (仅使用嵌入距离模式)
    if pred_result and proto_store.num_classes > 0:
        pred_class = pred_result["pred_class"][0]
        score = pred_result["scores"][0].item()
        target_proto = proto_store.prototypes[pred_class]
        grad_cam = GradCAM(model, mode="embedding")
        cam = grad_cam(x, target_proto=target_proto)
    else:
        pred_class = None
        score = None
        grad_cam = GradCAM(model, mode="embedding")
        cam = grad_cam(x)

    with open(Path(cfg.prepared_dir) / "grid_info.json") as f:
        grid_info = json.load(f)

    rt_range = cfg.rt_range or (0.0, 40.0)
    mz_range = tuple(grid_info.get("mz_range", cfg.mz_range))

    regions = find_top_regions(cam, rt_range, mz_range, top_k=10)
    print(f"\n样本 {sample['sample_id']} 的预测结果:")
    if pred_class:
        print(f"  产品识别: {pred_class}")
        print(f"  一致性分数: {score:.4f}")
    print(f"\n  Top-10 关键区域:")
    for j, r in enumerate(regions):
        print(f"    {j+1}. RT={r['rt']:.2f} min, "
              f"m/z={r['mz']:.1f}, importance={r['importance']:.3f}")

    x_np = sample["input"].numpy()
    plot_interpretation(x_np, cam, rt_range, mz_range,
                        sample_id=sample["sample_id"],
                        consistency_score=score,
                        pred_class=pred_class,
                        save_dir=Path(cfg.output_dir) / "interpretations")
    print(f"解释图已保存")


def main():
    parser = argparse.ArgumentParser(
        description="GC-MS 跨批次一致性深度学习流水线 (统一度量学习框架)"
    )
    parser.add_argument("command",
                        choices=["prepare", "train", "evaluate",
                                 "interpret", "compare"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--methods", type=str, default=None,
                        help="对比方法 (逗号分隔), 默认全部运行")

    # 数据准备选项
    parser.add_argument("--save_plot", dest="save_prepare_plots",
                        action="store_true", default=False)
    parser.add_argument("--no-save_plot", dest="save_prepare_plots",
                        action="store_false")
    parser.add_argument("--save_table", dest="save_prepare_tables",
                        action="store_true", default=False)
    parser.add_argument("--no-save_table", dest="save_prepare_tables",
                        action="store_false")

    # 范围参数
    parser.add_argument("--rt_min", type=float, default=3.17)
    parser.add_argument("--rt_max", type=float, default=36.91)
    parser.add_argument("--mz_min", type=float, default=30)
    parser.add_argument("--mz_max", type=float, default=200)

    args = parser.parse_args()

    cfg = Config()

    cfg.save_prepare_plots = bool(args.save_prepare_plots)
    cfg.save_prepare_tables = bool(args.save_prepare_tables)

    if (args.rt_min is not None) or (args.rt_max is not None):
        rt_min = cfg.rt_range[0] if args.rt_min is None else args.rt_min
        rt_max = cfg.rt_range[1] if args.rt_max is None else args.rt_max
        if rt_max <= rt_min:
            raise ValueError(f"RT 范围非法: rt_min={rt_min}, rt_max={rt_max}")
        cfg.rt_range = (float(rt_min), float(rt_max))

    if (args.mz_min is not None) or (args.mz_max is not None):
        mz_min = cfg.mz_range[0] if args.mz_min is None else args.mz_min
        mz_max = cfg.mz_range[1] if args.mz_max is None else args.mz_max
        if mz_max <= mz_min:
            raise ValueError(f"m/z 范围非法: mz_min={mz_min}, mz_max={mz_max}")
        cfg.mz_range = (float(mz_min), float(mz_max))

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if args.command == "prepare":
        cmd_prepare(cfg)
    elif args.command == "train":
        cmd_train(cfg)
    elif args.command == "evaluate":
        cmd_evaluate(cfg)
    elif args.command == "interpret":
        cmd_interpret(cfg, fold_idx=args.fold, sample_idx=args.sample_idx)
    elif args.command == "compare":
        cmd_compare(cfg, methods=args.methods)


def cmd_compare(cfg, methods=None):
    """运行对比实验: 传统方法 / DL基线 / 消融变体 / 本文方法。"""
    from compare import run_comparison, ALL_METHODS
    if methods:
        method_list = [m.strip() for m in methods.split(",")]
    else:
        method_list = list(ALL_METHODS)
    run_comparison(cfg, methods=method_list)


if __name__ == "__main__":
    main()