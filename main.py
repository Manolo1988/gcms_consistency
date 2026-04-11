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
    from dataset import create_data_split
    metadata = scan_dataset(cfg.dataset_root)
    print("\n产品分布:")
    print(metadata["code"].value_counts().to_string())
    convert_all(metadata, cfg.prepared_dir, cfg)

    # 创建固定数据划分
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")
    create_data_split(metadata_csv, cfg, product_col=product_col)


def cmd_train(cfg):
    """训练单一最终模型。"""
    from train import train_single_model
    train_single_model(cfg)


def cmd_evaluate(cfg):
    """加载已保存模型, 运行 Setting A/B/C 评估。"""
    from evaluate import evaluate_single_model
    evaluate_single_model(cfg)


def cmd_register(cfg, new_data_dir):
    """增量注册新产品: 微调编码器 + 球面重分布。

    new_data_dir 下应包含已 prepare 好的 .npz 张量文件,
    以及 metadata.csv (同 prepared_data 格式)。
    """
    from dataset import GCMSDataset, GCMSAugmentation, load_data_split
    from models import GCMSConsistencyNet
    from register import PrototypeStore, finetune_for_new_product
    from config import get_device
    from torch.utils.data import DataLoader

    device = get_device()
    model_dir = Path(cfg.output_dir) / "final_model"

    # 加载已训练模型
    with open(model_dir / "train_meta.json") as f:
        meta = json.load(f)
    model = GCMSConsistencyNet(meta["num_batches"], cfg).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt",
                                     map_location=device,
                                     weights_only=True))

    # 加载旧原型
    old_store = PrototypeStore()
    old_store.load(model_dir / "prototypes")
    print(f"已加载旧模型, {old_store.num_classes} 个已知产品: "
          f"{old_store.class_names}")

    # 加载旧训练数据 (经验回放)
    split = load_data_split(cfg)
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")
    ds_old = GCMSDataset(metadata_csv, product_col=product_col,
                         augmentation=GCMSAugmentation(cfg),
                         indices=split["train_idx"])
    old_label_names = ds_old.get_label_name_map()
    loader_old = DataLoader(ds_old, batch_size=cfg.batch_size,
                            shuffle=True, num_workers=0)

    # 加载新产品数据
    new_metadata_csv = str(Path(new_data_dir) / "metadata.csv")
    ds_new = GCMSDataset(new_metadata_csv, product_col=product_col,
                         augmentation=GCMSAugmentation(cfg))
    # 重新编码: 新类标签偏移, 避免与旧类冲突
    max_old_label = max(old_label_names.keys()) + 1
    new_product_names = ds_new.get_product_names()
    new_label_names = {i + max_old_label: name
                       for i, name in enumerate(new_product_names)}
    ds_new.df["product_label"] = ds_new.df["product_label"] + max_old_label
    loader_new = DataLoader(ds_new, batch_size=cfg.batch_size,
                            shuffle=True, num_workers=0)

    print(f"新产品: {new_product_names}")
    print(f"新数据: {len(ds_new)} 样本")

    # 微调
    model, new_store = finetune_for_new_product(
        model, old_store, loader_new, loader_old,
        cfg, device,
        new_label_names=new_label_names,
        old_label_names=old_label_names,
    )

    # 保存更新后的模型和原型
    torch.save(model.state_dict(), model_dir / "model.pt")
    new_store.save(model_dir / "prototypes")
    with open(model_dir / "product_classes.json", "w") as f:
        all_names = list(new_store.class_names)
        json.dump(all_names, f)
    print(f"\n注册完成, 共 {new_store.num_classes} 个产品")


def cmd_interpret(cfg, fold_idx=0, sample_idx=0):
    """对指定样本做 Grad-CAM 解释 (基于嵌入距离)。"""
    from dataset import GCMSDataset, load_data_split
    from models import GCMSConsistencyNet
    from interpret import GradCAM, find_top_regions, plot_interpretation
    from register import PrototypeStore

    from config import get_device
    device = get_device()
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")

    split = load_data_split(cfg)

    # 使用 Setting A 测试集 (留出批次) 作为解释对象
    test_idx = split["test_batch_idx"] or split["val_idx"]
    ds_test = GCMSDataset(metadata_csv, product_col=product_col,
                          augmentation=None, indices=test_idx)

    model_dir = Path(cfg.output_dir) / "final_model"
    model = GCMSConsistencyNet(ds_test.num_batches, cfg).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt",
                                     map_location=device,
                                     weights_only=True))

    proto_store = PrototypeStore()
    proto_dir = model_dir / "prototypes"
    if proto_dir.exists():
        proto_store.load(proto_dir)

    sample = ds_test[sample_idx]
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
                                 "interpret", "compare", "register"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--new_data_dir", type=str, default=None,
                        help="新产品数据目录 (register 命令使用)")
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
    elif args.command == "register":
        if not args.new_data_dir:
            print("错误: register 命令需要 --new_data_dir 参数")
            sys.exit(1)
        cmd_register(cfg, new_data_dir=args.new_data_dir)


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