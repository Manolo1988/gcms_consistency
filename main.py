"""
CLI 入口: 数据准备 → 训练 → 注册 → 评估 → 解释
用法:
  python main.py prepare
  python main.py train [--split_mode closed|open|fewshot]
  python main.py register --fold 0
  python main.py evaluate [--split_mode closed|open|fewshot]
  python main.py interpret --sample_idx 0 --fold 0
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
    """根据 split_mode 执行不同的训练策略。"""
    if cfg.split_mode == "closed":
        _train_closed(cfg)
    elif cfg.split_mode == "open":
        _train_open(cfg)
    elif cfg.split_mode == "fewshot":
        _train_fewshot(cfg)
    else:
        raise ValueError(f"未知 split_mode: {cfg.split_mode}")


def _train_closed(cfg):
    """闭集跨批次: leave-one-batch-out 训练 + 评估。"""
    from train import train_all_folds
    from evaluate import evaluate_all_folds
    fold_results = train_all_folds(cfg)
    evaluate_all_folds(fold_results, cfg)


def _train_open(cfg):
    """开集: 留出部分类，训练后评估开集识别能力。"""
    from train import set_seed, build_loaders, run_fold
    from dataset import open_set_splits, GCMSDataset
    from evaluate import (collect_predictions, classification_metrics,
                          open_set_metrics, batch_robustness_metrics)
    from register import register_from_loader
    from torch.utils.data import DataLoader

    set_seed(cfg.seed)
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = "product_fine" if cfg.product_granularity == "fine" else "product_coarse"

    split = open_set_splits(
        metadata_csv, product_col=product_col,
        num_test_classes=cfg.num_open_test_classes,
        num_val_classes=cfg.num_open_val_classes,
        seed=cfg.seed,
    )

    print(f"\n开集划分:")
    print(f"  训练类 ({len(split['train_classes'])}): {split['train_classes']}")
    print(f"  验证类 ({len(split['val_classes'])}): {split['val_classes']}")
    print(f"  测试类 ({len(split['test_classes'])}): {split['test_classes']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练 (单次, 不做 leave-one-batch-out)
    model, proto_store, ds_train, ds_val, loader_val = run_fold(
        0, split["train_idx"], split["val_known_idx"],
        "open_val", metadata_csv, cfg
    )

    # 评估: 已知类
    known_records = collect_predictions(
        model, loader_val, proto_store, device,
        reject_factor=cfg.reject_threshold_factor
    )
    cls_m = classification_metrics(known_records)
    print(f"\n已知类分类: Acc={cls_m['accuracy']:.4f}, F1={cls_m['macro_f1']:.4f}")

    # 评估: 未知类
    ds_test = GCMSDataset(metadata_csv, product_col=product_col,
                          augmentation=None, indices=split["test_idx"])
    loader_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)
    unknown_records = collect_predictions(
        model, loader_test, proto_store, device,
        reject_factor=cfg.reject_threshold_factor
    )

    # 开集指标
    os_m = open_set_metrics(known_records, unknown_records)
    print(f"\n开集识别: AUROC={os_m['open_set_AUROC']:.4f}, "
          f"F1@FPR5%={os_m['F1_at_FPR5pct']:.4f}")
    print(f"  已知类平均分数: {os_m['known_score_mean']:.4f}")
    print(f"  未知类平均分数: {os_m['unknown_score_mean']:.4f}")

    # 保存结果
    out_dir = Path(cfg.output_dir) / "open_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "split": {k: v for k, v in split.items()
                      if k in ("train_classes", "val_classes", "test_classes")},
            "classification": cls_m,
            "open_set": os_m,
        }, f, indent=2, ensure_ascii=False, default=str)


def _train_fewshot(cfg):
    """少样本: 全类训练，测试时模拟 N-shot 注册。"""
    from train import train_all_folds
    from dataset import few_shot_splits, GCMSDataset
    from evaluate import few_shot_evaluate

    fold_results = train_all_folds(cfg)

    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = "product_fine" if cfg.product_granularity == "fine" else "product_coarse"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fs = few_shot_splits(metadata_csv, product_col=product_col,
                         n_shot_values=cfg.n_shot_values, seed=cfg.seed)

    print(f"\n{'='*60}")
    print(f"少样本评估")
    print(f"{'='*60}")

    all_results = {}
    for n_shot in cfg.n_shot_values:
        evals = fs["fewshot_evals"][n_shot]
        accs = []
        for i, (fr, ev) in enumerate(zip(fold_results, evals)):
            if not ev["test_idx"]:
                continue
            ds = GCMSDataset(metadata_csv, product_col=product_col,
                             augmentation=None)
            label_names = ds.get_label_name_map()
            m = few_shot_evaluate(
                fr["model"], ds, ev["ref_idx"], ev["test_idx"],
                label_names, device, cfg
            )
            accs.append(m["accuracy"])

        mean_acc = np.nanmean(accs) if accs else np.nan
        std_acc = np.nanstd(accs) if accs else np.nan
        print(f"  {n_shot}-shot: Acc = {mean_acc:.4f} ± {std_acc:.4f}")
        all_results[n_shot] = {"mean": float(mean_acc), "std": float(std_acc)}

    out_dir = Path(cfg.output_dir)
    with open(out_dir / "fewshot_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


def cmd_register(cfg, fold_idx=0):
    """从训练集注册原型 (用于新产品快速接入)。"""
    from dataset import GCMSDataset, leave_one_batch_out_splits
    from models import GCMSConsistencyNet
    from register import register_from_loader
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = "product_fine" if cfg.product_granularity == "fine" else "product_coarse"
    splits = leave_one_batch_out_splits(metadata_csv)

    train_idx, val_idx, bname = splits[fold_idx]
    fold_dir = Path(cfg.output_dir) / f"fold_{fold_idx}"

    if not (fold_dir / "model.pt").exists():
        print(f"Fold {fold_idx} 模型不存在")
        return

    # 加载模型
    with open(fold_dir / "product_classes.json") as f:
        classes = json.load(f)

    ds = GCMSDataset(metadata_csv, product_col=product_col,
                     augmentation=None, indices=train_idx)
    model = GCMSConsistencyNet(
        len(classes), ds.num_batches, cfg
    ).to(device)
    model.load_state_dict(torch.load(fold_dir / "model.pt",
                                     map_location=device,
                                     weights_only=True))

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    label_names = ds.get_label_name_map()

    proto_store, _, _ = register_from_loader(
        model, loader, label_names, device,
        percentile=cfg.accept_percentile
    )
    proto_store.save(fold_dir / "prototypes")
    proto_store.summary()
    print(f"\n原型已保存到 {fold_dir / 'prototypes'}")


def cmd_evaluate(cfg):
    """加载已保存模型进行评估。"""
    from dataset import GCMSDataset, leave_one_batch_out_splits
    from models import GCMSConsistencyNet
    from evaluate import evaluate_fold
    from register import PrototypeStore
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    splits = leave_one_batch_out_splits(metadata_csv)
    product_col = "product_fine" if cfg.product_granularity == "fine" else "product_coarse"

    for i, (train_idx, val_idx, bname) in enumerate(splits):
        fold_dir = Path(cfg.output_dir) / f"fold_{i}"
        if not (fold_dir / "model.pt").exists():
            print(f"Fold {i} 模型不存在，跳过")
            continue

        ds_val = GCMSDataset(metadata_csv, product_col=product_col,
                             augmentation=None, indices=val_idx)
        with open(fold_dir / "product_classes.json") as f:
            classes = json.load(f)
        ds_val.product_enc.classes_ = np.array(classes)

        loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

        model = GCMSConsistencyNet(
            len(classes), len(ds_val.batch_enc.classes_), cfg
        ).to(device)
        model.load_state_dict(torch.load(fold_dir / "model.pt",
                                         map_location=device,
                                         weights_only=True))

        # 加载原型库
        proto_store = PrototypeStore()
        proto_dir = fold_dir / "prototypes"
        if proto_dir.exists():
            proto_store.load(proto_dir)
        else:
            print(f"  警告: 原型库不存在, 使用辅助分类头进行评估")
            # fallback: 从验证集自注册
            from register import register_from_loader
            label_names = {i: c for i, c in enumerate(classes)}
            proto_store, _, _ = register_from_loader(
                model, loader, label_names, device,
                percentile=cfg.accept_percentile
            )

        evaluate_fold(model, loader, proto_store, device,
                      fold_name=bname,
                      save_dir=Path(cfg.output_dir) / "visualizations",
                      reject_factor=cfg.reject_threshold_factor)


def cmd_interpret(cfg, fold_idx=0, sample_idx=0):
    """对指定样本做 Grad-CAM 解释 (支持嵌入距离模式)。"""
    from dataset import GCMSDataset, leave_one_batch_out_splits
    from models import GCMSConsistencyNet
    from interpret import GradCAM, find_top_regions, plot_interpretation
    from register import PrototypeStore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    splits = leave_one_batch_out_splits(metadata_csv)

    train_idx, val_idx, bname = splits[fold_idx]
    product_col = "product_fine" if cfg.product_granularity == "fine" else "product_coarse"
    ds_val = GCMSDataset(metadata_csv, product_col=product_col,
                         augmentation=None, indices=val_idx)

    fold_dir = Path(cfg.output_dir) / f"fold_{fold_idx}"
    with open(fold_dir / "product_classes.json") as f:
        classes = json.load(f)

    model = GCMSConsistencyNet(
        len(classes), len(ds_val.batch_enc.classes_), cfg
    ).to(device)
    model.load_state_dict(torch.load(fold_dir / "model.pt",
                                     map_location=device,
                                     weights_only=True))

    # 加载原型库
    proto_store = PrototypeStore()
    proto_dir = fold_dir / "prototypes"
    if proto_dir.exists():
        proto_store.load(proto_dir)

    sample = ds_val[sample_idx]
    x = sample["input"].unsqueeze(0).to(device)

    # 先获取预测结果
    z = model.encode(x)
    pred_result = proto_store.predict(z) if proto_store.num_classes > 0 else None

    # Grad-CAM (优先使用嵌入距离模式)
    if pred_result and proto_store.num_classes > 0:
        pred_class = pred_result["pred_class"][0]
        score = pred_result["scores"][0].item()
        target_proto = proto_store.prototypes[pred_class]
        grad_cam = GradCAM(model, mode="embedding")
        cam = grad_cam(x, target_proto=target_proto)
    else:
        pred_class = None
        score = None
        grad_cam = GradCAM(model, mode="logits")
        cam = grad_cam(x)

    # 读取 grid info
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
        description="GC-MS 跨批次一致性深度学习流水线 (度量学习框架)"
    )
    parser.add_argument("command",
                        choices=["prepare", "train", "register",
                                 "evaluate", "interpret"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sample_idx", type=int, default=0)

    # 实验模式
    parser.add_argument("--split_mode", type=str, default="closed",
                        choices=["closed", "open", "fewshot"],
                        help="数据划分策略: closed(闭集跨批次), "
                             "open(开集留类), fewshot(少样本)")

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

    cfg.split_mode = args.split_mode
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
    elif args.command == "register":
        cmd_register(cfg, fold_idx=args.fold)
    elif args.command == "evaluate":
        cmd_evaluate(cfg)
    elif args.command == "interpret":
        cmd_interpret(cfg, fold_idx=args.fold, sample_idx=args.sample_idx)


if __name__ == "__main__":
    main()