"""
CLI 入口: 数据准备 → 训练 → 评估 → 解释
用法:
  python main.py prepare
  python main.py train
  python main.py evaluate
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
    from train import train_all_folds
    from evaluate import evaluate_all_folds
    fold_results = train_all_folds(cfg)
    evaluate_all_folds(fold_results, cfg)


def cmd_evaluate(cfg):
    """加载已保存模型进行评估。"""
    from dataset import GCMSDataset, leave_one_batch_out_splits
    from models import GCMSConsistencyNet
    from evaluate import evaluate_fold
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
        # 加载产品编码
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

        with open(fold_dir / "thresholds.json") as f:
            thresholds = {int(k): v for k, v in json.load(f).items()}

        evaluate_fold(model, loader, thresholds, device, fold_name=bname)


def cmd_interpret(cfg, fold_idx=0, sample_idx=0):
    """对指定样本做 Grad-CAM 解释。"""
    from dataset import GCMSDataset, leave_one_batch_out_splits
    from models import GCMSConsistencyNet
    from interpret import GradCAM, find_top_regions, plot_interpretation
    from torch.utils.data import DataLoader

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

    sample = ds_val[sample_idx]
    x = sample["input"].unsqueeze(0).to(device)

    grad_cam = GradCAM(model)
    cam = grad_cam(x)

    # 读取 grid info
    with open(Path(cfg.prepared_dir) / "grid_info.json") as f:
        grid_info = json.load(f)

    rt_range = cfg.rt_range or (0.0, 40.0)
    mz_range = tuple(grid_info.get("mz_range", cfg.mz_range))

    regions = find_top_regions(cam, rt_range, mz_range, top_k=10)
    print(f"\n样本 {sample['sample_id']} 的 Top-10 关键区域:")
    for j, r in enumerate(regions):
        print(f"  {j+1}. RT={r['rt']:.2f} min, "
              f"m/z={r['mz']:.1f}, importance={r['importance']:.3f}")

    x_np = sample["input"].numpy()
    plot_interpretation(x_np, cam, rt_range, mz_range,
                        sample_id=sample["sample_id"],
                        save_dir=Path(cfg.output_dir) / "interpretations")
    print(f"解释图已保存")


def main():
    parser = argparse.ArgumentParser(description="GC-MS 跨批次一致性深度学习流水线")
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "interpret"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--save_plot", dest="save_prepare_plots", action="store_true", default=False,
                        help="数据准备阶段保存 plot 图")
    parser.add_argument("--no-save_plot", dest="save_prepare_plots", action="store_false",
                        help="数据准备阶段不保存 plot 图")
    parser.add_argument("--save_table", dest="save_prepare_tables", action="store_true", default=False,
                        help="数据准备阶段保存 table 表格")
    parser.add_argument("--no-save_table", dest="save_prepare_tables", action="store_false",
                        help="数据准备阶段不保存 table 表格")
    parser.add_argument("--rt_min", type=float, default=3.17, help="手动设置 RT 最小值（min）")
    parser.add_argument("--rt_max", type=float, default=36.91, help="手动设置 RT 最大值（min）")
    parser.add_argument("--mz_min", type=float, default=0, help="手动设置 m/z 最小值")
    parser.add_argument("--mz_max", type=float, default=200, help="手动设置 m/z 最大值")
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


if __name__ == "__main__":
    main()