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
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

from config import Config, apply_tic_config, config_to_dict


class _TeeStream:
    """Write console output to both terminal and a run-local log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, s):
        self.stream.write(s)
        self.log_file.write(s)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def _run_with_log(cfg, log_name, fn):
    """Run a command and save stdout/stderr into cfg.output_dir/log_name."""
    log_path = Path(cfg.output_dir) / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_config_path = Path(cfg.output_dir) / "run_config.json"
    if not run_config_path.exists():
        with open(run_config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "config": config_to_dict(cfg),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    with open(log_path, "w", encoding="utf-8") as log_f:
        sys.stdout = _TeeStream(orig_stdout, log_f)
        sys.stderr = _TeeStream(orig_stderr, log_f)
        try:
            return fn()
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            print(f"日志已保存到 {log_path}")


def cmd_prepare(cfg):
    def _impl():
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

    return _run_with_log(cfg, "prepare.log", _impl)


def cmd_train(cfg):
    """训练单一最终模型。"""
    from train import train_single_model
    return _run_with_log(cfg, "train.log", lambda: train_single_model(cfg))


def cmd_evaluate(cfg):
    """加载已保存模型, 运行 Setting A/B/C 评估。"""
    from evaluate import evaluate_single_model
    return _run_with_log(cfg, "eval.log", lambda: evaluate_single_model(cfg))


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

    with open(model_dir / "train_meta.json") as f:
        meta = json.load(f)
    apply_tic_config(cfg, meta)
    if bool(meta.get("input_raw_pca_enabled", False)):
        cfg.mz_bins = int(meta.get("input_raw_pca_components", cfg.mz_bins))
    cfg.rt_bins = int(meta.get("input_raw_pca_rt_bins", cfg.rt_bins))

    input_transform = None
    input_pca_path = model_dir / "input_rt_pca.pkl"
    if input_pca_path.exists():
        from input_pca import load_rt_axis_pca, RtAxisPcaTransform

        input_pca_model = load_rt_axis_pca(input_pca_path)
        input_transform = RtAxisPcaTransform(input_pca_model)
        cfg.mz_bins = int(getattr(input_pca_model, "n_components_", cfg.mz_bins))

    # 加载已训练模型
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
                         indices=split["train_idx"],
                         input_transform=input_transform)
    old_label_names = ds_old.get_label_name_map()
    loader_old = DataLoader(ds_old, batch_size=cfg.batch_size,
                            shuffle=True, num_workers=0)

    # 加载新产品数据
    new_metadata_csv = str(Path(new_data_dir) / "metadata.csv")
    ds_new = GCMSDataset(new_metadata_csv, product_col=product_col,
                         augmentation=GCMSAugmentation(cfg),
                         input_transform=input_transform)
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

    if bool(getattr(cfg, "tic_branch_enabled", False)):
        for name, ds in (("old replay", ds_old), ("new product", ds_new)):
            tic_dim = ds.get_tic_dim()
            if tic_dim is None:
                raise RuntimeError(
                    f"TIC branch is enabled, but {name} metadata has no usable "
                    "tic_pca_path. Prepare that dataset with --enable_tic_branch."
                )
            if int(tic_dim) != int(cfg.tic_pca_components):
                raise RuntimeError(
                    f"TIC feature dim mismatch for {name}: metadata has {tic_dim}, "
                    f"model expects {cfg.tic_pca_components}."
                )

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
    model_dir = Path(cfg.output_dir) / "final_model"

    meta_path = model_dir / "train_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        apply_tic_config(cfg, meta)
        if bool(meta.get("input_raw_pca_enabled", False)):
            cfg.mz_bins = int(meta.get("input_raw_pca_components", cfg.mz_bins))
        cfg.rt_bins = int(meta.get("input_raw_pca_rt_bins", cfg.rt_bins))

    input_transform = None
    input_pca_path = model_dir / "input_rt_pca.pkl"
    if input_pca_path.exists():
        from input_pca import load_rt_axis_pca, RtAxisPcaTransform

        input_pca_model = load_rt_axis_pca(input_pca_path)
        input_transform = RtAxisPcaTransform(input_pca_model)
        cfg.mz_bins = int(getattr(input_pca_model, "n_components_", cfg.mz_bins))

    # 使用 Setting A 测试集 (留出批次) 作为解释对象
    test_idx = split["test_batch_idx"] or split["val_idx"]
    ds_test = GCMSDataset(metadata_csv, product_col=product_col,
                          augmentation=None, indices=test_idx,
                          input_transform=input_transform)

    num_batches_model = ds_test.num_batches
    num_products_model = None
    if meta:
        num_batches_model = int(meta.get("num_batches", num_batches_model))
        num_products_model = meta.get("num_products")

    model = GCMSConsistencyNet(num_batches_model, cfg, num_products=num_products_model).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt",
                                     map_location=device,
                                     weights_only=True))

    proto_store = PrototypeStore()
    proto_dir = model_dir / "prototypes"
    if proto_dir.exists():
        proto_store.load(proto_dir)

    sample = ds_test[sample_idx]
    x = sample["input"].unsqueeze(0).to(device)
    tic = sample.get("tic")
    if tic is not None:
        tic = tic.unsqueeze(0).to(device)

    z = model.encode(x, tic=tic)
    pred_result = proto_store.predict(z) if proto_store.num_classes > 0 else None

    # Grad-CAM (仅使用嵌入距离模式)
    if pred_result and proto_store.num_classes > 0:
        pred_class = pred_result["pred_class"][0]
        score = pred_result["scores"][0].item()
        target_proto = proto_store.prototypes[pred_class]
        grad_cam = GradCAM(model, mode="embedding")
        cam = grad_cam(x, tic=tic, target_proto=target_proto)
    else:
        pred_class = None
        score = None
        grad_cam = GradCAM(model, mode="embedding")
        cam = grad_cam(x, tic=tic)

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
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录。prepare/train 默认会在该目录下创建 run_时间戳 子目录")
    parser.add_argument("--prepared_dir", type=str, default=None,
                        help="prepared data 目录")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="训练/评估 batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率")
    parser.add_argument("--lambda_hard_pair", type=float, default=None,
                        help="易混产品对 hard margin 权重")
    parser.add_argument("--lambda_tic_residual", type=float, default=None,
                        help="TIC 残差幅度正则权重")
    parser.add_argument("--lambda_tic_anchor", type=float, default=None,
                        help="融合嵌入贴近主干嵌入的正则权重")
    parser.add_argument("--hard_pair_margin", type=float, default=None,
                        help="易混产品对 hard margin 距离")
    parser.add_argument("--open_score_base_weight", type=float, default=None,
                        help="开集分数 base score 权重")
    parser.add_argument("--open_score_margin_weight", type=float, default=None,
                        help="开集分数 margin score 权重")
    parser.add_argument("--eval_interval", type=int, default=None,
                        help="验证间隔 epoch")
    parser.add_argument("--stress_test_batches", type=str, default=None,
                        help="额外压力测试批次, 逗号分隔")
    parser.add_argument("--no_auto_create_split_on_train", action="store_true",
                        help="训练前不自动刷新 split.json")
    parser.add_argument("--skip_readme_baselines", action="store_true",
                        help="评估时跳过 README baselines")
    parser.add_argument("--enable_tic_branch", action="store_true",
                        help="启用 TIC 辅助分支")
    parser.add_argument("--disable_tic_branch", action="store_true",
                        help="禁用 TIC 辅助分支")
    parser.add_argument("--tic_source", type=str, default=None,
                        choices=["from_tensor", "raw_file"],
                        help="TIC 来源标记")
    parser.add_argument("--tic_encoder", type=str, default=None,
                        choices=["mlp", "cnn1d", "cnn", "transformer"],
                        help="TIC 编码器")
    parser.add_argument("--tic_embed_dim", type=int, default=None,
                        help="TIC 编码器输出维度")
    parser.add_argument("--tic_fusion_mode", type=str, default=None,
                        choices=[
                            "orthogonal_residual", "orthogonal",
                            "residual_gated", "film", "concat", "gated", "sum",
                        ],
                        help="TIC 与主嵌入融合方式")
    parser.add_argument("--tic_fusion_output_dim", type=int, default=None,
                        help="TIC 融合后嵌入维度")
    parser.add_argument("--tic_pca_components", type=int, default=None,
                        help="TIC PCA 特征维度")
    parser.add_argument("--aug_tic_jitter", type=float, default=None,
                        help="TIC 增强抖动幅度")
    parser.add_argument("--tic_residual_scale", type=float, default=None,
                        help="TIC residual/FiLM 最大影响强度")
    parser.add_argument("--tic_gate_bias", type=float, default=None,
                        help="residual_gated 初始门控偏置")
    parser.add_argument("--tic_warmup_epochs", type=int, default=None,
                        help="TIC 残差线性放开的 epoch 数")
    parser.add_argument("--tic_residual_dropout", type=float, default=None,
                        help="训练时 TIC 残差随机丢弃概率")

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

    if args.output_dir:
        cfg.output_dir = str(Path(args.output_dir))
    if args.prepared_dir:
        cfg.prepared_dir = str(Path(args.prepared_dir))
    for name in (
        "seed", "epochs", "batch_size", "lr",
        "lambda_hard_pair", "lambda_tic_residual", "lambda_tic_anchor",
        "hard_pair_margin",
        "open_score_base_weight", "open_score_margin_weight",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.eval_interval is not None:
        cfg.eval_interval = int(args.eval_interval)
        cfg.eval_interval_search = int(args.eval_interval)
        cfg.eval_interval_final = int(args.eval_interval)
    if args.stress_test_batches is not None:
        cfg.stress_test_batches = tuple(
            b.strip() for b in args.stress_test_batches.split(",")
            if b.strip()
        )
    if args.no_auto_create_split_on_train:
        cfg.auto_create_split_on_train = False
    if args.skip_readme_baselines:
        cfg.evaluate_readme_baselines = False
    if args.enable_tic_branch:
        cfg.tic_branch_enabled = True
    if args.disable_tic_branch:
        cfg.tic_branch_enabled = False
    for name in (
        "tic_source",
        "tic_encoder",
        "tic_embed_dim",
        "tic_fusion_mode",
        "tic_fusion_output_dim",
        "tic_pca_components",
        "tic_residual_scale",
        "tic_gate_bias",
        "tic_warmup_epochs",
        "tic_residual_dropout",
        "aug_tic_jitter",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

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

    if args.command == "evaluate" and Path(cfg.output_dir).name == "final_model":
        cfg.output_dir = str(Path(cfg.output_dir).parent)
    if args.command in ("prepare", "train"):
        import time as _time
        cfg.output_dir = str(
            Path(cfg.output_dir) / f"run_{_time.strftime('%Y%m%d_%H%M%S')}"
        )

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
