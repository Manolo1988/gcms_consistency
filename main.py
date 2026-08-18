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

from config import Config


def _parse_int_tuple(value: str):
        parts = [p.strip() for p in str(value).split(",") if p.strip()]
        if not parts:
                raise ValueError("encoder_channels 不能为空")
        return tuple(int(p) for p in parts)


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


def _persist_run_metadata(cfg, args):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = out_dir / "run_config.json"
    timestamp = datetime.now().isoformat(timespec="seconds")

    if args.command == "evaluate" and run_config_path.exists():
        with open(run_config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["last_evaluate"] = {
            "timestamp": timestamp,
            "args": vars(args),
        }
    else:
        payload = {
            "command": args.command,
            "timestamp": timestamp,
            "args": vars(args),
            "config": cfg.__dict__,
        }

    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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


def cmd_summarize_runs(cfg, sort_by="b_auroc", limit=20):
    """汇总多个 run 的 evaluation_summary.json, 方便按指标挑模型。"""
    import glob

    root = Path(cfg.output_dir)
    patterns = [
        str(root / "**" / "evaluation_summary.json"),
        str(root / "**" / "final_model" / "evaluation_summary.json"),
    ]
    paths = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})

    rows = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception as exc:
            print(f"[skip] {p}: {exc}")
            continue

        a = s.get("setting_a", {}) or {}
        b = s.get("setting_b", {}) or {}
        c = s.get("setting_c", {}) or {}
        c3 = c.get("3", {}) or c.get(3, {}) or {}
        blend = s.get("setting_b_score_blend_used", {}) or {}
        run_dir = Path(p).parent
        if run_dir.name == "final_model":
            run_dir = run_dir.parent
        rows.append({
            "path": str(run_dir),
            "a_acc": float(a.get("accuracy", float("nan"))),
            "a_macro": float(a.get("macro_f1", float("nan"))),
            "b_auroc": float(b.get("open_set_AUROC", float("nan"))),
            "b_fpr95": float(b.get("FPR_at_95TPR", float("nan"))),
            "c3_acc": float(c3.get("accuracy", float("nan"))),
            "blend": f"{blend.get('base_weight', float('nan')):.2f}/"
                     f"{blend.get('margin_weight', float('nan')):.2f}"
                     if blend else "-",
        })

    sort_key = str(sort_by or "b_auroc").lower()
    reverse = sort_key != "b_fpr95"
    rows = [r for r in rows if sort_key in r and np.isfinite(r[sort_key])]
    rows.sort(key=lambda r: r[sort_key], reverse=reverse)
    rows = rows[:max(int(limit or 20), 1)]

    print(f"\n共找到 {len(paths)} 个 evaluation_summary.json")
    print(f"按 {sort_key} {'降序' if reverse else '升序'}展示前 {len(rows)} 个:")
    print("rank  A_acc   A_F1    B_AUROC  B_FPR95  C_3shot  blend     run")
    for i, r in enumerate(rows, 1):
        print(
            f"{i:>4d}  {r['a_acc']:.4f}  {r['a_macro']:.4f}  "
            f"{r['b_auroc']:.4f}   {r['b_fpr95']:.4f}   "
            f"{r['c3_acc']:.4f}   {r['blend']:>7s}   {r['path']}"
        )


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

    input_transform = None
    input_pca_path = model_dir / "input_rt_pca.pkl"
    if input_pca_path.exists():
        from input_pca import load_rt_axis_pca, RtAxisPcaTransform

        input_pca_model = load_rt_axis_pca(input_pca_path)
        input_transform = RtAxisPcaTransform(input_pca_model)
        cfg.mz_bins = int(getattr(input_pca_model, "n_components_", cfg.mz_bins))

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
                                 "interpret", "compare", "register",
                                 "summarize_runs"])
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
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="权重衰减")
    parser.add_argument("--lambda_adv", type=float, default=None,
                        help="批次对抗损失权重")
    parser.add_argument("--lambda_supcon",type=float,default=None,
                        help="监督对比损失权重")
    parser.add_argument("--lambda_proto", type=float, default=None,
                        help="原型紧凑损失权重")
    parser.add_argument("--lambda_recon", type=float, default=None,
                        help="重建正则损失权重")
    parser.add_argument("--lambda_cls", type=float, default=None,
                        help="分类辅助头损失权重")
    parser.add_argument("--lambda_hard_pair", type=float, default=None,
                        help="易混产品对 hard margin 权重")
    parser.add_argument("--supcon_temperature", type=float, default=None,
                        help="SupCon 温度参数")
    parser.add_argument("--embed_dim", "--feature_dim", dest="feature_dim",
                        type=int, default=None,
                        help="最终 embedding 维度")
    parser.add_argument("--proj_dim", type=int, default=None,
                        help="SupCon 投影头维度")
    parser.add_argument("--hard_pair_margin", type=float, default=None,
                        help="易混产品对 hard margin 距离")
    parser.add_argument("--accept_percentile", type=float, default=None,
                        help="一致性半径百分位")
    parser.add_argument("--reject_threshold_factor", type=float, default=None,
                        help="开集拒识半径倍率")
    parser.add_argument("--open_score_base_weight", type=float, default=None,
                        help="开集分数 base score 权重")
    parser.add_argument("--open_score_margin_weight", type=float, default=None,
                        help="开集分数 margin score 权重")
    parser.add_argument("--no_auto_open_score_blend", action="store_true",
                        help="关闭Setting B自动选择最佳open-score混合")
    parser.add_argument("--open_score_calibration_products", type=str, default=None,
                        help="开发期伪未知产品，逗号分隔；不使用最终留出产品")
    parser.add_argument("--disable_open_score_calibration_apply", action="store_true",
                        help="仅报告伪未知校准开发结果，不应用到最终 Setting B")
    parser.add_argument("--eval_interval", type=int, default=None,
                        help="验证间隔 epoch")
    parser.add_argument("--eval_interval_search", type=int, default=None,
                        help="搜索阶段验证间隔")
    parser.add_argument("--eval_interval_final", type=int, default=None,
                        help="后期收敛阶段验证间隔")
    parser.add_argument("--eval_final_start_ratio", type=float, default=None,
                        help="切换到后期收敛验证间隔的训练进度比例")
    parser.add_argument("--model_select_metric", type=str, default=None,
                        choices=["metric", "acc", "auroc", "auroc_correct"],
                        help="训练中 best checkpoint 的选择指标")
    parser.add_argument("--model_select_min_epoch", type=int, default=None,
                        help="选模最小 epoch (只在该 epoch 之后才允许更新 best checkpoint)")
    parser.add_argument("--swa_enabled", action="store_true",
                        help="启用 SWA 多 checkpoint 权重平均 (替代单点 best)")
    parser.add_argument("--swa_start_epoch", type=int, default=None,
                        help="SWA 平均起始 epoch (默认 100)")
    parser.add_argument("--early_stop_patience", type=int, default=None,
                        help="早停耐心")
    parser.add_argument("--min_epochs_before_early_stop", type=int, default=None,
                        help="允许早停前最少训练 epoch")
    parser.add_argument("--min_epoch_ratio_before_early_stop", type=float, default=None,
                        help="允许早停前最少训练比例")
    parser.add_argument("--early_stop_min_lr_ratio", type=float, default=None,
                        help="仅当 lr 下降到初始 lr*ratio 后允许早停")
    parser.add_argument("--early_stop_min_delta", type=float, default=None,
                        help="判定提升的最小增量")
    parser.add_argument("--proto_val_subset_ratio", type=float, default=None,
                        help="训练中期验证时原型构建使用的训练子集比例")
    parser.add_argument("--proto_val_subset_min_samples", type=int, default=None,
                        help="训练中期验证原型子集最少样本数")
    parser.add_argument("--proto_val_subset_max_samples", type=int, default=None,
                        help="训练中期验证原型子集最多样本数")
    parser.add_argument("--proto_val_full_every", type=int, default=None,
                        help="每隔 N 次验证执行一次全量原型验证")
    parser.add_argument("--open_score_blend_objective", type=str, default=None,
                        choices=["fpr95", "auroc", "balanced"],
                        help="Setting B 自动 blend 的优化目标")
    parser.add_argument("--warmup_guard_enabled", action="store_true",
                        help="启用前期 warmup guard 淘汰")
    parser.add_argument("--warmup_guard_epoch", type=int, default=None,
                        help="warmup guard 对比轮次")
    parser.add_argument("--warmup_guard_best_at_epoch", type=float, default=None,
                        help="warmup guard 参考最优方案在指定轮次的 val_acc")
    parser.add_argument("--warmup_guard_compare_best", action="store_true",
                        help="warmup guard 直接对比当前最优运行")
    parser.add_argument("--warmup_guard_no_compare_best", action="store_true",
                        help="warmup guard 改为按比例阈值比较")
    parser.add_argument("--warmup_guard_min_ratio", type=float, default=None,
                        help="warmup guard 的最小比例阈值")
    parser.add_argument("--dataloader_workers", type=int, default=None,
                        help="DataLoader workers 数")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=None,
                        help="DataLoader prefetch factor")
    parser.add_argument("--disable_dataloader_pin_memory", action="store_true",
                        help="关闭 DataLoader pin_memory")
    parser.add_argument("--disable_dataloader_persistent_workers", action="store_true",
                        help="关闭 DataLoader persistent_workers")
    parser.add_argument("--deterministic", action="store_true",
                        help="启用严格可复现训练：固定 worker/采样/CUDA 随机路径，关闭 AMP/TF32")
    parser.add_argument("--main_backbone", type=str, default=None,
                        help="主干模型: gcms/transformer/resnet18/resnet50/wide_resnet50_2")
    parser.add_argument("--main_backbone_model", type=str, default=None,
                        help="主干模型预训练权重路径")
    parser.add_argument("--main_feature_layers", type=str, default=None,
                        help="主干特征层, 逗号分隔")
    parser.add_argument("--main_feature_fuse", type=str, default=None,
                        help="主干多层特征融合方式 concat/last")
    parser.add_argument("--transformer_patch_size", type=int, default=None,
                        help="Transformer patch size")
    parser.add_argument("--transformer_embed_dim", type=int, default=None,
                        help="Transformer token 维度")
    parser.add_argument("--transformer_depth", type=int, default=None,
                        help="Transformer block 层数")
    parser.add_argument("--transformer_num_heads", type=int, default=None,
                        help="Transformer 注意力头数")
    parser.add_argument("--transformer_mlp_ratio", type=float, default=None,
                        help="Transformer FFN 扩展比例")
    parser.add_argument("--encoder_channels", type=str, default=None,
                        help="编码器通道, 逗号分隔, 如 32,64,128,256")
    parser.add_argument("--blocks_per_stage", type=int, default=None,
                        help="每阶段 ResBlock 数")
    parser.add_argument("--num_axial_heads", type=int, default=None,
                        help="双轴注意力头数")
    parser.add_argument("--dropout", type=float, default=None,
                        help="主干 dropout")
    parser.add_argument("--primary_model", type=str, default=None,
                        help="主算法: deep_consistency/raw_pca_mlp")
    parser.add_argument("--raw_pca_components", type=int, default=None,
                        help="raw_pca_mlp 的 PCA 维度")
    parser.add_argument("--raw_pca_hidden", type=str, default=None,
                        help="raw_pca_mlp 隐藏层, 逗号分隔")
    parser.add_argument("--raw_pca_max_iter", type=int, default=None,
                        help="raw_pca_mlp 最大迭代次数")
    parser.add_argument("--raw_pca_alpha", type=float, default=None,
                        help="raw_pca_mlp alpha")
    parser.add_argument("--raw_pca_lr_init", type=float, default=None,
                        help="raw_pca_mlp 初始学习率")
    parser.add_argument("--input_raw_pca_components", type=int, default=None,
                        help="deep_consistency 输入 PCA 维度")
    parser.add_argument("--enable_input_raw_pca", action="store_true",
                        help="启用输入 PCA 通路")
    parser.add_argument("--disable_input_raw_pca", action="store_true",
                        help="关闭输入 PCA 通路")
    parser.add_argument("--rt_bins", type=int, default=None,
                        help="覆盖 RT bins")
    parser.add_argument("--mz_bins", type=int, default=None,
                        help="覆盖 m/z bins")
    parser.add_argument("--aug_peak_broaden_prob", type=float, default=None,
                        help="峰展宽增强触发概率")
    parser.add_argument("--aug_rt_warp_prob", type=float, default=None,
                        help="RT 扭曲增强触发概率")
    parser.add_argument("--sort_by", type=str, default="b_auroc",
                        choices=["a_acc", "a_macro", "b_auroc", "b_fpr95", "c3_acc"],
                        help="summarize_runs 排序指标")
    parser.add_argument("--limit", type=int, default=20,
                        help="summarize_runs 展示条数")
    parser.add_argument("--stress_test_batches", type=str, default=None,
                        help="额外压力测试批次, 逗号分隔")
    parser.add_argument("--no_auto_create_split_on_train", action="store_true",
                        help="训练前不自动刷新 split.json")
    parser.add_argument("--skip_readme_baselines", action="store_true",
                        help="评估时跳过 README baselines")
    parser.add_argument("--fewshot_repeats", type=int, default=None,
                        help="每个 N-shot 重复抽样次数 (1=不重复, >1 取平均以降低少样本评估噪声)")

    # 数据准备选项
    parser.add_argument("--save_plot", dest="save_prepare_plots",
                        action="store_true", default=False)
    parser.add_argument("--no-save_plot", dest="save_prepare_plots",
                        action="store_false")
    parser.add_argument("--save_table", dest="save_prepare_tables",
                        action="store_true", default=False)
    parser.add_argument("--no-save_table", dest="save_prepare_tables",
                        action="store_false")
    parser.add_argument("--save_visualizations", dest="evaluate_save_visualizations",
                        action="store_true", default=None,
                        help="评估时保存 t-SNE/score distribution 图")
    parser.add_argument("--no_save_visualizations", dest="evaluate_save_visualizations",
                        action="store_false",
                        help="评估时不保存可视化图")

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
        "seed", "epochs", "batch_size", "lr", "weight_decay",
        "lambda_supcon", "lambda_adv", "lambda_proto", "lambda_recon", "lambda_cls",
        "lambda_hard_pair", "hard_pair_margin",
        "supcon_temperature", "feature_dim", "proj_dim",
        "accept_percentile", "reject_threshold_factor",
        "open_score_base_weight", "open_score_margin_weight",
        "open_score_calibration_products",
        "eval_interval_search", "eval_interval_final", "eval_final_start_ratio",
        "early_stop_patience", "min_epochs_before_early_stop",
        "min_epoch_ratio_before_early_stop", "early_stop_min_lr_ratio",
        "early_stop_min_delta", "proto_val_subset_ratio",
        "proto_val_subset_min_samples", "proto_val_subset_max_samples",
        "proto_val_full_every", "dataloader_workers", "dataloader_prefetch_factor",
        "warmup_guard_epoch", "warmup_guard_best_at_epoch", "warmup_guard_min_ratio",
        "main_backbone", "main_backbone_model", "main_feature_layers", "main_feature_fuse",
        "transformer_patch_size", "transformer_embed_dim", "transformer_depth",
        "transformer_num_heads", "transformer_mlp_ratio",
        "blocks_per_stage", "num_axial_heads", "dropout",
        "primary_model", "raw_pca_components", "raw_pca_hidden",
        "raw_pca_max_iter", "raw_pca_alpha", "raw_pca_lr_init",
        "input_raw_pca_components", "rt_bins", "mz_bins",
        "aug_peak_broaden_prob", "aug_rt_warp_prob",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.eval_interval is not None:
        cfg.eval_interval = int(args.eval_interval)
        cfg.eval_interval_search = int(args.eval_interval)
        cfg.eval_interval_final = int(args.eval_interval)
    if args.model_select_metric is not None:
        cfg.model_select_metric = args.model_select_metric
    if args.model_select_min_epoch is not None:
        cfg.model_select_min_epoch = int(args.model_select_min_epoch)
    if args.swa_enabled:
        cfg.swa_enabled = True
    if args.swa_start_epoch is not None:
        cfg.swa_start_epoch = int(args.swa_start_epoch)
    if args.open_score_blend_objective is not None:
        cfg.open_score_blend_objective = args.open_score_blend_objective
    if args.stress_test_batches is not None:
        cfg.stress_test_batches = tuple(
            b.strip() for b in args.stress_test_batches.split(",")
            if b.strip()
        )
    if args.no_auto_create_split_on_train:
        cfg.auto_create_split_on_train = False
    if args.no_auto_open_score_blend:
        cfg.open_score_auto_blend = False
    if args.disable_open_score_calibration_apply:
        cfg.open_score_calibration_apply = False
    if args.skip_readme_baselines:
        cfg.evaluate_readme_baselines = False
    if args.fewshot_repeats is not None:
        cfg.fewshot_repeats = int(args.fewshot_repeats)
    if args.enable_input_raw_pca:
        cfg.input_raw_pca_enabled = True
    if args.disable_input_raw_pca:
        cfg.input_raw_pca_enabled = False
    if args.disable_dataloader_pin_memory:
        cfg.dataloader_pin_memory = False
    if args.disable_dataloader_persistent_workers:
        cfg.dataloader_persistent_workers = False
    if args.deterministic:
        cfg.deterministic = True
    if args.warmup_guard_enabled:
        cfg.warmup_guard_enabled = True
    if args.warmup_guard_compare_best:
        cfg.warmup_guard_compare_best = True
    if args.warmup_guard_no_compare_best:
        cfg.warmup_guard_compare_best = False
    if args.encoder_channels is not None:
        cfg.encoder_channels = _parse_int_tuple(args.encoder_channels)

    cfg.save_prepare_plots = bool(args.save_prepare_plots)
    cfg.save_prepare_tables = bool(args.save_prepare_tables)
    if args.evaluate_save_visualizations is not None:
        cfg.evaluate_save_visualizations = bool(args.evaluate_save_visualizations)

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
    output_name = Path(cfg.output_dir).name
    explicit_run_dir = (
        output_name.startswith("run_")
        or output_name.startswith("run_seed")
        or output_name.startswith("iter_auto")
    )
    if args.command in ("prepare", "train") and not explicit_run_dir:
        import time as _time
        cfg.output_dir = str(
            Path(cfg.output_dir) / f"run_{_time.strftime('%Y%m%d_%H%M%S')}"
        )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    if args.command in ("prepare", "train", "evaluate"):
        _persist_run_metadata(cfg, args)

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
    elif args.command == "summarize_runs":
        cmd_summarize_runs(cfg, sort_by=args.sort_by, limit=args.limit)


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
