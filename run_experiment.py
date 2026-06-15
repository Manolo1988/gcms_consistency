"""Experiment runner with per-run output dir and hyperparameter overrides.

Usage example:
  /home/ubuntu/sunlong/gcms_consistency/.venv/bin/python run_experiment.py \
    --name iter2_tuned --epochs 60 --batch_size 16 --lr 2e-4 \
    --lambda_adv 0.2 --lambda_proto 0.8
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from config import Config
from train import train_single_model
from evaluate import evaluate_single_model


def _parse_int_tuple(value: str):
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError("encoder_channels 不能为空")
    return tuple(int(p) for p in parts)


def _append_progress(log_path: Path, lines):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n")
        for line in lines:
            f.write(line + "\n")


def _load_summary(summary_path: Path):
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run one GCMS experiment")
    parser.add_argument("--name", required=True,
                        help="Experiment name; outputs to outputs/<name>")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lambda_adv", type=float, default=None)
    parser.add_argument("--lambda_proto", type=float, default=None)
    parser.add_argument("--lambda_recon", type=float, default=None)
    parser.add_argument("--supcon_temperature", type=float, default=None)
    parser.add_argument("--reject_threshold_factor", type=float, default=None)
    parser.add_argument("--accept_percentile", type=float, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_interval_search", type=int, default=None)
    parser.add_argument("--eval_interval_final", type=int, default=None)
    parser.add_argument("--eval_final_start_ratio", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--min_epochs_before_early_stop", type=int, default=None)
    parser.add_argument("--min_epoch_ratio_before_early_stop", type=float, default=None)
    parser.add_argument("--early_stop_min_lr_ratio", type=float, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument("--proto_val_subset_ratio", type=float, default=None)
    parser.add_argument("--proto_val_subset_min_samples", type=int, default=None)
    parser.add_argument("--proto_val_subset_max_samples", type=int, default=None)
    parser.add_argument("--proto_val_full_every", type=int, default=None)
    parser.add_argument("--dataloader_workers", type=int, default=None)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=None)
    parser.add_argument("--disable_dataloader_pin_memory", action="store_true")
    parser.add_argument("--disable_dataloader_persistent_workers", action="store_true")
    parser.add_argument("--enable_dataset_cache", action="store_true",
                        help="Cache loaded npz tensors inside each Dataset worker")
    parser.add_argument("--dataset_cache_max_items", type=int, default=None)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default=None,
                        help="float16/bfloat16")
    parser.add_argument("--disable_channels_last", action="store_true")
    parser.add_argument("--enable_torch_compile", action="store_true")
    parser.add_argument("--disable_cuda_benchmark", action="store_true")
    parser.add_argument("--warmup_guard_enabled", action="store_true",
                        help="Enable warmup guard against current best strategy")
    parser.add_argument("--warmup_guard_epoch", type=int, default=None)
    parser.add_argument("--warmup_guard_best_at_epoch", type=float, default=None)
    parser.add_argument("--warmup_guard_compare_best", action="store_true",
                        help="Compare directly with current best completed iteration")
    parser.add_argument("--warmup_guard_no_compare_best", action="store_true",
                        help="Disable direct compare-best mode and use ratio threshold")
    parser.add_argument("--warmup_guard_min_ratio", type=float, default=None)
    parser.add_argument("--pretrained_feature_model", type=str, default=None,
                        help="Local pretrained weight path for baseline feature extraction")
    parser.add_argument("--pretrained_feature_arch", type=str, default=None,
                        help="auto/resnet18/resnet50/wide_resnet50_2")
    parser.add_argument("--main_backbone", type=str, default=None,
                        help="gcms/transformer/resnet18/resnet50/wide_resnet50_2")
    parser.add_argument("--main_backbone_model", type=str, default=None,
                        help="Local pretrained weight path for main model backbone")
    parser.add_argument("--main_feature_layers", type=str, default=None,
                        help="Comma-separated backbone layers, e.g. layer3,layer4")
    parser.add_argument("--main_feature_fuse", type=str, default=None,
                        help="concat/last")
    parser.add_argument("--transformer_patch_size", type=int, default=None,
                        help="Patch size for transformer main backbone")
    parser.add_argument("--transformer_embed_dim", type=int, default=None,
                        help="Token embedding dimension for transformer main backbone")
    parser.add_argument("--transformer_depth", type=int, default=None,
                        help="Transformer block depth for transformer main backbone")
    parser.add_argument("--transformer_num_heads", type=int, default=None,
                        help="Attention heads for transformer main backbone")
    parser.add_argument("--transformer_mlp_ratio", type=float, default=None,
                        help="MLP ratio for transformer main backbone")
    parser.add_argument("--encoder_channels", type=str, default=None,
                        help="Comma-separated encoder channels, e.g. 32,64,160,320")
    parser.add_argument("--blocks_per_stage", type=int, default=None)
    parser.add_argument("--num_axial_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--primary_model", type=str, default=None,
                        help="raw_pca_mlp/deep_consistency")
    parser.add_argument("--raw_pca_components", type=int, default=None,
                        help="PCA components for raw_pca_mlp")
    parser.add_argument("--raw_pca_hidden", type=str, default=None,
                        help="MLP hidden sizes, e.g. 256,128")
    parser.add_argument("--raw_pca_max_iter", type=int, default=None,
                        help="MLP max_iter for raw_pca_mlp")
    parser.add_argument("--raw_pca_alpha", type=float, default=None,
                        help="MLP alpha for raw_pca_mlp")
    parser.add_argument("--raw_pca_lr_init", type=float, default=None,
                        help="MLP learning_rate_init for raw_pca_mlp")
    parser.add_argument("--raw_open_score_blend", type=float, default=None,
                        help="Blend weight between max_prob and distance-knownness")
    parser.add_argument("--raw_distance_percentile", type=float, default=None,
                        help="Percentile to estimate class radius in PCA space")
    parser.add_argument("--raw_fewshot_c_3shot", type=float, default=None,
                        help="SVM C used for 3-shot in Setting C (raw_pca_mlp)")
    parser.add_argument("--input_raw_pca_components", type=int, default=None,
                        help="PCA components for deep_consistency input projection")
    parser.add_argument("--prepared_dir", type=str, default=None,
                        help="Prepared data directory (e.g. prepared_data_pca171)")
    parser.add_argument("--rt_bins", type=int, default=None,
                        help="Override RT bins (decoder target/input shape hint)")
    parser.add_argument("--mz_bins", type=int, default=None,
                        help="Override m/z bins (decoder target/input shape hint)")
    parser.add_argument("--enable_input_raw_pca", action="store_true",
                        help="Enable raw->PCA input projection for deep_consistency")
    parser.add_argument("--disable_input_raw_pca", action="store_true",
                        help="Disable raw->PCA input projection for deep_consistency")
    parser.add_argument("--aug_peak_broaden_prob", type=float, default=None,
                        help="Probability for gaussian peak broadening augmentation")
    parser.add_argument("--aug_rt_warp_prob", type=float, default=None,
                        help="Probability for RT warping augmentation")

    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training and only run evaluation")
    parser.add_argument("--skip_evaluate", action="store_true",
                        help="Skip evaluation and only run training")
    parser.add_argument("--progress_log",
                        default="outputs/PROJECT_PROGRESS.md",
                        help="Progress markdown file")
    parser.add_argument("--show_progress", action="store_true",
                        help="Enable tqdm progress bars (default: off)")

    args = parser.parse_args()

    # 默认关闭 tqdm，避免进度条干扰日志；需要时可显式开启
    os.environ["GCMS_SHOW_PROGRESS"] = "1" if args.show_progress else "0"

    cfg = Config()
    project_root = Path(__file__).resolve().parent
    cfg.output_dir = str(project_root / "outputs" / args.name)

    # Apply overrides
    for key in [
        "epochs", "batch_size", "lr", "weight_decay",
        "lambda_adv", "lambda_proto", "lambda_recon",
        "supcon_temperature", "reject_threshold_factor", "accept_percentile",
        "eval_interval", "early_stop_patience",
        "eval_interval_search", "eval_interval_final", "eval_final_start_ratio",
        "min_epochs_before_early_stop", "min_epoch_ratio_before_early_stop",
        "early_stop_min_lr_ratio", "early_stop_min_delta",
        "proto_val_subset_ratio", "proto_val_subset_min_samples",
        "proto_val_subset_max_samples", "proto_val_full_every",
        "dataloader_workers", "dataloader_prefetch_factor",
        "dataset_cache_max_items", "amp_dtype",
        "warmup_guard_epoch", "warmup_guard_best_at_epoch", "warmup_guard_min_ratio",
        "pretrained_feature_model", "pretrained_feature_arch",
        "main_backbone", "main_backbone_model", "main_feature_layers", "main_feature_fuse",
        "transformer_patch_size", "transformer_embed_dim", "transformer_depth",
        "transformer_num_heads", "transformer_mlp_ratio",
        "blocks_per_stage", "num_axial_heads", "dropout",
        "prepared_dir", "rt_bins", "mz_bins",
        "primary_model", "raw_pca_components", "raw_pca_hidden", "raw_pca_max_iter",
        "raw_pca_alpha", "raw_pca_lr_init", "raw_open_score_blend",
        "raw_distance_percentile", "raw_fewshot_c_3shot",
        "input_raw_pca_components", "aug_peak_broaden_prob", "aug_rt_warp_prob",
    ]:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    if args.enable_input_raw_pca:
        cfg.input_raw_pca_enabled = True
    if args.disable_input_raw_pca:
        cfg.input_raw_pca_enabled = False
    if args.disable_dataloader_pin_memory:
        cfg.dataloader_pin_memory = False
    if args.disable_dataloader_persistent_workers:
        cfg.dataloader_persistent_workers = False
    if args.enable_dataset_cache:
        cfg.dataset_cache_in_memory = True
    if args.disable_amp:
        cfg.amp_enabled = False
    if args.disable_channels_last:
        cfg.channels_last = False
    if args.enable_torch_compile:
        cfg.torch_compile = True
    if args.disable_cuda_benchmark:
        cfg.cuda_benchmark = False
    if args.warmup_guard_enabled:
        cfg.warmup_guard_enabled = True
    if args.warmup_guard_compare_best:
        cfg.warmup_guard_compare_best = True
    if args.warmup_guard_no_compare_best:
        cfg.warmup_guard_compare_best = False
    if args.encoder_channels is not None:
        cfg.encoder_channels = _parse_int_tuple(args.encoder_channels)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist exact run config for reproducibility
    config_dump = {
        "name": args.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.__dict__,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2, ensure_ascii=False)

    # 保留每次迭代参数文档, 便于后续追溯(即使清理非最优模型产物)
    _append_jsonl(project_root / "outputs" / "ITERATION_PARAMS.jsonl", {
        "name": args.name,
        "timestamp": config_dump["timestamp"],
        "config": config_dump["config"],
    })

    progress_log = project_root / args.progress_log
    _append_progress(progress_log, [
        f"- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START {args.name}",
        f"  - output_dir: {cfg.output_dir}",
        f"  - overrides: epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
        f"lr={cfg.lr}, lambda_adv={cfg.lambda_adv}, lambda_proto={cfg.lambda_proto}, "
        f"lambda_recon={cfg.lambda_recon}, eval_interval={cfg.eval_interval}, "
        f"eval_interval_search={cfg.eval_interval_search}, "
        f"eval_interval_final={cfg.eval_interval_final}, "
        f"eval_final_start_ratio={cfg.eval_final_start_ratio}, "
        f"early_stop_patience={cfg.early_stop_patience}, "
        f"min_epochs_before_early_stop={cfg.min_epochs_before_early_stop}, "
        f"min_epoch_ratio_before_early_stop={cfg.min_epoch_ratio_before_early_stop}, "
        f"early_stop_min_lr_ratio={cfg.early_stop_min_lr_ratio}, "
        f"early_stop_min_delta={cfg.early_stop_min_delta}, "
        f"proto_val_subset_ratio={cfg.proto_val_subset_ratio}, "
        f"proto_val_full_every={cfg.proto_val_full_every}, "
        f"dataloader_workers={cfg.dataloader_workers}, "
        f"dataloader_prefetch_factor={cfg.dataloader_prefetch_factor}, "
        f"dataloader_pin_memory={cfg.dataloader_pin_memory}, "
        f"dataloader_persistent_workers={cfg.dataloader_persistent_workers}, "
        f"encoder_channels={cfg.encoder_channels}, "
        f"blocks_per_stage={cfg.blocks_per_stage}, "
        f"num_axial_heads={cfg.num_axial_heads}, "
        f"dropout={cfg.dropout}, "
        f"primary_model={cfg.primary_model}, "
        f"raw_pca_components={cfg.raw_pca_components}, "
        f"raw_pca_hidden={cfg.raw_pca_hidden}, "
        f"raw_pca_max_iter={cfg.raw_pca_max_iter}, "
        f"raw_pca_alpha={cfg.raw_pca_alpha}, "
        f"raw_pca_lr_init={cfg.raw_pca_lr_init}, "
        f"raw_open_score_blend={cfg.raw_open_score_blend}, "
        f"raw_distance_percentile={cfg.raw_distance_percentile}, "
        f"raw_fewshot_c_3shot={cfg.raw_fewshot_c_3shot}, "
        f"prepared_dir={cfg.prepared_dir}, "
        f"rt_bins={cfg.rt_bins}, "
        f"mz_bins={cfg.mz_bins}, "
        f"input_raw_pca_enabled={cfg.input_raw_pca_enabled}, "
        f"input_raw_pca_components={cfg.input_raw_pca_components}, "
        f"aug_peak_broaden_prob={cfg.aug_peak_broaden_prob}, "
        f"aug_rt_warp_prob={cfg.aug_rt_warp_prob}, "
        f"pretrained_feature_model={cfg.pretrained_feature_model}, "
        f"pretrained_feature_arch={cfg.pretrained_feature_arch}, "
        f"main_backbone={cfg.main_backbone}, "
        f"main_backbone_model={cfg.main_backbone_model}, "
        f"main_feature_layers={cfg.main_feature_layers}, "
        f"main_feature_fuse={cfg.main_feature_fuse}, "
        f"transformer_patch_size={cfg.transformer_patch_size}, "
        f"transformer_embed_dim={cfg.transformer_embed_dim}, "
        f"transformer_depth={cfg.transformer_depth}, "
        f"transformer_num_heads={cfg.transformer_num_heads}, "
        f"transformer_mlp_ratio={cfg.transformer_mlp_ratio}, "
        f"warmup_guard_enabled={cfg.warmup_guard_enabled}, "
        f"warmup_guard_epoch={cfg.warmup_guard_epoch}, "
        f"warmup_guard_best_at_epoch={cfg.warmup_guard_best_at_epoch}, "
        f"warmup_guard_compare_best={cfg.warmup_guard_compare_best}, "
        f"warmup_guard_min_ratio={cfg.warmup_guard_min_ratio}, "
        f"show_progress={args.show_progress}",
    ])

    if not args.skip_train:
        train_single_model(cfg)

    if not args.skip_evaluate:
        evaluate_single_model(cfg)

    summary = _load_summary(out_dir / "evaluation_summary.json")
    if summary and "setting_a" in summary:
        sa = summary.get("setting_a", {})
        sb = summary.get("setting_b", {})
        baseline = summary.get("baseline_tic_pca_mlp", {})
        baseline_sb = baseline.get("setting_b", {})
        baseline_sc3 = baseline.get("setting_c", {}).get("3", {})
        cmp_res = summary.get("main_vs_baseline", {})
        cmp_sb = cmp_res.get("setting_b", {})
        cmp_sc3 = cmp_res.get("setting_c", {}).get("3", {})
        readme_baselines = summary.get("baselines_readme", {})
        readme_cmp = summary.get("main_vs_readme_baselines", {})
        pretrained_info = summary.get("pretrained_feature_extractor", {})

        readme_lines = []
        for key in ["pca_mahalanobis", "pls_da", "svm_rbf", "tic_pca_mlp"]:
            item = readme_baselines.get(key)
            if not item:
                continue
            name = item.get("name", key)
            mode = item.get("feature_mode", "raw")
            item_sb = (item.get("setting_b") or {})
            item_sc3 = (item.get("setting_c") or {}).get("3", {})
            d_sb = (readme_cmp.get(key) or {}).get("setting_b", {})
            d_sc3 = ((readme_cmp.get(key) or {}).get("setting_c", {}) or {}).get("3", {})
            readme_lines.append(
                "  - Baseline[{name}]({mode}): ".format(name=name, mode=mode)
                + "auroc={auroc}, fpr95={fpr95}, shot3={shot3}, ".format(
                    auroc=item_sb.get("open_set_AUROC"),
                    fpr95=item_sb.get("FPR_at_95TPR"),
                    shot3=item_sc3.get("accuracy"),
                )
                + "d_auroc={d_auroc}, d_fpr95={d_fpr95}, d_shot3={d_shot3}".format(
                    d_auroc=d_sb.get("open_set_AUROC"),
                    d_fpr95=d_sb.get("FPR_at_95TPR"),
                    d_shot3=d_sc3.get("accuracy"),
                )
            )

        _append_progress(progress_log, [
            f"- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DONE {args.name}",
            f"  - Setting A: acc={sa.get('accuracy')}, macro_f1={sa.get('macro_f1')}",
            f"  - Setting B: auroc={sb.get('open_set_AUROC')}, fpr95={sb.get('FPR_at_95TPR')}",
            (
                "  - pretrained_feature_extractor: "
                f"enabled={pretrained_info.get('enabled')}, "
                f"arch={pretrained_info.get('arch')}, "
                f"model={pretrained_info.get('model_path')}"
            ),
            (
                "  - Baseline(TIC+PCA+MLP): "
                f"auroc={baseline_sb.get('open_set_AUROC')}, "
                f"fpr95={baseline_sb.get('FPR_at_95TPR')}, "
                f"shot3={baseline_sc3.get('accuracy')}"
            ),
            (
                "  - Main-Baseline: "
                f"d_auroc={cmp_sb.get('open_set_AUROC')}, "
                f"d_fpr95={cmp_sb.get('FPR_at_95TPR')}, "
                f"d_shot3={cmp_sc3.get('accuracy')}"
            ),
            *readme_lines,
            f"  - summary: outputs/{args.name}/evaluation_summary.json",
        ])
    else:
        _append_progress(progress_log, [
            f"- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DONE {args.name} (no summary)",
        ])


if __name__ == "__main__":
    main()
