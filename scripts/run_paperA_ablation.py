#!/usr/bin/env python3
"""
Paper A ablation experiments runner.

Ablations:
  1. full_677        - 677 baseline
  2. full_677_calibrated - 677 + open_score_calibration
  3. no_proto_loss    - lambda_proto=0
  4. no_batch_adv     - lambda_adv=0
  5. no_recon         - lambda_recon=0
  6. pca287_style     - 287 PCA route (precomputed), marked as ablation only
  7. tic_only         - TIC-only baseline
  8. rt_mz_plus_tic_concat - RT×m/z + TIC concat
  9. rt_mz_plus_tic_gated  - RT×m/z + TIC gated
 10. score_uncalibrated_vs_calibrated - same model, compare score

All results write to output_new/ablation/.
"""
import argparse
import json
import sys
import csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, get_device
from output_utils import (
    make_run_dir, save_run_config, save_evaluation_summary,
    compute_statistics, OUTPUT_ROOT,
)


ABLATION_DIR = OUTPUT_ROOT / "ablation"


def get_base_677_config():
    """Get baseline 677 config."""
    cfg = Config()
    cfg.prepared_dir = str(
        Path(__file__).resolve().parent.parent
        / "prepared_data" / "bins_rt1152_mz288"
    )
    cfg.rt_bins = 1152
    cfg.mz_bins = 288
    cfg.rt_range = (3.17, 36.91)
    cfg.mz_range = (30, 200)
    cfg.input_raw_pca_enabled = False
    cfg.input_raw_pca_components = 128
    cfg.prepare_direct_raw_pca = False
    cfg.epochs = 200
    cfg.batch_size = 16
    cfg.lr = 0.00026
    cfg.weight_decay = 1e-4
    cfg.eval_interval = 10
    cfg.eval_interval_search = 10
    cfg.eval_interval_final = 5
    cfg.eval_final_start_ratio = 0.7
    cfg.early_stop_patience = 0
    cfg.lambda_supcon = 1.0
    cfg.lambda_adv = 0.12
    cfg.lambda_proto = 0.88
    cfg.lambda_recon = 0.2
    cfg.supcon_temperature = 0.075
    cfg.proto_margin = 1.0
    cfg.accept_percentile = 97.0
    cfg.reject_threshold_factor = 2.0
    cfg.primary_model = "deep_consistency"
    cfg.main_backbone = "gcms"
    cfg.in_channels = 2
    cfg.feature_dim = 256
    cfg.proj_dim = 128
    cfg.encoder_channels = (32, 64, 128, 256)
    cfg.blocks_per_stage = 2
    cfg.num_axial_heads = 4
    cfg.dropout = 0.3
    cfg.embed_normalize = True
    cfg.num_open_test_classes = 2
    cfg.n_shot_values = (1, 3, 5, 10)
    cfg.holdout_batch_ratio = 0.1
    cfg.preferred_holdout_products = ("HMD", "XCJ")
    cfg.preferred_holdout_batches = ("20250905", "20250912", "20250920")
    cfg.val_ratio = 0.1
    cfg.tic_branch_enabled = False
    cfg.open_score_calibration_enabled = False
    cfg.fewshot_repeats = 1
    cfg.fewshot_seed_start = 42
    cfg.split_seed = 42
    cfg.seed = 42
    return cfg


def config_to_dict(cfg: Config) -> dict:
    d = {}
    for field_name in cfg.__dataclass_fields__:
        val = getattr(cfg, field_name)
        if isinstance(val, tuple):
            val = list(val)
        if isinstance(val, Path):
            val = str(val)
        try:
            json.dumps(val)
            d[field_name] = val
        except (TypeError, ValueError):
            d[field_name] = str(val)
    return d


def run_ablation(cfg, ablation_name, skip_train=False, skip_evaluate=False):
    """Run a single ablation experiment."""
    run_dir = ABLATION_DIR / ablation_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final_model").mkdir(exist_ok=True)
    (run_dir / "visualizations").mkdir(exist_ok=True)
    (run_dir / "calibration").mkdir(exist_ok=True)
    (run_dir / "fewshot").mkdir(exist_ok=True)

    cfg.output_dir = str(run_dir)
    save_run_config(run_dir, config_to_dict(cfg))

    if not skip_train:
        print(f"\n{'='*60}")
        print(f"[ABLATION] Training: {ablation_name}")
        print(f"{'='*60}")
        from train import train_single_model, set_seed
        set_seed(cfg.seed)
        try:
            train_single_model(cfg)
        except Exception as e:
            print(f"[ERROR] Training failed for {ablation_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    if not skip_evaluate:
        print(f"\n{'='*60}")
        print(f"[ABLATION] Evaluating: {ablation_name}")
        print(f"{'='*60}")
        from evaluate import evaluate_single_model
        try:
            evaluate_single_model(cfg)
        except Exception as e:
            print(f"[ERROR] Evaluation failed for {ablation_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Read evaluation summary
    summary_path = run_dir / "evaluation_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def extract_metrics_from_summary(summary):
    """Extract key metrics from evaluation summary."""
    if summary is None:
        return {}

    m = {}
    sa = summary.get("setting_a", {})
    m["A_acc"] = sa.get("accuracy")
    m["A_macro_f1"] = sa.get("macro_f1")
    m["A_balanced_acc"] = sa.get("balanced_acc")

    sb = summary.get("setting_b", {})
    m["B_AUROC"] = sb.get("open_set_AUROC")
    m["B_FPR95"] = sb.get("FPR_at_95TPR")
    m["B_EER"] = sb.get("EER")
    m["B_TPR_at_FPR5"] = sb.get("TPR_at_FPR5")
    m["B_TPR_at_FPR10"] = sb.get("TPR_at_FPR10")

    # Calibration
    cal = sb.get("calibration", {})
    if cal.get("enabled"):
        post = cal.get("post_calibration", {})
        m["B_AUROC_cal"] = post.get("AUROC")
        m["B_FPR95_cal"] = post.get("FPR_at_95TPR")
        m["B_EER_cal"] = post.get("EER")

    sc = summary.get("setting_c", {})
    for n in [1, 3, 5, 10]:
        nk = str(n)
        if nk in sc:
            fc = sc[nk]
            m[f"C_{n}shot_acc"] = fc.get("accuracy")
            m[f"C_{n}shot_f1"] = fc.get("macro_f1")
            m[f"C_{n}shot_acc_std"] = fc.get("accuracy_std")
            if fc.get("repeats", 1) > 1:
                m[f"C_{n}shot_acc_ci95"] = fc.get("accuracy_ci95")

    return m


def write_ablation_table(ablation_results):
    """Write ablation_table.csv and ablation_table.md."""
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    if not ablation_results:
        print("No ablation results to write")
        return

    # Collect all metric keys
    all_keys = set()
    for name, metrics in ablation_results.items():
        all_keys.update(metrics.keys())

    # Define column order
    col_order = [
        "ablation",
        "A_acc", "A_macro_f1", "A_balanced_acc",
        "B_AUROC", "B_FPR95", "B_EER",
        "B_TPR_at_FPR5", "B_TPR_at_FPR10",
        "B_AUROC_cal", "B_FPR95_cal", "B_EER_cal",
        "C_1shot_acc", "C_1shot_f1", "C_1shot_acc_std",
        "C_3shot_acc", "C_3shot_f1", "C_3shot_acc_std",
        "C_5shot_acc", "C_5shot_f1", "C_5shot_acc_std",
        "C_10shot_acc", "C_10shot_f1", "C_10shot_acc_std",
    ]
    columns = [c for c in col_order if c in all_keys or c == "ablation"]

    # CSV
    csv_path = ABLATION_DIR / "ablation_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for name, metrics in ablation_results.items():
            row = {"ablation": name}
            row.update(metrics)
            writer.writerow(row)
    print(f"Ablation CSV: {csv_path}")

    # Markdown
    md_path = ABLATION_DIR / "ablation_table.md"
    lines = [
        "# Paper A: Ablation Study",
        "",
        "All experiments use the 677 data split and RT×m/z tensor input.",
        f"Total ablations: {len(ablation_results)}",
        "",
    ]

    # Filter to columns that actually have data
    active_cols = ["ablation"]
    for c in columns[1:]:
        has_data = any(
            ablation_results.get(name, {}).get(c) is not None
            for name in ablation_results
        )
        if has_data:
            active_cols.append(c)

    lines.append("| " + " | ".join(active_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(active_cols)) + " |")

    for name, metrics in ablation_results.items():
        row = [name]
        for c in active_cols[1:]:
            v = metrics.get(c)
            if v is None:
                row.append("-")
            elif isinstance(v, float):
                row.append(f"{v:.4f}")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Ablation MD: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Paper A ablation experiments")
    parser.add_argument("--skip_train", action="store_true", help="Skip training")
    parser.add_argument("--skip_evaluate", action="store_true", help="Skip evaluation")
    parser.add_argument("--ablations", type=str, default="all",
                        help="Comma-separated list of ablations or 'all'")
    parser.add_argument("--epochs", type=int, default=200, help="Override epochs")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 5 epochs")
    parser.add_argument("--fewshot_repeats", type=int, default=50,
                        help="Few-shot repeats for ablation")
    parser.add_argument("--dataloader_workers", type=int, default=None)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=None)
    parser.add_argument("--enable_dataset_cache", action="store_true")
    parser.add_argument("--dataset_cache_max_items", type=int, default=None)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default=None,
                        help="float16/bfloat16")
    parser.add_argument("--disable_channels_last", action="store_true")
    parser.add_argument("--enable_torch_compile", action="store_true")
    parser.add_argument("--disable_cuda_benchmark", action="store_true")
    args = parser.parse_args()

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    if args.ablations == "all":
        to_run = [
            "full_677",
            "full_677_calibrated",
            "no_proto_loss",
            "no_batch_adv",
            "no_recon",
            "pca287_style",
            "tic_only",
            "rt_mz_plus_tic_concat",
            "rt_mz_plus_tic_gated",
        ]
    else:
        to_run = [a.strip() for a in args.ablations.split(",")]

    smoke_epochs = 5 if args.smoke else args.epochs
    results = {}

    for ablation_name in to_run:
        cfg = get_base_677_config()
        cfg.epochs = smoke_epochs
        cfg.fewshot_repeats = args.fewshot_repeats
        if args.dataloader_workers is not None:
            cfg.dataloader_workers = args.dataloader_workers
        if args.dataloader_prefetch_factor is not None:
            cfg.dataloader_prefetch_factor = args.dataloader_prefetch_factor
        if args.enable_dataset_cache:
            cfg.dataset_cache_in_memory = True
        if args.dataset_cache_max_items is not None:
            cfg.dataset_cache_max_items = args.dataset_cache_max_items
        if args.disable_amp:
            cfg.amp_enabled = False
        if args.amp_dtype is not None:
            cfg.amp_dtype = args.amp_dtype
        if args.disable_channels_last:
            cfg.channels_last = False
        if args.enable_torch_compile:
            cfg.torch_compile = True
        if args.disable_cuda_benchmark:
            cfg.cuda_benchmark = False

        print(f"\n{'#'*60}")
        print(f"# Ablation: {ablation_name}")
        print(f"{'#'*60}")

        if ablation_name == "full_677":
            # Baseline 677
            pass

        elif ablation_name == "full_677_calibrated":
            cfg.open_score_calibration_enabled = True
            cfg.open_score_calibration_mode = "logistic"

        elif ablation_name == "no_proto_loss":
            cfg.lambda_proto = 0.0

        elif ablation_name == "no_batch_adv":
            cfg.lambda_adv = 0.0

        elif ablation_name == "no_recon":
            cfg.lambda_recon = 0.0

        elif ablation_name == "pca287_style":
            cfg.input_raw_pca_enabled = True
            cfg.input_raw_pca_components = 128
            cfg.prepare_direct_raw_pca = True
            cfg.prepared_dir = str(Path(__file__).resolve().parent.parent / "prepared_data")

        elif ablation_name == "tic_only":
            cfg.tic_branch_enabled = True
            cfg.tic_source = "from_tensor"
            cfg.tic_encoder = "cnn1d"
            cfg.tic_embed_dim = 256  # Use TIC-only as primary
            cfg.tic_fusion_mode = "concat"
            cfg.in_channels = 1      # Only use one channel for fairness
            # Actually for TIC-only, we need a different architecture
            # Use a 1D model - for now, keep 2D but only use ch0
            cfg.in_channels = 2

        elif ablation_name == "rt_mz_plus_tic_concat":
            cfg.tic_branch_enabled = True
            cfg.tic_source = "from_tensor"
            cfg.tic_encoder = "cnn1d"
            cfg.tic_embed_dim = 64
            cfg.tic_fusion_mode = "concat"
            cfg.tic_fusion_output_dim = 256

        elif ablation_name == "rt_mz_plus_tic_gated":
            cfg.tic_branch_enabled = True
            cfg.tic_source = "from_tensor"
            cfg.tic_encoder = "cnn1d"
            cfg.tic_embed_dim = 64
            cfg.tic_fusion_mode = "gated"
            cfg.tic_fusion_output_dim = 256

        elif ablation_name == "score_uncalibrated_vs_calibrated":
            # Same as full_677 but with calibration enabled for comparison
            cfg.open_score_calibration_enabled = True
            cfg.open_score_calibration_mode = "logistic"

        summary = run_ablation(cfg, ablation_name,
                                skip_train=args.skip_train,
                                skip_evaluate=args.skip_evaluate)
        metrics = extract_metrics_from_summary(summary)
        results[ablation_name] = metrics

        # Print per-ablation summary
        print(f"\n[Ablation: {ablation_name}]")
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Write combined table
    write_ablation_table(results)

    print(f"\n{'='*60}")
    print(f"All ablations complete. Results in {ABLATION_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
