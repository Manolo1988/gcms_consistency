#!/usr/bin/env python3
"""
Reproduce the 677 route with multiple seeds and multiple splits.

677 configuration core:
  prepared_dir = bins_rt1152_mz288
  input_raw_pca_enabled = False
  rt_bins = 1152, mz_bins = 288
  primary_model = deep_consistency
  main_backbone = gcms
  epochs = 200
  batch_size = 16
  lr = 0.00026
  lambda_adv = 0.12
  lambda_proto = 0.88
  lambda_recon = 0.2
  supcon_temperature = 0.075
  accept_percentile = 97
  reject_threshold_factor = 2.0

Usage:
  python scripts/reproduce_677.py --seeds 42,43,44,45,46
  python scripts/reproduce_677.py --seeds 42,43 --split_seeds 42,43
  python scripts/reproduce_677.py --skip_train --skip_evaluate
  python scripts/reproduce_677.py --run_prefix reproduce677
"""
import argparse
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from config import Config, get_device
from output_utils import (
    make_run_dir, save_run_config, save_evaluation_summary,
    collect_all_runs, write_all_runs_csv, write_all_runs_md,
    compute_statistics, SUMMARY_DIR,
)


def get_677_config():
    """Get a Config with 677 default settings."""
    cfg = Config()

    # ── 677 core path ──
    cfg.prepared_dir = str(
        Path(__file__).resolve().parent.parent
        / "prepared_data" / "bins_rt1152_mz288"
    )

    # ── 677 core grid ──
    cfg.rt_bins = 1152
    cfg.mz_bins = 288
    cfg.rt_range = (3.17, 36.91)
    cfg.mz_range = (30, 200)

    # ── No input PCA ──
    cfg.input_raw_pca_enabled = False
    cfg.input_raw_pca_components = 128
    cfg.prepare_direct_raw_pca = False

    # ── 677 core training ──
    cfg.epochs = 200
    cfg.batch_size = 16
    cfg.lr = 0.00026
    cfg.weight_decay = 1e-4
    cfg.eval_interval = 10
    cfg.eval_interval_search = 10
    cfg.eval_interval_final = 5
    cfg.eval_final_start_ratio = 0.7
    cfg.early_stop_patience = 0
    cfg.early_stop_min_lr_ratio = 0.2

    # ── 677 core loss ──
    cfg.lambda_supcon = 1.0
    cfg.lambda_adv = 0.12
    cfg.lambda_proto = 0.88
    cfg.lambda_recon = 0.2
    cfg.supcon_temperature = 0.075
    cfg.proto_margin = 1.0

    # ── 677 core prototype ──
    cfg.accept_percentile = 97.0
    cfg.reject_threshold_factor = 2.0

    # ── Model architecture ──
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

    # ── Split settings ──
    cfg.num_open_test_classes = 2
    cfg.n_shot_values = (1, 3, 5, 10)
    cfg.holdout_batch_ratio = 0.1
    cfg.preferred_holdout_products = ("HMD", "XCJ")
    cfg.preferred_holdout_batches = ("20250905", "20250912", "20250920")
    cfg.val_ratio = 0.1

    # ── Default new features: disabled for baseline ──
    cfg.tic_branch_enabled = False
    cfg.open_score_calibration_enabled = False
    cfg.fewshot_repeats = 1
    cfg.fewshot_seed_start = 42
    cfg.split_seed = 42

    # ── Augmentation ──
    cfg.aug_intensity_scale = (0.8, 1.2)
    cfg.aug_noise_std = 0.05
    cfg.aug_mask_ratio = 0.15
    cfg.aug_rt_shift_max = 8
    cfg.aug_mz_shift_max = 2

    return cfg


def config_to_dict(cfg: Config) -> dict:
    """Convert Config to dict for JSON serialization."""
    d = {}
    for field_name in cfg.__dataclass_fields__:
        val = getattr(cfg, field_name)
        if isinstance(val, tuple):
            val = list(val)
        if isinstance(val, Path):
            val = str(val)
        # Skip non-serializable
        try:
            json.dumps(val)
            d[field_name] = val
        except (TypeError, ValueError):
            d[field_name] = str(val)
    return d


def run_single_seed(cfg, seed, run_dir, skip_train=False, skip_evaluate=False):
    """Run a single seed experiment."""
    print(f"\n{'='*70}")
    print(f"Running seed={seed} -> {run_dir}")
    print(f"{'='*70}")

    cfg.seed = seed
    cfg.output_dir = str(run_dir)

    # Save config
    save_run_config(run_dir, config_to_dict(cfg))

    if not skip_train:
        print(f"\n[Training] seed={seed}...")
        from train import train_single_model, set_seed
        set_seed(seed)
        try:
            train_single_model(cfg)
        except Exception as e:
            print(f"[ERROR] Training failed for seed={seed}: {e}")
            import traceback
            traceback.print_exc()
            with open(run_dir / "run.log", "a") as f:
                f.write(f"TRAINING FAILED: {e}\n")
            return False

    if not skip_evaluate:
        print(f"\n[Evaluating] seed={seed}...")
        from evaluate import evaluate_single_model
        try:
            evaluate_single_model(cfg)
        except Exception as e:
            print(f"[ERROR] Evaluation failed for seed={seed}: {e}")
            import traceback
            traceback.print_exc()
            with open(run_dir / "run.log", "a") as f:
                f.write(f"EVALUATION FAILED: {e}\n")
            return False

    return True


def run_multiple_splits(cfg, seeds, split_seeds, run_prefix, skip_train, skip_evaluate):
    """Run across multiple seeds and splits."""
    all_run_info = []

    for split_seed in split_seeds:
        for seed in seeds:
            run_name = f"{run_prefix}_split{split_seed}_seed{seed}"
            run_dir = make_run_dir(run_name)
            run_cfg = Config()
            # Copy all fields from base config
            for field_name in cfg.__dataclass_fields__:
                setattr(run_cfg, field_name, getattr(cfg, field_name))
            run_cfg.seed = seed
            run_cfg.split_seed = split_seed

            success = run_single_seed(run_cfg, seed, run_dir,
                                       skip_train=skip_train,
                                       skip_evaluate=skip_evaluate)
            all_run_info.append({
                "run_name": run_name,
                "seed": seed,
                "split_seed": split_seed,
                "success": success,
            })

    return all_run_info


def run_multi_seed(cfg, seeds, run_prefix, skip_train, skip_evaluate):
    """Run across multiple seeds with fixed split."""
    all_run_info = []

    for seed in seeds:
        run_name = f"{run_prefix}_seed{seed}"
        run_dir = make_run_dir(run_name)

        run_cfg = Config()
        for field_name in cfg.__dataclass_fields__:
            setattr(run_cfg, field_name, getattr(cfg, field_name))
        run_cfg.seed = seed

        success = run_single_seed(run_cfg, seed, run_dir,
                                   skip_train=skip_train,
                                   skip_evaluate=skip_evaluate)
        all_run_info.append({
            "run_name": run_name,
            "seed": seed,
            "split_seed": cfg.split_seed,
            "success": success,
        })

    return all_run_info


def generate_summary(all_run_info, run_prefix):
    """Generate summary across seeds."""
    print(f"\n{'='*70}")
    print(f"Generating summary for {run_prefix}")
    print(f"{'='*70}")

    # Collect all runs
    all_runs = collect_all_runs()
    matched = [r for r in all_runs
               if r["run_name"].startswith(run_prefix)]

    if not matched:
        print(f"No runs found matching prefix '{run_prefix}'")
        return

    print(f"Found {len(matched)} runs")

    # Write all_runs files
    write_all_runs_csv(all_runs)
    write_all_runs_md(all_runs)

    # Generate paper_main_table for this prefix
    metrics = {}
    metric_keys = [
        "A_acc", "A_macro_f1", "A_balanced_acc",
        "B_AUROC", "B_FPR95", "B_EER",
    ]
    for n in [1, 3, 5, 10]:
        metric_keys.append(f"C_{n}shot_acc")

    valid_keys = []
    for key in metric_keys:
        vals = []
        for r in matched:
            v = r.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
        if vals:
            stats = compute_statistics(vals)
            metrics[key] = stats
            valid_keys.append(key)

    # Write paper_main_table.md
    if valid_keys:
        lines = [
            f"# 677 Reproduction Results: {run_prefix}",
            "",
            f"Seeds: {len(matched)} runs",
            f"Total successful runs: {len([r for r in matched if r.get('A_acc') is not None])}",
            "",
            "## Main Metrics (mean ± std [95% CI])",
            "",
            "| Metric | Mean | Std | 95% CI Low | 95% CI High | N |",
            "|--------|------|-----|-----------|-------------|---|",
        ]
        for key in valid_keys:
            s = metrics[key]
            lines.append(
                f"| {key} | {s['mean']:.4f} | {s['std']:.4f} | "
                f"{s['ci95_low']:.4f} | {s['ci95_high']:.4f} | {s['n']} |"
            )

        paper_table_path = SUMMARY_DIR / "paper_main_table.md"
        with open(paper_table_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Paper main table written to {paper_table_path}")

        # CSV version
        csv_path = SUMMARY_DIR / "paper_main_table.csv"
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Mean", "Std", "CI95_Low", "CI95_High", "N"])
            for key in valid_keys:
                s = metrics[key]
                writer.writerow([key, s['mean'], s['std'], s['ci95_low'], s['ci95_high'], s['n']])
        print(f"Paper main table CSV written to {csv_path}")

    # Write run info JSON
    with open(SUMMARY_DIR / f"{run_prefix}_run_info.json", "w") as f:
        json.dump(all_run_info, f, indent=2)

    print(f"\nSummary complete. See {SUMMARY_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce 677 route with multiple seeds"
    )
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46",
                        help="Comma-separated list of seeds")
    parser.add_argument("--split_seeds", type=str, default="42",
                        help="Comma-separated list of split seeds")
    parser.add_argument("--output_root", type=str, default="output_new")
    parser.add_argument("--run_prefix", type=str, default="reproduce677")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training (evaluate only)")
    parser.add_argument("--skip_evaluate", action="store_true",
                        help="Skip evaluation (train only)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Override epochs (use small value for smoke test)")
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
    parser.add_argument("--multi_split", action="store_true",
                        help="Run with multiple split_seeds")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test: 5 epochs only")

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    split_seeds = [int(s.strip()) for s in args.split_seeds.split(",")]

    print(f"Seeds: {seeds}")
    print(f"Split seeds: {split_seeds}")
    print(f"Run prefix: {args.run_prefix}")

    cfg = get_677_config()
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

    if args.smoke:
        cfg.epochs = 5
        cfg.eval_interval = 2
        cfg.eval_interval_search = 2
        cfg.eval_interval_final = 2
        print(f"[SMOKE TEST] epochs={cfg.epochs}")

    if args.epochs != 200:
        cfg.epochs = args.epochs

    if args.multi_split:
        all_run_info = run_multiple_splits(
            cfg, seeds, split_seeds, args.run_prefix,
            skip_train=args.skip_train,
            skip_evaluate=args.skip_evaluate,
        )
    else:
        all_run_info = run_multi_seed(
            cfg, seeds, args.run_prefix,
            skip_train=args.skip_train,
            skip_evaluate=args.skip_evaluate,
        )

    # Generate summary
    generate_summary(all_run_info, args.run_prefix)

    # Print success/failure summary
    successes = [r for r in all_run_info if r["success"]]
    failures = [r for r in all_run_info if not r["success"]]
    print(f"\n{'='*70}")
    print(f"Overall: {len(successes)} succeeded, {len(failures)} failed")
    if failures:
        print("Failed runs:")
        for f in failures:
            print(f"  - {f['run_name']} (seed={f['seed']}, split_seed={f['split_seed']})")


if __name__ == "__main__":
    main()
