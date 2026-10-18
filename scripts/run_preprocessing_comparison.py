#!/usr/bin/env python3
"""Run the three frozen inputs sequentially without any input augmentation."""

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_reporting import save_comparison
from paper_protocol import load_filtered_metadata, save_json


VARIANTS = ("fixed_grid_log1p", "dual_channel_no_pca", "dual_channel_pca")
ROLE_KEYS = ("train_idx", "val_idx", "test_batch_idx", "test_unknown_idx")


def validate_inputs(manifest_dir):
    """Fail before training if any representation uses another sample/split order."""
    original = json.loads((manifest_dir / "split.json").read_text(encoding="utf-8"))
    reference = load_filtered_metadata(manifest_dir / "metadata.csv")
    frozen_columns = ["sample_id", "product_fine", "batch_idx"]
    for name in VARIANTS:
        directory = manifest_dir / "representations" / name
        split = json.loads((directory / "split.json").read_text(encoding="utf-8"))
        grid = json.loads((directory / "grid_info.json").read_text(encoding="utf-8"))
        if not split.get("fixed_split") or split.get("stage") != "prepared":
            raise ValueError(f"{name}: input is not a frozen prepared split")
        if grid.get("input_representation") != name:
            raise ValueError(f"{name}: representation name mismatch")
        fingerprint = original.get("manifest_fingerprint")
        if not fingerprint or split.get("manifest_fingerprint") != fingerprint or grid.get("source_manifest_fingerprint") != fingerprint:
            raise ValueError(f"{name}: manifest fingerprint mismatch")
        if any(split[key] != original[key] for key in ROLE_KEYS):
            raise ValueError(f"{name}: split indices differ")
        frame = load_filtered_metadata(directory / "metadata.csv")
        if not reference[frozen_columns].astype(str).equals(frame[frozen_columns].astype(str)):
            raise ValueError(f"{name}: sample order or labels differ")
    sets = [set(original[key]) for key in ROLE_KEYS]
    if any(not group for group in sets):
        raise ValueError("All four split roles must be nonempty")
    if any(len(group) != len(original[key]) for group, key in zip(sets, ROLE_KEYS)):
        raise ValueError("Duplicate sample indices inside a split")
    if any(left & right for i, left in enumerate(sets) for right in sets[i + 1:]):
        raise ValueError("Sample overlap between split roles")
    if set.union(*sets) != set(range(len(reference))):
        raise ValueError("Split indices do not cover the metadata exactly")
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("new_prepared_data_preproc_v1"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("result") / f"preproc_compare_noaug_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true", help="Validate frozen sample/split contracts without loading torch")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error("epochs/batch-size/lr must be positive; num-workers must be nonnegative")
    manifest_dir = args.manifest_dir.resolve()
    split = validate_inputs(manifest_dir)
    print(f"Frozen inputs verified: {dict((key, len(split[key])) for key in ROLE_KEYS)}", flush=True)
    if args.check_only:
        return

    import torch
    from run_paper_dl_baseline import main as run_baseline

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json({**vars(args), "manifest_dir": str(manifest_dir), "output_dir": str(output_dir),
               "augmentation_enabled": False, "variants": list(VARIANTS),
               "checkpoint_selection": "validation_accuracy",
               "interpretation": "Single training seed; few-shot episodes reuse samples and are not independent training repeats."},
              output_dir / "comparison_config.json")
    save_json({"status": "running", "completed_variants": []}, output_dir / "comparison_status.json")
    rows = []
    for name in VARIANTS:
        print(f"\nStarting {name}: no augmentation, seed={args.seed}", flush=True)
        directory = output_dir / name
        run_baseline([
            "--method", "plain_cnn_ce", "--no-augmentation", "--seed", str(args.seed),
            "--prepared-dir", str(manifest_dir / "representations" / name),
            "--output-dir", str(directory), "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--lr", str(args.lr),
            "--num-workers", str(args.num_workers),
        ])
        result = json.loads((directory / "paper_gate_result.json").read_text(encoding="utf-8"))
        row = {"representation": name, "seed": args.seed, **result["closed_set"],
               "val_accuracy": result["validation"]["accuracy"],
               "val_balanced_acc": result["validation"]["balanced_acc"],
               "val_macro_f1": result["validation"]["macro_f1"],
               "best_epoch": result["best_epoch"], "training_seconds": result["training_seconds"],
               "parameter_count": result["parameter_count"]}
        row.update({f"fewshot_{item['shot']}shot_accuracy": item["mean_accuracy"]
                    for item in result["fewshot_summary"] if item["scope"] == "pooled"})
        rows.append(row)
        save_comparison(rows, output_dir)
        save_json({"status": "complete" if len(rows) == len(VARIANTS) else "running",
                   "completed_variants": [item["representation"] for item in rows]},
                  output_dir / "comparison_status.json")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Completed all three variants: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
