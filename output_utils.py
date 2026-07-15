"""
Output directory management and result summary utilities for output_new/.
All new experiments write to output_new/ and DO NOT overwrite outputs/ or output/.
"""
import json
import csv
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np


OUTPUT_ROOT = Path(__file__).resolve().parent / "output_new"
SUMMARY_DIR = OUTPUT_ROOT / "summary"


def ensure_output_dirs():
    """Create output_new directory structure."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT


def make_run_dir(run_name: str, suffix: str = "") -> Path:
    """Create a run directory under output_new/<run_name>.
    If it exists and suffix is empty, auto-append _1, _2, etc.
    Returns the run directory path.
    """
    base = OUTPUT_ROOT / run_name
    if suffix:
        base = OUTPUT_ROOT / f"{run_name}{suffix}"

    run_dir = base
    counter = 1
    while run_dir.exists():
        run_dir = OUTPUT_ROOT / f"{base.name}_{counter}"
        counter += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    # Create subdirectories
    (run_dir / "final_model").mkdir(exist_ok=True)
    (run_dir / "visualizations").mkdir(exist_ok=True)
    (run_dir / "calibration").mkdir(exist_ok=True)
    (run_dir / "fewshot").mkdir(exist_ok=True)
    return run_dir


def get_run_dir(run_name: str) -> Optional[Path]:
    """Get existing run directory, or None."""
    run_dir = OUTPUT_ROOT / run_name
    if run_dir.exists():
        return run_dir
    return None


def save_run_config(run_dir: Path, config_dict: dict):
    """Save run configuration to run_config.json."""
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, default=_json_default)


def save_evaluation_summary(run_dir: Path, summary: dict):
    """Save evaluation summary."""
    with open(run_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _safe_float(v):
    """Convert value to float, returning NaN for None/invalid."""
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


# ─────────────────────────────────────────────────────────
#  All-runs CSV/MD summary
# ─────────────────────────────────────────────────────────

def collect_all_runs() -> List[Dict]:
    """Scan output_new/ for run directories and collect basic info."""
    runs = []
    if not OUTPUT_ROOT.exists():
        return runs

    for d in sorted(OUTPUT_ROOT.iterdir()):
        if not d.is_dir() or d.name == "summary":
            continue
        config_path = d / "run_config.json"
        eval_path = d / "evaluation_summary.json"
        if not config_path.exists() and not eval_path.exists():
            continue

        entry = {"run_name": d.name, "run_dir": str(d)}

        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            entry["config"] = cfg
            entry["seed"] = cfg.get("seed", "?")
            entry["rt_bins"] = cfg.get("rt_bins", "?")
            entry["mz_bins"] = cfg.get("mz_bins", "?")
            entry["lr"] = cfg.get("lr", "?")
            entry["lambda_adv"] = cfg.get("lambda_adv", "?")
            entry["lambda_proto"] = cfg.get("lambda_proto", "?")
            entry["input_raw_pca_enabled"] = cfg.get("input_raw_pca_enabled", False)
            entry["tic_branch_enabled"] = cfg.get("tic_branch_enabled", False)
            entry["open_score_calibration_enabled"] = cfg.get("open_score_calibration_enabled", False)

        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)
            entry["eval"] = eval_data

            # Extract key metrics
            sa = eval_data.get("setting_a", {})
            entry["A_acc"] = _safe_float(sa.get("accuracy"))
            entry["A_macro_f1"] = _safe_float(sa.get("macro_f1"))
            entry["A_balanced_acc"] = _safe_float(sa.get("balanced_acc"))

            sb = eval_data.get("setting_b", {})
            entry["B_AUROC"] = _safe_float(sb.get("open_set_AUROC"))
            entry["B_FPR95"] = _safe_float(sb.get("FPR_at_95TPR"))
            entry["B_EER"] = _safe_float(sb.get("EER"))

            sc = eval_data.get("setting_c", {})
            for n in [1, 3, 5, 10]:
                n_key = str(n)
                if n_key in sc:
                    entry[f"C_{n}shot_acc"] = _safe_float(sc[n_key].get("accuracy"))
                    entry[f"C_{n}shot_acc_mean"] = _safe_float(sc[n_key].get("accuracy_mean"))
                    entry[f"C_{n}shot_acc_std"] = _safe_float(sc[n_key].get("accuracy_std"))

        runs.append(entry)

    return runs


def write_all_runs_csv(runs: List[Dict]):
    """Write all_runs.csv."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_DIR / "all_runs.csv"

    if not runs:
        return

    # Determine all column keys
    columns = ["run_name", "seed", "rt_bins", "mz_bins", "lr",
               "lambda_adv", "lambda_proto",
               "input_raw_pca_enabled", "tic_branch_enabled", "open_score_calibration_enabled",
               "A_acc", "A_macro_f1", "A_balanced_acc",
               "B_AUROC", "B_FPR95", "B_EER"]
    for n in [1, 3, 5, 10]:
        columns.extend([f"C_{n}shot_acc", f"C_{n}shot_acc_mean", f"C_{n}shot_acc_std"])

    existing_cols = set()
    for r in runs:
        existing_cols.update(r.keys())
    columns = [c for c in columns if c in existing_cols]
    # Add any extra columns not in predefined list
    for c in sorted(existing_cols - set(columns)):
        if not c.startswith(("config", "eval")):
            columns.append(c)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            writer.writerow(r)

    print(f"[output_utils] all_runs.csv written: {csv_path}")


def write_all_runs_md(runs: List[Dict]):
    """Write all_runs.md markdown table."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    md_path = SUMMARY_DIR / "all_runs.md"

    if not runs:
        with open(md_path, "w") as f:
            f.write("# All Runs\n\n(No runs found)\n")
        return

    lines = ["# All Runs Summary", "", f"Total runs: {len(runs)}", ""]

    # Main metrics table
    headers = ["Run", "Seed", "A_acc", "A_F1", "B_AUROC", "B_FPR95",
               "C_1shot", "C_3shot", "C_5shot", "C_10shot", "PCA", "TIC", "Calib"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in runs:
        row = [
            r.get("run_name", "?"),
            str(r.get("seed", "?")),
            f"{r.get('A_acc', float('nan')):.4f}" if not np.isnan(_safe_float(r.get('A_acc'))) else "?",
            f"{r.get('A_macro_f1', float('nan')):.4f}" if not np.isnan(_safe_float(r.get('A_macro_f1'))) else "?",
            f"{r.get('B_AUROC', float('nan')):.4f}" if not np.isnan(_safe_float(r.get('B_AUROC'))) else "?",
            f"{r.get('B_FPR95', float('nan')):.4f}" if not np.isnan(_safe_float(r.get('B_FPR95'))) else "?",
            f"{r.get('C_1shot_acc', float('nan')):.4f}" if not np.isnan(_safe_float(r.get('C_1shot_acc'))) else "?",
            f"{r.get('C_3shot_acc', float('nan')):.4f}" if not np.isnan(_safe_float(r.get('C_3shot_acc'))) else "?",
            f"{r.get('C_5shot_acc', float('nan')):.4f}" if not np.isnan(_safe_float(r.get('C_5shot_acc'))) else "?",
            f"{r.get('C_10shot_acc', float('nan')):.4f}" if not np.isnan(_safe_float(r.get('C_10shot_acc'))) else "?",
            "Y" if r.get("input_raw_pca_enabled") else "N",
            "Y" if r.get("tic_branch_enabled") else "N",
            "Y" if r.get("open_score_calibration_enabled") else "N",
        ]
        lines.append("| " + " | ".join(row) + " |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[output_utils] all_runs.md written: {md_path}")


def compute_statistics(values: List[float]) -> Dict:
    """Compute mean, std, and 95% CI for a list of values."""
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}

    arr = np.array(clean, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) > 1:
        ci = 1.96 * std / np.sqrt(len(arr))
    else:
        ci = 0.0

    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
        "n": len(arr),
        "values": [float(v) for v in clean],
    }


def aggregate_seed_results(run_prefix: str) -> Dict:
    """Aggregate results across multiple seeds for the same run prefix."""
    all_runs = collect_all_runs()
    matched = [r for r in all_runs if r["run_name"].startswith(run_prefix)]

    if not matched:
        return {"error": f"No runs found with prefix '{run_prefix}'"}

    metrics_keys = [
        ("A_acc", "A_acc"), ("A_macro_f1", "A_macro_f1"),
        ("A_balanced_acc", "A_balanced_acc"),
        ("B_AUROC", "B_AUROC"), ("B_FPR95", "B_FPR95"), ("B_EER", "B_EER"),
    ]
    for n in [1, 3, 5, 10]:
        metrics_keys.append((f"C_{n}shot_acc", f"C_{n}shot_acc"))
        metrics_keys.append((f"C_{n}shot_acc_mean", f"C_{n}shot_acc_mean"))

    result = {"run_prefix": run_prefix, "n_seeds": len(matched), "runs": [r["run_name"] for r in matched]}
    for key, display in metrics_keys:
        vals = [_safe_float(r.get(key)) for r in matched]
        result[display] = compute_statistics(vals)

    return result


# ─────────────────────────────────────────────────────────
#  Metadata
# ─────────────────────────────────────────────────────────

def build_prepare_metadata(cfg) -> dict:
    """Build metadata dict about data preparation, for leakage audit."""
    from pathlib import Path
    meta = {
        "prepare_mode": "bins" if not cfg.input_raw_pca_enabled else "pca_precomputed",
        "rt_bins": cfg.rt_bins,
        "mz_bins": cfg.mz_bins,
        "rt_range": list(cfg.rt_range) if cfg.rt_range else None,
        "mz_range": list(cfg.mz_range),
        "input_raw_pca_enabled": cfg.input_raw_pca_enabled,
        "pca_fit_scope": "unknown",  # updated during train/eval
        "tic_source": cfg.tic_source if hasattr(cfg, 'tic_source') else "disabled",
    }

    # Try to read grid_info for observed mz range stats
    grid_info_path = Path(cfg.prepared_dir) / "grid_info.json"
    if grid_info_path.exists():
        with open(grid_info_path) as f:
            gi = json.load(f)
        meta["observed_mz_range"] = gi.get("mz_range", None)
        meta["grid_info"] = gi

    return meta


def build_leakage_check_report(cfg) -> dict:
    """Check for potential data leakage in the pipeline and return a report."""
    issues = []
    warnings = []

    # Check PCA fit scope
    if cfg.input_raw_pca_enabled:
        grid_info_path = Path(cfg.prepared_dir) / "grid_info.json"
        if grid_info_path.exists():
            with open(grid_info_path) as f:
                gi = json.load(f)
            if gi.get("input_pca_precomputed", False):
                pca_scope = gi.get("input_pca_fit_scope", "unknown")
                if pca_scope == "full_dataset":
                    issues.append(
                        "PCA was fit on full dataset during prepare stage. "
                        "This is a potential data leakage. The PCA route can only "
                        "be reported as 'transductive/precomputed PCA ablation', "
                        "not as the main model result."
                    )
                elif pca_scope == "unknown":
                    warnings.append(
                        "PCA fit scope is unknown. Verify that PCA was fit on train_idx only."
                    )

    # Check if scalers/normalizers are train-only
    # (This is checked at runtime in train.py)

    return {
        "has_issues": len(issues) > 0,
        "has_warnings": len(warnings) > 0,
        "issues": issues,
        "warnings": warnings,
    }


def scan_existing_runs() -> list:
    """Quick scan of output_new for existing runs, return list of names."""
    if not OUTPUT_ROOT.exists():
        return []
    return sorted([
        d.name for d in OUTPUT_ROOT.iterdir()
        if d.is_dir() and d.name != "summary" and (d / "run_config.json").exists()
    ])
