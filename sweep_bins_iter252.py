"""Sweep rt_bins/mz_bins based on iter_auto252 hyperparameters.

This script does NOT delete any historical output folders or run logs.
Each bins combo writes into its own output directory under outputs/.

Example:
  .venv/bin/python sweep_bins_iter252.py \
    --base_run outputs/iter_auto252_bs16_lr172_a7_p90_fewr50 \
    --combos 768x192 896x224 1024x256 1152x288 \
    --epochs_override 5
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from data_prepare import convert_all, scan_dataset
from dataset import create_data_split
from evaluate import evaluate_single_model
from train import train_single_model

OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_combo(text: str) -> Tuple[int, int]:
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid combo format: {text}, expected <rt>x<mz>")
    rt = int(parts[0])
    mz = int(parts[1])
    if rt <= 0 or mz <= 0:
        raise ValueError(f"Invalid combo values: {text}")
    return rt, mz


def _rank_key(metrics: Dict) -> Tuple[float, float, float]:
    auroc = metrics.get("open_set_AUROC")
    fpr95 = metrics.get("FPR_at_95TPR")
    shot3 = metrics.get("shot3_acc")
    auroc_v = -1.0 if auroc is None else float(auroc)
    fpr_v = 1e9 if fpr95 is None else float(fpr95)
    shot3_v = -1.0 if shot3 is None else float(shot3)
    return (auroc_v, -fpr_v, shot3_v)


def _extract_metrics(summary: Dict) -> Dict:
    sb = summary.get("setting_b", {})
    sc3 = summary.get("setting_c", {}).get("3", {})
    sa = summary.get("setting_a", {})
    return {
        "setting_a_acc": sa.get("accuracy"),
        "setting_a_macro_f1": sa.get("macro_f1"),
        "open_set_AUROC": sb.get("open_set_AUROC"),
        "FPR_at_95TPR": sb.get("FPR_at_95TPR"),
        "shot3_acc": sc3.get("accuracy"),
    }


def _apply_base_config(cfg: Config, base_cfg: Dict):
    for k, v in base_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)


def _need_prepare(cfg: Config) -> bool:
    prepared_dir = Path(cfg.prepared_dir)
    grid_info = prepared_dir / "grid_info.json"
    metadata = prepared_dir / "metadata.csv"
    split_json = prepared_dir / "split.json"
    if not (grid_info.exists() and metadata.exists() and split_json.exists()):
        return True
    try:
        info = _load_json(grid_info)
        return not (
            int(info.get("rt_bins", -1)) == int(cfg.rt_bins)
            and int(info.get("mz_bins", -1)) == int(cfg.mz_bins)
        )
    except Exception:
        return True


def _ensure_prepared(cfg: Config):
    if not _need_prepare(cfg):
        print(f"[{_now()}] prepared_data hit cache: {cfg.prepared_dir}")
        return

    print(f"[{_now()}] preparing data for rt_bins={cfg.rt_bins}, mz_bins={cfg.mz_bins}")
    metadata = scan_dataset(cfg.dataset_root)
    convert_all(metadata, cfg.prepared_dir, cfg)
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = "product_fine" if cfg.product_granularity == "fine" else "product_coarse"
    create_data_split(metadata_csv, cfg, product_col=product_col)


def _build_cfg(base_cfg: Dict, run_name: str, rt_bins: int, mz_bins: int, epochs_override: int | None):
    cfg = Config()
    _apply_base_config(cfg, base_cfg)
    cfg.output_dir = str(OUTPUTS_DIR / run_name)
    cfg.prepared_dir = str(PROJECT_ROOT / "prepared_data" / f"bins_rt{rt_bins}_mz{mz_bins}")
    cfg.rt_bins = int(rt_bins)
    cfg.mz_bins = int(mz_bins)
    # Sweep时仅保留训练必需产物，避免prepare阶段I/O成为瓶颈
    cfg.save_prepare_plots = False
    cfg.save_prepare_tables = False
    cfg.prepare_plot_max_samples = 0
    if epochs_override is not None:
        cfg.epochs = int(epochs_override)
        cfg.min_epochs_before_early_stop = min(
            int(getattr(cfg, "min_epochs_before_early_stop", 0) or 0),
            cfg.epochs,
        )
    return cfg


def _write_run_config(run_dir: Path, run_name: str, cfg: Config):
    _save_json(
        run_dir / "run_config.json",
        {
            "name": run_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(cfg),
        },
    )


def run_one(base_cfg: Dict, run_name: str, rt_bins: int, mz_bins: int, epochs_override: int | None):
    run_dir = OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "evaluation_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        return {
            "name": run_name,
            "rt_bins": rt_bins,
            "mz_bins": mz_bins,
            "status": "cached",
            "metrics": _extract_metrics(summary),
            "summary": str(summary_path),
        }

    cfg = _build_cfg(base_cfg, run_name, rt_bins, mz_bins, epochs_override)
    _write_run_config(run_dir, run_name, cfg)

    log_path = run_dir / "run.log"
    with open(log_path, "a", encoding="utf-8") as lf:
        tee = Tee(os.sys.stdout, lf)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"\n[{_now()}] START {run_name}")
            print(f"output_dir={cfg.output_dir}")
            print(f"prepared_dir={cfg.prepared_dir}")
            print(f"rt_bins={cfg.rt_bins}, mz_bins={cfg.mz_bins}, epochs={cfg.epochs}")
            _ensure_prepared(cfg)
            train_single_model(cfg)
            evaluate_single_model(cfg)
            print(f"[{_now()}] DONE {run_name}")

    summary = _load_json(summary_path)
    return {
        "name": run_name,
        "rt_bins": rt_bins,
        "mz_bins": mz_bins,
        "status": "done",
        "metrics": _extract_metrics(summary),
        "summary": str(summary_path),
    }


def write_report(results: List[Dict], out_json: Path, out_md: Path):
    _save_json(out_json, {"generated_at": datetime.now().isoformat(), "results": results})

    ranked = sorted(results, key=lambda x: _rank_key(x.get("metrics", {})), reverse=True)
    best = ranked[0] if ranked else None

    lines = [
        "# Bins Sweep Report (iter_auto252)",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- total_runs: {len(results)}",
    ]

    if best is not None:
        m = best["metrics"]
        lines += [
            "",
            "## Best",
            f"- name: {best['name']}",
            f"- rt_bins: {best['rt_bins']}",
            f"- mz_bins: {best['mz_bins']}",
            f"- AUROC: {m.get('open_set_AUROC')}",
            f"- FPR@95: {m.get('FPR_at_95TPR')}",
            f"- 3-shot: {m.get('shot3_acc')}",
            "",
        ]

    lines += [
        "## All Results",
        "| name | rt_bins | mz_bins | AUROC | FPR@95 | 3-shot | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for item in ranked:
        m = item.get("metrics", {})
        lines.append(
            f"| {item['name']} | {item['rt_bins']} | {item['mz_bins']} | "
            f"{m.get('open_set_AUROC')} | {m.get('FPR_at_95TPR')} | {m.get('shot3_acc')} | {item['status']} |"
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Sweep bins for iter_auto252 config")
    parser.add_argument(
        "--base_run",
        default="outputs/iter_auto252_bs16_lr172_a7_p90_fewr50",
        help="Base run directory containing run_config.json",
    )
    parser.add_argument(
        "--combos",
        nargs="+",
        default=["1024x256", "1152x288", "1280x320"],
        help="rt_bins x mz_bins combinations",
    )
    parser.add_argument(
        "--epochs_override",
        type=int,
        default=5,
        help="Override epochs for faster sweep; set <=0 to disable",
    )
    args = parser.parse_args()

    base_run = PROJECT_ROOT / args.base_run
    base_cfg = _load_json(base_run / "run_config.json").get("config", {})
    epochs_override = args.epochs_override if args.epochs_override and args.epochs_override > 0 else None

    results = []
    for combo in args.combos:
        rt_bins, mz_bins = _parse_combo(combo)
        run_name = f"iter252_bins_rt{rt_bins}_mz{mz_bins}"
        if epochs_override is not None:
            run_name += f"_e{epochs_override}"
        item = run_one(base_cfg, run_name, rt_bins, mz_bins, epochs_override)
        results.append(item)

    out_json = OUTPUTS_DIR / "BIN_SWEEP_252_RESULTS.json"
    out_md = OUTPUTS_DIR / "BIN_SWEEP_252_REPORT.md"
    write_report(results, out_json, out_md)

    ranked = sorted(results, key=lambda x: _rank_key(x.get("metrics", {})), reverse=True)
    if ranked:
        best = ranked[0]
        print("\nBEST:")
        print(
            f"{best['name']} (rt_bins={best['rt_bins']}, mz_bins={best['mz_bins']}) "
            f"AUROC={best['metrics'].get('open_set_AUROC')} "
            f"FPR95={best['metrics'].get('FPR_at_95TPR')} "
            f"shot3={best['metrics'].get('shot3_acc')}"
        )
        print(f"report={out_md}")


if __name__ == "__main__":
    main()
