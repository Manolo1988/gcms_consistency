"""Auto reporter for iteration progress.

Reads outputs/AUTO_SEARCH_RESULTS.jsonl and appends concise progress reports to
outputs/ITERATION_PROGRESS.md for AUTO3 runs.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_JSONL = OUTPUTS_DIR / "AUTO_SEARCH_RESULTS.jsonl"
REPORT_MD = OUTPUTS_DIR / "ITERATION_PROGRESS.md"
STATE_JSON = OUTPUTS_DIR / ".auto_report_state.json"

TARGETS = {
    "auroc_min": 0.61,
    "fpr95_max": 0.70,
    "shot3_min": 0.85,
}


def _rank_metrics(m: dict) -> tuple:
    auroc = m.get("open_set_AUROC")
    fpr95 = m.get("FPR_at_95TPR")
    shot3 = m.get("shot3_acc")
    auroc_v = -1.0 if auroc is None else float(auroc)
    fpr_v = 1e9 if fpr95 is None else float(fpr95)
    shot3_v = -1.0 if shot3 is None else float(shot3)
    return (auroc_v, -fpr_v, shot3_v)


def _fmt_float(v):
    if v is None:
        return "NA"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "NA"


def _load_state() -> dict:
    if not STATE_JSON.exists():
        return {"last_index": 0}
    try:
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {"last_index": 0}


def _save_state(state: dict) -> None:
    STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_lines() -> list[str]:
    if not RESULTS_JSONL.exists():
        return []
    return RESULTS_JSONL.read_text(encoding="utf-8").splitlines()


def _parse_jsonl(lines: list[str]) -> list[dict]:
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _collect_completed_metrics_from_summaries() -> list[tuple[str, dict]]:
    pairs = []
    for p in OUTPUTS_DIR.glob("iter*/evaluation_summary.json"):
        run = p.parent.name
        try:
            summary = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        sb = summary.get("setting_b", {})
        sc = summary.get("setting_c", {}).get("3", {})
        m = {
            "open_set_AUROC": sb.get("open_set_AUROC"),
            "FPR_at_95TPR": sb.get("FPR_at_95TPR"),
            "shot3_acc": sc.get("accuracy"),
        }
        pairs.append((run, m))
    return pairs


def _best_run_snapshot() -> tuple[str | None, dict | None]:
    pairs = _collect_completed_metrics_from_summaries()
    if not pairs:
        return None, None
    run, m = max(pairs, key=lambda x: _rank_metrics(x[1]))
    return run, m


def _progress_block(obj: dict) -> list[str]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = obj.get("name")
    status = obj.get("status")
    m = obj.get("metrics") or {}

    auroc = m.get("open_set_AUROC")
    fpr95 = m.get("FPR_at_95TPR")
    shot3 = m.get("shot3_acc")

    b_auroc = m.get("baseline_open_set_AUROC")
    b_fpr95 = m.get("baseline_FPR_at_95TPR")
    b_shot3 = m.get("baseline_shot3_acc")

    d_auroc = m.get("delta_vs_baseline_open_set_AUROC")
    d_fpr95 = m.get("delta_vs_baseline_FPR_at_95TPR")
    d_shot3 = m.get("delta_vs_baseline_shot3_acc")
    readme_baselines = m.get("readme_baselines") or {}
    readme_deltas = m.get("main_minus_readme_baselines") or {}
    pretrained_info = m.get("pretrained_feature_extractor") or {}

    delta_auroc = None if auroc is None else max(0.0, TARGETS["auroc_min"] - float(auroc))
    delta_fpr95 = None if fpr95 is None else max(0.0, float(fpr95) - TARGETS["fpr95_max"])
    delta_shot3 = None if shot3 is None else max(0.0, TARGETS["shot3_min"] - float(shot3))

    best_run, best_m = _best_run_snapshot()

    lines = [
        "",
        f"- [{now}] REPORT {name}",
        f"  - phase={obj.get('phase')}, status={status}",
        (
            f"  - metrics: AUROC={_fmt_float(auroc)}, "
            f"FPR95={_fmt_float(fpr95)}, 3-shot={_fmt_float(shot3)}"
        ),
        (
            f"  - gap_to_target: d_AUROC={_fmt_float(delta_auroc)}, "
            f"d_FPR95={_fmt_float(delta_fpr95)}, d_3shot={_fmt_float(delta_shot3)}"
        ),
    ]

    if any(v is not None for v in (b_auroc, b_fpr95, b_shot3)):
        lines.append(
            "  - baseline_tic_pca_mlp: "
            + f"AUROC={_fmt_float(b_auroc)}, "
            + f"FPR95={_fmt_float(b_fpr95)}, "
            + f"3-shot={_fmt_float(b_shot3)}"
        )

    if any(v is not None for v in (d_auroc, d_fpr95, d_shot3)):
        lines.append(
            "  - main_minus_baseline: "
            + f"d_AUROC={_fmt_float(d_auroc)}, "
            + f"d_FPR95={_fmt_float(d_fpr95)}, "
            + f"d_3shot={_fmt_float(d_shot3)}"
        )

    if pretrained_info:
        lines.append(
            "  - pretrained_feature_extractor: "
            + f"enabled={pretrained_info.get('enabled')}, "
            + f"arch={pretrained_info.get('arch')}, "
            + f"model={pretrained_info.get('model_path')}"
        )

    for key in ["pca_mahalanobis", "pls_da", "svm_rbf", "tic_pca_mlp"]:
        b = readme_baselines.get(key) or {}
        d = readme_deltas.get(key) or {}
        if not b:
            continue
        lines.append(
            f"  - readme_baseline[{b.get('name', key)}|{b.get('feature_mode')}]: "
            + f"AUROC={_fmt_float(b.get('open_set_AUROC'))}, "
            + f"FPR95={_fmt_float(b.get('FPR_at_95TPR'))}, "
            + f"3-shot={_fmt_float(b.get('shot3_acc'))}, "
            + f"d_AUROC={_fmt_float(d.get('delta_open_set_AUROC'))}, "
            + f"d_FPR95={_fmt_float(d.get('delta_FPR_at_95TPR'))}, "
            + f"d_3shot={_fmt_float(d.get('delta_shot3_acc'))}"
        )

    if best_run and best_m:
        lines.append(
            "  - current_best: "
            + f"{best_run} (AUROC={_fmt_float(best_m.get('open_set_AUROC'))}, "
            + f"FPR95={_fmt_float(best_m.get('FPR_at_95TPR'))}, "
            + f"3-shot={_fmt_float(best_m.get('shot3_acc'))})"
        )

    return lines


def process_once() -> int:
    state = _load_state()
    last_index = int(state.get("last_index", 0))

    lines = _load_lines()
    if last_index >= len(lines):
        return 0

    objs = _parse_jsonl(lines)
    new_objs = objs[last_index:]

    report_lines = []
    for obj in new_objs:
        if obj.get("phase") != "AUTO3":
            continue
        if obj.get("status") not in {"done", "failed", "skip_exists"}:
            continue
        report_lines.extend(_progress_block(obj))

    if report_lines:
        with open(REPORT_MD, "a", encoding="utf-8") as f:
            for line in report_lines:
                f.write(line + "\n")

    _save_state({"last_index": len(objs)})
    return len(report_lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Auto report iteration progress")
    parser.add_argument("--watch", action="store_true", help="Run forever and poll periodically")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    args = parser.parse_args()

    if not args.watch:
        process_once()
        return 0

    while True:
        process_once()
        time.sleep(max(args.interval, 5))


if __name__ == "__main__":
    raise SystemExit(main())
