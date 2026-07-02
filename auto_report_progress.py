"""Auto reporter for unified new_outputs iteration progress.

Reads AUTO3 result streams from new_outputs and appends concise progress reports to
new_outputs/ITERATION_PROGRESS.md.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "new_outputs"
RESULTS_JSONL = OUTPUTS_DIR / "AUTO_SEARCH_RESULTS.jsonl"
PRACTICAL_RESULTS_JSONL = OUTPUTS_DIR / "AUTO_PRACTICAL_CANDIDATES.jsonl"
REPORT_MD = OUTPUTS_DIR / "ITERATION_PROGRESS.md"
STATE_JSON = OUTPUTS_DIR / ".auto_report_state.json"

TARGETS = {
    "setting_b_open_set_AUROC_min": 0.88,
    "setting_b_fpr95_max": 0.35,
    "setting_c_1shot_acc_min": 0.70,
    "setting_c_3shot_acc_min": 0.95,
    "known_unknown_gap_min": 0.20,
}

SEARCH_GUARDS = {
    "setting_a_accuracy_min": 0.84,
    "setting_a_balanced_acc_min": 0.66,
    "known_unknown_gap_min": 0.30,
}


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except Exception:
        return default


def _fmt_float(value: Any) -> str:
    try:
        result = float(value)
        if math.isnan(result):
            return "NA"
        return f"{result:.4f}"
    except Exception:
        return "NA"


def _rank_metrics(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    fpr95 = _safe_float(metrics.get("FPR_at_95TPR"), 1e9)
    auroc = _safe_float(metrics.get("open_set_AUROC"), -1.0)
    gap = _safe_float(metrics.get("known_unknown_gap"), -1.0)
    a_acc = _safe_float(metrics.get("setting_a_accuracy"), -1.0)
    a_bal = _safe_float(metrics.get("setting_a_balanced_acc"), -1.0)
    shot1 = _safe_float(metrics.get("shot1_acc"), -1.0)
    return (-fpr95, auroc, gap, a_acc + 0.4 * a_bal, shot1)


def _load_state() -> dict[str, Any]:
    if not STATE_JSON.exists():
        return {"last_index": 0}
    try:
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {"last_index": 0}


def _save_state(state: dict[str, Any]) -> None:
    STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_jsonl_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _parse_jsonl(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except Exception:
            continue
    return rows


def _collect_completed_metrics_from_summaries() -> list[tuple[str, dict[str, Any]]]:
    pairs: list[tuple[str, dict[str, Any]]] = []
    for path in OUTPUTS_DIR.glob("**/evaluation_summary.json"):
        run_dir = path.parent
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        sb = summary.get("setting_b", {}) or {}
        sa = summary.get("setting_a", {}) or {}
        sc1 = (summary.get("setting_c", {}) or {}).get("1", {}) or {}
        sc3 = (summary.get("setting_c", {}) or {}).get("3", {}) or {}
        known_mean = sb.get("known_score_mean")
        unknown_mean = sb.get("unknown_score_mean")
        gap = None
        if known_mean is not None and unknown_mean is not None:
            try:
                gap = float(known_mean) - float(unknown_mean)
            except Exception:
                gap = None

        metrics = {
            "setting_a_accuracy": sa.get("accuracy"),
            "setting_a_balanced_acc": sa.get("balanced_acc"),
            "open_set_AUROC": sb.get("open_set_AUROC"),
            "FPR_at_95TPR": sb.get("FPR_at_95TPR"),
            "known_unknown_gap": gap,
            "shot1_acc": sc1.get("accuracy"),
            "shot3_acc": sc3.get("accuracy"),
        }
        pairs.append((str(run_dir.relative_to(OUTPUTS_DIR)), metrics))
    return pairs


def _best_run_snapshot() -> tuple[str | None, dict[str, Any] | None]:
    pairs = _collect_completed_metrics_from_summaries()
    if not pairs:
        return None, None
    run_name, metrics = max(pairs, key=lambda item: _rank_metrics(item[1]))
    return run_name, metrics


def _gap_to_target(metrics: dict[str, Any]) -> dict[str, Any]:
    auroc_gap = max(0.0, TARGETS["setting_b_open_set_AUROC_min"] - _safe_float(metrics.get("open_set_AUROC"), -1.0))
    fpr_gap = max(0.0, _safe_float(metrics.get("FPR_at_95TPR"), 1e9) - TARGETS["setting_b_fpr95_max"])
    shot1_gap = max(0.0, TARGETS["setting_c_1shot_acc_min"] - _safe_float(metrics.get("shot1_acc"), -1.0))
    shot3_gap = max(0.0, TARGETS["setting_c_3shot_acc_min"] - _safe_float(metrics.get("shot3_acc"), -1.0))
    gap_gap = max(0.0, TARGETS["known_unknown_gap_min"] - _safe_float(metrics.get("known_unknown_gap"), -1.0))
    return {
        "d_AUROC": auroc_gap,
        "d_FPR95": fpr_gap,
        "d_1shot": shot1_gap,
        "d_3shot": shot3_gap,
        "d_gap": gap_gap,
    }


def _guard_status(metrics: dict[str, Any]) -> str:
    a_acc_ok = _safe_float(metrics.get("setting_a_accuracy"), -1.0) >= SEARCH_GUARDS["setting_a_accuracy_min"]
    a_bal_ok = _safe_float(metrics.get("setting_a_balanced_acc"), -1.0) >= SEARCH_GUARDS["setting_a_balanced_acc_min"]
    gap_ok = _safe_float(metrics.get("known_unknown_gap"), -1.0) >= SEARCH_GUARDS["known_unknown_gap_min"]
    if a_acc_ok and a_bal_ok and gap_ok:
        return "practical"
    failed = []
    if not a_acc_ok:
        failed.append("A_acc")
    if not a_bal_ok:
        failed.append("A_bal")
    if not gap_ok:
        failed.append("gap")
    return "guard_fail:" + ",".join(failed)


def _candidate_summary(config: dict[str, Any]) -> str:
    return (
        f"dir={config.get('search_directive')}, "
        f"bs={config.get('batch_size')}, lr={config.get('lr')}, "
        f"adv={config.get('lambda_adv')}, proto={config.get('lambda_proto')}, "
        f"temp={config.get('supcon_temperature')}, accp={config.get('accept_percentile')}, "
        f"reject={config.get('reject_threshold_factor')}, blocks={config.get('blocks_per_stage')}, "
        f"dropout={config.get('dropout')}, auto_blend={config.get('open_score_auto_blend')}, "
        f"blend_obj={config.get('open_score_blend_objective')}"
    )


def _progress_block(obj: dict[str, Any]) -> list[str]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = obj.get("name")
    status = obj.get("status")
    metrics = obj.get("metrics") or {}
    config = obj.get("config") or {}
    result = obj.get("result") or {}
    best_run, best_metrics = _best_run_snapshot()
    gap = _gap_to_target(metrics)

    lines = [
        "",
        f"- [{now}] REPORT {name}",
        f"  - phase={obj.get('phase')}, status={status}, guard={_guard_status(metrics)}",
        f"  - candidate: {_candidate_summary(config)}",
        (
            f"  - metrics: A_acc={_fmt_float(metrics.get('setting_a_accuracy'))}, "
            f"A_bal={_fmt_float(metrics.get('setting_a_balanced_acc'))}, "
            f"AUROC={_fmt_float(metrics.get('open_set_AUROC'))}, "
            f"FPR95={_fmt_float(metrics.get('FPR_at_95TPR'))}, "
            f"gap={_fmt_float(metrics.get('known_unknown_gap'))}, "
            f"1-shot={_fmt_float(metrics.get('shot1_acc'))}, "
            f"3-shot={_fmt_float(metrics.get('shot3_acc'))}"
        ),
        (
            f"  - gap_to_target: d_AUROC={_fmt_float(gap['d_AUROC'])}, "
            f"d_FPR95={_fmt_float(gap['d_FPR95'])}, d_1shot={_fmt_float(gap['d_1shot'])}, "
            f"d_3shot={_fmt_float(gap['d_3shot'])}, d_gap={_fmt_float(gap['d_gap'])}"
        ),
        (
            f"  - exits: train_exit={result.get('train_exit')}, "
            f"eval_exit={result.get('eval_exit')}"
        ),
    ]

    if best_run and best_metrics:
        lines.append(
            "  - current_best: "
            + f"{best_run} (A_acc={_fmt_float(best_metrics.get('setting_a_accuracy'))}, "
            + f"A_bal={_fmt_float(best_metrics.get('setting_a_balanced_acc'))}, "
            + f"AUROC={_fmt_float(best_metrics.get('open_set_AUROC'))}, "
            + f"FPR95={_fmt_float(best_metrics.get('FPR_at_95TPR'))}, "
            + f"gap={_fmt_float(best_metrics.get('known_unknown_gap'))}, "
            + f"1-shot={_fmt_float(best_metrics.get('shot1_acc'))}, "
            + f"3-shot={_fmt_float(best_metrics.get('shot3_acc'))})"
        )

    return lines


def process_once() -> int:
    state = _load_state()
    last_index = int(state.get("last_index", 0))

    lines = _load_jsonl_lines(RESULTS_JSONL)
    if last_index >= len(lines):
        _save_state({"last_index": len(_parse_jsonl(lines))})
        return 0

    objs = _parse_jsonl(lines)
    new_objs = objs[last_index:]

    report_lines: list[str] = []
    for obj in new_objs:
        if obj.get("phase") != "AUTO3":
            continue
        if obj.get("status") not in {"done", "failed"}:
            continue
        report_lines.extend(_progress_block(obj))

    if report_lines:
        with open(REPORT_MD, "a", encoding="utf-8") as f:
            for line in report_lines:
                f.write(line + "\n")

    _save_state({"last_index": len(objs)})
    return len(report_lines)


def summarize_practical_candidates(limit: int = 10) -> list[str]:
    lines = _load_jsonl_lines(PRACTICAL_RESULTS_JSONL)
    rows = _parse_jsonl(lines)
    rows = [row for row in rows if row.get("phase") == "AUTO3"]
    rows.sort(key=lambda row: _rank_metrics(row.get("metrics") or {}), reverse=True)

    summary_lines: list[str] = []
    for idx, row in enumerate(rows[:limit], start=1):
        m = row.get("metrics") or {}
        summary_lines.append(
            f"{idx}. {row.get('name')}: "
            f"A_acc={_fmt_float(m.get('setting_a_accuracy'))}, "
            f"A_bal={_fmt_float(m.get('setting_a_balanced_acc'))}, "
            f"AUROC={_fmt_float(m.get('open_set_AUROC'))}, "
            f"FPR95={_fmt_float(m.get('FPR_at_95TPR'))}, "
            f"gap={_fmt_float(m.get('known_unknown_gap'))}, "
            f"1-shot={_fmt_float(m.get('shot1_acc'))}, "
            f"3-shot={_fmt_float(m.get('shot3_acc'))}"
        )
    return summary_lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto report unified iteration progress")
    parser.add_argument("--watch", action="store_true", help="Run forever and poll periodically")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument("--show_practical", action="store_true", help="Print current practical candidate ranking")
    args = parser.parse_args()

    if args.show_practical:
        print("=== Practical Candidates ===")
        for line in summarize_practical_candidates():
            print(line)

    if not args.watch:
        process_once()
        return 0

    while True:
        process_once()
        time.sleep(max(args.interval, 5))


if __name__ == "__main__":
    raise SystemExit(main())
