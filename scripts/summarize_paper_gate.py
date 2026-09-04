#!/usr/bin/env python3
"""Aggregate paper gate runs and apply the predefined acceptance criteria."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper_protocol import paired_bootstrap_ci, percentile_ci, save_json


def _closed_rows(run_dir, gate_result):
    seed = int(gate_result["train_seed"])
    rows = [{
        "method": gate_result["method"],
        "train_seed": seed,
        "accuracy": float(gate_result.get("closed_set", {}).get("accuracy", np.nan)),
        "balanced_accuracy": float(
            gate_result.get("closed_set", {}).get("balanced_acc", np.nan)
        ),
    }]
    summary_path = run_dir / "evaluation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, block in (summary.get("baselines_readme") or {}).items():
            setting = block.get("setting_a") or {}
            rows.append({
                "method": key,
                "train_seed": seed,
                "accuracy": float(setting.get("accuracy", np.nan)),
                "balanced_accuracy": float(setting.get("balanced_acc", np.nan)),
            })
    return rows


def _metric_summary(frame, group_columns):
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["accuracy"].astype(float).to_numpy()
        low, high = percentile_ci(values)
        row = dict(zip(group_columns, keys))
        row.update({
            "mean_accuracy": float(np.mean(values)),
            "std_accuracy": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ci95_low": low,
            "ci95_high": high,
            "n": int(len(values)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _closed_gate(closed, main_method):
    means = closed.groupby("method")["accuracy"].mean().sort_values(ascending=False)
    baselines = means.drop(labels=[main_method], errors="ignore")
    if baselines.empty:
        return {"status": "incomplete", "reason": "no closed-set baseline results"}
    strongest = str(baselines.index[0])
    paired = closed[closed["method"].isin([main_method, strongest])].pivot(
        index="train_seed", columns="method", values="accuracy"
    ).dropna()
    if main_method not in paired or strongest not in paired:
        return {"status": "incomplete", "reason": "main/baseline seeds are not paired"}
    stats = paired_bootstrap_ci(paired[main_method], paired[strongest])
    positive_seeds = int((paired[main_method] > paired[strongest]).sum())
    required_positive = 4 if len(paired) >= 5 else len(paired)
    passed = (
        float(means[main_method]) > float(means[strongest])
        and positive_seeds >= required_positive
        and stats["ci95_low"] > 0.0
    )
    return {
        "status": "pass" if passed else "fail",
        "strongest_baseline": strongest,
        "main_mean_accuracy": float(means[main_method]),
        "baseline_mean_accuracy": float(means[strongest]),
        "paired_seed_count": int(len(paired)),
        "positive_seed_count": positive_seeds,
        "required_positive_seed_count": required_positive,
        **stats,
    }


def _fewshot_gate(fewshot, main_method):
    pooled = fewshot[fewshot["scope"] == "pooled"].copy()
    results = {}
    for shot, shot_frame in pooled.groupby("shot"):
        means = shot_frame.groupby("method")["accuracy"].mean().sort_values(ascending=False)
        baselines = means.drop(labels=[main_method], errors="ignore")
        if baselines.empty:
            results[str(int(shot))] = {"status": "incomplete", "reason": "no baseline"}
            continue
        strongest = str(baselines.index[0])
        paired = shot_frame[shot_frame["method"].isin([main_method, strongest])].pivot_table(
            index=["train_seed", "episode_seed"],
            columns="method",
            values="accuracy",
            aggfunc="first",
        ).dropna()
        stats = paired_bootstrap_ci(paired[main_method], paired[strongest])
        seed_means = paired.assign(
            delta=paired[main_method] - paired[strongest]
        ).groupby(level="train_seed")["delta"].mean()
        positive_seeds = int((seed_means > 0).sum())
        required_positive = 4 if len(seed_means) >= 5 else len(seed_means)
        passed = (
            float(means[main_method]) > float(means[strongest])
            and positive_seeds >= required_positive
            and stats["ci95_low"] > 0.0
        )
        results[str(int(shot))] = {
            "status": "pass" if passed else "fail",
            "strongest_baseline": strongest,
            "main_mean_accuracy": float(means[main_method]),
            "baseline_mean_accuracy": float(means[strongest]),
            "paired_episode_count": int(len(paired)),
            "paired_seed_count": int(len(seed_means)),
            "positive_seed_count": positive_seeds,
            "required_positive_seed_count": required_positive,
            **stats,
        }
    return results


def _write_markdown(path, closed_summary, fewshot_summary, decision):
    def markdown_table(frame):
        columns = list(frame.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = ["# Paper Gate Summary", "", "## Closed-Set Accuracy", ""]
    lines.append(markdown_table(closed_summary))
    lines.extend(["", "## Few-Shot Accuracy", "", markdown_table(fewshot_summary)])
    lines.extend(["", "## Acceptance Decision", "", "```json"])
    lines.append(json.dumps(decision, indent=2, ensure_ascii=False))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--main-method", default="main")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "paper_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    closed_rows = []
    fewshot_frames = []
    for gate_path in root.rglob("paper_gate_result.json"):
        run_dir = gate_path.parent
        gate_result = json.loads(gate_path.read_text(encoding="utf-8"))
        closed_rows.extend(_closed_rows(run_dir, gate_result))
        csv_path = run_dir / gate_result["fewshot_results_csv"]
        if csv_path.exists():
            fewshot_frames.append(pd.read_csv(csv_path))
    if not closed_rows or not fewshot_frames:
        raise SystemExit("No complete paper gate results found")

    closed = pd.DataFrame(closed_rows)
    fewshot = pd.concat(fewshot_frames, ignore_index=True)
    closed.to_csv(output_dir / "closed_set_raw.csv", index=False)
    fewshot.to_csv(output_dir / "fewshot_raw.csv", index=False)
    closed_summary = _metric_summary(closed, ["method"])
    fewshot_summary = _metric_summary(fewshot, ["method", "shot", "scope", "product"])
    closed_summary.to_csv(output_dir / "closed_set_summary.csv", index=False)
    fewshot_summary.to_csv(output_dir / "fewshot_summary.csv", index=False)

    decision = {
        "closed_set": _closed_gate(closed, args.main_method),
        "fewshot": _fewshot_gate(fewshot, args.main_method),
    }
    decision["all_gates_pass"] = (
        decision["closed_set"].get("status") == "pass"
        and all(block.get("status") == "pass" for block in decision["fewshot"].values())
    )
    save_json(decision, output_dir / "gate_decision.json")
    _write_markdown(output_dir / "paper_gate_summary.md", closed_summary, fewshot_summary, decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
