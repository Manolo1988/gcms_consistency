# -*- coding: utf-8 -*-
"""按批次的 Setting A 体检表。

只读评估目录里的 setting_a_predictions.csv, 不修改评估流程、不重新训练。
每一批输出: 样本数 / 判对 / 判错 / 被拒 / 判对但被拒 / 半径中位 / 主要错误去向。

为什么要看这张表: 总准确率掉几个点时, 往往只是某一批整批翻面
(例如整批被判成相邻产品、或被阈值全部拒掉)。这类问题在汇总指标里
看不出来, 在按批的表里一眼就能看到。

用法 (在项目根目录):
  .venv/bin/python scripts/batch_report.py <目录或csv> [更多目录 ...]
  .venv/bin/python scripts/batch_report.py a/seed41 b/seed41 --label 基线 --label proto
"""
import argparse
import csv
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = "setting_a_predictions.csv"


def disp_width(text):
    """按终端显示宽度计数, 中文算 2 格。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
               for c in str(text))


def pad(text, width, align="left"):
    text = str(text)
    gap = max(0, width - disp_width(text))
    return text + " " * gap if align == "left" else " " * gap + text


def find_csvs(path):
    path = Path(path)
    if path.is_file():
        return [path]
    direct = path / TARGET
    if direct.exists():
        return [direct]
    found = sorted(path.glob("**/" + TARGET))
    if not found:
        raise SystemExit(f"[错误] {path} 下没有找到 {TARGET}")
    return found


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    groups = {}
    for row in rows:
        key = row.get("batch_name") or row.get("batch_label") or "?"
        groups.setdefault(key, []).append(row)

    table = []
    for batch in sorted(groups):
        rs = groups[batch]
        n = len(rs)
        correct = sum(1 for x in rs if str(x.get("correct")) == "1")
        rejected = sum(1 for x in rs if str(x.get("is_known")) == "0")
        correct_rejected = sum(
            1 for x in rs
            if str(x.get("correct")) == "1" and str(x.get("is_known")) == "0"
        )
        radii = [to_float(x.get("radius_norm")) for x in rs]
        radii = [v for v in radii if v == v]
        errors = Counter(
            f'{x.get("true_class")}→{x.get("pred_class")}'
            for x in rs if str(x.get("correct")) != "1"
        )
        top = "  ".join(f"{name}×{cnt}" for name, cnt in errors.most_common(2))
        table.append({
            "batch": batch,
            "n": n,
            "correct": correct,
            "wrong": n - correct,
            "rejected": rejected,
            "correct_rejected": correct_rejected,
            "radius_median": statistics.median(radii) if radii else float("nan"),
            "top_errors": top or "-",
        })
    return table


def print_run(label, csv_path, table):
    if not table:
        print(f"\n[{label}] 空表, 跳过: {csv_path}")
        return
    total_n = sum(r["n"] for r in table)
    total_ok = sum(r["correct"] for r in table)
    total_rej = sum(r["rejected"] for r in table)
    total_ok_rej = sum(r["correct_rejected"] for r in table)

    print(f"\n[{label}]  {csv_path}")
    print("  " + pad("批次", 12) + pad("n", 5, "right") + pad("对", 5, "right")
          + pad("错", 5, "right") + pad("拒", 5, "right")
          + pad("对但拒", 8, "right") + pad("半径中位", 10, "right")
          + "  主要错误去向")
    for r in table:
        print("  " + pad(r["batch"], 12)
              + pad(r["n"], 5, "right") + pad(r["correct"], 5, "right")
              + pad(r["wrong"], 5, "right") + pad(r["rejected"], 5, "right")
              + pad(r["correct_rejected"], 8, "right")
              + pad(f"{r['radius_median']:.2f}", 10, "right")
              + "  " + r["top_errors"])
    print("  " + pad("合计", 12)
          + pad(total_n, 5, "right") + pad(total_ok, 5, "right")
          + pad(total_n - total_ok, 5, "right") + pad(total_rej, 5, "right")
          + pad(total_ok_rej, 8, "right")
          + pad("", 10, "right")
          + f"  准确率 {total_ok / total_n:.3f}")


def print_compare(labels, tables):
    batches = sorted({r["batch"] for t in tables.values() for r in t})
    if len(labels) < 2 or not batches:
        return
    print("\n跨运行对照 (判对/总数)")
    print("  " + pad("批次", 12) + "".join(pad(lbl, 18) for lbl in labels))
    for batch in batches:
        cells = []
        for lbl in labels:
            hit = next((r for r in tables[lbl] if r["batch"] == batch), None)
            cells.append(pad(f"{hit['correct']}/{hit['n']}" if hit else "-", 18))
        print("  " + pad(batch, 12) + "".join(cells))


def main():
    parser = argparse.ArgumentParser(description="按批次的 Setting A 体检表")
    parser.add_argument("inputs", nargs="+", help="评估目录、任务目录, 或 setting_a_predictions.csv")
    parser.add_argument("--label", action="append", default=None,
                        help="每份结果的显示名, 与 inputs 顺序对应; 可重复")
    parser.add_argument("--out", default=None, help="输出目录, 默认 new_result/reports/batch_report_<时间>")
    args = parser.parse_args()

    csvs = []
    for item in args.inputs:
        csvs.extend(find_csvs(item))

    labels = []
    for i, path in enumerate(csvs):
        if args.label and i < len(args.label):
            labels.append(args.label[i])
        else:
            labels.append(path.parent.name)
    seen = Counter(labels)
    labels = [f"{lbl}@{p.parent.parent.name[-8:]}" if seen[lbl] > 1 else lbl
              for lbl, p in zip(labels, csvs)]

    tables = {}
    for label, path in zip(labels, csvs):
        table = summarize(path)
        tables[label] = table
        print_run(label, path, table)
    print_compare(labels, tables)

    out_dir = Path(args.out) if args.out else (
        REPO / "new_result" / "reports" / f"batch_report_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "per_batch.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "batch", "n", "correct", "wrong", "rejected",
                         "correct_but_rejected", "accuracy", "radius_median",
                         "top_errors", "source_csv"])
        for label, path in zip(labels, csvs):
            for r in tables[label]:
                writer.writerow([label, r["batch"], r["n"], r["correct"], r["wrong"],
                                 r["rejected"], r["correct_rejected"],
                                 f"{r['correct'] / r['n']:.4f}",
                                 f"{r['radius_median']:.4f}",
                                 r["top_errors"], str(path)])
    print(f"\n明细已写入: {out_csv}")


if __name__ == "__main__":
    main()
