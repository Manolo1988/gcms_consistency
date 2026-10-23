"""Verify that batch-level RT alignment actually removed the drift.

Runs on the server and touches only a handful of tensors, so the output stays
small enough to download: two CSVs, two PNGs, one JSON.

For each batch it builds a pseudo-TIC from channel 0 (sum over m/z), then
estimates the shift against the reference batch by cross-correlation. The
pre-alignment directory acts as a built-in control: it must reproduce the
drift that was measured from the raw .D files, otherwise the estimate itself
is not trustworthy.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_MIN = 0.005          # 0.3 s resampling grid, finer than the 1.17 s tensor axis
RT_LO, RT_HI = 0.0, 40.0
GRID = np.arange(RT_LO, RT_HI + STEP_MIN / 2, STEP_MIN)


def read_metadata(prepared: Path):
    import pandas as pd

    frame = pd.read_csv(prepared / "metadata.csv")
    frame["tensor_abs"] = [
        prepared / str(value).replace("\\", "/") for value in frame["tensor_path"]
    ]
    return frame


def pseudo_tic(path: Path, rt_bins: int) -> np.ndarray:
    """Sum channel 0 over m/z, then resample onto a finer RT grid."""
    tensor = np.load(path)["tensor"]
    profile = tensor[0].sum(axis=1).astype(np.float64)
    src = np.linspace(RT_LO, RT_HI, rt_bins, endpoint=False)
    return np.interp(GRID, src, profile, left=0.0, right=0.0)


def normalize(values: np.ndarray) -> np.ndarray:
    values = values - values.mean()
    norm = np.linalg.norm(values)
    return values / norm if norm > 1e-12 else values


def best_shift(ref: np.ndarray, cur: np.ndarray, max_lag_min=1.5):
    """Return lag in minutes so that cur[i] ~ ref[i + lag]."""
    n = len(ref)
    max_lag = int(round(max_lag_min / STEP_MIN))
    best_lag, best_r = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = ref[lag:], cur[: n - lag]
        else:
            a, b = ref[: n + lag], cur[-lag:]
        if len(a) < 2000:
            continue
        a = a - a.mean()
        b = b - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            continue
        r = float(np.dot(a, b) / denom)
        if r > best_r:
            best_lag, best_r = lag, r
    return best_lag * STEP_MIN, best_r


def batch_offsets(prepared: Path, ref_batch: str, samples_per_batch: int, rt_bins: int):
    frame = read_metadata(prepared)
    frame = frame.sort_values(["batch_name", "seq_no"], na_position="last")

    ref_traces = {}
    for product in frame.loc[frame["batch_name"] == ref_batch, "product_fine"].unique():
        rows = frame[(frame["batch_name"] == ref_batch) & (frame["product_fine"] == product)]
        traces = [normalize(pseudo_tic(p, rt_bins)) for p in rows["tensor_abs"].head(samples_per_batch)
                  if Path(p).exists()]
        if traces:
            ref_traces[product] = np.mean(traces, axis=0)

    per_batch, detail = {}, []
    for batch, group in frame.groupby("batch_name"):
        offsets, corrs = [], []
        for product, ref in ref_traces.items():
            rows = group[group["product_fine"] == product]
            for _, row in rows.head(samples_per_batch).iterrows():
                path = Path(row["tensor_abs"])
                if not path.exists():
                    continue
                lag, corr = best_shift(ref, normalize(pseudo_tic(path, rt_bins)))
                offsets.append(lag * 60.0)
                corrs.append(corr)
                detail.append({"batch": batch, "product": product,
                               "sample_id": row["sample_id"],
                               "offset_seconds": round(lag * 60.0, 2),
                               "correlation": round(corr, 4)})
        per_batch[batch] = {
            "offset_seconds": round(float(np.median(offsets)), 2) if offsets else None,
            "n_products_used": len(offsets),
            "min_correlation": round(min(corrs), 4) if corrs else None,
        }
    return per_batch, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-dir", type=Path, required=True,
                        help="对齐前的 representations/dual_channel_no_pca")
    parser.add_argument("--after-dir", type=Path, required=True,
                        help="对齐后的 representations/dual_channel_no_pca")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ref-batch", default="20240304")
    parser.add_argument("--samples-per-batch", type=int, default=2)
    parser.add_argument("--rt-bins", type=int, default=2048)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"对齐前: {args.before_dir}")
    before, before_detail = batch_offsets(
        args.before_dir, args.ref_batch, args.samples_per_batch, args.rt_bins)
    print(f"对齐后: {args.after_dir}")
    after, after_detail = batch_offsets(
        args.after_dir, args.ref_batch, args.samples_per_batch, args.rt_bins)

    batches = sorted(set(before) | set(after))
    rows = []
    for batch in batches:
        b = before.get(batch, {}).get("offset_seconds")
        a = after.get(batch, {}).get("offset_seconds")
        rows.append({
            "batch": batch,
            "offset_before_seconds": b,
            "offset_after_seconds": a,
            "abs_after_seconds": None if a is None else round(abs(a), 2),
            "delta_seconds": None if (a is None or b is None) else round(a - b, 2),
            "n_products_used": after.get(batch, {}).get("n_products_used"),
            "min_correlation_after": after.get(batch, {}).get("min_correlation"),
        })

    with open(args.output_dir / "rt_shift_before_after.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.output_dir / "rt_shift_samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "batch", "product", "sample_id",
                                               "offset_seconds", "correlation"])
        writer.writeheader()
        for stage, detail in (("before", before_detail), ("after", after_detail)):
            for item in detail:
                writer.writerow({"stage": stage, **item})

    before_vals = np.array([r["offset_before_seconds"] for r in rows
                            if r["offset_before_seconds"] is not None], dtype=float)
    after_vals = np.array([r["offset_after_seconds"] for r in rows
                           if r["offset_after_seconds"] is not None], dtype=float)
    zero_batches = [r["batch"] for r in rows if r["offset_before_seconds"] == 0.0]
    zero_after = [r["offset_after_seconds"] for r in rows
                  if r["offset_before_seconds"] == 0.0 and r["offset_after_seconds"] is not None]

    summary = {
        "reference_batch": args.ref_batch,
        "samples_per_batch": args.samples_per_batch,
        "resampling_step_seconds": round(STEP_MIN * 60, 3),
        "n_batches": len(rows),
        "before": {
            "min": round(float(before_vals.min()), 2) if before_vals.size else None,
            "max": round(float(before_vals.max()), 2) if before_vals.size else None,
            "mean_abs": round(float(np.abs(before_vals).mean()), 2) if before_vals.size else None,
        },
        "after": {
            "min": round(float(after_vals.min()), 2) if after_vals.size else None,
            "max": round(float(after_vals.max()), 2) if after_vals.size else None,
            "mean_abs": round(float(np.abs(after_vals).mean()), 2) if after_vals.size else None,
            "max_abs": round(float(np.abs(after_vals).max()), 2) if after_vals.size else None,
            "n_within_1s": int((np.abs(after_vals) <= 1.0).sum()) if after_vals.size else None,
        },
        "control_batches_offset_zero": {
            "batches": zero_batches,
            "after_seconds": zero_after,
        },
        "note": "positive = elutes earlier than the reference batch; "
                "the before/after columns must reproduce the raw-.D drift before the after column is trusted",
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 5))
    b = [np.nan if r["offset_before_seconds"] is None else r["offset_before_seconds"] for r in rows]
    a = [np.nan if r["offset_after_seconds"] is None else r["offset_after_seconds"] for r in rows]
    ax.plot(x, b, "o-", ms=4, lw=1.2, color="#D85A30", label="before alignment")
    ax.plot(x, a, "s-", ms=4, lw=1.2, color="#1D9E75", label="after alignment")
    ax.axhline(0, color="#2C2C2A", lw=0.8)
    ax.axhline(1, color="#888780", lw=0.6, ls="--")
    ax.axhline(-1, color="#888780", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([r["batch"] for r in rows], rotation=90, fontsize=7)
    ax.set_ylabel("estimated RT shift (s)")
    ax.set_title("residual RT shift per batch, before vs after alignment")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output_dir / "rt_shift_before_after.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(np.abs(after_vals), bins=20, color="#9FE1CB", edgecolor="#0F6E56", linewidth=0.6)
    ax.axvline(1.0, color="#A32D2D", lw=1, ls="--")
    ax.set_xlabel("|residual shift| after alignment (s)")
    ax.set_ylabel("batches")
    ax.set_title("how close to zero did the alignment land?")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(args.output_dir / "rt_shift_after_hist.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n批次明细：")
    for r in rows:
        print(f"{r['batch']}  before={str(r['offset_before_seconds']):>7}  "
              f"after={str(r['offset_after_seconds']):>6}  n={r['n_products_used']}")


if __name__ == "__main__":
    main()
