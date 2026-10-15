#!/usr/bin/env python3
"""Audit raw GC-MS data for the fixed preprocessing manifest.

Run from the project root. The default backend uses the project's reader and
records TIC-only fallbacks explicitly; use --require-matrix for a full matrix
audit that must not silently accept TIC-only data.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_reader import _read_tic_csv, read_gcms_data
from prepare_preprocessing_manifest import manifest_fingerprint


DEFAULT_MANIFEST = PROJECT_ROOT / "new_prepared_data_preproc_v1"


def _monotonic(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        return False
    return bool(np.all(np.diff(values) >= 0))


def _record_metrics(row, mode, rts, data, mz_axis=None):
    rt = np.asarray(rts, dtype=np.float64).reshape(-1)
    flags = []
    if rt.size == 0:
        flags.append("empty_rt_axis")
    elif not np.isfinite(rt).all():
        flags.append("nonfinite_rt")

    metrics = {
        "data_mode": mode,
        "n_scans": int(rt.size),
        "rt_min": float(np.nanmin(rt)) if rt.size and np.isfinite(rt).any() else np.nan,
        "rt_max": float(np.nanmax(rt)) if rt.size and np.isfinite(rt).any() else np.nan,
        "rt_nondecreasing": _monotonic(rt),
    }
    if rt.size > 1 and np.isfinite(rt).all() and not metrics["rt_nondecreasing"]:
        flags.append("rt_not_nondecreasing")
    if rt.size and np.isfinite(rt).all() and (float(np.min(rt)) < 0 or float(np.max(rt)) > 1000):
        flags.append("rt_range_unusual")

    if mode == "matrix":
        signal = np.asarray(data, dtype=np.float64)
        mz = np.asarray(mz_axis, dtype=np.float64).reshape(-1)
        metrics.update({
            "matrix_shape": "x".join(str(value) for value in signal.shape),
            "n_mz": int(mz.size),
            "mz_min": float(np.nanmin(mz)) if mz.size and np.isfinite(mz).any() else np.nan,
            "mz_max": float(np.nanmax(mz)) if mz.size and np.isfinite(mz).any() else np.nan,
            "mz_nondecreasing": _monotonic(mz),
        })
        if mz.size == 0:
            flags.append("empty_mz_axis")
        elif not np.isfinite(mz).all():
            flags.append("nonfinite_mz")
        if mz.size > 1 and np.isfinite(mz).all() and not metrics["mz_nondecreasing"]:
            flags.append("mz_not_nondecreasing")
        if mz.size and np.isfinite(mz).all() and (float(np.min(mz)) < 0 or float(np.max(mz)) > 5000):
            flags.append("mz_range_unusual")
        if signal.ndim != 2 or signal.shape != (rt.size, mz.size):
            flags.append("matrix_shape_mismatch")
        if signal.size == 0:
            flags.append("empty_signal")
        else:
            finite = np.isfinite(signal)
            metrics["signal_size"] = int(signal.size)
            metrics["nonfinite_signal_count"] = int(signal.size - finite.sum())
            metrics["negative_signal_count"] = int(np.count_nonzero(signal[finite] < 0))
            metrics["nonzero_signal_count"] = int(np.count_nonzero(signal[finite]))
            metrics["signal_sum"] = float(np.sum(signal[finite], dtype=np.float64))
            metrics["signal_min"] = float(np.min(signal[finite])) if finite.any() else np.nan
            metrics["signal_max"] = float(np.max(signal[finite])) if finite.any() else np.nan
            if not finite.all():
                flags.append("nonfinite_signal")
            if finite.any() and not np.any(signal[finite] != 0):
                flags.append("all_zero_signal")
            if metrics["negative_signal_count"]:
                flags.append("negative_signal")
    elif mode == "tic":
        signal = np.asarray(data, dtype=np.float64).reshape(-1)
        metrics["signal_size"] = int(signal.size)
        finite = np.isfinite(signal)
        metrics["nonfinite_signal_count"] = int(signal.size - finite.sum())
        metrics["negative_signal_count"] = int(np.count_nonzero(signal[finite] < 0))
        metrics["nonzero_signal_count"] = int(np.count_nonzero(signal[finite]))
        metrics["signal_sum"] = float(np.sum(signal[finite], dtype=np.float64))
        metrics["signal_min"] = float(np.min(signal[finite])) if finite.any() else np.nan
        metrics["signal_max"] = float(np.max(signal[finite])) if finite.any() else np.nan
        if signal.size != rt.size:
            flags.append("tic_rt_length_mismatch")
        if signal.size == 0:
            flags.append("empty_signal")
        elif not finite.all():
            flags.append("nonfinite_signal")
        elif not np.any(signal != 0):
            flags.append("all_zero_signal")
        else:
            flags.append("tic_only_matrix_unchecked")
        if metrics["negative_signal_count"]:
            flags.append("negative_signal")
    elif mode == "spectra":
        spectra = data or []
        mz_blocks = [np.asarray(item[0], dtype=np.float64).reshape(-1) for item in spectra]
        signal_blocks = [np.asarray(item[1], dtype=np.float64).reshape(-1) for item in spectra]
        mz = np.concatenate(mz_blocks) if mz_blocks else np.asarray([], dtype=np.float64)
        signal = np.concatenate(signal_blocks) if signal_blocks else np.asarray([], dtype=np.float64)
        finite_mz = np.isfinite(mz)
        finite = np.isfinite(signal)
        metrics.update({
            "matrix_shape": "",
            "n_mz": int(mz.size),
            "mz_min": float(np.min(mz[finite_mz])) if finite_mz.any() else np.nan,
            "mz_max": float(np.max(mz[finite_mz])) if finite_mz.any() else np.nan,
            "mz_nondecreasing": all(_monotonic(block) for block in mz_blocks if block.size > 1),
            "signal_size": int(signal.size),
            "nonfinite_signal_count": int(signal.size - finite.sum()),
            "negative_signal_count": int(np.count_nonzero(signal[finite] < 0)),
            "nonzero_signal_count": int(np.count_nonzero(signal[finite])),
            "signal_sum": float(np.sum(signal[finite], dtype=np.float64)),
            "signal_min": float(np.min(signal[finite])) if finite.any() else np.nan,
            "signal_max": float(np.max(signal[finite])) if finite.any() else np.nan,
        })
        if len(spectra) != rt.size:
            flags.append("spectra_rt_length_mismatch")
        if not finite_mz.all():
            flags.append("nonfinite_mz")
        if not finite.all():
            flags.append("nonfinite_signal")
        if not np.any(finite):
            flags.append("empty_signal")
        elif not np.any(signal[finite] != 0):
            flags.append("all_zero_signal")
        if metrics["negative_signal_count"]:
            flags.append("negative_signal")
    else:
        raise ValueError(f"Unsupported data mode: {mode}")

    hard_errors = {
        "empty_rt_axis", "nonfinite_rt", "empty_mz_axis", "nonfinite_mz",
        "matrix_shape_mismatch", "empty_signal", "nonfinite_signal",
        "all_zero_signal", "tic_rt_length_mismatch", "spectra_rt_length_mismatch",
    }
    if hard_errors.intersection(flags):
        status = "invalid"
    elif mode == "tic":
        status = "tic_only"
    elif mode == "spectra":
        status = "spectra_only"
    elif flags:
        status = "review"
    else:
        status = "pass"
    metrics["quality_status"] = status
    metrics["quality_flags"] = ";".join(flags)
    return metrics


def _audit_one(row, require_matrix, tic_only=False):
    started = time.perf_counter()
    result = {
        "metadata_index": int(row["metadata_index"]),
        "record_id": str(row["record_id"]),
        "sample_id": str(row["sample_id"]),
        "source_relpath": str(row["source_relpath"]),
        "product_fine": str(row["product_fine"]),
        "batch_name": str(row["batch_name"]),
        "split_role": str(row["split_role"]),
        "raw_relpath": str(row["raw_relpath"]),
        "quality_status": "error",
        "data_mode": "",
        "quality_flags": "",
        "error": "",
        "reader_warnings": "",
    }
    warning_messages = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if tic_only:
                rts, data = _read_tic_csv(Path(row["d_path"]))
                mode, mz_axis = "tic", None
            elif require_matrix:
                mode, rts, data, mz_axis = read_gcms_data(row["d_path"], backend="rainbow")
            else:
                mode, rts, data, mz_axis = read_gcms_data(row["d_path"], backend="auto")
            warning_messages.extend(str(item.message) for item in caught)
        if require_matrix and mode != "matrix":
            raise ValueError(f"Expected matrix mode, got {mode}")
        if mode == "matrix":
            result.update(_record_metrics(row, mode, rts, mz_axis, data))
        else:
            result.update(_record_metrics(row, mode, rts, data))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if require_matrix and "rainbow" in result["error"].lower() and "no module" in result["error"].lower():
            result["quality_status"] = "blocked_missing_reader"
    if warning_messages:
        result["reader_warnings"] = " | ".join(warning_messages)
    result["read_seconds"] = round(time.perf_counter() - started, 4)
    return result


def _fingerprint(metadata):
    columns = ["record_id", "source_relpath", "product_fine", "batch_name", "split_role", "raw_relpath"]
    payload = metadata[columns].fillna("").astype(str).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_audit(manifest_dir, require_matrix=False, tic_only=False, limit=None, progress=False):
    manifest_dir = Path(manifest_dir)
    metadata = pd.read_csv(manifest_dir / "metadata.csv", keep_default_na=False)
    expected_fingerprint = json.loads((manifest_dir / "split.json").read_text(encoding="utf-8"))["manifest_fingerprint"]
    if metadata.empty:
        raise ValueError("Manifest contains no included samples")
    actual_fingerprint = manifest_fingerprint(metadata)
    if actual_fingerprint != expected_fingerprint:
        raise ValueError("Manifest fingerprint mismatch; refusing to audit changed metadata")
    rows = metadata if limit is None else metadata.head(max(0, int(limit)))
    results = []
    for number, (_, row) in enumerate(rows.iterrows(), start=1):
        results.append(_audit_one(row, require_matrix, tic_only=tic_only))
        if progress and (number == 1 or number % 100 == 0 or number == len(rows)):
            print(f"Audited {number}/{len(rows)} samples", file=sys.stderr, flush=True)
    table = pd.DataFrame(results)
    counts = table["quality_status"].value_counts().to_dict()
    modes = table["data_mode"].replace("", "read_error").value_counts().to_dict()
    flag_counts = {}
    for value in table["quality_flags"].fillna(""):
        for flag in filter(None, str(value).split(";")):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    product_status = {}
    if "product_fine" in table and "quality_status" in table:
        for (product, status), count in table.groupby(
            ["product_fine", "quality_status"], dropna=False
        ).size().items():
            product_status.setdefault(str(product), {})[str(status)] = int(count)
    summary = {
        "manifest_dir": str(manifest_dir.resolve()),
        "manifest_fingerprint": expected_fingerprint,
        "metadata_fingerprint_verified": True,
        "samples_requested": int(len(rows)),
        "samples_in_manifest": int(len(metadata)),
        "scope": (
            "matrix_required" if require_matrix else
            "tic_only" if tic_only else
            "reader_auto_with_tic_fallback"
        ),
        "matrix_checked_count": int((table.get("data_mode", pd.Series(dtype=str)) == "matrix").sum()),
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "mode_counts": {str(key): int(value) for key, value in modes.items()},
        "quality_flag_counts": flag_counts,
        "quality_status_by_product": product_status,
        "median_read_seconds": float(table["read_seconds"].median()) if "read_seconds" in table else None,
        "all_included_samples_pass_matrix_audit": bool(
            limit is None and len(table) == len(metadata)
            and (table["quality_status"] == "pass").all()
            and (table["data_mode"] == "matrix").all()
        ),
        "note": (
            "Matrix mode was required; TIC fallback was not accepted."
            if require_matrix else
            "TIC files were parsed directly; matrix files were not read."
            if tic_only else
            "TIC-only results do not establish raw matrix readability or validity."
        ),
    }
    return table, summary


def save_audit(manifest_dir, table, summary, refresh=False, output_prefix="raw_data_audit"):
    manifest_dir = Path(manifest_dir)
    csv_path = manifest_dir / f"{output_prefix}.csv"
    json_path = manifest_dir / f"{output_prefix}_summary.json"
    if not refresh and (csv_path.exists() or json_path.exists()):
        raise FileExistsError("Audit output exists; pass --refresh to replace these audit files")
    matrix = table[table["data_mode"] == "matrix"] if "data_mode" in table else table.iloc[0:0]
    if not matrix.empty:
        summary["matrix_dimensions"] = {
            "n_scans_min": int(matrix["n_scans"].min()),
            "n_scans_median": float(matrix["n_scans"].median()),
            "n_scans_max": int(matrix["n_scans"].max()),
            "n_scans_distribution": {
                str(key): int(value) for key, value in matrix["n_scans"].value_counts().sort_index().items()
            },
            "n_mz_min": int(matrix["n_mz"].min()),
            "n_mz_median": float(matrix["n_mz"].median()),
            "n_mz_max": int(matrix["n_mz"].max()),
            "n_mz_unique_values": int(matrix["n_mz"].nunique()),
            "unique_matrix_shapes": int(matrix["matrix_shape"].nunique()),
            "rt_min": float(matrix["rt_min"].min()),
            "rt_max": float(matrix["rt_max"].max()),
            "mz_min": float(matrix["mz_min"].min()),
            "mz_max": float(matrix["mz_max"].max()),
        }
        summary["matrix_signal_checks"] = {
            "nonfinite_values": int(matrix["nonfinite_signal_count"].sum()),
            "negative_values": int(matrix["negative_signal_count"].sum()),
            "all_zero_matrices": int((matrix["nonzero_signal_count"] == 0).sum()),
            "samples_max_at_8388096": int((matrix["signal_max"] == 8388096).sum()),
            "samples_max_at_or_above_8000000": int((matrix["signal_max"] >= 8000000).sum()),
        }
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output_prefix == "raw_matrix_audit" or (
        "data_mode" in table and (table["data_mode"] == "matrix").any()
    ):
        _save_matrix_dimension_plot(manifest_dir / "raw_matrix_dimensions.png", table)
    elif output_prefix == "raw_tic_audit" and (manifest_dir / "raw_data_audit.csv").is_file():
        comparison, comparison_summary = compare_matrix_to_tic(
            manifest_dir / "raw_data_audit.csv", csv_path
        )
        comparison.to_csv(
            manifest_dir / "raw_matrix_tic_comparison.csv", index=False, encoding="utf-8-sig"
        )
        summary["matrix_tic_comparison"] = comparison_summary
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compare_matrix_to_tic(matrix_csv, tic_csv, rt_tolerance_minutes=0.01):
    matrix = pd.read_csv(matrix_csv, keep_default_na=False)
    tic = pd.read_csv(tic_csv, keep_default_na=False)
    left = matrix[[
        "record_id", "sample_id", "source_relpath", "product_fine", "batch_name",
        "n_scans", "rt_min", "rt_max", "quality_status",
    ]].rename(columns={
        "n_scans": "matrix_n_scans", "rt_min": "matrix_rt_min",
        "rt_max": "matrix_rt_max", "quality_status": "matrix_status",
    })
    right = tic[["record_id", "n_scans", "rt_min", "rt_max", "quality_status"]].rename(columns={
        "n_scans": "tic_n_scans", "rt_min": "tic_rt_min",
        "rt_max": "tic_rt_max", "quality_status": "tic_status",
    })
    comparison = left.merge(right, on="record_id", how="outer", indicator=True)
    comparison["matrix_tic_scan_delta"] = (
        pd.to_numeric(comparison["matrix_n_scans"], errors="coerce")
        - pd.to_numeric(comparison["tic_n_scans"], errors="coerce")
    )
    comparison["rt_start_delta_minutes"] = (
        pd.to_numeric(comparison["matrix_rt_min"], errors="coerce")
        - pd.to_numeric(comparison["tic_rt_min"], errors="coerce")
    )
    comparison["rt_end_delta_minutes"] = (
        pd.to_numeric(comparison["matrix_rt_max"], errors="coerce")
        - pd.to_numeric(comparison["tic_rt_max"], errors="coerce")
    )
    same_count = comparison["matrix_tic_scan_delta"] == 0
    same_range = (
        comparison["rt_start_delta_minutes"].abs().le(rt_tolerance_minutes)
        & comparison["rt_end_delta_minutes"].abs().le(rt_tolerance_minutes)
    )
    comparison["coverage_comparison"] = np.select(
        [comparison["_merge"] != "both", ~same_count, ~same_range],
        [comparison["_merge"].map({"left_only": "tic_missing", "right_only": "matrix_missing"}),
         "scan_count_mismatch", "rt_range_mismatch"],
        default="match",
    )
    counts = comparison["coverage_comparison"].value_counts().to_dict()
    mismatches = comparison[comparison["coverage_comparison"] != "match"]
    return comparison.drop(columns="_merge"), {
        "samples_compared": int(len(comparison)),
        "coverage_status_counts": {str(key): int(value) for key, value in counts.items()},
        "mismatch_record_ids": mismatches["record_id"].astype(str).tolist(),
        "rt_endpoint_tolerance_minutes": float(rt_tolerance_minutes),
        "interpretation": "Coverage differences are review flags; they do not by themselves prove corrupted spectra.",
    }


def inspect_coverage_mismatches(manifest_dir):
    manifest_dir = Path(manifest_dir)
    comparisons = pd.read_csv(manifest_dir / "raw_matrix_tic_comparison.csv")
    metadata = pd.read_csv(manifest_dir / "metadata.csv", keep_default_na=False).set_index("record_id")
    mismatches = comparisons[comparisons["coverage_comparison"] != "match"]
    rows = []
    for _, item in mismatches.iterrows():
        record = metadata.loc[item["record_id"]]
        mode, rts, mzs, matrix = read_gcms_data(record["d_path"], backend="rainbow")
        if mode != "matrix":
            raise ValueError(f"Expected matrix mode for {record['source_relpath']}, got {mode}")
        rts = np.asarray(rts, dtype=np.float64)
        mzs = np.asarray(mzs, dtype=np.float64)
        matrix = np.asarray(matrix, dtype=np.float64)
        cutoff = float(item["tic_rt_min"])
        early = rts < cutoff
        early_matrix = matrix[early]
        peak_flat_index = int(np.argmax(early_matrix)) if early_matrix.size else -1
        if peak_flat_index >= 0:
            peak_row, peak_col = np.unravel_index(peak_flat_index, early_matrix.shape)
            early_rts = rts[early]
            early_peak_rt = float(early_rts[peak_row])
            early_peak_mz = float(mzs[peak_col])
            early_max = float(early_matrix[peak_row, peak_col])
        else:
            early_peak_rt = np.nan
            early_peak_mz = np.nan
            early_max = np.nan
        rows.append({
            "record_id": str(item["record_id"]),
            "sample_id": str(item["sample_id"]),
            "source_relpath": str(item["source_relpath"]),
            "product_fine": str(item["product_fine"]),
            "batch_name": str(item["batch_name"]),
            "split_role": str(record["split_role"]),
            "matrix_n_scans": int(len(rts)),
            "tic_n_scans": int(item["tic_n_scans"]),
            "matrix_tic_scan_delta": int(item["matrix_tic_scan_delta"]),
            "tic_rt_start": cutoff,
            "matrix_rt_start": float(rts.min()),
            "matrix_rt_end": float(rts.max()),
            "pre_tic_scan_count": int(early.sum()),
            "pre_tic_signal_sum": float(np.sum(early_matrix, dtype=np.float64)),
            "matrix_signal_sum": float(np.sum(matrix, dtype=np.float64)),
            "pre_tic_signal_fraction": float(np.sum(early_matrix) / np.sum(matrix)) if np.sum(matrix) else np.nan,
            "pre_tic_max_signal": early_max,
            "pre_tic_max_rt": early_peak_rt,
            "pre_tic_max_mz": early_peak_mz,
            "review_reason": "Matrix has substantial signal before TIC begins; confirm acquisition/source behavior.",
        })
    return pd.DataFrame(rows)


def _save_matrix_dimension_plot(path, table):
    matrix = table[table["data_mode"] == "matrix"]
    if matrix.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, (scan_axis, mz_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    scans = matrix["n_scans"].value_counts().sort_index()
    scan_axis.bar(scans.index.astype(str), scans.values, color="#429B83", width=0.68)
    scan_axis.set_title("每个样本的 RT 扫描点数", loc="left")
    scan_axis.set_xlabel("扫描点数")
    scan_axis.set_ylabel("样本数")
    scan_axis.tick_params(axis="x", rotation=25)
    scan_axis.bar_label(scan_axis.containers[0], padding=3, fontsize=9)
    mz_axis.hist(matrix["n_mz"], bins=18, color="#E5A33E", edgecolor="white")
    mz_axis.axvline(
        matrix["n_mz"].median(), color="#386CB0", linestyle="--", linewidth=1.5,
        label=f"中位数 {matrix['n_mz'].median():.0f}",
    )
    mz_axis.set_title("每个样本的 m/z 轴点数", loc="left")
    mz_axis.set_xlabel("m/z 轴点数")
    mz_axis.set_ylabel("样本数")
    mz_axis.legend(frameon=False)
    for axis in (scan_axis, mz_axis):
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#DDDDDD", alpha=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle("全量原始矩阵维度", x=0.07, ha="left", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--require-matrix", action="store_true",
                        help="Require rainbow matrix parsing; never accept TIC fallback")
    parser.add_argument("--tic-only", action="store_true",
                        help="Read tic_front.csv directly without reading data.ms")
    parser.add_argument("--inspect-coverage-mismatches", action="store_true",
                        help="Inspect raw matrix signal before TIC begins for flagged records")
    parser.add_argument("--reader-package-path", default="",
                        help="Optional local package directory containing rainbow-api")
    parser.add_argument("--limit", type=int, default=None, help="Audit only the first N manifest rows")
    parser.add_argument("--refresh", action="store_true", help="Replace prior raw audit output files")
    args = parser.parse_args()
    if args.require_matrix and args.tic_only:
        parser.error("--require-matrix and --tic-only cannot be combined")
    if args.reader_package_path:
        reader_path = Path(args.reader_package_path).resolve()
        if not reader_path.is_dir():
            parser.error(f"Reader package path does not exist: {reader_path}")
        sys.path.append(str(reader_path))
    if args.inspect_coverage_mismatches:
        try:
            review = inspect_coverage_mismatches(args.manifest_dir)
        except FileExistsError as exc:
            parser.error(str(exc))
        review_path = Path(args.manifest_dir) / "raw_coverage_mismatch_review.csv"
        if review_path.exists() and not args.refresh:
            parser.error("Coverage review exists; pass --refresh to replace it")
        review.to_csv(review_path, index=False, encoding="utf-8-sig")
        print(review.to_string(index=False))
        return
    table, summary = run_audit(
        args.manifest_dir, require_matrix=args.require_matrix, tic_only=args.tic_only,
        limit=args.limit, progress=args.limit is None,
    )
    prefix = "raw_data_audit" if args.require_matrix else "raw_tic_audit" if args.tic_only else "raw_data_audit"
    save_audit(args.manifest_dir, table, summary, refresh=args.refresh, output_prefix=prefix)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_matrix and summary["matrix_checked_count"] != summary["samples_requested"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
