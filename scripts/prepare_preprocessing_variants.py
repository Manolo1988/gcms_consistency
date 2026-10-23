"""Generate comparable input tensors from a frozen sample manifest and split."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import IncrementalPCA
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_preprocessing_manifest import manifest_fingerprint

SITE_PACKAGES = ROOT / ".venv" / "Lib" / "site-packages"
if SITE_PACKAGES.is_dir() and str(SITE_PACKAGES) not in sys.path:
    sys.path.append(str(SITE_PACKAGES))

from data_reader import build_dual_channel, read_gcms_data


VARIANTS = {
    "fixed_grid_log1p": "固定物理 m/z 网格 + log1p + 双通道",
    "dual_channel_no_pca": "统一物理 m/z 轴 + 双通道，不做 PCA",
    "dual_channel_pca": "统一物理 m/z 轴 + 双通道 + 训练集 PCA",
}
H99_MISMATCH_REL_PATH = "20240304/033#H99CS124030403.D"
SIGNAL_CEILING = 8_388_096.0


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_inputs(manifest_dir: Path):
    metadata = pd.read_csv(manifest_dir / "metadata.csv")
    split = json.loads((manifest_dir / "split.json").read_text(encoding="utf-8"))
    if metadata.empty:
        raise ValueError("metadata.csv 为空")
    if "metadata_index" not in metadata or metadata["metadata_index"].astype(int).tolist() != list(range(len(metadata))):
        raise ValueError("metadata_index 必须与 metadata.csv 行位置一致且从 0 连续编号")
    fingerprint = manifest_fingerprint(metadata)
    if fingerprint != split.get("manifest_fingerprint"):
        raise ValueError("metadata.csv 与 split.json 指纹不一致，拒绝生成")

    role_keys = ("train_idx", "val_idx", "test_batch_idx", "test_unknown_idx")
    roles = {}
    for key in role_keys:
        values = [int(i) for i in split.get(key, [])]
        if any(i < 0 or i >= len(metadata) for i in values):
            raise ValueError(f"{key} 包含越界索引")
        roles[key] = values
    joined = [i for values in roles.values() for i in values]
    if len(joined) != len(set(joined)) or set(joined) != set(range(len(metadata))):
        raise ValueError("固定划分必须无重叠并覆盖 metadata.csv 中全部样本")

    metadata = metadata.copy()
    metadata["preprocessing_quality_flags"] = ""
    metadata["preprocessing_quality_note"] = ""
    mismatch_path = manifest_dir / "raw_coverage_mismatch_review.csv"
    if mismatch_path.exists():
        mismatch = pd.read_csv(mismatch_path)
        for _, item in mismatch.iterrows():
            relpath = str(item["source_relpath"])
            mask = metadata["source_relpath"].astype(str) == relpath
            if mask.any():
                metadata.loc[mask, "preprocessing_quality_flags"] = "MATRIX_TIC_RT_MISMATCH"
                metadata.loc[mask, "preprocessing_quality_note"] = (
                    "矩阵包含 TIC 起始前 501 个扫描，早期区间约占矩阵总信号 54%；保留并标记。"
                )

    audit_path = manifest_dir / "raw_data_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        ceiling_rows = audit[pd.to_numeric(audit["signal_max"], errors="coerce") == SIGNAL_CEILING]
        ceiling_paths = set(ceiling_rows["source_relpath"].astype(str))
        mask = metadata["source_relpath"].astype(str).isin(ceiling_paths)
        metadata.loc[mask, "preprocessing_quality_flags"] = metadata.loc[mask, "preprocessing_quality_flags"].map(
            lambda value: ";".join(filter(None, [value, "AT_MAX_INTENSITY_8388096"]))
        )
        empty_note = metadata.loc[mask, "preprocessing_quality_note"].eq("")
        note_index = metadata.loc[mask].index[empty_note]
        metadata.loc[note_index, "preprocessing_quality_note"] = (
            "矩阵最大强度达到 8388096；保留，后续结合仪器信息检查是否存在信号上限。"
        )
    return metadata, split, roles, fingerprint


def _read_raw_matrix(row):
    mode, rts, mzs, matrix = read_gcms_data(str(row["d_path"]), backend="rainbow")
    if mode != "matrix":
        raise RuntimeError(f"要求原始矩阵，实际读取模式为 {mode}")
    rts = np.asarray(rts, dtype=np.float64)
    mzs = np.asarray(mzs, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (len(mzs), len(rts)):
        matrix = matrix.T
    if matrix.shape != (len(rts), len(mzs)):
        raise ValueError(f"矩阵坐标维度不匹配: {matrix.shape}, {len(rts)}, {len(mzs)}")
    if not np.isfinite(rts).all() or not np.isfinite(mzs).all() or not np.isfinite(matrix).all():
        raise ValueError("原始数据包含 NaN 或 Inf")
    if np.any(np.diff(rts) <= 0):
        raise ValueError("RT 坐标未严格递增")
    if np.any(np.diff(mzs) <= 0):
        raise ValueError("m/z 坐标未严格递增")
    if np.any(matrix < 0):
        raise ValueError("原始矩阵包含负强度")
    return rts.astype(np.float32), mzs.astype(np.float32), matrix.astype(np.float32)


def _load_rt_offset_table(path):
    """Read batch -> RT offset in seconds; empty map when no table is given."""
    if path is None:
        return {}
    table = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            batch = (row.get("batch") or "").strip()
            raw = (row.get("offset_seconds") or "").strip()
            if batch and raw:
                table[batch] = float(raw)
    return table


def _resample_raw_to_physical_axes(rts, mzs, matrix, rt_axis, mz_axis, rt_offset_min=0.0):
    logged = np.log1p(np.maximum(matrix, 0.0)).astype(np.float32)
    aligned_mz = interp1d(
        mzs, logged, axis=1, kind="linear", bounds_error=False,
        fill_value=0.0, assume_sorted=True,
    )(mz_axis).astype(np.float32)
    # A positive offset means this run eluted early, so shift the sample later:
    # y'(t) = y(t - offset), i.e. sample the original trace at rt_axis - offset.
    target_rt = rt_axis - rt_offset_min if rt_offset_min else rt_axis
    aligned_rt = interp1d(
        rts, aligned_mz, axis=0, kind="linear", bounds_error=False,
        fill_value=0.0, assume_sorted=True,
    )(target_rt).astype(np.float32)
    return aligned_rt


def _fixed_grid_log1p(rts, mzs, matrix, rt_bins, rt_range, mz_bins, mz_range, rt_offset_min=0.0):
    rt_lo, rt_hi = rt_range
    mz_lo, mz_hi = mz_range
    shifted_rts = rts + rt_offset_min if rt_offset_min else rts
    keep_rt = (shifted_rts >= rt_lo) & (shifted_rts <= rt_hi)
    keep_mz = (mzs >= mz_lo) & (mzs < mz_hi)
    if not keep_rt.any() or not keep_mz.any():
        raise ValueError("样本不在指定 RT 或 m/z 范围内")
    rt_indices = np.floor((shifted_rts[keep_rt] - rt_lo) / (rt_hi - rt_lo) * rt_bins).astype(np.int64)
    rt_indices = np.clip(rt_indices, 0, rt_bins - 1)
    mz_indices = np.floor((mzs[keep_mz] - mz_lo) / (mz_hi - mz_lo) * mz_bins).astype(np.int64)
    mz_indices = np.clip(mz_indices, 0, mz_bins - 1)
    selected = matrix[np.ix_(keep_rt, keep_mz)].astype(np.float64, copy=False)
    flat_indices = np.repeat(rt_indices, len(mz_indices)) * mz_bins + np.tile(mz_indices, len(rt_indices))
    binned = np.bincount(
        flat_indices,
        weights=selected.reshape(-1),
        minlength=rt_bins * mz_bins,
    ).reshape(rt_bins, mz_bins)
    return np.log1p(binned).astype(np.float32)


def _row_path(row, sample_index):
    batch = str(row["batch_name"])
    product = str(row["product_fine"])
    filename = f"{sample_index:04d}_{row['sample_id']}.npz"
    return Path("tensors") / batch / product / filename


def _tensor_shape(tensor):
    """Shape text for the QA table; empty when this variant was not generated."""
    return "x".join(map(str, tensor.shape)) if tensor is not None else ""


def _tensor_nonzero(tensor):
    return int(np.count_nonzero(tensor)) if tensor is not None else ""


def _write_variant_metadata(directory, metadata, original_indices, split, grid_info):
    """Describe generated tensors using the training entry point's input contract."""
    variant_metadata = metadata.copy()
    variant_metadata["tensor_path"] = [
        _row_path(row, index).as_posix()
        for index, (_, row) in zip(original_indices, variant_metadata.iterrows())
    ]
    variant_metadata["tensor_ready"] = True
    variant_metadata.to_csv(directory / "metadata.csv", index=False, encoding="utf-8-sig")
    info = dict(grid_info)
    precomputed = info["input_representation"] == "dual_channel_pca"
    info["preprocessing_frozen"] = True
    info["runtime_input_pca_allowed"] = False
    info["input_pca_precomputed"] = precomputed
    info["input_pca_applied"] = precomputed
    info["input_pca_components"] = int(info["mz_bins"]) if precomputed else None
    info["input_pca_rt_bins"] = int(info["rt_bins"]) if precomputed else None
    _write_json(directory / "grid_info.json", info)
    if split is not None:
        prepared_split = dict(split)
        prepared_split["stage"] = "prepared"
        _write_json(directory / "split.json", prepared_split)


def _save_tensor(path: Path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temp_path, tensor=np.asarray(tensor, dtype=np.float32))
    os.replace(temp_path, path)


def _fit_train_pca(metadata, train_indices, rt_axis, mz_axis, components, batch_rows, progress_every):
    pca = IncrementalPCA(n_components=components)
    pending = []
    pending_rows = 0
    fit_rows = 0
    started = time.monotonic()
    for count, index in enumerate(train_indices, start=1):
        row = metadata.iloc[index]
        rts, mzs, matrix = _read_raw_matrix(row)
        aligned = _resample_raw_to_physical_axes(rts, mzs, matrix, rt_axis, mz_axis)
        dual = build_dual_channel(aligned)
        rows = np.concatenate((dual[0], dual[1]), axis=0)
        pending.append(rows)
        pending_rows += len(rows)
        if pending_rows >= batch_rows:
            batch = np.concatenate(pending, axis=0)
            pca.partial_fit(batch)
            fit_rows += len(batch)
            pending.clear()
            pending_rows = 0
        if count % progress_every == 0 or count == len(train_indices):
            print(f"PCA 拟合训练样本 {count}/{len(train_indices)}，累计拟合行 {fit_rows + pending_rows}", flush=True)
    if pending:
        batch = np.concatenate(pending, axis=0)
        if len(batch) < components:
            raise RuntimeError(f"PCA 最后一批只有 {len(batch)} 行，小于 {components} 维")
        pca.partial_fit(batch)
        fit_rows += len(batch)
    if not hasattr(pca, "components_"):
        raise RuntimeError("训练集 PCA 拟合失败")
    return pca, fit_rows, time.monotonic() - started


def _plot_shapes(output_dir: Path, shapes):
    names = list(shapes)
    labels = {
        "fixed_grid_log1p": "Fixed grid + log1p",
        "dual_channel_no_pca": "Dual channel, no PCA",
        "dual_channel_pca": "Dual channel + PCA",
    }
    widths = [shapes[name][-1] for name in names]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#1B7F79", "#C66A3D", "#5B6C9A"]
    bars = ax.barh([labels[name] for name in names], widths, color=colors[:len(names)], height=0.56)
    ax.set_xlabel("m/z feature width")
    ax.set_title("Feature width by input representation")
    ax.grid(axis="x", color="#D8DDE3", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, width, name in zip(bars, widths, names):
        ax.text(width + max(widths) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{width}  |  {shapes[name]}", va="center", fontsize=9)
    ax.set_xlim(0, max(widths) * 1.35)
    fig.tight_layout()
    fig.savefig(output_dir / "preprocessing_shapes.png", dpi=160)
    plt.close(fig)


def _validate_overwrite_target(output_dir, manifest_dir, fingerprint):
    """Only replace this manifest's previously generated representations directory."""
    if output_dir != (manifest_dir / "representations").resolve():
        raise ValueError("覆盖仅允许目标清单下的 representations，不能覆盖其他目录")
    if not output_dir.is_relative_to(ROOT.resolve()) or output_dir.is_symlink():
        raise ValueError("覆盖目标必须是项目内的普通目录")
    old = json.loads((output_dir / "prepare_summary.json").read_text(encoding="utf-8"))
    if old.get("stage") != "prepared" or old.get("manifest_fingerprint") != fingerprint:
        raise ValueError("旧输入必须是同一冻结清单生成的完整版本")
    for name in (old.get("variants") or VARIANTS):
        info = json.loads((output_dir / name / "grid_info.json").read_text(encoding="utf-8"))
        if info.get("source_manifest_fingerprint") != fingerprint or info.get("input_representation") != name:
            raise ValueError(f"覆盖目标 {name} 的输入身份不匹配")


def _publish(stage, output_dir, manifest_dir, fingerprint, overwrite):
    if not output_dir.exists():
        stage.rename(output_dir)
        return
    if not overwrite:
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_dir}")
    _validate_overwrite_target(output_dir, manifest_dir, fingerprint)
    retired = output_dir.with_name(f".{output_dir.name}.replaced-{os.getpid()}-{time.time_ns()}")
    # Resolve both rename and deletion targets before touching existing files.
    if (retired.parent.resolve() != output_dir.parent.resolve()
            or not retired.resolve().is_relative_to(ROOT.resolve()) or retired.exists()):
        raise ValueError("旧版本临时路径不安全")
    output_dir.rename(retired)
    try:
        stage.rename(output_dir)
    except BaseException:
        retired.rename(output_dir)
        raise
    print("新版本已替换原目录；删除旧版张量。", flush=True)
    if retired.resolve().parent != manifest_dir.resolve() or retired.is_symlink():
        raise ValueError("旧版本删除路径不安全，新版本已发布但旧目录未删除")
    shutil.rmtree(retired)


def prepare(args):
    manifest_dir = args.manifest_dir.resolve()
    output_dir = args.output_dir.resolve()
    overwrite = bool(getattr(args, "overwrite", False))
    requested = tuple(getattr(args, "variants", None) or VARIANTS)
    unknown = [name for name in requested if name not in VARIANTS]
    if unknown:
        raise ValueError(f"未知的输入变体: {unknown}")
    variants = {name: VARIANTS[name] for name in VARIANTS if name in requested}
    if not variants:
        raise ValueError("至少要选择一个输入变体")
    need_fixed = "fixed_grid_log1p" in variants
    need_dual = bool({"dual_channel_no_pca", "dual_channel_pca"} & set(variants))
    need_pca = "dual_channel_pca" in variants
    if overwrite and set(variants) != set(VARIANTS):
        raise ValueError("覆盖已发布目录时必须生成全部变体，避免留下残缺的旧版本")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_dir}")
    if overwrite and args.sample_limit is not None:
        raise ValueError("抽样生成不能覆盖完整输入")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    metadata, split, roles, fingerprint = _load_inputs(manifest_dir)
    if output_dir.exists():
        _validate_overwrite_target(output_dir, manifest_dir, fingerprint)
    original_count = len(metadata)
    if args.sample_limit is not None:
        if args.sample_limit < 1:
            raise ValueError("--sample-limit 必须大于 0")
        metadata = metadata.iloc[:args.sample_limit].copy().reset_index(drop=True)
        selected_original_indices = metadata["metadata_index"].astype(int).tolist()
        selected_set = set(selected_original_indices)
        train_indices = [i for i in roles["train_idx"] if i in selected_set]
        if need_pca and not train_indices:
            raise ValueError("抽样范围没有训练样本，无法做 PCA 冒烟检查")
        print(f"冒烟检查: 处理前 {len(metadata)}/{original_count} 条样本；不是完整数据集。")
    else:
        selected_original_indices = list(range(original_count))
        train_indices = roles["train_idx"]

    rt_axis = np.linspace(args.rt_range[0], args.rt_range[1], args.rt_bins, dtype=np.float32)
    mz_axis = np.arange(args.raw_mz_min, args.raw_mz_max + args.raw_mz_step / 2,
                        args.raw_mz_step, dtype=np.float32)
    if need_pca and len(mz_axis) < args.pca_components:
        raise ValueError("统一物理 m/z 轴宽度小于 PCA 维数")
    if need_pca and len(train_indices) * 2 * args.rt_bins < args.pca_components:
        raise ValueError("训练样本行数不足以拟合指定 PCA 维数")

    stage = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}-{int(time.time())}")
    stage.mkdir(parents=True, exist_ok=False)
    print(f"统一物理 m/z 轴: {mz_axis[0]:g}..{mz_axis[-1]:g}, step={args.raw_mz_step:g}, width={len(mz_axis)}")
    print(f"固定划分训练样本: {len(train_indices)}；PCA 维数: {args.pca_components}")
    print(f"RT 点数: {args.rt_bins}；生成变体: {list(variants)}；临时生成目录: {stage}", flush=True)

    if need_pca:
        pca, fit_rows, fit_seconds = _fit_train_pca(
            metadata=metadata,
            train_indices=train_indices,
            rt_axis=rt_axis,
            mz_axis=mz_axis,
            components=args.pca_components,
            batch_rows=args.pca_batch_rows,
            progress_every=args.progress_every,
        )
    else:
        pca, fit_rows, fit_seconds = None, 0, 0.0
        print("未选择 dual_channel_pca，跳过 PCA 拟合。", flush=True)

    variant_dirs = {name: stage / name for name in variants}
    for directory in variant_dirs.values():
        (directory / "tensors").mkdir(parents=True, exist_ok=True)
    qa_rows = []
    started = time.monotonic()
    offset_map = _load_rt_offset_table(args.rt_offset_table)
    if args.rt_offset_table is not None:
        missing = sorted({str(b) for b in metadata["batch_name"].unique()} - set(offset_map))
        if missing:
            raise ValueError(f"RT 偏移表缺少批次，拒绝按 0 静默处理: {missing}")
        peak = max(abs(v) for v in offset_map.values()) if offset_map else 0.0
        print(f"RT 对齐已启用：{len(offset_map)} 个批次，最大偏移 {peak:.2f} 秒", flush=True)
    for order, (original_index, (_, row)) in enumerate(zip(selected_original_indices, metadata.iterrows()), start=1):
        rts, mzs, matrix = _read_raw_matrix(row)
        rt_offset_min = offset_map.get(str(row["batch_name"]), 0.0) / 60.0
        fixed_tensor = dual_tensor = pca_tensor = None
        tensors = {}
        if need_fixed:
            grid = _fixed_grid_log1p(
                rts, mzs, matrix, args.rt_bins, args.rt_range, args.mz_bins, args.mz_range,
                rt_offset_min=rt_offset_min,
            )
            fixed_tensor = build_dual_channel(grid)
            tensors["fixed_grid_log1p"] = fixed_tensor
        if need_dual:
            raw_base = _resample_raw_to_physical_axes(
                rts, mzs, matrix, rt_axis, mz_axis, rt_offset_min=rt_offset_min,
            )
            dual_tensor = build_dual_channel(raw_base)
            if "dual_channel_no_pca" in variants:
                tensors["dual_channel_no_pca"] = dual_tensor
            if need_pca:
                pca_tensor = np.stack(
                    [pca.transform(dual_tensor[channel]).astype(np.float32)
                     for channel in range(dual_tensor.shape[0])],
                    axis=0,
                )
                tensors["dual_channel_pca"] = pca_tensor
        for name, tensor in tensors.items():
            if tensor.shape[0] != 2 or tensor.shape[1] != args.rt_bins:
                raise ValueError(f"{name} 张量形状不符合预期: {tensor.shape}")
            if not np.isfinite(tensor).all():
                raise ValueError(f"{name} 张量包含 NaN 或 Inf: {row['sample_id']}")
            relative = _row_path(row, original_index)
            _save_tensor(variant_dirs[name] / relative, tensor)

        qa_rows.append({
            "metadata_index": original_index,
            "sample_id": row["sample_id"],
            "source_relpath": row["source_relpath"],
            "split_role": row["split_role"],
            "quality_flags": row["preprocessing_quality_flags"],
            "raw_matrix_shape": f"{len(rts)}x{len(mzs)}",
            "fixed_grid_shape": _tensor_shape(fixed_tensor),
            "dual_no_pca_shape": _tensor_shape(dual_tensor),
            "dual_pca_shape": _tensor_shape(pca_tensor),
            "all_finite": True,
            "fixed_grid_nonzero": _tensor_nonzero(fixed_tensor),
            "dual_no_pca_nonzero": _tensor_nonzero(dual_tensor),
            "dual_pca_nonzero": _tensor_nonzero(pca_tensor),
        })
        if order % args.progress_every == 0 or order == len(metadata):
            elapsed = time.monotonic() - started
            print(f"张量生成 {order}/{len(metadata)}，用时 {elapsed:.1f}s", flush=True)

    all_shapes = {
        "fixed_grid_log1p": [2, args.rt_bins, args.mz_bins],
        "dual_channel_no_pca": [2, args.rt_bins, len(mz_axis)],
        "dual_channel_pca": [2, args.rt_bins, args.pca_components],
    }
    shapes = {name: all_shapes[name] for name in variants}
    for name, directory in variant_dirs.items():
        grid_info = {
            "rt_bins": args.rt_bins,
            "rt_range": list(args.rt_range),
            "mz_bins": shapes[name][-1],
            "mz_range": list(args.mz_range) if name == "fixed_grid_log1p" else [float(mz_axis[0]), float(mz_axis[-1])],
            "in_channels": 2,
            "input_representation": name,
            "source_manifest_fingerprint": fingerprint,
            "split_file": "split.json" if args.sample_limit is None else None,
            "pca_fit_scope": "train_idx_only" if name == "dual_channel_pca" else "not_applicable",
            "pca_components": args.pca_components if name == "dual_channel_pca" else None,
            "rt_alignment_applied": args.rt_offset_table is not None,
        }
        _write_variant_metadata(
            directory, metadata, selected_original_indices,
            split if args.sample_limit is None else None, grid_info,
        )

    if need_pca:
        pca_dir = variant_dirs["dual_channel_pca"]
        with (pca_dir / "input_mz_pca.pkl").open("wb") as handle:
            pickle.dump(pca, handle)
        np.save(pca_dir / "reference_mz_axis.npy", mz_axis)
        np.save(pca_dir / "reference_rt_axis.npy", rt_axis)
        fit_ids = metadata.iloc[train_indices][["metadata_index", "sample_id", "source_relpath"]].copy()
        fit_ids.to_csv(pca_dir / "pca_fit_samples.csv", index=False, encoding="utf-8-sig")

    summary = {
        "stage": "smoke_test_subset" if args.sample_limit is not None else "prepared",
        "manifest_fingerprint": fingerprint,
        "samples_in_source_manifest": original_count,
        "samples_processed": len(metadata),
        "train_samples_used_for_pca_fit": len(train_indices),
        "pca_fit_scope": "train_idx_only",
        "pca_fit_rows": fit_rows,
        "pca_fit_seconds": round(fit_seconds, 2),
        "pca_components": args.pca_components,
        "pca_explained_variance_ratio_sum": (
            float(pca.explained_variance_ratio_.sum()) if pca is not None else None),
        "pca_batch_rows": args.pca_batch_rows,
        "rt_axis": {"bins": args.rt_bins, "range": list(args.rt_range)},
        "fixed_grid_mz_axis": {"bins": args.mz_bins, "range": list(args.mz_range)},
        "dual_mz_axis": {"min": float(mz_axis[0]), "max": float(mz_axis[-1]), "step": args.raw_mz_step, "bins": len(mz_axis)},
        "rt_alignment": {
            "applied": args.rt_offset_table is not None,
            "table": str(args.rt_offset_table) if args.rt_offset_table else None,
            "n_batches": len(offset_map),
            "max_abs_offset_seconds": round(
                max((abs(v) for v in offset_map.values()), default=0.0), 3),
            "direction": "positive offset shifts the sample later along RT",
        },
        "tensor_shapes": shapes,
        "variants": variants,
        "elapsed_tensor_generation_seconds": round(time.monotonic() - started, 2),
        "sample_limit": args.sample_limit,
        "quality_flags": {
            "matrix_tic_rt_mismatch": H99_MISMATCH_REL_PATH,
            "signal_ceiling_exact": SIGNAL_CEILING,
        },
    }
    _write_json(stage / "prepare_summary.json", summary)
    pd.DataFrame(qa_rows).to_csv(stage / "tensor_qa.csv", index=False, encoding="utf-8-sig")
    flagged = metadata[metadata["preprocessing_quality_flags"] != ""]
    flagged.to_csv(stage / "preprocessing_quality_flags.csv", index=False, encoding="utf-8-sig")
    _plot_shapes(stage, shapes)
    readme_lines = [
        "# 预处理输入对照",
        "",
        f"本次生成的输入组：{', '.join(f'{n}（{variants[n]}）' for n in variants)}。",
        "所有组复用同一份样本清单和固定划分，均使用双通道。",
        "固定网格组先在物理 RT/m/z 网格中累加原始强度，再取 log1p；",
        "统一 m/z 轴的两组先对原始强度取 log1p，再插值到公共 m/z 轴与固定 RT 轴，二者仅差是否做 PCA。",
        f"时间轴为 {args.rt_range[0]:g}～{args.rt_range[1]:g} 分钟、{args.rt_bins} 点或格子。",
    ]
    if need_pca:
        readme_lines.append(
            f"PCA 只在 train_idx 对应样本上拟合，随后转换全部样本；当前 {args.pca_components} 维保留的训练输入方差比例为 "
            f"{float(pca.explained_variance_ratio_.sum()):.4%}，这不是分类性能指标。")
    if args.rt_offset_table is not None:
        readme_lines.append(
            f"已启用按批次 RT 对齐，偏移表：{args.rt_offset_table}；正值表示该批次更早出峰、已向后平移。")
    readme_lines += [
        "未训练模型。",
        "训练/评估入口会读取 grid_info.json 中的输入尺寸并沿用冻结划分，不再执行输入 PCA。",
        "双通道会逐样本标准化、裁剪；第二通道使用 log 强度比例，并非原始峰面积比例。",
        "`preprocessing_quality_flags.csv` 记录保留的异常样本和达到信号上限值的样本。",
        "",
    ]
    (stage / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    if overwrite:
        from audit_preprocessing_inputs import audit
        audit_dir = ROOT / "result" / f"preprocessing_input_review_rt{args.rt_bins}_{time.time_ns()}"
        print("全量生成完成；发布前检查完整性、划分及抽样 PCA 变换。", flush=True)
        audit(manifest_dir, audit_dir, 2, variants_dir=stage)
        print(f"发布前检查通过: {audit_dir}", flush=True)
    _publish(stage, output_dir, manifest_dir, fingerprint, overwrite)
    print(f"完成: {output_dir}")
    print(f"张量质量表: {output_dir / 'tensor_qa.csv'}")
    print(f"各组张量形状: {json.dumps(shapes, ensure_ascii=False)}")


def main():
    parser = argparse.ArgumentParser(description="按冻结的样本清单生成三种预处理张量")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "new_prepared_data_preproc_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "new_prepared_data_preproc_v1" / "representations")
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="仅用于小批冒烟检查；不生成可训练的完整数据集")
    parser.add_argument("--overwrite", action="store_true",
                        help="完整生成并检查后替换同一清单下的 representations，删除旧版")
    parser.add_argument("--rt-bins", type=int, default=2048)
    parser.add_argument("--rt-range", type=float, nargs=2, default=(0.0, 40.0))
    parser.add_argument("--mz-bins", type=int, default=256)
    parser.add_argument("--mz-range", type=float, nargs=2, default=(0.0, 550.0))
    parser.add_argument("--raw-mz-min", type=float, default=30.0)
    parser.add_argument("--raw-mz-max", type=float, default=532.0)
    parser.add_argument("--raw-mz-step", type=float, default=1.0)
    parser.add_argument("--rt-offset-table", type=Path, default=None,
                        help="批次→RT 偏移秒数 CSV（列 batch,offset_seconds）；"
                             "不提供则完全沿用旧的未对齐流程")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS),
                        help="要生成的输入变体；默认三组全生成，可只指定一组以节省磁盘")
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--pca-batch-rows", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4,
                        help="限制数值库线程数，避免小批 PCA 过度并行")
    args = parser.parse_args()
    if args.rt_bins < 2 or args.mz_bins < 2 or args.pca_batch_rows < args.pca_components:
        parser.error("网格尺寸需 >=2，且 pca-batch-rows 不小于 PCA 维数")
    if args.rt_range[0] >= args.rt_range[1] or args.mz_range[0] >= args.mz_range[1]:
        parser.error("RT 和 m/z 范围的下限必须小于上限")
    if args.raw_mz_step <= 0 or args.raw_mz_min >= args.raw_mz_max:
        parser.error("统一 m/z 轴范围或步长无效")
    if args.threads < 1 or args.progress_every < 1:
        parser.error("threads 和 progress-every 必须为正整数")
    with threadpool_limits(limits=args.threads):
        prepare(args)


if __name__ == "__main__":
    main()
