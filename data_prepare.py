"""
扫描 dataset 目录 -> 建元数据表 -> 逐样本转换为 .npz 张量。
运行方式: python data_prepare.py
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
from scipy.interpolate import interp1d

from config import Config
from data_reader import build_dual_channel, d_folder_to_tensor, read_gcms_data

cfg = Config()

# 规格代号 -> 规格名称
SPEC_NAME_MAP = {
    "H88": "红河（88）",
    "H99": "红河（99）",
    "HRJ": "红河（软甲）",
    "HDC": "红河（道）彩膜版",
    "HV8": "红河（V8）",
    "HYZ": "云烟（紫）",
    "HRY": "云烟（软如意）",
    "HYX": "云烟（印象）",
    "HRL": "云烟（软礼印象）",
    "XCJ": "云烟（中支小重九）",
    "RJD": "红塔山（软经典）",
    "YJD": "红塔山（硬经典）",
    "HMD": "牡丹（软）",
}

# 后前三位作为产品分类，剩余部分作为测试信息
_RE_STD = re.compile(r"^([A-Za-z][A-Za-z0-9]{2})(.*)$", re.IGNORECASE)


def _special(code: str, spec_name: str, stem: str) -> dict:
    seq_match = re.match(r"^(\d+)#", stem)
    return {
        "seq_no": int(seq_match.group(1)) if seq_match else None,
        "code": code,
        "spec_name": spec_name,
        "line_code": "",
        "lot_id": "",
        "test_case": stem,
        "is_special": True,
    }


def parse_d_name(folder_name: str) -> dict:
    """从 .D 文件夹名解析样品元数据。"""
    stem = folder_name.removesuffix(".D").removesuffix(".d").strip()
    stem_nsp = stem.replace(" ", "")
    stem_up = stem_nsp.upper()

    if any(k in stem_up for k in ("空白", "BLANK", "BALNK")):
        return _special("BLANK", "空白样", stem)

    for keywords, code, name in (
        (("管道清洗剂",), "CLEANER", "管道清洗剂"),
        (("热熔胶",), "HOTMELT", "热熔胶"),
        (("糖料",), "SUGAR", "糖料"),
        (("内标",), "INSTD", "内标"),
    ):
        if any(k in stem for k in keywords):
            return _special(code, name, stem)

    if any(k in stem for k in ("凝似被污染", "污染")):
        return _special("CONTAM", "疑似污染", stem)

    no_seq = re.sub(r"^\d+#", "", stem_nsp)
    if no_seq.upper().startswith("WYZX"):
        return _special("WYZX", "乌兰参照样", stem)

    if re.match(r"^\d+号取样点", no_seq):
        return _special("ENV", "环境取样", stem)

    seq_match = re.match(r"^(\d+)#", stem_nsp)
    seq_no = int(seq_match.group(1)) if seq_match else None

    body = no_seq
    if re.match(r"^[89][89][A-C]CS", body, re.IGNORECASE):
        body = "H" + body

    m = _RE_STD.match(body)
    if m:
        code = m.group(1).upper()
        tail = m.group(2).strip()

        line_code = ""
        lot_id = ""
        cs_match = re.match(r"^([A-Za-z])CS(.*)$", tail, re.IGNORECASE)
        if cs_match:
            line_code = cs_match.group(1).upper()
            lot_id = cs_match.group(2).strip()
        else:
            lot_match = re.search(r"(\d{6,})", tail)
            lot_id = lot_match.group(1) if lot_match else ""

        lot_id = re.sub(r"\(.*", "", lot_id).strip()
        spec_name = SPEC_NAME_MAP.get(code, code)
        is_special = code not in SPEC_NAME_MAP

        return {
            "seq_no": seq_no,
            "code": code,
            "spec_name": spec_name,
            "line_code": line_code,
            "lot_id": lot_id,
            "test_case": tail,
            "is_special": is_special,
        }

    for ch_kw, code in (
        ("云烟（紫）", "HYZ"),
        ("云烟(紫)", "HYZ"),
        ("中支小重九", "XCJ"),
    ):
        if ch_kw in stem:
            spec_name = SPEC_NAME_MAP.get(code, code)
            is_ref = "对照" in stem or "参照" in stem
            return {
                "seq_no": seq_no if seq_match else None,
                "code": code,
                "spec_name": spec_name,
                "line_code": "",
                "lot_id": "",
                "test_case": no_seq,
                "is_special": is_ref,
            }

    return _special("UNKNOWN", "未知", stem)


def _print_scan_summary_tables(df: pd.DataFrame):
    """扫描完成后输出两张统计表：总体表 + 分批次表。"""
    total_table = (
        df.groupby(["code", "spec_name"], as_index=False)
        .agg(sample_count=("sample_id", "count"))
        .sort_values(["sample_count", "code"], ascending=[False, True])
    )

    batch_table = (
        df.groupby(["batch_name", "code", "spec_name"], as_index=False)
        .agg(sample_count=("sample_id", "count"))
        .sort_values(["batch_name", "sample_count", "code"], ascending=[True, False, True])
    )

    print("\n统计表1（总信息）: code / 产品类型 / 产品数据数量")
    print(total_table.rename(columns={"spec_name": "product_type"}).to_string(index=False))

    print("\n统计表2（每个批次）: batch / code / 产品类型 / 产品数据数量")
    print(batch_table.rename(columns={"spec_name": "product_type"}).to_string(index=False))


def scan_dataset(root: str) -> pd.DataFrame:
    """扫描 dataset/ 目录，收集所有 .D 文件夹元数据。"""
    root = Path(root)
    rows = []

    batch_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)

    for batch_idx, batch_dir in enumerate(batch_dirs):
        batch_name = batch_dir.name
        d_folders = sorted([p for p in batch_dir.iterdir() if p.is_dir() and p.suffix.lower() == ".d"])

        for d_path in d_folders:
            info = parse_d_name(d_path.name)
            seq_str = f"{info['seq_no']:04d}" if info["seq_no"] is not None else "0000"
            sample_id = f"B{batch_idx:02d}_{seq_str}_{info['lot_id'] or d_path.stem}"
            rows.append({
                "sample_id": sample_id,
                "d_path": str(d_path),
                "d_name": d_path.name,
                "batch_idx": batch_idx,
                "batch_name": batch_name,
                "seq_no": info["seq_no"],
                "code": info["code"],
                "spec_name": info["spec_name"],
                "line_code": info["line_code"],
                "lot_id": info["lot_id"],
                "test_case": info["test_case"],
                # 向后兼容: 下游仍可能读取 fine/coarse 列
                "fine_code": info["code"],
                "coarse_code": info["code"],
                "product_code": info["code"],
                "product_fine": info["code"],
                "product_coarse": info["code"],
                "is_special": info["is_special"],
            })

    df = pd.DataFrame(rows)
    print(
        f"扫描完成: {len(df)} 个样本, "
        f"{df['batch_idx'].nunique()} 个批次, "
        f"{df['code'].nunique()} 种产品(code)"
    )
    _print_scan_summary_tables(df)
    return df


def _safe_tag(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"[^0-9A-Za-z_\-]+", "_", s)
    return s.strip("_") or "NA"


def _save_prepare_plot(
    plot_path: Path,
    grid: np.ndarray,
    sample_id: str,
    d_name: str,
    batch_name: str,
    fine_code: str,
    rt_range: tuple,
    mz_range: tuple,
    dpi: int = 120,
):
    """保存数据准备阶段的二维网格可视化图。"""
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.5))
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[float(mz_range[0]), float(mz_range[1]), float(rt_range[0]), float(rt_range[1])],
    )
    ax.set_xlabel("m/z")
    ax.set_ylabel("RT")
    ax.set_title(f"{sample_id} | {fine_code} | {batch_name}\n{d_name}")
    fig.colorbar(im, ax=ax, label="log intensity")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)


def _save_prepare_table(
    table_path: Path,
    grid: np.ndarray,
    rt_range: tuple,
    mz_range: tuple,
):
    """保存二维网格为长表，便于与仪器软件导出结果逐点比对。"""
    h, w = grid.shape
    rt_axis = np.linspace(float(rt_range[0]), float(rt_range[1]), h, endpoint=False)
    mz_axis = np.linspace(float(mz_range[0]), float(mz_range[1]), w, endpoint=False)

    rt_grid, mz_grid = np.meshgrid(rt_axis, mz_axis, indexing="ij")
    table_df = pd.DataFrame({
        "rt_min": rt_grid.reshape(-1),
        "mz": mz_grid.reshape(-1),
        "log_intensity": grid.reshape(-1),
    })
    table_df.to_csv(table_path, index=False, encoding="utf-8-sig")


def _raw_direct_cache_key(metadata_csv: Path, cfg: Config, n_components: int) -> str:
    st = metadata_csv.stat()
    payload = {
        "metadata_csv": str(metadata_csv.resolve()),
        "metadata_mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "metadata_size": int(st.st_size),
        "n_components": int(n_components),
        "rt_range": list(cfg.rt_range) if cfg.rt_range is not None else None,
        "mz_range": list(cfg.mz_range),
        "log_transform": bool(cfg.log_transform),
        "rt_target_bins": int(cfg.rt_bins),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    import hashlib

    return hashlib.sha1(raw).hexdigest()[:16]


def _read_raw_matrix_no_bins(d_path: str, cfg: Config):
    """直接读取原始 RT×m/z 矩阵，不做栅格化累加。"""
    mode, rts, a, b = read_gcms_data(d_path, backend="auto")
    if mode != "matrix":
        raise RuntimeError(f"当前样本非 matrix 模式: mode={mode}")

    rts = np.asarray(rts, dtype=np.float64)
    mzs = np.asarray(a, dtype=np.float64)
    mat = np.asarray(b, dtype=np.float64)

    if mat.shape == (len(mzs), len(rts)):
        mat = mat.T
    if mat.shape != (len(rts), len(mzs)):
        raise ValueError(
            f"矩阵形状不匹配: mat={mat.shape}, len(rts)={len(rts)}, len(mzs)={len(mzs)}"
        )

    if cfg.rt_range is not None:
        rt_lo, rt_hi = float(cfg.rt_range[0]), float(cfg.rt_range[1])
        rt_mask = (rts >= rt_lo) & (rts <= rt_hi)
        if np.any(rt_mask):
            rts = rts[rt_mask]
            mat = mat[rt_mask, :]

    mz_lo, mz_hi = float(cfg.mz_range[0]), float(cfg.mz_range[1])
    mz_mask = (mzs >= mz_lo) & (mzs <= mz_hi)
    if np.any(mz_mask):
        mzs = mzs[mz_mask]
        mat = mat[:, mz_mask]

    if mat.size == 0 or mat.shape[0] < 2 or mat.shape[1] < 2:
        raise RuntimeError(f"有效原始矩阵为空或过小: shape={mat.shape}")

    mat = np.maximum(mat, 0.0)
    if bool(cfg.log_transform):
        mat = np.log1p(mat)

    return rts.astype(np.float32), mzs.astype(np.float32), mat.astype(np.float32)


def _align_mz_axis_linear(mat_rt_mz: np.ndarray, src_mz: np.ndarray, ref_mz: np.ndarray) -> np.ndarray:
    """线性插值对齐 m/z 轴（非强度栅格化）。"""
    src_mz = np.asarray(src_mz, dtype=np.float32)
    ref_mz = np.asarray(ref_mz, dtype=np.float32)

    if src_mz.ndim != 1 or ref_mz.ndim != 1:
        raise ValueError("m/z 轴必须是一维")

    if len(src_mz) == len(ref_mz) and np.allclose(src_mz, ref_mz, atol=1e-6):
        return mat_rt_mz.astype(np.float32)

    order = np.argsort(src_mz)
    src_sorted = src_mz[order]
    mat_sorted = mat_rt_mz[:, order]
    f = interp1d(
        src_sorted,
        mat_sorted,
        axis=1,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    out = f(ref_mz)
    return np.asarray(out, dtype=np.float32)


def _resample_rt_linear(mat_rt_k: np.ndarray, src_rts: np.ndarray, target_len: int) -> np.ndarray:
    """RT 轴线性重采样，避免 DataLoader 因长度不同无法组 batch。"""
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError("target_len 必须为正")

    if mat_rt_k.shape[0] == target_len:
        return mat_rt_k.astype(np.float32)

    src_rts = np.asarray(src_rts, dtype=np.float32)
    if src_rts.ndim != 1 or len(src_rts) != mat_rt_k.shape[0]:
        src_rts = np.linspace(0.0, 1.0, mat_rt_k.shape[0], dtype=np.float32)

    if np.any(np.diff(src_rts) <= 0):
        src_rts = np.linspace(0.0, 1.0, mat_rt_k.shape[0], dtype=np.float32)

    dst_rts = np.linspace(float(src_rts[0]), float(src_rts[-1]), target_len, dtype=np.float32)
    f = interp1d(
        src_rts,
        mat_rt_k,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    out = f(dst_rts)
    return np.asarray(out, dtype=np.float32)


def _fit_or_load_raw_direct_input_pca(metadata_csv: Path, cfg: Config, n_components: int):
    """基于原始 RT×m/z 矩阵拟合/加载 PCA 模型。"""
    cache_root = Path(cfg.prepared_dir) / "cache" / "input_pca"
    cache_root.mkdir(parents=True, exist_ok=True)

    key = _raw_direct_cache_key(metadata_csv, cfg, n_components)
    model_path = cache_root / f"raw_direct_rt_axis_pca_{key}.pkl"
    meta_path = cache_root / f"raw_direct_rt_axis_pca_{key}.json"
    ref_axis_path = cache_root / f"raw_direct_rt_axis_pca_{key}_ref_mz.npy"

    if model_path.exists() and ref_axis_path.exists():
        import pickle

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        ref_mz_axis = np.load(ref_axis_path)
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        meta.setdefault("cache_key", key)
        return model, ref_mz_axis.astype(np.float32), meta, True, model_path

    meta_df = pd.read_csv(metadata_csv)
    if meta_df.empty:
        raise RuntimeError("metadata 为空，无法拟合 PCA")

    ipca = None
    ref_mz_axis = None
    n_tensors = 0
    n_rows = 0
    skipped = 0

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="拟合原始PCA"):
        try:
            _rts, mzs, raw_mat = _read_raw_matrix_no_bins(row["d_path"], cfg)
        except Exception as e:
            skipped += 1
            print(f"\n  ! 拟合跳过样本 {row.get('d_name', 'NA')}: {e}")
            continue

        if ref_mz_axis is None:
            ref_mz_axis = mzs.astype(np.float32)

        aligned = _align_mz_axis_linear(raw_mat, mzs, ref_mz_axis)
        tensor = build_dual_channel(aligned)

        if ipca is None:
            width = int(tensor.shape[2])
            n_comp_use = min(int(n_components), width - 1)
            if n_comp_use < 2:
                raise RuntimeError(f"原始 m/z 维度过小，无法做 PCA: width={width}")
            ipca = IncrementalPCA(n_components=n_comp_use)

        for c in range(tensor.shape[0]):
            ipca.partial_fit(tensor[c])
            n_rows += int(tensor.shape[1])
        n_tensors += 1

    if ipca is None:
        raise RuntimeError("PCA 拟合失败: 没有可用样本")

    import pickle
    import time

    with open(model_path, "wb") as f:
        pickle.dump(ipca, f)
    np.save(ref_axis_path, ref_mz_axis.astype(np.float32))

    meta = {
        "cache_key": key,
        "n_tensors": int(n_tensors),
        "n_rows": int(n_rows),
        "n_skipped": int(skipped),
        "input_width": int(len(ref_mz_axis)),
        "n_components": int(getattr(ipca, "n_components_", n_components)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cache_model_path": str(model_path),
        "cache_ref_mz_axis_path": str(ref_axis_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return ipca, ref_mz_axis.astype(np.float32), meta, False, model_path


def _precompute_input_pca_tensors(metadata_csv: Path, cfg: Config):
    """对原始 RT×m/z 数据做 PCA 并直接落盘 (不走栅格化 bins)。"""
    if not bool(getattr(cfg, "input_raw_pca_enabled", False)):
        return None

    n_comp = int(getattr(cfg, "input_raw_pca_components", 128) or 128)
    meta_df = pd.read_csv(metadata_csv)
    if meta_df.empty:
        print("\n  [Input PCA] 跳过: metadata 为空")
        return {
            "enabled": True,
            "applied": False,
            "reason": "empty_metadata",
            "n_components": n_comp,
        }

    pca_model, ref_mz_axis, pca_meta, cache_hit, cache_model_path = _fit_or_load_raw_direct_input_pca(
        metadata_csv=Path(metadata_csv),
        cfg=cfg,
        n_components=n_comp,
    )

    n_comp_real = int(getattr(pca_model, "n_components_", n_comp))
    rt_target_bins = int(getattr(cfg, "rt_bins", 1024) or 1024)

    source = "cache_hit" if cache_hit else "cache_miss_fit"
    print(
        f"  [Input PCA] 模型就绪({source}): "
        f"rows={pca_meta.get('n_rows', 'na')}, width={pca_meta.get('input_width', 'na')}->{n_comp_real}, model={cache_model_path}"
    )

    converted = 0
    failed = 0
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="写回PCA tensor"):
        npz_path = Path(row["tensor_path"])
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            rts, mzs, raw_mat = _read_raw_matrix_no_bins(row["d_path"], cfg)
            aligned = _align_mz_axis_linear(raw_mat, mzs, ref_mz_axis)
            tensor = build_dual_channel(aligned)

            out = []
            for c in range(tensor.shape[0]):
                ch = pca_model.transform(tensor[c]).astype(np.float32)
                ch = _resample_rt_linear(ch, rts, rt_target_bins)
                out.append(ch)
            tensor_pca = np.stack(out, axis=0).astype(np.float32)

            np.savez_compressed(npz_path, tensor=tensor_pca, grid=np.zeros(1, dtype=np.float32))
            converted += 1
        except Exception as e:
            print(f"\n  ✗ 预计算失败 {row['d_name']}: {e}")
            empty = np.zeros((cfg.in_channels, rt_target_bins, n_comp_real), dtype=np.float32)
            np.savez_compressed(npz_path, tensor=empty, grid=np.zeros(1, dtype=np.float32))
            failed += 1

    return {
        "enabled": True,
        "applied": converted > 0,
        "reason": "ok" if failed == 0 else "partial_failed",
        "n_components": int(n_comp_real),
        "current_width": int(pca_meta.get("input_width", 0) or 0),
        "converted": int(converted),
        "failed": int(failed),
        "rt_target_bins": int(rt_target_bins),
        "ref_mz_axis_len": int(len(ref_mz_axis)),
        "cache_model_path": str(cache_model_path),
        "cache_hit": bool(cache_hit),
    }


def _build_tensor_paths(metadata: pd.DataFrame, tensor_dir: Path, cfg: Config):
    tensor_paths = []
    for idx, row in metadata.iterrows():
        batch_tag = _safe_tag(row["batch_name"])
        product_tag = _safe_tag(row["code"])
        sample_tag = _safe_tag(row["sample_id"])

        if cfg.tag_output_with_batch_and_product:
            tensor_subdir = tensor_dir / batch_tag / product_tag
            tensor_subdir.mkdir(parents=True, exist_ok=True)
            npz_path = tensor_subdir / f"{idx:04d}_{sample_tag}.npz"
        else:
            npz_path = tensor_dir / f"{idx:04d}.npz"
        tensor_paths.append(str(npz_path))
    return tensor_paths


def _convert_all_raw_direct_pca(metadata: pd.DataFrame, out_dir: Path, cfg: Config):
    """主路径：不做 bins，直接原始矩阵 PCA 后落盘。"""
    tensor_dir = out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    metadata = metadata.copy()
    metadata["tensor_path"] = _build_tensor_paths(metadata, tensor_dir, cfg)

    meta_path = out_dir / "metadata.csv"
    metadata.to_csv(meta_path, index=False, encoding="utf-8-sig")

    print(
        "\n[Prepare] 使用原始 RT×m/z 直读 PCA 流程(不做栅格化 bins): "
        f"samples={len(metadata)}, rt_target={cfg.rt_bins}, pca_components={cfg.input_raw_pca_components}"
    )
    pca_prepare_info = _precompute_input_pca_tensors(meta_path, cfg)

    info = {
        "rt_bins": int(cfg.rt_bins),
        "mz_bins": int(cfg.input_raw_pca_components),
        "rt_range": list(cfg.rt_range) if cfg.rt_range is not None else None,
        "mz_range": list(cfg.mz_range),
        "success": int(pca_prepare_info.get("converted", 0) - pca_prepare_info.get("failed", 0)),
        "fail": int(pca_prepare_info.get("failed", 0)),
        "plots_saved": False,
        "plot_dir": "",
        "tables_saved": False,
        "table_dir": "",
        "prepare_mode": "raw_rt_mz_direct_pca",
        "input_pca_source": "raw_rt_mz_no_bins",
    }

    if pca_prepare_info is not None:
        info["input_pca_precomputed"] = bool(pca_prepare_info.get("enabled", False))
        info["input_pca_applied"] = bool(pca_prepare_info.get("applied", False))
        info["input_pca_reason"] = str(pca_prepare_info.get("reason", "unknown"))
        info["input_pca_requested_components"] = int(
            getattr(cfg, "input_raw_pca_components", cfg.mz_bins)
        )
        info["input_pca_components"] = int(
            pca_prepare_info.get("n_components", cfg.input_raw_pca_components)
        )
        info["input_pca_rt_bins"] = int(pca_prepare_info.get("rt_target_bins", cfg.rt_bins))
        info["input_pca_cache_model_path"] = str(pca_prepare_info.get("cache_model_path", ""))
        info["input_pca_cache_hit"] = bool(pca_prepare_info.get("cache_hit", False))
        info["input_pca_converted"] = int(pca_prepare_info.get("converted", 0))
        info["input_pca_failed"] = int(pca_prepare_info.get("failed", 0))
        info["input_pca_ref_mz_axis_len"] = int(pca_prepare_info.get("ref_mz_axis_len", 0))
        info["mz_bins"] = int(info["input_pca_components"])

    (out_dir / "grid_info.json").write_text(json.dumps(info, indent=2))
    print(f"\n转换完成(原始直读PCA): 成功 {info['success']}, 失败 {info['fail']}")
    print(f"元数据: {meta_path}")
    return metadata


def convert_all(metadata: pd.DataFrame, out_dir: str, cfg: Config):
    """逐样本读取 .D 并生成 .npz 张量。"""
    out_dir = Path(out_dir)

    if bool(getattr(cfg, "prepare_direct_raw_pca", True)) and bool(
        getattr(cfg, "input_raw_pca_enabled", False)
    ):
        return _convert_all_raw_direct_pca(metadata, out_dir, cfg)

    tensor_dir = out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / "tables"
    if cfg.save_prepare_tables:
        table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    if cfg.save_prepare_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    success, fail = 0, 0
    tensor_paths = []
    mz_global_min = np.inf
    mz_global_max = -np.inf
    mz_stat_samples = 0

    rt_range_use, mz_range_use = cfg.rt_range, cfg.mz_range
    print(f"\n使用固定范围: RT={rt_range_use}, m/z={mz_range_use}")

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="转换中"):
        batch_tag = _safe_tag(row["batch_name"])
        product_tag = _safe_tag(row["code"])
        sample_tag = _safe_tag(row["sample_id"])

        if cfg.tag_output_with_batch_and_product:
            tensor_subdir = tensor_dir / batch_tag / product_tag
            tensor_subdir.mkdir(parents=True, exist_ok=True)
            npz_path = tensor_subdir / f"{idx:04d}_{sample_tag}.npz"
        else:
            npz_path = tensor_dir / f"{idx:04d}.npz"
        tensor_paths.append(str(npz_path))

        if cfg.save_prepare_tables:
            if cfg.tag_output_with_batch_and_product:
                table_subdir = table_dir / batch_tag / product_tag
                table_subdir.mkdir(parents=True, exist_ok=True)
                table_path = table_subdir / f"{idx:04d}_{sample_tag}.csv"
            else:
                table_path = table_dir / f"{idx:04d}.csv"
        else:
            table_path = None

        if cfg.save_prepare_plots:
            if cfg.tag_output_with_batch_and_product:
                plot_subdir = plot_dir / batch_tag / product_tag
                plot_subdir.mkdir(parents=True, exist_ok=True)
                plot_path = plot_subdir / f"{idx:04d}_{sample_tag}.png"
            else:
                plot_path = plot_dir / f"{idx:04d}.png"
        else:
            plot_path = None

        if npz_path.exists():
            if cfg.save_prepare_tables and (table_path is not None) and (not table_path.exists()):
                try:
                    with np.load(npz_path) as npz:
                        grid_cached = npz["grid"]
                    if getattr(grid_cached, "ndim", 0) == 2:
                        rt_for_table = rt_range_use if rt_range_use is not None else (0.0, float(grid_cached.shape[0]))
                        _save_prepare_table(
                            table_path=table_path,
                            grid=grid_cached,
                            rt_range=rt_for_table,
                            mz_range=mz_range_use,
                        )
                except Exception as e:
                    print(f"\n  ! {row['d_name']} 补保存表格失败: {e}")

            if cfg.save_prepare_plots and ((cfg.prepare_plot_max_samples is None) or (idx < cfg.prepare_plot_max_samples)):
                if (plot_path is not None) and (not plot_path.exists()):
                    try:
                        with np.load(npz_path) as npz:
                            grid_cached = npz["grid"]
                        if getattr(grid_cached, "ndim", 0) == 2:
                            rt_for_plot = rt_range_use if rt_range_use is not None else (0.0, float(grid_cached.shape[0]))
                            _save_prepare_plot(
                                plot_path=plot_path,
                                grid=grid_cached,
                                sample_id=row["sample_id"],
                                d_name=row["d_name"],
                                batch_name=row["batch_name"],
                                fine_code=row["code"],
                                rt_range=rt_for_plot,
                                mz_range=mz_range_use,
                                dpi=cfg.prepare_plot_dpi,
                            )
                    except Exception as e:
                        print(f"\n  ! {row['d_name']} 补保存图失败: {e}")
            success += 1
            continue

        try:
            tensor, grid, actual_rt, observed_mz_range = d_folder_to_tensor(
                row["d_path"],
                rt_bins=cfg.rt_bins,
                mz_bins=cfg.mz_bins,
                rt_range=rt_range_use,
                mz_range=mz_range_use,
                return_mz_stats=True,
            )

            if observed_mz_range is not None:
                mz_global_min = min(mz_global_min, float(observed_mz_range[0]))
                mz_global_max = max(mz_global_max, float(observed_mz_range[1]))
                mz_stat_samples += 1

            np.savez_compressed(npz_path, tensor=tensor, grid=grid)

            if cfg.save_prepare_tables and (table_path is not None):
                _save_prepare_table(
                    table_path=table_path,
                    grid=grid,
                    rt_range=actual_rt,
                    mz_range=mz_range_use,
                )

            if cfg.save_prepare_plots and ((cfg.prepare_plot_max_samples is None) or (idx < cfg.prepare_plot_max_samples)):
                _save_prepare_plot(
                    plot_path=plot_path,
                    grid=grid,
                    sample_id=row["sample_id"],
                    d_name=row["d_name"],
                    batch_name=row["batch_name"],
                    fine_code=row["code"],
                    rt_range=actual_rt,
                    mz_range=mz_range_use,
                    dpi=cfg.prepare_plot_dpi,
                )

            success += 1
        except Exception as e:
            print(f"\n  ✗ {row['d_name']}: {e}")
            empty = np.zeros((cfg.in_channels, cfg.rt_bins, cfg.mz_bins), dtype=np.float32)
            np.savez_compressed(npz_path, tensor=empty, grid=np.zeros(1))
            fail += 1

    metadata = metadata.copy()
    metadata["tensor_path"] = tensor_paths

    meta_path = out_dir / "metadata.csv"
    metadata.to_csv(meta_path, index=False, encoding="utf-8-sig")

    pca_prepare_info = _precompute_input_pca_tensors(meta_path, cfg)

    info = {
        "rt_bins": cfg.rt_bins,
        "mz_bins": cfg.mz_bins,
        "rt_range": list(rt_range_use) if rt_range_use is not None else None,
        "mz_range": list(mz_range_use),
        "success": success,
        "fail": fail,
        "plots_saved": bool(cfg.save_prepare_plots),
        "plot_dir": str((out_dir / "plots").resolve()) if cfg.save_prepare_plots else "",
        "tables_saved": bool(cfg.save_prepare_tables),
        "table_dir": str((out_dir / "tables").resolve()) if cfg.save_prepare_tables else "",
    }

    if pca_prepare_info is not None:
        info["input_pca_precomputed"] = bool(pca_prepare_info.get("enabled", False))
        info["input_pca_applied"] = bool(pca_prepare_info.get("applied", False))
        info["input_pca_reason"] = str(pca_prepare_info.get("reason", "unknown"))
        info["input_pca_requested_components"] = int(
            getattr(cfg, "input_raw_pca_components", cfg.mz_bins)
        )
        info["input_pca_components"] = int(
            pca_prepare_info.get("n_components", cfg.input_raw_pca_components)
        )
        info["input_pca_cache_model_path"] = str(pca_prepare_info.get("cache_model_path", ""))
        info["input_pca_cache_hit"] = bool(pca_prepare_info.get("cache_hit", False))
        info["input_pca_converted"] = int(pca_prepare_info.get("converted", 0))

        if bool(pca_prepare_info.get("enabled", False)):
            info["mz_bins"] = int(info["input_pca_components"])

    (out_dir / "grid_info.json").write_text(json.dumps(info, indent=2))
    print(f"\n转换完成: 成功 {success}, 失败 {fail}")
    if mz_stat_samples > 0 and np.isfinite(mz_global_min) and np.isfinite(mz_global_max):
        print(
            f"读取样本 m/z 统计: min={mz_global_min:.6f}, "
            f"max={mz_global_max:.6f} (基于本次实际读取的 {mz_stat_samples} 个样本)"
        )
    else:
        print("读取样本 m/z 统计: 无有效数据（本次可能全部命中缓存未重新读取）")
    print(f"元数据: {meta_path}")
    return metadata


if __name__ == "__main__":
    metadata = scan_dataset(cfg.dataset_root)
    print("\n产品分布（code）:")
    print(metadata[["code", "spec_name"]].drop_duplicates().sort_values("code").to_string(index=False))
    print("\n各规格样本数:")
    print(metadata["code"].value_counts().to_string())
    print("\n批次分布:")
    print(metadata.groupby("batch_name")["sample_id"].count().to_string())
    print(f"\nUNKNOWN 样本数: {(metadata['code'] == 'UNKNOWN').sum()}")

    unknown_df = metadata[metadata["code"] == "UNKNOWN"][["batch_name", "d_name"]]
    if len(unknown_df):
        print(unknown_df.to_string(index=False))

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--yes", action="store_true", help="跳过交互确认并直接开始转换")
    args, _ = parser.parse_known_args()

    if args.yes:
        ans = "y"
    else:
        ans = input("\n开始转换张量? (y/n): ").strip().lower()

    if ans == "y":
        convert_all(metadata, cfg.prepared_dir, cfg)
