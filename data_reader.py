"""
从安捷伦 .D 文件夹读取完整 GC-MS scan 数据，
栅格化为 RT × m/z 二维张量。

读取后端优先级：rainbow-api → pyteomics(mzML) → TIC CSV fallback
"""
import warnings
from pathlib import Path
import numpy as np


# ═══════════════════════════════════════════════════════════
#  后端 1：rainbow-api（推荐）
# ═══════════════════════════════════════════════════════════
def _read_with_rainbow(d_path: Path):
    """返回 (rts, mz_axis, intensity_matrix) 或 (rts, spectra_list)。"""
    import rainbow as rb

    d = rb.read(str(d_path))
    ms = None
    for f in d.datafiles:
        name = getattr(f, "name", "")
        if name.lower().endswith(".ms"):
            ms = f
            break
    if ms is None:
        raise ValueError(f"rainbow 未找到 MS 数据: {d_path}")

    rts = np.asarray(ms.ylabels, dtype=np.float64)
    mzs = np.asarray(ms.xlabels, dtype=np.float64)
    data = np.asarray(ms.data, dtype=np.float64)
    return rts, mzs, data


# ═══════════════════════════════════════════════════════════
#  后端 2：pyteomics 读 mzML（需先用 msconvert 转换）
# ═══════════════════════════════════════════════════════════
def _read_with_pyteomics(mzml_path: Path):
    from pyteomics import mzml as mzml_reader

    rts, spectra = [], []
    with mzml_reader.read(str(mzml_path)) as reader:
        for spec in reader:
            if int(spec.get("ms level", 1)) != 1:
                continue
            scan_info = spec.get("scanList", {}).get("scan", [{}])[0]
            rt = float(scan_info.get("scan start time", 0.0))
            rts.append(rt)
            spectra.append((
                np.asarray(spec["m/z array"], dtype=np.float64),
                np.asarray(spec["intensity array"], dtype=np.float64),
            ))
    return np.array(rts), spectra


# ═══════════════════════════════════════════════════════════
#  后端 3：TIC CSV fallback（仅一维 TIC）
# ═══════════════════════════════════════════════════════════
def _read_tic_csv(d_path: Path):
    csv_path = d_path / "tic_front.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"TIC CSV 不存在: {csv_path}")
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("start"):
            start = i + 1
            break
    rts, tics = [], []
    for line in lines[start:]:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                rts.append(float(parts[0]))
                tics.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(rts), np.array(tics)


# ═══════════════════════════════════════════════════════════
#  统一入口
# ═══════════════════════════════════════════════════════════
def read_gcms_data(d_path, backend="auto"):
    """
    读取一个 .D 文件夹的 GC-MS 数据。

    返回值:
        mode = "matrix":  (rts, mz_axis, intensity_2d)
        mode = "spectra": (rts, spectra_list)  # list of (mz, int) tuples
        mode = "tic":     (rts, tic_1d)
    """
    d_path = Path(d_path)

    if backend in ("auto", "rainbow"):
        try:
            rts, mzs, data = _read_with_rainbow(d_path)
            return "matrix", rts, mzs, data
        except Exception as e:
            if backend == "rainbow":
                raise
            warnings.warn(f"rainbow 读取失败 ({e})，尝试下一后端")

    if backend in ("auto", "mzml"):
        mzml_candidates = list(d_path.parent.glob(d_path.stem + "*.mzML"))
        if mzml_candidates:
            try:
                rts, spectra = _read_with_pyteomics(mzml_candidates[0])
                return "spectra", rts, spectra, None
            except Exception as e:
                warnings.warn(f"pyteomics 读取失败 ({e})")

    # fallback: TIC
    try:
        rts, tics = _read_tic_csv(d_path)
        warnings.warn(f"仅读取到 TIC 数据: {d_path.name}")
        return "tic", rts, tics, None
    except Exception:
        pass

    raise RuntimeError(f"所有后端均无法读取: {d_path}")


# ═══════════════════════════════════════════════════════════
#  栅格化
# ═══════════════════════════════════════════════════════════
def rasterize(
    mode, rts, data_or_spectra, mz_axis_or_none,
    rt_bins=1024, mz_bins=256,
    rt_range=None, mz_range=(35.0, 550.0),
    log_transform=True,
):
    """
    把原始数据栅格化到 (rt_bins, mz_bins) 固定网格。
    """
    mz_min, mz_max = mz_range
    if rt_range is None:
        rt_min, rt_max = float(rts.min()), float(rts.max())
    else:
        rt_min, rt_max = rt_range

    grid = np.zeros((rt_bins, mz_bins), dtype=np.float64)
    rt_edges = np.linspace(rt_min, rt_max, rt_bins + 1)
    mz_edges = np.linspace(mz_min, mz_max, mz_bins + 1)

    if mode == "matrix":
        mz_axis = data_or_spectra  # 这里传进来的顺序需要注意
        intensity_matrix = mz_axis_or_none
        # 实际调用时: rasterize(mode, rts, mz_axis, intensity_matrix, ...)
        # 重新理解参数
        mz_vals = data_or_spectra
        int_mat = mz_axis_or_none
        for i, rt in enumerate(rts):
            if rt < rt_min or rt > rt_max:
                continue
            rt_idx = min(int((rt - rt_min) / (rt_max - rt_min) * rt_bins), rt_bins - 1)
            for j, mz in enumerate(mz_vals):
                if mz < mz_min or mz >= mz_max:
                    continue
                mz_idx = min(int((mz - mz_min) / (mz_max - mz_min) * mz_bins), mz_bins - 1)
                grid[rt_idx, mz_idx] += int_mat[i, j]

    elif mode == "spectra":
        spectra = data_or_spectra
        for i, rt in enumerate(rts):
            if rt < rt_min or rt > rt_max:
                continue
            rt_idx = min(int((rt - rt_min) / (rt_max - rt_min) * rt_bins), rt_bins - 1)
            mzs, ints = spectra[i]
            mask = (mzs >= mz_min) & (mzs < mz_max) & (ints > 0)
            mzs_f, ints_f = mzs[mask], ints[mask]
            mz_idx = np.clip(
                ((mzs_f - mz_min) / (mz_max - mz_min) * mz_bins).astype(int),
                0, mz_bins - 1
            )
            np.add.at(grid[rt_idx], mz_idx, ints_f)

    elif mode == "tic":
        # TIC → 仅填充 m/z=0 列（1D退化模式）
        tic = data_or_spectra
        for i, rt in enumerate(rts):
            if rt < rt_min or rt > rt_max:
                continue
            rt_idx = min(int((rt - rt_min) / (rt_max - rt_min) * rt_bins), rt_bins - 1)
            grid[rt_idx, 0] += tic[i]

    if log_transform:
        grid = np.log1p(grid)

    return grid.astype(np.float32), (rt_min, rt_max)


def build_dual_channel(grid):
    """
    构建双通道输入:
      Ch0: z-score 标准化的绝对强度（关注物质结构）
      Ch1: 相对组成（关注含量比例）
    """
    # 通道 0：绝对强度
    ch0 = grid.copy()
    mu, std = ch0.mean(), ch0.std()
    if std > 0:
        ch0 = (ch0 - mu) / std
    ch0 = np.clip(ch0, -5, 5)

    # 通道 1：相对组成
    total = grid.sum()
    ch1 = grid / (total + 1e-8)
    ch1 = np.log1p(ch1 * 1e6)
    mu1, std1 = ch1.mean(), ch1.std()
    if std1 > 0:
        ch1 = (ch1 - mu1) / std1
    ch1 = np.clip(ch1, -5, 5)

    return np.stack([ch0, ch1], axis=0).astype(np.float32)


def d_folder_to_tensor(d_path, rt_bins=1024, mz_bins=256,
                       rt_range=None, mz_range=(35.0, 550.0),
                       backend="auto"):
    """一步到位：.D → (2, rt_bins, mz_bins) 张量。"""
    mode, rts, a, b = read_gcms_data(d_path, backend=backend)

    if mode == "matrix":
        grid, actual_rt = rasterize(
            mode, rts, a, b,
            rt_bins=rt_bins, mz_bins=mz_bins,
            rt_range=rt_range, mz_range=mz_range,
        )
    elif mode == "spectra":
        grid, actual_rt = rasterize(
            mode, rts, a, None,
            rt_bins=rt_bins, mz_bins=mz_bins,
            rt_range=rt_range, mz_range=mz_range,
        )
    else:  # tic
        grid, actual_rt = rasterize(
            mode, rts, a, None,
            rt_bins=rt_bins, mz_bins=mz_bins,
            rt_range=rt_range, mz_range=mz_range,
        )

    tensor = build_dual_channel(grid)
    return tensor, grid, actual_rt