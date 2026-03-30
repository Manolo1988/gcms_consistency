"""
从安捷伦 .D 文件夹读取完整 GC-MS scan 数据，
栅格化为 RT × m/z 二维张量。

读取后端优先级：rainbow-api → pyteomics(mzML) → TIC CSV fallback
"""
import warnings
from pathlib import Path
import numpy as np


def _axis_scores(vals: np.ndarray):
    """根据数值范围粗略判断该坐标轴更像 RT 还是 m/z。"""
    vals = np.asarray(vals, dtype=np.float64)
    if vals.ndim != 1 or vals.size < 8:
        return -1.0, -1.0

    finite = np.isfinite(vals)
    if not finite.any():
        return -1.0, -1.0
    vals = vals[finite]
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    span = vmax - vmin

    rt_score = 0.0
    mz_score = 0.0

    if vmin >= 0:
        rt_score += 0.5
    if 0 <= vmax <= 240:
        rt_score += 2.0
    if 0.2 <= span <= 300:
        rt_score += 1.0

    if 20 <= vmin <= 80:
        mz_score += 1.0
    if 80 <= vmax <= 1500:
        mz_score += 2.0
    if 50 <= span <= 2000:
        mz_score += 1.0

    return rt_score, mz_score


# ═══════════════════════════════════════════════════════════
#  后端 1：rainbow-api（推荐）
# ═══════════════════════════════════════════════════════════
def _read_with_rainbow(d_path: Path):
    """从 .D 中挑选最像 GC-MS 全扫描矩阵的数据并标准化为 RT x m/z。"""
    import rainbow as rb

    d = rb.read(str(d_path))

    best = None
    best_score = -1e9

    # 仅作为弱优先级：根目录 data.ms 通常是主 MS 二进制
    preferred = str((d_path / "data.ms").as_posix()).lower()

    for f in d.datafiles:
        name = getattr(f, "name", "")
        x = getattr(f, "xlabels", None)
        y = getattr(f, "ylabels", None)
        data = getattr(f, "data", None)
        if x is None or y is None or data is None:
            continue

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            continue

        # 规范到 arr.shape == (len(y), len(x))
        if arr.shape == (len(y), len(x)):
            mat_yx = arr
        elif arr.shape == (len(x), len(y)):
            mat_yx = arr.T
        else:
            continue

        x_rt, x_mz = _axis_scores(x)
        y_rt, y_mz = _axis_scores(y)

        # 方案 A: y=RT, x=m/z（无需转置）
        score_a = y_rt + x_mz - (x_rt + y_mz) * 0.25
        # 方案 B: x=RT, y=m/z（需要转置到 RT x m/z）
        score_b = x_rt + y_mz - (y_rt + x_mz) * 0.25

        bonus = 0.5 if str(name).lower() == preferred else 0.0
        if score_a + bonus >= score_b + bonus:
            cand_score = score_a + bonus
            cand_rts = y
            cand_mzs = x
            cand_mat = mat_yx
        else:
            cand_score = score_b + bonus
            cand_rts = x
            cand_mzs = y
            cand_mat = mat_yx.T

        if cand_score > best_score:
            best_score = cand_score
            best = (cand_rts, cand_mzs, cand_mat)

    if best is None:
        raise ValueError(f"rainbow 未找到可用二维 MS 数据: {d_path}")

    rts, mzs, data = best
    return np.asarray(rts, dtype=np.float64), np.asarray(mzs, dtype=np.float64), np.asarray(data, dtype=np.float64)


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
        mz_vals = np.asarray(data_or_spectra, dtype=np.float64)
        int_mat = np.asarray(mz_axis_or_none, dtype=np.float64)
        # 兜底：确保维度一致为 (len(rts), len(mz_vals))
        if int_mat.shape == (len(mz_vals), len(rts)):
            int_mat = int_mat.T
        if int_mat.shape != (len(rts), len(mz_vals)):
            raise ValueError(
                f"矩阵维度与坐标不一致: int_mat={int_mat.shape}, "
                f"len(rts)={len(rts)}, len(mz)={len(mz_vals)}"
            )
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