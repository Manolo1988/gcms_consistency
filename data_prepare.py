"""
扫描 dataset 目录 -> 建元数据表 -> 逐样本转换为 .npz 张量。
运行方式: python data_prepare.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from data_reader import d_folder_to_tensor

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


def convert_all(metadata: pd.DataFrame, out_dir: str, cfg: Config):
    """逐样本读取 .D 并生成 .npz 张量。"""
    out_dir = Path(out_dir)
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

    ans = input("\n开始转换张量? (y/n): ").strip().lower()
    if ans == "y":
        convert_all(metadata, cfg.prepared_dir, cfg)
