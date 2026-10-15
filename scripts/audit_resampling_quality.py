"""Diagnose m/z interpolation and RT sampling on training samples, without fitting."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths

from prepare_preprocessing_variants import (
    H99_MISMATCH_REL_PATH, ROOT, _fixed_grid_log1p, _load_inputs,
    _read_raw_matrix,
)


def _peak_metrics(rts, signal, rt_axis, identity, mz):
    """Inspect prominent peaks in one ion; metrics precede z-score and clipping."""
    maximum = float(signal.max())
    if maximum <= 0:
        return [], []
    peaks, props = find_peaks(signal, prominence=0.1 * maximum)
    if not len(peaks):
        return [], []
    widths = peak_widths(signal, peaks, rel_height=0.5,
                        prominence_data=(props["prominences"],
                                         props["left_bases"], props["right_bases"]))
    source_positions = np.arange(len(rts))
    left_rt = np.interp(widths[2], source_positions, rts)
    right_rt = np.interp(widths[3], source_positions, rts)
    sampled_log = np.interp(rt_axis, rts, np.log1p(signal), left=0.0, right=0.0)
    # Reverse log1p solely to express sampled height in the original intensity units.
    sampled_signal = np.expm1(sampled_log)
    rows, examples = [], []
    for j, peak in enumerate(peaks):
        center = float(rts[peak])
        if center <= rt_axis[0] or center >= rt_axis[-1]:
            continue
        position = int(np.searchsorted(rt_axis, center))
        candidates = [k for k in (position - 1, position)
                      if 0 <= k < len(rt_axis)
                      and rts[props["left_bases"][j]] <= rt_axis[k]
                      <= rts[props["right_bases"][j]]]
        ratio = (float(max(sampled_signal[candidates]) / signal[peak])
                 if candidates else None)
        row = dict(identity, mz=float(mz), peak_rt_minutes=center,
                   raw_peak_height=float(signal[peak]),
                   prominence=float(props["prominences"][j]),
                   fwhm_seconds=float((right_rt[j] - left_rt[j]) * 60),
                   grid_points_inside_fwhm=int(np.count_nonzero(
                       (rt_axis >= left_rt[j]) & (rt_axis <= right_rt[j]))),
                   bracketing_peak_height_ratio=ratio)
        rows.append(row)
        if ratio is not None and row["fwhm_seconds"] < float(np.diff(rt_axis).mean() * 60):
            examples.append((row, rts.copy(), signal.copy(), rt_axis.copy(), sampled_signal))
    return rows, examples


def audit(manifest_dir, output_dir, per_product, top_ions):
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有结果: {output_dir}")
    metadata, split, roles, fingerprint = _load_inputs(manifest_dir)
    variants = manifest_dir / "representations"
    info = json.loads((variants / "dual_channel_no_pca" / "grid_info.json").read_text(encoding="utf-8"))
    fixed = json.loads((variants / "fixed_grid_log1p" / "grid_info.json").read_text(encoding="utf-8"))
    if info["source_manifest_fingerprint"] != fingerprint or fixed["source_manifest_fingerprint"] != fingerprint:
        raise ValueError("输入目录与固定清单不一致")
    rt_axis = np.linspace(*info["rt_range"], info["rt_bins"], dtype=np.float32)
    mz_axis = np.arange(info["mz_range"][0], info["mz_range"][1] + 1, dtype=np.float32)
    if len(mz_axis) != info["mz_bins"]:
        raise ValueError("此检查只适用于当前间隔为1的公共m/z轴")
    rng = np.random.default_rng(42)
    sampled = []
    for _, group in metadata.iloc[roles["train_idx"]].groupby("product_fine", sort=True):
        sampled.extend(rng.choice(group.index.to_numpy(), min(per_product, len(group)), replace=False).tolist())
    flagged = metadata.index[metadata["source_relpath"] == H99_MISMATCH_REL_PATH].tolist()
    sampled = sorted(set(sampled + [i for i in flagged if i in roles["train_idx"]]))
    if not set(sampled).issubset(set(split["train_idx"])):
        raise ValueError("诊断参数只使用训练样本")
    output_dir.mkdir(parents=True)
    samples, peaks, plot_examples = [], [], []
    mz_example = None
    for count, index in enumerate(sampled, 1):
        row = metadata.iloc[index]
        identity = {"metadata_index": int(index), "sample_id": row["sample_id"],
                    "product": row["product_fine"], "split_role": row["split_role"],
                    "source_relpath": row["source_relpath"]}
        rts, mzs, matrix = _read_raw_matrix(row)
        if not np.allclose(mzs, np.round(mzs), rtol=0, atol=1e-5):
            raise ValueError("发现非整数原始m/z；不能用精确补零比较当前插值")
        logged = np.log1p(matrix)
        aligned = interp1d(mzs, logged, axis=1, bounds_error=False,
                           fill_value=0, assume_sorted=True)(mz_axis)
        missing = ~np.isin(mz_axis, mzs)
        added = float(aligned[:, missing].sum(dtype=np.float64))
        logged_sum = float(logged.sum(dtype=np.float64))
        keep_rt = (rts >= fixed["rt_range"][0]) & (rts <= fixed["rt_range"][1])
        keep_mz = (mzs >= fixed["mz_range"][0]) & (mzs < fixed["mz_range"][1])
        in_range_sum = float(matrix[np.ix_(keep_rt, keep_mz)].sum(dtype=np.float64))
        fixed_logged = _fixed_grid_log1p(rts, mzs, matrix, fixed["rt_bins"],
                                       fixed["rt_range"], fixed["mz_bins"], fixed["mz_range"])
        recovered_sum = float(np.expm1(fixed_logged.astype(np.float64)).sum())
        samples.append(dict(identity, raw_mz_columns=len(mzs),
                            missing_mz_inside_source_range=int(np.count_nonzero(
                                missing & (mz_axis >= mzs[0]) & (mz_axis <= mzs[-1]))),
                            mz_added_log_signal_ratio=added / max(logged_sum, 1e-12),
                            raw_rt_step_seconds=float(np.median(np.diff(rts)) * 60),
                            target_rt_step_seconds=float(np.diff(rt_axis.astype(np.float64)).mean() * 60),
                            outside_fixed_range_signal_ratio=1.0 - in_range_sum / float(matrix.sum(dtype=np.float64)),
                            fixed_grid_sum_relative_error=abs(recovered_sum - in_range_sum) / max(in_range_sum, 1e-12)))
        if mz_example is None and missing.any():
            totals = aligned.sum(axis=0, dtype=np.float64)
            largest_missing = int(np.argmax(np.where(missing, totals, -1)))
            support = np.zeros(len(mz_axis), dtype=np.float64)
            support[np.searchsorted(mz_axis, mzs)] = logged.sum(axis=0, dtype=np.float64)
            mz_example = (identity, mz_axis.copy(), totals, support, largest_missing)
        selected_ions = np.argsort(matrix.sum(axis=0, dtype=np.float64))[-top_ions:][::-1]
        for column in selected_ions:
            ion_peaks, examples = _peak_metrics(rts, matrix[:, column], rt_axis, identity, mzs[column])
            peaks.extend(ion_peaks)
            plot_examples.extend(examples)
        print(f"训练样本诊断 {count}/{len(sampled)}: {row['product_fine']}，累计主要离子峰 {len(peaks)}", flush=True)

    sample_frame, peak_frame = pd.DataFrame(samples), pd.DataFrame(peaks)
    if peak_frame.empty:
        raise ValueError("未检出满足阈值的峰，无法判断RT采样")
    sample_frame.to_csv(output_dir / "sample_resampling_metrics.csv", index=False, encoding="utf-8-sig")
    peak_frame.to_csv(output_dir / "major_ion_peak_metrics.csv", index=False, encoding="utf-8-sig")
    step = float(np.diff(rt_axis.astype(np.float64)).mean() * 60)
    narrow = peak_frame[peak_frame["fwhm_seconds"] < step]
    valid_heights = peak_frame["bracketing_peak_height_ratio"].dropna()
    narrow_heights = narrow["bracketing_peak_height_ratio"].dropna()
    summary = {
        "manifest_fingerprint": fingerprint, "sample_scope": "training_only",
        "sampling_rule": "seed42; up to two per known product plus flagged H99", "samples": len(samples),
        "sample_indices": sampled, "top_ions_per_sample": top_ions,
        "peak_selection": "ion raw peak prominence >= 10% of its maximum; RT endpoints excluded",
        "mz_added_log_signal_ratio_median": float(sample_frame["mz_added_log_signal_ratio"].median()),
        "mz_added_log_signal_ratio_max": float(sample_frame["mz_added_log_signal_ratio"].max()),
        "raw_rt_step_seconds_median": float(sample_frame["raw_rt_step_seconds"].median()),
        "target_rt_step_seconds": step, "peaks_checked": len(peaks), "narrow_peaks": len(narrow),
        "peaks_without_grid_point_inside_fwhm": int(peak_frame["grid_points_inside_fwhm"].eq(0).sum()),
        "bracketing_height_ratio_median": float(valid_heights.median()) if len(valid_heights) else None,
        "narrow_bracketing_height_ratio_median": float(narrow_heights.median()) if len(narrow_heights) else None,
        "height_ratio_unavailable_peaks": int(len(peak_frame) - len(valid_heights)),
        "fixed_grid_sum_relative_error_max": float(sample_frame["fixed_grid_sum_relative_error"].max()),
        "outside_fixed_range_signal_ratio_max": float(sample_frame["outside_fixed_range_signal_ratio"].max()),
        "limits": ["Train-only stratified diagnostic, not an unbiased dataset estimate or classification result.",
                   "m/z ratio is a log-space sum; the cause of missing source mass columns is unverified.",
                   "FWHM here is width at half peak prominence, in original intensity units.",
                   "Height uses the two target points bracketing the original peak within its prominence basin; adjacent peaks may affect it.",
                   "No grid point inside FWHM does not mean the entire peak signal disappeared.",
                   "Fixed-grid sum preservation precedes log1p, dual-channel normalization and clipping; it does not prove preservation of peak shape."]
    }
    if plot_examples:
        # Show one illustrative narrow peak near the median measured retention.
        plot_examples.sort(key=lambda item: item[0]["bracketing_peak_height_ratio"])
        example = plot_examples[len(plot_examples) // 2]
        summary["rt_plot_example"] = example[0]
    else:
        example = None
    (output_dir / "resampling_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(output_dir, sample_frame, peak_frame, mz_example, example, step)
    _write_report(output_dir, summary, sample_frame)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _plot(output_dir, samples, peaks, mz_example, rt_example, step):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    x = np.arange(len(samples))
    ax.bar(x, samples["mz_added_log_signal_ratio"] * 100, color="#318b85")
    ax.set_xticks(x, samples["product"], rotation=45)
    ax.set_title("m/z 插值补入的信号（15条训练样本）" if len(samples) == 15 else "m/z 插值补入的信号（训练抽样）")
    ax.set_ylabel("新增信号 / 原始信号（log空间，%）")
    ax = axes[0, 1]
    if mz_example:
        identity, axis, linear, zero, center = mz_example
        lo, hi = max(0, center - 5), min(len(axis), center + 6)
        area = slice(lo, hi)
        ax.plot(axis[area], linear[area], "o-", label="当前：线性插值", color="#c97943")
        ax.plot(axis[area], zero[area], "s--", label="比较：缺失位置补0", color="#318b85")
        ax.set_title(f"缺失 m/z 的补值示例（{identity['product']}）")
        ax.set_xlabel("m/z")
        ax.set_ylabel("全扫描 log1p 强度和")
        ax.legend(fontsize=9)
    ax = axes[1, 0]
    categories = ["窄于新采样间隔", "不窄于新采样间隔"]
    narrow = peaks["fwhm_seconds"] < step
    hit = peaks["grid_points_inside_fwhm"] > 0
    counts_hit = [int((narrow & hit).sum()), int((~narrow & hit).sum())]
    counts_missed = [int((narrow & ~hit).sum()), int((~narrow & ~hit).sum())]
    ax.bar(categories, counts_hit, color="#318b85", label="半峰宽内有采样点")
    ax.bar(categories, counts_missed, bottom=counts_hit, color="#c97943", label="半峰宽内无采样点")
    for i, total in enumerate(np.array(counts_hit) + np.array(counts_missed)):
        ax.text(i, total, str(total), ha="center", va="bottom")
    ax.set_title(f"主要离子峰的采样覆盖（间隔 {step:.2f}秒）")
    ax.set_ylabel("峰数量")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    if rt_example:
        row, rts, raw, target, sampled = rt_example
        center = row["peak_rt_minutes"]
        window = max(step * 3, row["fwhm_seconds"] * 2) / 60
        keep_raw = np.abs(rts - center) <= window
        keep_target = np.abs(target - center) <= window
        denominator = row["raw_peak_height"]
        ax.plot((rts[keep_raw] - center) * 60, raw[keep_raw] / denominator,
                "-", color="#318b85", label="原始扫描")
        ax.plot((target[keep_target] - center) * 60, sampled[keep_target] / denominator,
                "o--", color="#c97943", label="当前采样（逆log1p显示）")
        ax.set_title(f"窄峰示例：{row['product']}，m/z {row['mz']:g}，宽 {row['fwhm_seconds']:.2f}秒")
        ax.set_xlabel("相对原始峰顶的时间（秒）")
        ax.set_ylabel("强度 / 原始峰高")
        ax.legend(fontsize=9)
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("重采样诊断：只用训练样本；未训练模型、未改写已有输入", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "resampling_diagnostics.png", dpi=160)
    plt.close(fig)


def _write_report(output_dir, summary, samples):
    s = summary
    narrow_height = s["narrow_bracketing_height_ratio_median"]
    height_text = f"{narrow_height * 100:.2f}%" if narrow_height is not None else "不可用"
    text = f"""# 重采样抽样检查结论

只读取训练集{s['samples']}条原始矩阵：每个已知产品最多2条，随机种子42，另加入已保留的H99异常记录。未使用验证/测试样本挑参数，未改写三套输入。

| 检查 | 实测 | 含义 |
| --- | --- | --- |
| m/z插值新增log信号占比 | 中位数{s['mz_added_log_signal_ratio_median'] * 100:.4f}%，最大{s['mz_added_log_signal_ratio_max'] * 100:.4f}% | 存在相邻质量位置补值，整体占比小 |
| RT原始/目标间隔 | {s['raw_rt_step_seconds_median']:.3f}秒 / {s['target_rt_step_seconds']:.3f}秒 | 约每{s['target_rt_step_seconds'] / s['raw_rt_step_seconds_median']:.1f}次原始扫描取一个新点 |
| 主要离子峰 | {s['peaks_checked']}个，其中{s['narrow_peaks']}个窄于目标间隔 | 窄峰存在采样风险 |
| 半峰宽内无新采样点 | {s['peaks_without_grid_point_inside_fwhm']}个 | 峰中心区域可能跳过，不代表整个峰消失 |
| 窄峰相邻目标点峰高保留中位数 | {height_text} | 在log插值后逆log1p衡量，尚未做双通道标准化；不是最终输入损失或识别准确率 |
| 固定网格累加总量误差最大 | {s['fixed_grid_sum_relative_error_max']:.3g} | 范围内累加保留总量，不能据此证明峰形保留 |
| 固定范围外信号比例最大 | {s['outside_fixed_range_signal_ratio_max']:.3g} | 抽样未发现坐标裁剪丢信号 |

## 当前建议

1. ①整理、②固定划分、③三套输入生成均已完成；④检查补齐了m/z补值和RT窄峰诊断，但不能说预处理已经最优。
2. 先处理RT窄峰采样问题。可在同一训练样本比较RT累加/面积积分与现有插值；如果重视峰形，再检查2048点。现有固定网格累加可作为总量保留的参考，不能直接认定峰形更优。
3. 原始质量轴在本抽样中均为整数；m/z可比较按原有位置映射、缺失位置补0的方案。缺失列的仪器含义尚未核实，不把补0直接认定为真值。
4. 参数方案确认后，再按同一清单生成新版本，并让做/不做PCA使用相同前处理。正式训练对照前还要统一增强设置，避免把PCA列当作物理m/z平移。
5. PCA256压缩损失小仍成立，但分类帮助和最佳维数要由后续训练对照回答。

## 指标边界

- m/z占比在log空间求和，不是原始峰面积比例。
- 只对每条样本总强度最高的{s['top_ions_per_sample']}个离子找峰，筛选条件为突出程度至少为该离子最高峰的10%。半峰宽按原始强度中半突出程度宽度计算。
- 峰高取峰顶左右相邻目标点，且要求在原始峰的突出程度边界内；邻峰可能影响它。{s['height_ratio_unavailable_peaks']}个峰没有有效相邻点，不纳入峰高统计。
- 这些是训练集分层抽样诊断，不是全数据无偏估计、分类性能结果或最佳参数证明。
- H99异常记录继续保留；未裁除早期扫描、删除输入或训练模型。

图：resampling_diagnostics.png。逐样本及逐峰数值见两个CSV；清单指纹见resampling_summary.json。
"""
    (output_dir / "检查结论.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "new_prepared_data_preproc_v1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--per-product", type=int, default=2)
    parser.add_argument("--top-ions", type=int, default=5)
    args = parser.parse_args()
    if args.per_product < 1 or args.top_ions < 1:
        parser.error("抽样数及离子数必须大于0")
    output = args.output_dir or ROOT / "result" / f"resampling_review_{datetime.now():%Y%m%d_%H%M%S}"
    audit(args.manifest_dir.resolve(), output.resolve(), args.per_product, args.top_ions)


if __name__ == "__main__":
    main()
