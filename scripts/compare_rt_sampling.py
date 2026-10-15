"""Compare RT point counts on exactly the same training samples and raw peaks."""
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

from audit_resampling_quality import _peak_metrics
from prepare_preprocessing_variants import (
    ROOT, _load_inputs, _read_raw_matrix, _resample_raw_to_physical_axes,
)

PEAK_KEY = ["metadata_index", "mz", "peak_rt_minutes"]


def compare(manifest_dir, reference_dir, output_dir, point_counts):
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有结果: {output_dir}")
    metadata, _, roles, fingerprint = _load_inputs(manifest_dir)
    reference = json.loads((reference_dir / "resampling_summary.json").read_text(encoding="utf-8"))
    previous_peaks = pd.read_csv(reference_dir / "major_ion_peak_metrics.csv")
    info = json.loads((manifest_dir / "representations" / "dual_channel_no_pca" / "grid_info.json").read_text(encoding="utf-8"))
    if reference["manifest_fingerprint"] != fingerprint or info["source_manifest_fingerprint"] != fingerprint:
        raise ValueError("源清单、上次检查和当前输入指纹不一致")
    sampled = reference["sample_indices"]
    if not sampled or len(sampled) != len(set(sampled)) or not set(sampled).issubset(set(roles["train_idx"])):
        raise ValueError("必须沿用上次检查的训练样本，且不能重复")
    baseline = int(info["rt_bins"])
    if baseline not in point_counts:
        raise ValueError("比较必须包含现有输入的时间点数")
    rt_axes = {n: np.linspace(*info["rt_range"], n, dtype=np.float32) for n in point_counts}
    mz_axis = np.arange(info["mz_range"][0], info["mz_range"][1] + 1, dtype=np.float32)
    if len(mz_axis) != info["mz_bins"]:
        raise ValueError("此检查只适用于当前间隔为1的公共m/z轴")
    baseline_step = float(np.diff(rt_axes[baseline].astype(np.float64)).mean() * 60)
    if not np.isclose(baseline_step, reference["target_rt_step_seconds"]):
        raise ValueError("当前时间范围或点数与上次检查不同")
    output_dir.mkdir(parents=True)
    peak_rows, sample_rows = [], []
    example_curves = None
    example_info = reference.get("rt_plot_example")
    for count, index in enumerate(sampled, 1):
        row = metadata.iloc[index]
        rts, mzs, matrix = _read_raw_matrix(row)
        identity = {"metadata_index": int(index), "sample_id": row["sample_id"],
                    "product": row["product_fine"], "split_role": row["split_role"],
                    "source_relpath": row["source_relpath"]}
        columns = np.argsort(matrix.sum(axis=0, dtype=np.float64))[-reference["top_ions_per_sample"]:][::-1]
        source_mzs = mzs[columns]
        positions = np.searchsorted(mz_axis, source_mzs)
        if np.any(positions >= len(mz_axis)) or not np.array_equal(mz_axis[positions], source_mzs):
            raise ValueError("诊断离子必须精确位于公共m/z轴上")
        for n, rt_axis in rt_axes.items():
            # Verify that one-ion interpolation used for metrics matches the actual
            # preprocessing function at the same observed m/z columns, before dual channels.
            actual = _resample_raw_to_physical_axes(rts, mzs, matrix, rt_axis, mz_axis)
            expected = np.stack([np.interp(rt_axis, rts, np.log1p(matrix[:, c]),
                                          left=0.0, right=0.0) for c in columns], axis=1)
            if actual.shape != (n, len(mz_axis)) or not np.isfinite(actual).all():
                raise ValueError("新时间网格形状或数值异常")
            difference = float(np.max(np.abs(actual[:, positions] - expected)))
            if not np.allclose(actual[:, positions], expected, rtol=1e-5, atol=1e-5):
                raise ValueError("峰诊断插值与实际准备函数不一致")
            sample_rows.append(dict(identity, rt_bins=n,
                                    rt_step_seconds=float(np.diff(rt_axis.astype(np.float64)).mean() * 60),
                                    preprocessing_shape=f"{n}x{len(mz_axis)}",
                                    preprocessing_max_abs_difference=difference,
                                    pca_tensor_float32_mib=2 * n * 256 * 4 / 1024**2))
            for column in columns:
                records, _ = _peak_metrics(rts, matrix[:, column], rt_axis, identity, mzs[column])
                for record in records:
                    record["rt_bins"] = n
                    # Keep the same 22 peaks in the narrow subset at every resolution.
                    record["narrow_at_baseline"] = record["fwhm_seconds"] < baseline_step
                peak_rows.extend(records)
        if example_info and index == example_info["metadata_index"]:
            column = int(np.flatnonzero(mzs == example_info["mz"])[0])
            example_curves = (rts.copy(), matrix[:, column].copy())
        print(f"时间采样对照 {count}/{len(sampled)}: {row['product_fine']}", flush=True)

    peaks, samples = pd.DataFrame(peak_rows), pd.DataFrame(sample_rows)
    base_peaks = peaks[peaks["rt_bins"] == baseline].sort_values(PEAK_KEY).reset_index(drop=True)
    previous = previous_peaks.sort_values(PEAK_KEY).reset_index(drop=True)
    if (len(previous) != len(base_peaks)
            or not np.allclose(previous[PEAK_KEY], base_peaks[PEAK_KEY], rtol=0, atol=1e-12)
            or not np.allclose(previous["bracketing_peak_height_ratio"],
                               base_peaks["bracketing_peak_height_ratio"], equal_nan=True)):
        raise ValueError("1024点结果未复现上次同一批原始峰的检查")
    baseline_heights = base_peaks["bracketing_peak_height_ratio"]
    summary_rows = []
    for n in point_counts:
        group = peaks[peaks["rt_bins"] == n].sort_values(PEAK_KEY).reset_index(drop=True)
        if len(group) != len(base_peaks) or not np.array_equal(group[PEAK_KEY], base_peaks[PEAK_KEY]):
            raise ValueError("不同时间点数没有比较同一批原始峰")
        narrow = group[group["narrow_at_baseline"]]
        heights = group["bracketing_peak_height_ratio"]
        paired = heights.notna() & baseline_heights.notna()
        change = heights[paired] - baseline_heights[paired]
        summary_rows.append({
            "rt_bins": n, "rt_step_seconds": float(np.diff(rt_axes[n].astype(np.float64)).mean() * 60),
            "input_elements_multiple": n / baseline, "peaks": len(group), "fixed_narrow_peaks": len(narrow),
            "all_peak_height_ratio_median": float(heights.median()),
            "narrow_peak_height_ratio_median": float(narrow["bracketing_peak_height_ratio"].median()),
            "narrow_peak_height_ratio_min": float(narrow["bracketing_peak_height_ratio"].min()),
            "narrow_peak_grid_points_median": float(narrow["grid_points_inside_fwhm"].median()),
            "peaks_without_point_inside_fwhm": int(group["grid_points_inside_fwhm"].eq(0).sum()),
            "narrow_peaks_at_least_three_points_inside_fwhm": int(narrow["grid_points_inside_fwhm"].ge(3).sum()),
            "paired_peaks": int(paired.sum()), "height_unavailable_peaks": int(heights.isna().sum()),
            "higher_height_than_baseline_peaks": int((change > 1e-6).sum()),
            "lower_height_than_baseline_peaks": int((change < -1e-6).sum()),
        })
    summary_table = pd.DataFrame(summary_rows)
    summary = {
        "manifest_fingerprint": fingerprint, "scope": "training_only",
        "sample_indices": sampled, "samples": len(sampled),
        "baseline_rt_bins": baseline, "rt_range": info["rt_range"],
        "top_ions_per_sample": reference["top_ions_per_sample"],
        "peak_selection": reference["peak_selection"],
        "narrow_definition": "Same raw peaks narrower than the baseline RT interval at all point counts.",
        "baseline_reproduces_previous_audit": True,
        "preprocessing_max_abs_difference": float(samples["preprocessing_max_abs_difference"].max()),
        "comparisons": summary_rows, "rt_plot_example": example_info,
        "limits": [
            "Peak heights are measured after log1p interpolation reversed by expm1, before dual-channel normalization, clipping and PCA.",
            "The height metric is the larger of the two target points bracketing the raw peak within its prominence basin; adjacent peaks may affect it.",
            "These linspace grids have slightly different phases: individual peaks can get worse at a higher point count.",
            "Only major observed ions from 15 stratified training samples were diagnosed; this is not a classification result or a global optimum.",
            "The fixed-grid accumulation route was not compared here; no PCA was fitted and no complete tensors were saved.",
            "Input element counts scale exactly with RT point count; compressed disk size and training memory/runtime need measurement.",
        ],
    }
    peaks.to_csv(output_dir / "matched_peak_comparison.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(output_dir / "sample_sampling_checks.csv", index=False, encoding="utf-8-sig")
    summary_table.to_csv(output_dir / "rt_sampling_comparison.csv", index=False, encoding="utf-8-sig")
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(output_dir, summary_table, example_info, example_curves, rt_axes)
    _write_report(output_dir, summary)
    print(summary_table.to_string(index=False), flush=True)
    print(f"结果保存至: {output_dir}", flush=True)


def _plot(output_dir, table, example, curves, axes):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 11})
    fig, plot_axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#c97943", "#318b85", "#6774b0"]
    ax = plot_axes[0]
    x = np.arange(len(table))
    values = table["narrow_peak_height_ratio_median"] * 100
    bars = ax.bar(x, values, color=colors[:len(table)], width=0.55)
    ax.set_xticks(x, [f"{r.rt_bins}点\n间隔{r.rt_step_seconds:.2f}秒" for r in table.itertuples()])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center")
    ax.set_ylim(0, 108)
    ax.set_ylabel("采样后峰高 / 原始峰高（%，中位数）")
    ax.set_title(f"固定同一批{int(table.iloc[0]['fixed_narrow_peaks'])}个窄峰")
    ax = plot_axes[1]
    if example and curves:
        rts, signal = curves
        center = example["peak_rt_minutes"]
        raw_keep = np.abs(rts - center) <= 7 / 60
        ax.plot((rts[raw_keep] - center) * 60, signal[raw_keep] / example["raw_peak_height"],
                color="#414951", linewidth=2, label="原始扫描", zorder=4)
        for (n, rt_axis), color in zip(axes.items(), colors):
            sampled = np.expm1(np.interp(rt_axis, rts, np.log1p(signal), left=0.0, right=0.0))
            keep = np.abs(rt_axis - center) <= 7 / 60
            ax.plot((rt_axis[keep] - center) * 60, sampled[keep] / example["raw_peak_height"],
                    "o--", markersize=4, color=color, label=f"{n}点", alpha=0.85)
        ax.set_title(f"同一个{example['product']}峰：m/z {example['mz']:g}，宽{example['fwhm_seconds']:.2f}秒")
        ax.set_xlabel("相对原始峰顶的时间（秒）")
        ax.set_ylabel("强度 / 原始峰高")
        ax.legend(fontsize=9)
    for ax in plot_axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("时间采样点数对照：15条训练样本；峰高在标准化/PCA之前测量", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_dir / "rt_sampling_comparison.png", dpi=160)
    plt.close(fig)


def _write_report(output_dir, summary):
    comparisons = {r["rt_bins"]: r for r in summary["comparisons"]}
    baseline = comparisons[summary["baseline_rt_bins"]]
    lines = ["# 时间采样点数比较", "",
             f"固定{summary['samples']}条训练样本、同一批原始峰，只改变0～40分钟时间轴的点数。PCA未重拟合，三套已生成输入未改写。", "",
             f"| 时间点数 | 间隔 | 全部{baseline['peaks']}峰峰高保留中位数 | 同一批{baseline['fixed_narrow_peaks']}个窄峰峰高保留中位数 | 半峰宽内无采样点 | 输入元素量 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in summary["comparisons"]:
        lines.append(f"| {r['rt_bins']} | {r['rt_step_seconds']:.3f}秒 | {r['all_peak_height_ratio_median']:.2%} | {r['narrow_peak_height_ratio_median']:.2%} | {r['peaks_without_point_inside_fwhm']} | {r['input_elements_multiple']:g}倍 |")
    lines.extend(["", "## 结论", "",
                  "表中各方案使用完全相同的原始峰。峰高保留是中位数，不能理解为每个峰均达到该比例。"])
    if {1024, 2048, 4096}.issubset(comparisons):
        mid, high = comparisons[2048], comparisons[4096]
        lines.extend([
            f"同一批窄峰峰高保留中位数从1024点的{comparisons[1024]['narrow_peak_height_ratio_median']:.1%}，升到2048点的{mid['narrow_peak_height_ratio_median']:.1%}、4096点的{high['narrow_peak_height_ratio_median']:.1%}。2048/4096点半峰宽内无采样点的峰数分别为{mid['peaks_without_point_inside_fwhm']}/{high['peaks_without_point_inside_fwhm']}。",
            f"2048点可作为下一版时间插值输入的优先候选，输入元素量变为2倍。若重点保留窄峰形状，4096点值得保留为另一候选；窄峰半峰宽内有至少3个点的数量：2048点为{mid['narrow_peaks_at_least_three_points_inside_fwhm']}，4096点为{high['narrow_peaks_at_least_three_points_inside_fwhm']}。",
        ])
    lines.append("点数增加不能保证每个峰都变好，因为各方案的采样位置不是完全重合。配对结果如下：")
    for r in summary["comparisons"]:
        if r["rt_bins"] != summary["baseline_rt_bins"]:
            lines.append(f"- {r['rt_bins']}点比基线峰高更高{r['higher_height_than_baseline_peaks']}个、更低{r['lower_height_than_baseline_peaks']}个。")
    lines.extend(["",
                  "这里只比较双通道/PCA前的时间线性插值路线；固定网格累加路线尚未据此选定点数。后续新版本需要考虑计算资源，分类性能和最佳点数仍需后续训练判断。", "",
                  "## 检查与边界", "",
                  f"- {summary['baseline_rt_bins']}点结果复现上次{baseline['peaks']}个原始峰、其中{baseline['fixed_narrow_peaks']}个窄峰的诊断。窄峰集合固定按基线间隔定义，比较时不换峰。",
                  f"- {summary['samples'] * len(summary['comparisons'])}次实际准备函数调用覆盖{summary['samples']}条样本×{len(summary['comparisons'])}种时间点数，输出形状与数值均正常。抽查离子的实际插值与诊断插值最大log强度差为{summary['preprocessing_max_abs_difference']:.3g}。",
                  "- 峰高在log1p时间插值后逆变换测量，位于逐样本标准化、裁剪和PCA之前；不等于最终输入信息保留率、峰面积保留率或分类准确率。",
                  "- 峰顶取左右相邻目标点中较高者，要求在原始峰的突出程度边界内；邻峰可能影响它。半峰宽指半突出程度宽度。",
                  f"- 每条只诊断总强度最高的{summary['top_ions_per_sample']}个观测离子；结果不代表所有低强度峰。未使用验证/测试数据选参数。",
                  "- 2倍/4倍仅指输入元素量。压缩后的磁盘体积、训练内存和运行时间需要实际测量。",
                  "- 保留H99异常记录及完整早期信号。未训练模型、生成全量输入或修改默认时间点数。", "",
                  "图：rt_sampling_comparison.png。逐峰配对、逐样本检查和汇总见同目录CSV/JSON。"])
    (output_dir / "检查结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "new_prepared_data_preproc_v1")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "result" / "resampling_review_20261014_083651")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rt-bins", type=int, nargs="+", default=[1024, 2048, 4096])
    args = parser.parse_args()
    if len(args.rt_bins) != 3 or len(set(args.rt_bins)) != 3 or min(args.rt_bins) < 2:
        parser.error("提供3种不同的时间点数，每种不少于2点")
    output_dir = args.output_dir or ROOT / "result" / f"rt_sampling_comparison_{datetime.now():%Y%m%d_%H%M%S}"
    compare(args.manifest_dir.resolve(), args.reference_dir.resolve(), output_dir.resolve(), sorted(args.rt_bins))


if __name__ == "__main__":
    main()
