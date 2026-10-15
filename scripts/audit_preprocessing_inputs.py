"""Check frozen preprocessing inputs and sample PCA reconstruction, without training."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import Config, apply_prepared_input_config
from prepare_preprocessing_manifest import manifest_fingerprint
from prepare_preprocessing_variants import VARIANTS, H99_MISMATCH_REL_PATH

ROLE_KEYS = ("train_idx", "val_idx", "test_batch_idx", "test_unknown_idx")


def audit(manifest_dir, output_dir, per_group, variants_dir=None):
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有检查结果: {output_dir}")
    metadata = pd.read_csv(manifest_dir / "metadata.csv")
    split = json.loads((manifest_dir / "split.json").read_text(encoding="utf-8"))
    fingerprint = manifest_fingerprint(metadata)
    if fingerprint != split["manifest_fingerprint"]:
        raise ValueError("源清单与划分指纹不一致")
    indices = [int(i) for key in ROLE_KEYS for i in split[key]]
    if sorted(indices) != list(range(len(metadata))):
        raise ValueError("冻结划分包含重复或遗漏的样本")

    variants_dir = variants_dir or manifest_dir / "representations"
    frames, contracts = {}, {}
    for name in VARIANTS:
        directory = variants_dir / name
        frame = pd.read_csv(directory / "metadata.csv")
        if frame["sample_id"].tolist() != metadata["sample_id"].tolist():
            raise ValueError(f"{name} 样本或顺序不一致")
        if manifest_fingerprint(frame) != fingerprint or not frame["tensor_ready"].eq(True).all():
            raise ValueError(f"{name} 标签/划分不同或存在未就绪的张量")
        variant_split = json.loads((directory / "split.json").read_text(encoding="utf-8"))
        if any(variant_split[key] != split[key] for key in ROLE_KEYS):
            raise ValueError(f"{name} 划分索引与源清单不同")
        expected_files = {str(Path(value).as_posix()) for value in frame["tensor_path"]}
        actual_files = {p.relative_to(directory).as_posix() for p in (directory / "tensors").rglob("*.npz")}
        if expected_files != actual_files or len(actual_files) != len(frame):
            raise ValueError(f"{name} 存在缺失、多余或复用的张量文件")
        if any((directory / value).stat().st_size == 0 for value in expected_files):
            raise ValueError(f"{name} 有空文件")
        cfg = Config(prepared_dir=str(directory))
        info = apply_prepared_input_config(cfg)
        if info is None or cfg.auto_create_split_on_train or cfg.input_raw_pca_enabled:
            raise ValueError(f"{name} 训练入口未正确冻结输入")
        contracts[name] = {"samples": len(frame), "tensor_files": len(actual_files),
                           "shape": [cfg.in_channels, cfg.rt_bins, cfg.mz_bins],
                           "runtime_pca": cfg.input_raw_pca_enabled,
                           "refresh_split": cfg.auto_create_split_on_train}
        frames[name] = frame

    pca_dir = variants_dir / "dual_channel_pca"
    fit_samples = pd.read_csv(pca_dir / "pca_fit_samples.csv")
    if (fit_samples["metadata_index"].tolist() != split["train_idx"]
            or fit_samples["sample_id"].tolist() != metadata.iloc[split["train_idx"]]["sample_id"].tolist()):
        raise ValueError("PCA 拟合名单与冻结训练集不一致")
    # This pickle is the model generated locally by the preparation script.
    with (pca_dir / "input_mz_pca.pkl").open("rb") as handle:
        pca = pickle.load(handle)
    if int(pca.n_samples_seen_) != len(split["train_idx"]) * 2 * contracts["dual_channel_no_pca"]["shape"][1]:
        raise ValueError("PCA 拟合行数与训练输入不一致")

    rng = np.random.default_rng(42)
    sampled = []
    for _, group in metadata.groupby(["split_role", "product_fine"], sort=True):
        sampled.extend(sorted(rng.choice(group.index.to_numpy(), min(per_group, len(group)), replace=False).tolist()))
    flagged_index = metadata.index[metadata["source_relpath"] == H99_MISMATCH_REL_PATH].tolist()
    sampled = sorted(set(sampled + flagged_index))
    for frame in frames.values():
        if not frame.iloc[flagged_index]["preprocessing_quality_flags"].str.contains("MATRIX_TIC_RT_MISMATCH").all():
            raise ValueError("用户要求保留的 H99 标记丢失")

    quality_rows, reconstruction_rows = [], []
    with threadpool_limits(limits=4):
        for index in sampled:
            tensors = {}
            for name, frame in frames.items():
                with np.load(variants_dir / name / frame.iloc[index]["tensor_path"]) as data:
                    tensor = data["tensor"].copy()
                if list(tensor.shape) != contracts[name]["shape"] or not np.isfinite(tensor).all():
                    raise ValueError(f"{name} 样本 {index} 张量形状或数值异常")
                tensors[name] = tensor
                quality_rows.append({
                    "metadata_index": index, "sample_id": metadata.iloc[index]["sample_id"],
                    "split_role": metadata.iloc[index]["split_role"], "product": metadata.iloc[index]["product_fine"],
                    "variant": name, "minimum": float(tensor.min()), "maximum": float(tensor.max()),
                    "channel_correlation": float(np.corrcoef(tensor[0].ravel(), tensor[1].ravel())[0, 1]),
                    "at_clip_boundary_ratio": float(np.mean(np.abs(tensor) >= 4.99999)) if name != "dual_channel_pca" else None,
                })
            base = tensors["dual_channel_no_pca"].reshape(-1, pca.n_features_in_)
            scores = tensors["dual_channel_pca"].reshape(-1, pca.n_components_)
            expected = pca.transform(base).astype(np.float32)
            if not np.allclose(scores, expected, rtol=1e-5, atol=1e-5):
                raise ValueError(f"样本 {index} 的 PCA 张量与保存的变换不一致")
            reconstructed = pca.inverse_transform(scores)
            centered = base - pca.mean_
            denominator = float(np.sum(centered ** 2))
            error = float(np.sum((base - reconstructed) ** 2) / max(denominator, 1e-12))
            reconstruction_rows.append({
                "metadata_index": index, "sample_id": metadata.iloc[index]["sample_id"],
                "split_role": metadata.iloc[index]["split_role"], "product": metadata.iloc[index]["product_fine"],
                "relative_reconstruction_mse": error,
                "stored_pca_max_abs_difference": float(np.max(np.abs(scores - expected))),
            })

    quality = pd.DataFrame(quality_rows)
    reconstruction = pd.DataFrame(reconstruction_rows)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    summary = {
        "status": "pass", "scope": "input integrity and sampled reconstruction; no model training or evaluation",
        "manifest_fingerprint": fingerprint, "variants": contracts,
        "pca_fit_samples": len(fit_samples), "pca_fit_rows": int(pca.n_samples_seen_),
        "sampled_inputs": len(sampled), "sample_rule": f"up to {per_group} per product and split, seed 42, plus flagged H99",
        "reconstruction_by_split": reconstruction.groupby("split_role")["relative_reconstruction_mse"].agg(["count", "median", "max"]).to_dict("index"),
        "cumulative_training_variance": {str(k): float(cumulative[k - 1]) for k in (8, 16, 32, 64, 128, 256) if k <= len(cumulative)},
        "components_for_95pct_training_variance": int(np.searchsorted(cumulative, 0.95) + 1) if cumulative[-1] >= 0.95 else None,
        "components_for_99pct_training_variance": int(np.searchsorted(cumulative, 0.99) + 1) if cumulative[-1] >= 0.99 else None,
    }
    output_dir.mkdir(parents=True)
    quality.to_csv(output_dir / "sampled_tensor_quality.csv", index=False, encoding="utf-8-sig")
    reconstruction.to_csv(output_dir / "sampled_pca_reconstruction.csv", index=False, encoding="utf-8-sig")
    (output_dir / "input_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(np.arange(1, len(cumulative) + 1), cumulative * 100, color="#226baf")
    axes[0].axhline(99, ls="--", color="#888888", lw=0.8)
    axes[0].set(xlabel="PCA components", ylabel="Cumulative training variance (%)", title="Compression of training inputs", ylim=(0, 100.5))
    roles = [role for role in ("train", "val", "test_batch", "test_unknown") if role in reconstruction["split_role"].unique()]
    values = [reconstruction.loc[reconstruction["split_role"] == role, "relative_reconstruction_mse"].to_numpy() * 100 for role in roles]
    axes[1].boxplot(values, tick_labels=roles, showmeans=True)
    axes[1].set(ylabel="Reconstruction error / centered energy (%)", title=f"{pca.n_components_}-component PCA, {len(sampled)} sampled inputs")
    axes[1].tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_input_quality.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"检查结果: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="冻结预处理输入的轻量完整性和 PCA 还原检查，不训练模型")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "new_prepared_data_preproc_v1")
    parser.add_argument("--representations-dir", type=Path, default=None,
                        help="检查指定输入目录，也可用于发布前检查临时生成目录")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--per-group", type=int, default=2)
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error("per-group 必须为正整数")
    output_dir = args.output_dir or ROOT / "result" / f"preprocessing_input_review_{datetime.now():%Y%m%d_%H%M%S}"
    variants_dir = args.representations_dir.resolve() if args.representations_dir else None
    audit(args.manifest_dir.resolve(), output_dir.resolve(), args.per_group, variants_dir)


if __name__ == "__main__":
    main()
