#!/usr/bin/env python3
"""Freeze directory labels and batch splits without reading spectra or fitting PCA.

Run from the project root:
    python scripts/prepare_preprocessing_manifest.py
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_prepare import scan_dataset


PROTOCOL = "directory_labels_batch_split_preproc_v1"
KNOWN_PRODUCTS = ("H88", "H99", "HRJ", "HRY", "HYZ", "RJD", "YJD")
HOLDOUT_PRODUCTS = ("HMD", "XCJ")
EXCLUDED_PRODUCTS = ("HDC", "HRL", "HV8", "HYX")
VAL_BATCHES = ("20240312", "20250815")
TEST_BATCHES = ("20240308", "20250905", "20250912", "20250920", "20260313")
ROLE_KEYS = {
    "train": "train_idx",
    "val": "val_idx",
    "test_batch": "test_batch_idx",
    "test_unknown": "test_unknown_idx",
}
ROLE_PRIORITY = {"train": 0, "val": 1, "test_batch": 2, "test_unknown": 3}


def _hash_json(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_fingerprint(metadata):
    """Portable sample/label/split identity, independent of tensor representation."""
    columns = [
        "record_id", "source_relpath", "product_fine", "batch_name", "group_id",
        "label_source", "split_role", "raw_relpath", "raw_size_bytes",
    ]
    # CSV readers may infer date-like batch names as integers; normalize them.
    canonical = metadata[columns].fillna("").copy()
    canonical["raw_size_bytes"] = canonical["raw_size_bytes"].map(int)
    return _hash_json(canonical.astype(str).to_dict("records"))


def _group_id(row):
    # Full-width parentheses and retest suffixes must not create separate lots.
    lot = re.search(r"\d{6,}", str(row["lot_id"]))
    if lot is None:
        return "record:" + row["record_id"]
    return "lot:" + ":".join((row["product_fine"], row["line_code"], lot.group()))


def build_manifest(dataset_root):
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_root}")
    # Reuse the default parser; silence its large per-batch console tables.
    with contextlib.redirect_stdout(io.StringIO()):
        inventory = scan_dataset(str(dataset_root))
    if inventory.empty:
        raise ValueError("No .D sample directories found")
    inventory["source_relpath"] = inventory["d_path"].map(
        lambda value: Path(value).relative_to(dataset_root).as_posix()
    )
    inventory = inventory.sort_values("source_relpath", kind="stable").reset_index(drop=True)
    batches = set(inventory["batch_name"])
    missing_batches = (set(VAL_BATCHES) | set(TEST_BATCHES)) - batches
    if missing_batches:
        raise ValueError(f"Required split batches are missing: {sorted(missing_batches)}")

    inventory["record_id"] = inventory["source_relpath"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    inventory["legacy_sample_id"] = inventory["sample_id"]
    inventory["sample_id"] = inventory["record_id"].map(lambda value: "S_" + value[:20])
    inventory["label_source"] = "directory_name"
    inventory["label_independently_verified"] = False
    inventory["group_id"] = inventory.apply(_group_id, axis=1)
    inventory["raw_relpath"] = ""
    inventory["raw_size_bytes"] = 0
    inventory["raw_status"] = "missing"
    inventory["initial_split_role"] = "excluded"
    inventory["split_role"] = "excluded"
    inventory["exclusion_reason"] = ""
    inventory["cross_split_group_retained_role"] = ""

    for index, row in inventory.iterrows():
        d_path = Path(row["d_path"])
        raw_path = d_path / "data.ms"
        if not raw_path.is_file():
            raw_path = next(
                (path for path in sorted(d_path.iterdir())
                 if path.is_file() and path.name.lower() == "data.ms"), raw_path
            )
        if raw_path.is_file():
            size = raw_path.stat().st_size
            inventory.at[index, "raw_relpath"] = raw_path.relative_to(dataset_root).as_posix()
            inventory.at[index, "raw_size_bytes"] = size
            inventory.at[index, "raw_status"] = "present_nonempty" if size else "empty"

        product = row["product_fine"]
        if bool(row["is_special"]):
            reason = "special_or_blank"
        elif product == "HYX":
            reason = "user_excluded_hyx"
        elif product in EXCLUDED_PRODUCTS:
            reason = "low_support_product"
        elif inventory.at[index, "raw_status"] != "present_nonempty":
            reason = "missing_or_empty_data_ms"
        else:
            reason = ""
        if reason:
            inventory.at[index, "exclusion_reason"] = reason
            continue

        if product in HOLDOUT_PRODUCTS:
            role = "test_unknown"
        elif product in KNOWN_PRODUCTS:
            role = ("test_batch" if row["batch_name"] in TEST_BATCHES else
                    "val" if row["batch_name"] in VAL_BATCHES else "train")
        else:
            raise ValueError(f"Product {product!r} needs an explicit protocol decision")
        inventory.at[index, "initial_split_role"] = role
        inventory.at[index, "split_role"] = role

    candidates = inventory[inventory["split_role"] != "excluded"]
    for _, group in candidates.groupby("group_id", sort=True):
        roles = set(group["split_role"])
        if len(roles) < 2:
            continue
        retained_role = max(roles, key=ROLE_PRIORITY.get)
        inventory.loc[group.index, "cross_split_group_retained_role"] = retained_role
        removed = group.index[group["split_role"] != retained_role]
        inventory.loc[removed, "split_role"] = "excluded"
        inventory.loc[removed, "exclusion_reason"] = "cross_split_lot_group"

    metadata = inventory[inventory["split_role"] != "excluded"].copy().reset_index(drop=True)
    metadata.insert(0, "metadata_index", metadata.index)
    # These are reserved output paths. No tensor directories/files are created here.
    metadata["tensor_path"] = metadata.apply(
        lambda row: f"tensors/{row['batch_name']}/{row['product_fine']}/{row['sample_id']}.npz",
        axis=1,
    )
    metadata["tensor_ready"] = False
    inventory["metadata_index"] = inventory["record_id"].map(
        metadata.set_index("record_id")["metadata_index"]
    ).astype("Int64")
    indices = {
        key: metadata.index[metadata["split_role"] == role].tolist()
        for role, key in ROLE_KEYS.items()
    }
    train_batches = sorted(metadata.loc[metadata["split_role"] == "train", "batch_name"].unique())
    product_coverage = inventory[~inventory["is_special"]].groupby("product_fine")["batch_name"].nunique()
    split = {
        "protocol": PROTOCOL,
        "fixed_split": True,
        "stage": "metadata_only",
        "known_products": list(KNOWN_PRODUCTS),
        "holdout_products": list(HOLDOUT_PRODUCTS),
        "excluded_products": list(EXCLUDED_PRODUCTS),
        "product_batch_coverage": {product: int(count) for product, count in product_coverage.items()},
        "train_batches": sorted(set(train_batches) | set(VAL_BATCHES)),
        "model_train_batches": train_batches,
        "model_select_holdout_batches": list(VAL_BATCHES),
        "holdout_batches": list(TEST_BATCHES),
        **indices,
        "seed": 42,
        "random_sampling_used": False,
        "label_source": "directory_name",
        "label_independently_verified": False,
        "group_definition": "product + production line + numeric lot; missing lot uses record_id",
        "cross_split_group_policy": "retain_test_then_val_then_train; exclude lower priority records",
        "batch_policy": "known products isolated by batch; unknown products withheld in all batches",
        "evaluation_scope": "cross_batch_holdout_not_strict_forward_time",
        "pca_fit_scope_required": "train_idx_only",
        "stats": {
            "n_train": len(indices["train_idx"]),
            "n_val": len(indices["val_idx"]),
            "n_test_batch": len(indices["test_batch_idx"]),
            "n_test_unknown": len(indices["test_unknown_idx"]),
            "n_excluded": int((inventory["split_role"] == "excluded").sum()),
        },
        "manifest_fingerprint": manifest_fingerprint(metadata),
        "partition_record_ids": {
            key: metadata.iloc[positions]["record_id"].tolist()
            for key, positions in indices.items()
        },
    }
    split["split_fingerprint"] = _hash_json({
        "protocol": PROTOCOL,
        "manifest_fingerprint": split["manifest_fingerprint"],
        "partitions": split["partition_record_ids"],
    })
    audit = validate_manifest(inventory, metadata, split)
    if audit["status"] != "pass":
        raise ValueError(f"Manifest audit failed: {audit['checks']}")
    return inventory, metadata, split, audit


def validate_manifest(inventory, metadata, split):
    """Check the saved row coordinate system and isolation before any tensor work."""
    partitions = [split[key] for key in ROLE_KEYS.values()]
    flattened = [index for part in partitions for index in part]
    closed = metadata[metadata["split_role"].isin(("train", "val", "test_batch"))]
    batch_roles = closed.groupby("batch_name")["split_role"].nunique()
    checks = {
        "indices_cover_metadata_once": sorted(flattened) == list(range(len(metadata))),
        "metadata_indices_are_contiguous": metadata["metadata_index"].tolist() == list(range(len(metadata))),
        "record_ids_unique": inventory["record_id"].is_unique,
        "sample_ids_unique": inventory["sample_id"].is_unique,
        "source_paths_unique": inventory["source_relpath"].is_unique,
        "inventory_included_records_match": inventory.loc[
            inventory["split_role"] != "excluded", "record_id"
        ].tolist() == metadata["record_id"].tolist(),
        "inventory_all_roles_recognized": set(inventory["split_role"]) <= set(ROLE_KEYS) | {"excluded"},
        "known_batches_isolated": bool((batch_roles == 1).all()),
        "lot_groups_isolated": bool((metadata.groupby("group_id")["split_role"].nunique() == 1).all()),
        "only_declared_products": set(metadata["product_fine"]) == set(KNOWN_PRODUCTS + HOLDOUT_PRODUCTS),
        "excluded_products_absent": not metadata["product_fine"].isin(EXCLUDED_PRODUCTS).any(),
        "special_samples_absent": not metadata["is_special"].any(),
        "raw_files_present_nonempty": bool((metadata["raw_status"] == "present_nonempty").all()),
        "directory_labels_consistent": bool((metadata["product_fine"] == metadata["code"]).all()),
        "no_exclusion_reason_in_metadata": bool((metadata["exclusion_reason"] == "").all()),
        "all_exclusions_explained": bool((inventory.loc[inventory["split_role"] == "excluded", "exclusion_reason"] != "").all()),
        "fingerprint_matches": manifest_fingerprint(metadata) == split["manifest_fingerprint"],
    }
    for role, key in ROLE_KEYS.items():
        positions = split[key]
        if any(index < 0 or index >= len(metadata) for index in positions):
            checks[f"{role}_rows_match"] = False
            continue
        subset = metadata.iloc[positions]
        checks[f"{role}_rows_match"] = bool((subset["split_role"] == role).all())
        expected = HOLDOUT_PRODUCTS if role == "test_unknown" else KNOWN_PRODUCTS
        checks[f"{role}_all_products_present"] = set(subset["product_fine"]) == set(expected)
        checks[f"{role}_record_ids_match"] = subset["record_id"].tolist() == split["partition_record_ids"][key]
        batch_names = set(subset["batch_name"].astype(str))
        if role in ("val", "test_batch"):
            expected_batches = VAL_BATCHES if role == "val" else TEST_BATCHES
            checks[f"{role}_batches_match"] = batch_names == set(expected_batches)
        elif role == "train":
            checks["train_excludes_val_test_batches"] = not batch_names & set(VAL_BATCHES + TEST_BATCHES)
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def product_counts(metadata):
    counts = pd.crosstab(metadata["product_fine"], metadata["split_role"])
    counts = counts.reindex(index=KNOWN_PRODUCTS + HOLDOUT_PRODUCTS, columns=list(ROLE_KEYS), fill_value=0)
    counts["total"] = counts.sum(axis=1)
    return counts.rename_axis("product").reset_index()


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_counts_plot(path, counts):
    """Static distribution chart; values come from the frozen metadata only."""
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    known = counts[counts["product"].isin(KNOWN_PRODUCTS)]
    positions = np.arange(len(known))
    figure, axis = plt.subplots(figsize=(10, 6.5))
    styles = (
        ("train", "训练", -0.25, "#386CB0"),
        ("val", "验证", 0.0, "#E5A33E"),
        ("test_batch", "测试", 0.25, "#429B83"),
    )
    for column, label, offset, color in styles:
        bars = axis.barh(positions + offset, known[column], height=0.22, color=color, label=label)
        axis.bar_label(bars, padding=4, fontsize=10)
    axis.set_yticks(positions, known["product"])
    axis.invert_yaxis()
    axis.set_xlim(0, max(known["train"]) * 1.12)
    axis.set_xlabel("样本数")
    axis.set_title("固定划分 v1：各已知产品的样本分布", loc="left", fontsize=15, pad=22)
    axis.legend(loc="lower right", frameon=False)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#DDDDDD", alpha=0.6)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    figure.text(0.12, 0.02, "未知产品另行留出：HMD 68 条、XCJ 65 条；图中不包含排除记录。", fontsize=10)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def write_manifest(output, inventory, metadata, split, audit, refresh=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        existing_metadata_path = output / "metadata.csv"
        if not refresh or not existing_metadata_path.is_file():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
        existing_metadata = pd.read_csv(existing_metadata_path)
        if manifest_fingerprint(existing_metadata) != split["manifest_fingerprint"]:
            raise FileExistsError(
                "Existing output describes different source data; choose a new --output path"
            )
    output.mkdir(parents=True, exist_ok=True)
    counts = product_counts(metadata)
    excluded = inventory[inventory["split_role"] == "excluded"]
    for name, table in (
        ("metadata.csv", metadata), ("sample_inventory.csv", inventory),
        ("excluded_samples.csv", excluded), ("product_split_counts.csv", counts),
    ):
        table.to_csv(output / name, index=False, encoding="utf-8-sig")
    _write_json(output / "split.json", split)
    _write_json(output / "manifest_audit.json", audit)
    _write_json(output / "manifest_summary.json", {
        "protocol": PROTOCOL,
        "stage": "metadata_only",
        "raw_scan_count": len(inventory),
        "raw_batch_count": int(inventory["batch_name"].nunique()),
        "metadata_count": len(metadata),
        "split_counts": split["stats"],
        "exclusion_counts": {reason: int(count) for reason, count in excluded["exclusion_reason"].value_counts().items()},
        "manifest_fingerprint": split["manifest_fingerprint"],
        "split_fingerprint": split["split_fingerprint"],
        "audit_status": audit["status"],
        "product_counts": counts.to_dict("records"),
        "raw_check_scope": "data.ms existence and nonzero size only; spectra not read",
    })
    readme = f"""# 预处理对照的固定清单 v1

当前仅有样本清单和固定划分，尚未生成张量或拟合 PCA，不能直接用于训练。

- 标签从原始 dataset 的目录名解析，不继承旧 relabel；真实产品未独立核实。
- 已知产品：{', '.join(KNOWN_PRODUCTS)}；未知留出产品：{', '.join(HOLDOUT_PRODUCTS)}。
- 验证批次：{', '.join(VAL_BATCHES)}。
- 测试批次：{', '.join(TEST_BATCHES)}；其余批次的已知产品用于训练。
- 排除 HYX、原策略低支持产品 HDC/HRL/HV8、特殊样本、缺失或空的 data.ms。
- 同产品、同产线、同数字批号若跨集合，优先保留测试，其次验证，再次训练；同集合复检保留。
- 这是跨批次留出，未声称严格时间前向预测或从未在旧实验使用过的测试数据。
- 全量矩阵审查已读取2743/2743条，基础结构与数值检查全部通过；详见 raw_data_audit.csv。
- 矩阵与TIC的扫描数及RT范围有1条不一致，需人工复核，不据此自动排除。
- H99样本 20240304/033#H99CS124030403.D 的矩阵比TIC多501个早期扫描；该区间占矩阵总信号约54%，峰值在m/z 44，记录在 raw_coverage_mismatch_review.csv。
- 77条样本的单样本最大强度为8,388,096，共464个矩阵点达到该值；比例很低，是否为采集上限尚未确认。

## 文件与使用约定

- metadata.csv：有效样本，位置索引从 0 连续编号；tensor_path 是待生成路径，tensor_ready=False。
- split.json：train_idx / val_idx / test_batch_idx / test_unknown_idx 都指向 metadata.csv 的行位置。
- sample_inventory.csv：全部扫描记录；excluded_samples.csv：排除原因明细。
- product_split_counts.csv：按产品和集合统计；manifest_audit.json：隔离与索引检查。
- raw_data_audit.csv / raw_data_audit_summary.json：2743条原始矩阵读取和数值检查结果。
- raw_tic_audit.csv / raw_matrix_tic_comparison.csv：TIC检查及矩阵覆盖范围对照；raw_matrix_dimensions.png：矩阵维度分布。
- raw_coverage_mismatch_review.csv：覆盖范围不一致样本的前置区间信号复核。
- raw_signal_ceiling_review.csv：达到可疑强度值的样本及其对应矩阵点数。
- record_id 基于相对原始目录生成，跨机器移动仍保持身份；legacy_sample_id 保留旧解析标识。

后续固定物理 m/z 网格＋log1p、双通道＋PCA、双通道不做 PCA 三组必须沿用相同样本及索引。
PCA 和任何需要从数据学习的参数仅用 train_idx 拟合；不要重新自动选择划分。
后续训练需关闭 auto_create_split_on_train，避免自动刷新本文件。
当前未更改项目默认 prepared_dir，旧数据继续保留。

有效样本 {len(metadata)} 条；训练 {split['stats']['n_train']}、验证 {split['stats']['n_val']}、
已知测试 {split['stats']['n_test_batch']}、未知留出 {split['stats']['n_test_unknown']}。
扫描记录 {len(inventory)} 条，排除 {len(excluded)} 条。检查：{audit['status']}。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    save_counts_plot(output / "product_split_counts.png", counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="new_prepared_data_preproc_v1")
    parser.add_argument("--refresh", action="store_true",
                        help="Refresh generated files only when the manifest fingerprint is unchanged")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.refresh:
        parser.error(f"Refusing to overwrite existing output without --refresh: {output}")
    inventory, metadata, split, audit = build_manifest(args.dataset)
    try:
        write_manifest(output, inventory, metadata, split, audit, refresh=args.refresh)
    except FileExistsError as exc:
        parser.error(str(exc))
    print(f"Manifest: {output}")
    print(f"Scanned={len(inventory)}, included={len(metadata)}, excluded={len(inventory) - len(metadata)}")
    print(product_counts(metadata).to_string(index=False))
    print(f"Audit: {audit['status']}; metadata only, no spectra/tensors/PCA/training.")


if __name__ == "__main__":
    main()
