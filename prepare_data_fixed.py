#!/usr/bin/env python3
"""
重新准备数据脚本 —— 修复三个关键问题:

1. m/z 范围: (0,200) → (0,550), 匹配仪器实际扫描范围 (30-550 Da)
2. PCA 仅在训练集上拟合, 防止 BLANK/ENV/排除产品/留出产品泄漏到输入特征
3. 重新生成正确的 split.json (旧版 split 存在留出产品泄漏到训练集的问题)

用法: python prepare_data_fixed.py

输出: new_prepared_data/
  ├── metadata.csv
  ├── split.json
  ├── grid_info.json
  └── tensors/{batch}/{product}/{idx}_{sample_id}.npz
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

# ── 将项目根目录加入 sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from data_prepare import (
    scan_dataset,
    _align_mz_axis_linear,
    _build_tensor_paths,
    _read_raw_matrix_no_bins,
    _resample_rt_linear,
    _safe_tag,
)
from data_reader import build_dual_channel

# ─────────────────────────────────────────────
#  内联: dataset.py 中的核心函数 (避免 import torch)
# ─────────────────────────────────────────────

def _load_and_filter(metadata_csv, exclude_blanks=True, exclude_special=True):
    """过滤 BLANK 和特殊样品。"""
    df = pd.read_csv(metadata_csv)
    if exclude_blanks:
        df = df[df["product_fine"] != "BLANK"]
    if exclude_special:
        df = df[~df["is_special"]]
    return df


def _stratified_split(df_subset, product_col, val_ratio, rng):
    """分层抽样: 按产品类别分层划出验证集。"""
    train_idx = []
    val_idx = []
    for cls in df_subset[product_col].unique():
        cls_idx = df_subset[df_subset[product_col] == cls].index.tolist()
        n_val = max(1, int(len(cls_idx) * val_ratio))
        perm = rng.permutation(len(cls_idx))
        val_idx.extend([cls_idx[i] for i in perm[:n_val]])
        train_idx.extend([cls_idx[i] for i in perm[n_val:]])
    return train_idx, val_idx


def _print_split_summary(split, df, product_col):
    """打印划分摘要。"""
    print(f"\n{'='*60}")
    print("数据划分摘要")
    print(f"{'='*60}")
    print(f"  已知产品 ({len(split['known_products'])}): "
          f"{split['known_products']}")
    print(f"  留出产品 ({len(split['holdout_products'])}): "
          f"{split['holdout_products']}  → Setting B/C")
    print(f"  排除产品 ({len(split['excluded_products'])}): "
          f"{split['excluded_products']}  (样本不足)")
    print(f"  训练批次 ({len(split['train_batches'])}): "
          f"{split['train_batches']}")
    print(f"  伪验证批次 ({len(split.get('model_select_holdout_batches', []))}): "
          f"{split.get('model_select_holdout_batches', [])}  → 训练早停选模")
    print(f"  留出批次 ({len(split['holdout_batches'])}): "
          f"{split['holdout_batches']}  → Setting A")

    def _count_products_and_batches(indices, name):
        subset = df.loc[indices]
        product_counts = subset[product_col].value_counts().to_dict()
        batch_counts = subset["batch_name"].value_counts().to_dict()
        print(f"\n  {name}:")
        print(f"    产品数量: {len(product_counts)}")
        print(f"    批次数量: {len(batch_counts)}")
        print(f"    产品分布: {product_counts}")
        print(f"    批次分布: {batch_counts}")

    _count_products_and_batches(split['train_idx'], "训练集")
    _count_products_and_batches(split['val_idx'], "验证集")
    _count_products_and_batches(split['test_batch_idx'], "Setting A 测试集")
    _count_products_and_batches(split['test_unknown_idx'], "Setting B/C 测试集")

    s = split["stats"]
    print(f"\n  训练集: {s['n_train']} 样本")
    print(f"  验证集: {s['n_val']} 样本  (训练监控)")
    print(f"  Setting A 测试: {s['n_test_batch']} 样本  "
          f"(已知产品 × 留出批次)")
    print(f"  Setting B/C 测试: {s['n_test_unknown']} 样本  "
          f"(留出产品 × 全部批次)")
    print(f"  排除样本: {s['n_excluded']} (产品样本不足)")
    total = (s['n_train'] + s['n_val'] + s['n_test_batch']
             + s['n_test_unknown'] + s['n_excluded'])
    print(f"  总计: {total}")


def _create_data_split(metadata_csv, cfg, product_col="product_fine"):
    """
    创建确定性的数据划分并保存到 JSON 文件。

    划分逻辑:
      1. 排除样本数过少的产品 (< min_samples_per_product)
      2. 留出 num_open_test_classes 个产品类型 → Setting B/C 测试
      3. 留出约 holdout_batch_ratio 比例的批次 → Setting A 测试
      4. 在 train_batches 内再留出伪 holdout 批次做验证
    """
    rng = np.random.RandomState(cfg.seed)
    df = _load_and_filter(metadata_csv)
    df = df.reset_index(drop=True)  # 统一使用过滤后的位置索引 (0..2803)

    # ── 1. 排除样本过少/批次覆盖不足的产品 ──
    product_counts = df[product_col].value_counts()
    product_batch_coverage = df.groupby(product_col)["batch_name"].nunique()
    all_products = sorted(df[product_col].unique())
    excluded_products = sorted([
        p for p in all_products
        if (product_counts[p] < cfg.min_samples_per_product)
        or (product_batch_coverage[p] < cfg.min_batches_per_product)
    ])
    viable_products = sorted(
        [p for p in all_products if p not in excluded_products]
    )

    # ── 2. 留出产品类型 (Setting B/C) ──
    n_holdout = cfg.num_open_test_classes
    if n_holdout >= len(viable_products):
        raise ValueError(
            f"可用产品 {len(viable_products)} 不够留出 {n_holdout} 类"
        )

    preferred_products = [
        p for p in cfg.preferred_holdout_products
        if p in viable_products
    ]
    if len(preferred_products) >= n_holdout:
        holdout_products = sorted(preferred_products[:n_holdout])
    else:
        candidate_products = [
            p for p in viable_products
            if (product_counts[p] >= cfg.holdout_product_min_samples)
            and (product_batch_coverage[p] >= cfg.holdout_product_min_batches)
        ]
        if len(candidate_products) < n_holdout:
            candidate_products = viable_products.copy()
        candidate_products = sorted(
            candidate_products,
            key=lambda p: (product_counts[p], -product_batch_coverage[p], p),
        )
        holdout_products = sorted(candidate_products[:n_holdout])

    known_products = sorted(
        [p for p in viable_products if p not in holdout_products]
    )

    # ── 3. 留出批次 (Setting A) ──
    all_batches = sorted(df["batch_name"].unique().tolist())
    all_batches = [str(b) for b in all_batches]
    n_holdout_batches = max(1, int(len(all_batches) * cfg.holdout_batch_ratio))

    preferred_batches = [
        b for b in cfg.preferred_holdout_batches if b in all_batches
    ]
    if len(preferred_batches) >= n_holdout_batches:
        holdout_batches = sorted(preferred_batches[:n_holdout_batches])
    else:
        known_df = df[df[product_col].isin(known_products)]
        batch_stats = []
        for b in all_batches:
            b_df = known_df[known_df["batch_name"].astype(str) == b]
            batch_stats.append({
                "batch_name": b,
                "sample_count": int(len(b_df)),
                "class_count": int(b_df[product_col].nunique()),
            })
        candidate_batches = [
            s["batch_name"] for s in batch_stats
            if (s["sample_count"] >= cfg.holdout_batch_min_samples)
            and (s["class_count"] >= cfg.holdout_batch_min_classes)
        ]
        if len(candidate_batches) < n_holdout_batches:
            candidate_batches = [s["batch_name"] for s in sorted(
                batch_stats,
                key=lambda x: (-x["class_count"], -x["sample_count"], x["batch_name"]),
            )]
        else:
            candidate_batches = sorted(candidate_batches, reverse=True)
        holdout_batches = sorted(candidate_batches[:n_holdout_batches])

    train_batches = sorted(
        [b for b in all_batches if b not in holdout_batches]
    )

    # ── 4. 构建索引数组 ──
    df_viable = df[df[product_col].isin(viable_products)]
    df_viable = df_viable.copy()
    df_viable["batch_name"] = df_viable["batch_name"].astype(str)

    train_known_df = df_viable[
        df_viable[product_col].isin(known_products)
        & df_viable["batch_name"].isin(train_batches)
    ]

    n_pseudo_batches = max(1, int(len(train_batches) * cfg.val_ratio))
    n_pseudo_batches = min(max(n_pseudo_batches, 1), max(len(train_batches) - 1, 1))

    preferred_pseudo = [
        b for b in cfg.preferred_holdout_batches if b in train_batches
    ]
    if len(preferred_pseudo) >= n_pseudo_batches:
        pseudo_holdout_batches = sorted(preferred_pseudo[:n_pseudo_batches])
    else:
        batch_stats = []
        for b in train_batches:
            b_df = train_known_df[train_known_df["batch_name"] == b]
            batch_stats.append({
                "batch_name": b,
                "sample_count": int(len(b_df)),
                "class_count": int(b_df[product_col].nunique()),
            })
        min_samples = max(10, int(getattr(cfg, "holdout_batch_min_samples", 60) // 2))
        min_classes = max(3, int(getattr(cfg, "holdout_batch_min_classes", 5) // 2))
        candidate_batches = [
            s["batch_name"] for s in batch_stats
            if (s["sample_count"] >= min_samples)
            and (s["class_count"] >= min_classes)
        ]
        if len(candidate_batches) < n_pseudo_batches:
            candidate_batches = [
                s["batch_name"] for s in sorted(
                    batch_stats,
                    key=lambda x: (-x["class_count"], -x["sample_count"], x["batch_name"]),
                )
            ]
        else:
            candidate_batches = sorted(candidate_batches, reverse=True)
        pseudo_holdout_batches = sorted(candidate_batches[:n_pseudo_batches])

    model_train_batches = sorted(
        [b for b in train_batches if b not in pseudo_holdout_batches]
    )

    train_mask = (
        df_viable[product_col].isin(known_products)
        & df_viable["batch_name"].isin(model_train_batches)
    )
    val_mask = (
        df_viable[product_col].isin(known_products)
        & df_viable["batch_name"].isin(pseudo_holdout_batches)
    )
    train_idx = df_viable[train_mask].index.tolist()
    val_idx = df_viable[val_mask].index.tolist()

    if (len(train_idx) == 0) or (len(val_idx) == 0):
        train_mask = (
            df_viable[product_col].isin(known_products)
            & df_viable["batch_name"].isin(train_batches)
        )
        train_all_idx = df_viable[train_mask].index.tolist()
        train_idx, val_idx = _stratified_split(
            df_viable.loc[train_all_idx], product_col,
            val_ratio=cfg.val_ratio, rng=rng
        )
        pseudo_holdout_batches = []
        model_train_batches = train_batches.copy()

    # Setting A: 已知产品 × 留出批次
    test_batch_mask = (
        df_viable[product_col].isin(known_products)
        & df_viable["batch_name"].isin(holdout_batches)
    )
    test_batch_idx = df_viable[test_batch_mask].index.tolist()

    # Setting B/C: 留出产品 × 全部批次
    test_unknown_mask = df_viable[product_col].isin(holdout_products)
    test_unknown_idx = df_viable[test_unknown_mask].index.tolist()

    train_idx = [int(i) for i in train_idx]
    val_idx = [int(i) for i in val_idx]
    test_batch_idx = [int(i) for i in test_batch_idx]
    test_unknown_idx = [int(i) for i in test_unknown_idx]

    # ── 5. 保存 ──
    split = {
        "known_products": known_products,
        "holdout_products": holdout_products,
        "excluded_products": excluded_products,
        "product_batch_coverage": {
            p: int(product_batch_coverage[p])
            for p in viable_products + excluded_products
            if p in product_batch_coverage
        },
        "train_batches": train_batches,
        "model_train_batches": model_train_batches,
        "model_select_holdout_batches": pseudo_holdout_batches,
        "holdout_batches": holdout_batches,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_batch_idx": test_batch_idx,
        "test_unknown_idx": test_unknown_idx,
        "seed": cfg.seed,
        "stats": {
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test_batch": len(test_batch_idx),
            "n_test_unknown": len(test_unknown_idx),
            "n_excluded": int(
                df[df[product_col].isin(excluded_products)].shape[0]
            ),
        },
    }

    split_path = Path(cfg.prepared_dir) / "split.json"

    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(split_path, "w") as f:
        json.dump(split, f, indent=2, ensure_ascii=False, default=_json_default)

    _print_split_summary(split, df, product_col)
    return split


# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════

def get_fixed_config() -> Config:
    """返回修复后的 Config, 覆盖关键参数。"""
    cfg = Config()
    cfg.prepared_dir = str(PROJECT_ROOT / "new_prepared_data")
    cfg.mz_range = (0.0, 550.0)               # ← 匹配仪器扫描 30-550
    cfg.input_raw_pca_components = 256         # ← 与更大的 m/z 输入匹配
    cfg.preferred_holdout_products = ("HMD", "XCJ")
    cfg.preferred_holdout_batches = ("20250905", "20250912", "20250920")
    cfg.seed = 42
    cfg.rt_bins = 1024
    cfg.rt_range = (0.0, 40.0)
    cfg.save_prepare_plots = False             # 加速准备
    cfg.save_prepare_tables = False
    return cfg


# ═══════════════════════════════════════════════════════════════
#  Step 4: PCA — 仅在训练集上拟合
# ═══════════════════════════════════════════════════════════════

def fit_pca_on_train_only(metadata_csv: str, train_idx: list, cfg: Config):
    """
    仅在训练集样本上拟合 IncrementalPCA。

    参数
    ----
    metadata_csv : 完整 metadata 路径
    train_idx    : split.json 中的 train_idx (原始 CSV 行号)
    cfg          : 修复后的 Config

    返回
    ----
    (pca_model, ref_mz_axis)
    """
    n_comp = int(getattr(cfg, "input_raw_pca_components", 256))
    full_df = pd.read_csv(metadata_csv)

    # 与 GCMSDataset / _create_data_split 一致: 过滤后 reset_index
    filtered_df = full_df[
        (full_df["product_fine"] != "BLANK") & (~full_df["is_special"])
    ].reset_index(drop=True)

    # train_idx 现在是过滤后 reset 的位置索引 (0..2803)
    train_set = set(train_idx)
    train_samples = filtered_df.iloc[list(train_set)]
    print(f"  PCA 拟合样本数: {len(train_samples)} / {len(filtered_df)} (过滤后总样本)")

    if len(train_samples) == 0:
        raise RuntimeError("训练集为空, 无法拟合 PCA")

    ipca: IncrementalPCA | None = None
    ref_mz_axis: np.ndarray | None = None
    n_fitted = 0
    skipped = 0

    for _, row in tqdm(
        train_samples.iterrows(), total=len(train_samples), desc="  PCA拟合"
    ):
        try:
            _rts, mzs, raw_mat = _read_raw_matrix_no_bins(row["d_path"], cfg)
        except Exception:
            skipped += 1
            continue

        if ref_mz_axis is None:
            ref_mz_axis = mzs.astype(np.float32)

        aligned = _align_mz_axis_linear(raw_mat, mzs, ref_mz_axis)
        tensor = build_dual_channel(aligned)              # (2, H, W)

        if ipca is None:
            width = int(tensor.shape[2])
            n_comp_use = min(n_comp, width - 1)
            if n_comp_use < 2:
                raise RuntimeError(f"m/z 维度过小: width={width}")
            print(f"  PCA 输入 m/z 维度: {width}, 目标组件数: {n_comp_use}")
            ipca = IncrementalPCA(n_components=n_comp_use)

        for c in range(tensor.shape[0]):
            ipca.partial_fit(tensor[c])
        n_fitted += 1

    if ipca is None:
        raise RuntimeError("PCA 拟合失败: 无有效训练样本")

    n_comp_real = int(getattr(ipca, "n_components_", n_comp))
    print(f"  PCA 拟合完成: {n_fitted} 样本, 跳过 {skipped}")
    print(f"  m/z 输入维度: {len(ref_mz_axis)}, PCA 输出维度: {n_comp_real}")
    return ipca, ref_mz_axis


# ═══════════════════════════════════════════════════════════════
#  Step 5: 转换全部样本
# ═══════════════════════════════════════════════════════════════

def convert_all_samples(metadata_csv: str, pca_model, ref_mz_axis: np.ndarray, cfg: Config):
    """
    用训练集拟合的 PCA 转换全部样本 (含 BLANK / is_special / 排除产品)。

    返回 (converted, failed)
    """
    full_df = pd.read_csv(metadata_csv)
    rt_target_bins = int(getattr(cfg, "rt_bins", 1024))
    n_comp = int(getattr(pca_model, "n_components_", cfg.input_raw_pca_components))

    converted = 0
    failed = 0

    for _, row in tqdm(
        full_df.iterrows(), total=len(full_df), desc="  转换样本"
    ):
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
            print(f"\n  ✗ {row['d_name']}: {e}")
            empty = np.zeros((cfg.in_channels, rt_target_bins, n_comp), dtype=np.float32)
            np.savez_compressed(npz_path, tensor=empty, grid=np.zeros(1, dtype=np.float32))
            failed += 1

    return converted, failed


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    cfg = get_fixed_config()
    out_dir = Path(cfg.prepared_dir)
    tensor_dir = out_dir / "tensors"

    # 如果目标目录已存在, 先清空
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 扫描数据集 ──
    print("=" * 60)
    print("Step 1/6: 扫描数据集 ...")
    metadata = scan_dataset(cfg.dataset_root)
    print(f"  发现 {len(metadata)} 个样本, {metadata['batch_name'].nunique()} 个批次")
    print(f"  产品数: {metadata['product_fine'].nunique()}")

    # ── Step 2: 保存 metadata.csv ──
    print("\nStep 2/6: 保存 metadata.csv ...")
    metadata = metadata.copy()
    metadata["tensor_path"] = _build_tensor_paths(metadata, tensor_dir, cfg)
    meta_path = out_dir / "metadata.csv"
    metadata.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {meta_path}")

    # ── Step 3: 创建数据划分 ──
    print("\nStep 3/6: 创建数据划分 (split.json) ...")
    product_col = "product_fine" if cfg.product_granularity == "fine" else "product_coarse"
    split = _create_data_split(str(meta_path), cfg, product_col=product_col)

    # ── Step 4: 仅训练集拟合 PCA ──
    print("\nStep 4/6: 仅用训练集拟合 PCA ...")
    pca_model, ref_mz_axis = fit_pca_on_train_only(
        str(meta_path), split["train_idx"], cfg
    )

    # ── Step 5: 转换全部样本 ──
    print("\nStep 5/6: 转换全部样本 ...")
    converted, failed = convert_all_samples(
        str(meta_path), pca_model, ref_mz_axis, cfg
    )

    # ── Step 6: 保存 grid_info.json ──
    print("\nStep 6/6: 保存 grid_info.json ...")
    n_comp = int(getattr(pca_model, "n_components_", cfg.input_raw_pca_components))
    info = {
        "rt_bins": int(cfg.rt_bins),
        "mz_bins": int(n_comp),
        "rt_range": list(cfg.rt_range) if cfg.rt_range is not None else None,
        "mz_range": list(cfg.mz_range),
        "success": converted,
        "fail": failed,
        "prepare_mode": "raw_rt_mz_direct_pca_train_only",
        "input_pca_fit_source": "training_samples_only",
        "input_pca_precomputed": True,
        "input_pca_applied": True,
        "input_pca_reason": "ok" if failed == 0 else "partial_failed",
        "input_pca_components": int(n_comp),
        "input_pca_rt_bins": int(cfg.rt_bins),
        "input_pca_ref_mz_axis_len": int(len(ref_mz_axis)),
        "input_pca_ref_mz_min": float(ref_mz_axis.min()),
        "input_pca_ref_mz_max": float(ref_mz_axis.max()),
        "preferred_holdout_products": list(cfg.preferred_holdout_products),
        "preferred_holdout_batches": list(cfg.preferred_holdout_batches),
        "seed": cfg.seed,
    }
    (out_dir / "grid_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False)
    )

    # ── 完成 ──
    print(f"\n{'=' * 60}")
    print(f"  数据准备完成!")
    print(f"  成功: {converted}, 失败: {failed}")
    print(f"  输出目录: {out_dir}")
    print(f"  Tensor 形状: (2, {cfg.rt_bins}, {n_comp})")
    s = split["stats"]
    print(f"  样本划分: train={s['n_train']}, val={s['n_val']}, "
          f"test_batch={s['n_test_batch']}, test_unknown={s['n_test_unknown']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
