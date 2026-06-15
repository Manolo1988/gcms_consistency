"""
PyTorch Dataset + 数据增强 + 批次/开集/少样本切分。
"""
import json
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


class GCMSAugmentation:
    """RT × m/z 二维张量数据增强 (含 GC-MS 专用变换)。"""

    def __init__(self, cfg):
        self.intensity_lo, self.intensity_hi = cfg.aug_intensity_scale
        self.noise_std = cfg.aug_noise_std
        self.mask_ratio = cfg.aug_mask_ratio
        self.rt_shift = cfg.aug_rt_shift_max
        self.mz_shift = cfg.aug_mz_shift_max
        # GC-MS 专用
        self.baseline_amp = cfg.aug_baseline_wander_amp
        self.baseline_freq = cfg.aug_baseline_wander_freq
        self.peak_broaden_sigma = cfg.aug_peak_broaden_sigma
        self.peak_broaden_prob = float(
            min(max(getattr(cfg, "aug_peak_broaden_prob", 0.1), 0.0), 1.0)
        )
        self.rt_warp_strength = cfg.aug_rt_warp_strength
        self.rt_warp_prob = float(
            min(max(getattr(cfg, "aug_rt_warp_prob", 0.2), 0.0), 1.0)
        )
        self.mz_channel_drop = cfg.aug_mz_channel_drop
        self.tic_jitter = cfg.aug_tic_jitter

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (C, H, W) float32"""
        x = x.copy()
        C, H, W = x.shape

        # 1. 随机强度缩放
        scale = np.random.uniform(self.intensity_lo, self.intensity_hi)
        x *= scale

        # 2. 高斯噪声
        if self.noise_std > 0:
            x += np.random.randn(*x.shape).astype(np.float32) * self.noise_std

        # 3. 随机 patch mask
        if self.mask_ratio > 0:
            n_patches = int(H * W * self.mask_ratio / (16 * 16))
            for _ in range(max(1, n_patches)):
                rh = np.random.randint(4, 17)
                rw = np.random.randint(4, 17)
                y0 = np.random.randint(0, max(1, H - rh))
                x0 = np.random.randint(0, max(1, W - rw))
                x[:, y0:y0+rh, x0:x0+rw] = 0.0

        # 4. 随机 RT/m/z 平移
        if self.rt_shift > 0 or self.mz_shift > 0:
            dy = np.random.randint(-self.rt_shift, self.rt_shift + 1)
            dx = np.random.randint(-self.mz_shift, self.mz_shift + 1)
            x = np.roll(x, dy, axis=1)
            x = np.roll(x, dx, axis=2)

        # 5. 基线漂移 (低频正弦叠加, 模拟色谱基线漂移)
        if self.baseline_amp > 0 and np.random.rand() < 0.5:
            n_freq = np.random.randint(1, self.baseline_freq + 1)
            t = np.linspace(0, 2 * np.pi * n_freq, H, dtype=np.float32)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0, self.baseline_amp)
            baseline = amp * np.sin(t + phase)  # (H,)
            x += baseline[np.newaxis, :, np.newaxis]

        # 6. 峰展宽/收窄 (沿 RT 轴高斯卷积, 模拟色谱柱效率变化)
        if self.peak_broaden_sigma > 0 and np.random.rand() < self.peak_broaden_prob:
            from scipy.ndimage import gaussian_filter1d
            sigma = np.random.uniform(0.5, self.peak_broaden_sigma)
            for c in range(C):
                x[c] = gaussian_filter1d(x[c], sigma=sigma, axis=0)

        # 7. RT 非线性扭曲 (保留时间漂移的真实模拟)
        if self.rt_warp_strength > 0 and np.random.rand() < self.rt_warp_prob:
            # 生成随机控制点构造单调映射
            n_ctrl = np.random.randint(3, 6)
            ctrl_x = np.linspace(0, 1, n_ctrl)
            ctrl_y = ctrl_x + np.random.randn(n_ctrl).astype(np.float32) \
                     * self.rt_warp_strength
            ctrl_y = np.sort(ctrl_y)  # 保持单调
            ctrl_y = (ctrl_y - ctrl_y[0]) / (ctrl_y[-1] - ctrl_y[0] + 1e-8)
            # 插值到完整 RT 轴
            warp_map = np.interp(
                np.linspace(0, 1, H), ctrl_x, ctrl_y
            )
            idx_map = np.clip(warp_map * (H - 1), 0, H - 1).astype(np.int64)
            x = x[:, idx_map, :]

        # 8. m/z 通道随机丢弃 (模拟检测器噪声/离子抑制)
        if self.mz_channel_drop > 0 and np.random.rand() < 0.3:
            n_drop = max(1, int(W * self.mz_channel_drop))
            drop_idx = np.random.choice(W, n_drop, replace=False)
            x[:, :, drop_idx] = 0.0

        # 9. TIC 归一化抖动 (模拟进样量波动)
        if self.tic_jitter > 0 and np.random.rand() < 0.5:
            jitter = 1.0 + np.random.uniform(-self.tic_jitter,
                                              self.tic_jitter)
            x *= jitter

        return x


class GCMSDataset(Dataset):
    """
    加载预处理好的 RT × m/z 张量。

    metadata_csv 必须包含列:
        tensor_path, product_fine, product_coarse, batch_idx, is_special
    """

    def __init__(self, metadata_csv, product_col="product_fine",
                 augmentation=None, exclude_blanks=True,
                 exclude_special=True, indices=None,
                 input_transform=None, cfg=None):
        df = pd.read_csv(metadata_csv)

        if exclude_blanks:
            df = df[df["product_fine"] != "BLANK"]
        if exclude_special:
            df = df[~df["is_special"]]

        df = df.reset_index(drop=True)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.df = df
        self.product_col = product_col
        self.aug = augmentation
        self.input_transform = input_transform
        self.cache_enabled = bool(
            getattr(cfg, "dataset_cache_in_memory", False)
        ) if cfg is not None else False
        self.cache_max_items = int(
            getattr(cfg, "dataset_cache_max_items", 4096)
        ) if cfg is not None else 4096
        self.cache_max_items = max(self.cache_max_items, 0)
        self._tensor_cache = OrderedDict()

        # 标签编码
        self.product_enc = LabelEncoder()
        self.df["product_label"] = self.product_enc.fit_transform(
            self.df[product_col]
        )
        self.batch_enc = LabelEncoder()
        self.df["batch_label"] = self.batch_enc.fit_transform(
            self.df["batch_idx"]
        )

        self.num_products = len(self.product_enc.classes_)
        self.num_batches = len(self.batch_enc.classes_)

    def __len__(self):
        return len(self.df)

    def _load_tensor(self, tensor_path):
        if not self.cache_enabled or self.cache_max_items <= 0:
            with np.load(tensor_path) as data:
                return data["tensor"].astype(np.float32)

        cached = self._tensor_cache.get(tensor_path)
        if cached is not None:
            self._tensor_cache.move_to_end(tensor_path)
            return cached.copy()

        with np.load(tensor_path) as data:
            x = data["tensor"].astype(np.float32)
        if len(self._tensor_cache) >= self.cache_max_items:
            self._tensor_cache.popitem(last=False)
        self._tensor_cache[tensor_path] = x
        return x.copy()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self._load_tensor(row["tensor_path"])

        if self.input_transform is not None:
            x = self.input_transform(x)

        if self.aug is not None:
            x = self.aug(x)

        tic = x[0].sum(axis=1).astype(np.float32)
        tic_min = float(tic.min()) if tic.size else 0.0
        tic_max = float(tic.max()) if tic.size else 0.0
        tic = (tic - tic_min) / (tic_max - tic_min + 1e-8)

        return {
            "input": torch.from_numpy(x),
            "tic": torch.from_numpy(tic),
            "product": torch.tensor(row["product_label"], dtype=torch.long),
            "batch": torch.tensor(row["batch_label"], dtype=torch.long),
            "sample_id": row["sample_id"],
        }

    def get_product_names(self):
        return list(self.product_enc.classes_)

    def get_batch_names(self):
        return list(self.batch_enc.classes_)

    def get_label_name_map(self):
        """返回 {label_idx: product_name} 映射。"""
        return {i: c for i, c in enumerate(self.product_enc.classes_)}


# ─────────────────────────────────────────────────────────
#  切分策略
# ─────────────────────────────────────────────────────────
def _load_and_filter(metadata_csv, exclude_blanks=True, exclude_special=True):
    df = pd.read_csv(metadata_csv)
    if exclude_blanks:
        df = df[df["product_fine"] != "BLANK"]
    if exclude_special:
        df = df[~df["is_special"]]
    return df.reset_index(drop=True)


def leave_one_batch_out_splits(metadata_csv, product_col="product_fine",
                               exclude_blanks=True, exclude_special=True):
    """
    闭集: leave-one-batch-out 切分。
    返回 list of (train_indices, val_indices, test_batch_name)。
    """
    df = _load_and_filter(metadata_csv, exclude_blanks, exclude_special)

    batches = sorted(df["batch_idx"].unique())
    splits = []
    for test_batch in batches:
        test_idx = df[df["batch_idx"] == test_batch].index.tolist()
        train_idx = df[df["batch_idx"] != test_batch].index.tolist()
        batch_name = df[df["batch_idx"] == test_batch]["batch_name"].iloc[0]
        splits.append((train_idx, test_idx, batch_name))
    return splits


def unified_splits(metadata_csv, product_col="product_fine",
                   num_open_classes=3,
                   exclude_blanks=True, exclude_special=True, seed=42):
    """
    统一切分: 留出若干类 + leave-one-batch-out。
    训练一个模型, 三个 Setting 共用:
      Setting A (闭集跨批次): 在已知类上评估跨批次性能
      Setting B (开放集): 已知类 vs 未知类判别
      Setting C (少样本): N-shot 注册未知类

    返回 dict:
        known_classes:   list[str]
        unknown_classes: list[str]
        unknown_idx:     list[int] — 未知类样本在 df 中的索引
        folds:           list[dict] — leave-one-batch-out 各 fold
    """
    rng = np.random.RandomState(seed)
    df = _load_and_filter(metadata_csv, exclude_blanks, exclude_special)

    all_classes = sorted(df[product_col].unique())
    n_total = len(all_classes)

    if num_open_classes >= n_total:
        raise ValueError(f"类别不足: {n_total} 类, 需留出 {num_open_classes} 类")

    perm = rng.permutation(n_total)
    unknown_classes = [all_classes[i] for i in perm[:num_open_classes]]
    known_classes = [all_classes[i] for i in perm[num_open_classes:]]

    # 已知类样本
    known_mask = df[product_col].isin(known_classes)
    known_df = df[known_mask]

    # 未知类样本索引
    unknown_idx = df[~known_mask].index.tolist()

    # 已知类上 leave-one-batch-out
    batches = sorted(known_df["batch_idx"].unique())
    folds = []
    for i, test_batch in enumerate(batches):
        val_idx = known_df[known_df["batch_idx"] == test_batch].index.tolist()
        train_idx = known_df[known_df["batch_idx"] != test_batch].index.tolist()
        batch_name = known_df[
            known_df["batch_idx"] == test_batch
        ]["batch_name"].iloc[0]
        folds.append({
            "fold_idx": i,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_batch": batch_name,
        })

    return {
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "unknown_idx": unknown_idx,
        "folds": folds,
    }


def few_shot_from_unknown(unknown_idx, metadata_csv, product_col="product_fine",
                          n_shot_values=(1, 3, 5, 10),
                          exclude_blanks=True, exclude_special=True, seed=42,
                          repeats=1, seed_start=42):
    """
    从未知类样本中划分 N-shot 注册集和测试集。

    支持多次重复抽样:
      - repeats=1: 单次抽样 (兼容旧版), 返回 {n_shot: {"ref_idx": [...], "test_idx": [...]}}
      - repeats>1: 多次抽样, 返回 {n_shot: [{"ref_idx": [...], "test_idx": [...]}, ...]}

    Args:
        unknown_idx:   未知类在 metadata_csv 中的索引列表
        metadata_csv:  metadata CSV 路径
        product_col:   产品列名
        n_shot_values: N-shot 值元组
        seed:          基础随机种子 (repeats=1 时使用)
        repeats:       每个 N 重复抽样次数
        seed_start:    重复抽样起始种子 (repeat_idx 会加到此种子)
    """
    df = _load_and_filter(metadata_csv, exclude_blanks, exclude_special)
    unknown_df = df.iloc[unknown_idx]

    # 建立原始行号 → unknown_ds 内部位置的映射
    orig_to_local = {orig: local for local, orig in enumerate(unknown_idx)}

    results = {}
    for n_shot in n_shot_values:
        if repeats <= 1:
            rng = np.random.RandomState(seed)
            ref_idx_list = []
            test_idx_list = []
            for cls in unknown_df[product_col].unique():
                cls_indices = unknown_df[
                    unknown_df[product_col] == cls
                ].index.tolist()
                if len(cls_indices) <= n_shot:
                    ref_idx_list.extend(
                        [orig_to_local[i] for i in cls_indices])
                    continue
                perm = rng.permutation(len(cls_indices))
                ref_idx_list.extend(
                    [orig_to_local[cls_indices[j]] for j in perm[:n_shot]])
                test_idx_list.extend(
                    [orig_to_local[cls_indices[j]] for j in perm[n_shot:]])
            results[n_shot] = {"ref_idx": ref_idx_list, "test_idx": test_idx_list}
        else:
            # Multiple repeats
            repeats_list = []
            for r in range(repeats):
                rng = np.random.RandomState(seed_start + r)
                ref_idx_list = []
                test_idx_list = []
                for cls in unknown_df[product_col].unique():
                    cls_indices = unknown_df[
                        unknown_df[product_col] == cls
                    ].index.tolist()
                    if len(cls_indices) <= n_shot:
                        ref_idx_list.extend(
                            [orig_to_local[i] for i in cls_indices])
                        continue
                    perm = rng.permutation(len(cls_indices))
                    ref_idx_list.extend(
                        [orig_to_local[cls_indices[j]] for j in perm[:n_shot]])
                    test_idx_list.extend(
                        [orig_to_local[cls_indices[j]] for j in perm[n_shot:]])
                repeats_list.append({
                    "ref_idx": ref_idx_list,
                    "test_idx": test_idx_list,
                    "repeat_idx": r,
                })
            results[n_shot] = repeats_list
    return results


# ─────────────────────────────────────────────────────────
#  固定切分: 单模型训练 + 三 Setting 测试
# ─────────────────────────────────────────────────────────
def create_data_split(metadata_csv, cfg, product_col="product_fine"):
    """
    创建确定性的数据划分并保存到 JSON 文件。

    划分逻辑:
      1. 排除样本数过少的产品 (< min_samples_per_product)
      2. 留出 num_open_test_classes 个产品类型 → Setting B/C 测试
      3. 留出约 holdout_batch_ratio 比例的批次 → Setting A 测试
    4. 在 train_batches 内再留出伪 holdout 批次做验证 (训练监控用)

    结果:
      train_idx:        已知产品 × 训练批次 (实际训练)
    val_idx:          已知产品 × 伪 holdout 批次 (批次外推验证)
      test_batch_idx:   已知产品 × 留出批次 → Setting A
      test_unknown_idx: 留出产品 × 全部批次 → Setting B/C
    """
    split_seed = int(getattr(cfg, "split_seed", cfg.seed))
    rng = np.random.RandomState(split_seed)
    df = _load_and_filter(metadata_csv)

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
    # 目标: 既保证留出类可做 N-shot(1/3/5/10), 又具备跨批次覆盖。
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
        # 先筛掉无法支持少样本实验的类别
        candidate_products = [
            p for p in viable_products
            if (product_counts[p] >= cfg.holdout_product_min_samples)
            and (product_batch_coverage[p] >= cfg.holdout_product_min_batches)
        ]
        if len(candidate_products) < n_holdout:
            candidate_products = viable_products.copy()

        candidate_products = sorted(
            candidate_products,
            key=lambda p: (
                product_counts[p],
                -product_batch_coverage[p],
                p,
            ),
        )
        if split_seed == 42:
            holdout_products = sorted(candidate_products[:n_holdout])
        else:
            pool = candidate_products[:max(n_holdout, min(len(candidate_products), n_holdout * 4))]
            holdout_products = sorted(rng.choice(pool, size=n_holdout, replace=False).tolist())

    known_products = sorted(
        [p for p in viable_products if p not in holdout_products]
    )

    # ── 3. 留出批次 (Setting A) ──
    all_batches = sorted(df["batch_name"].unique().tolist())
    all_batches = [str(b) for b in all_batches]  # 确保是 str
    n_holdout_batches = max(1, int(len(all_batches) * cfg.holdout_batch_ratio))

    preferred_batches = [
        b for b in cfg.preferred_holdout_batches
        if b in all_batches
    ]
    if len(preferred_batches) >= n_holdout_batches:
        holdout_batches = sorted(preferred_batches[:n_holdout_batches])
    else:
        # 仅在已知产品子集上计算批次难度与代表性
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
                key=lambda x: (
                    -x["class_count"],
                    -x["sample_count"],
                    x["batch_name"],
                ),
            )]
        else:
            # 倾向选择时间靠后的批次，验证时序外推鲁棒性
            candidate_batches = sorted(candidate_batches, reverse=True)

        if split_seed == 42:
            holdout_batches = sorted(candidate_batches[:n_holdout_batches])
        else:
            pool = candidate_batches[:max(n_holdout_batches, min(len(candidate_batches), n_holdout_batches * 4))]
            holdout_batches = sorted(rng.choice(pool, size=n_holdout_batches, replace=False).tolist())

    train_batches = sorted(
        [b for b in all_batches if b not in holdout_batches]
    )

    # ── 4. 构建索引数组 ──
    # 排除过少产品后的 DataFrame
    df_viable = df[df[product_col].isin(viable_products)]
    # 确保 batch_name 比较用 str
    df_viable = df_viable.copy()
    df_viable["batch_name"] = df_viable["batch_name"].astype(str)

    # 在 train_batches 内再留出伪 holdout 批次作为验证集，
    # 让 early-stop/model selection 更贴近 Setting A 的跨批次外推场景。
    train_known_df = df_viable[
        df_viable[product_col].isin(known_products)
        & df_viable["batch_name"].isin(train_batches)
    ]

    n_pseudo_batches = max(1, int(len(train_batches) * cfg.val_ratio))
    n_pseudo_batches = min(max(n_pseudo_batches, 1), max(len(train_batches) - 1, 1))

    preferred_pseudo = [
        b for b in cfg.preferred_holdout_batches
        if b in train_batches
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
                    key=lambda x: (
                        -x["class_count"],
                        -x["sample_count"],
                        x["batch_name"],
                    ),
                )
            ]
        else:
            # 倾向选择时间靠后的批次做伪外推验证
            candidate_batches = sorted(candidate_batches, reverse=True)

        if split_seed == 42:
            pseudo_holdout_batches = sorted(candidate_batches[:n_pseudo_batches])
        else:
            pool = candidate_batches[:max(n_pseudo_batches, min(len(candidate_batches), n_pseudo_batches * 4))]
            pseudo_holdout_batches = sorted(rng.choice(pool, size=n_pseudo_batches, replace=False).tolist())

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

    # 兜底：若批次法未能形成有效验证集，则回退到分层抽样。
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

    # Setting A 测试: 已知产品 × 留出批次
    test_batch_mask = (
        df_viable[product_col].isin(known_products)
        & df_viable["batch_name"].isin(holdout_batches)
    )
    test_batch_idx = df_viable[test_batch_mask].index.tolist()

    # Setting B/C 测试: 留出产品 × 全部批次
    test_unknown_mask = df_viable[product_col].isin(holdout_products)
    test_unknown_idx = df_viable[test_unknown_mask].index.tolist()

    # 确保所有索引为 Python int (JSON 序列化)
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
        "split_seed": split_seed,
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
    split_path.parent.mkdir(parents=True, exist_ok=True)

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


def _stratified_split(df_subset, product_col, val_ratio, rng):
    """分层抽样: 按产品类别分层划出验证集。"""
    train_idx = []
    val_idx = []
    for cls in df_subset[product_col].unique():
        cls_idx = df_subset[
            df_subset[product_col] == cls
        ].index.tolist()
        n_val = max(1, int(len(cls_idx) * val_ratio))
        perm = rng.permutation(len(cls_idx))
        val_idx.extend([cls_idx[i] for i in perm[:n_val]])
        train_idx.extend([cls_idx[i] for i in perm[n_val:]])
    return train_idx, val_idx


def _print_split_summary(split, df, product_col):
    """打印划分摘要，包含各数据集的产品和批次数量。"""
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
    print(f"  实训批次 ({len(split.get('model_train_batches', split['train_batches']))}): "
          f"{split.get('model_train_batches', split['train_batches'])}")
    print(f"  留出批次 ({len(split['holdout_batches'])}): "
          f"{split['holdout_batches']}  → Setting A")

    # 统计各数据集的产品和批次数量
    def _count_products_and_batches(indices, name):
        subset = df.iloc[indices]
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


def load_data_split(cfg):
    """加载已保存的数据划分。"""
    split_path = Path(cfg.prepared_dir) / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"未找到 {split_path}, 请先运行 python main.py prepare"
        )
    with open(split_path) as f:
        return json.load(f)
