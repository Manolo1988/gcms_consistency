"""
PyTorch Dataset + 数据增强 + 批次/开集/少样本切分。
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


class GCMSAugmentation:
    """RT × m/z 二维张量数据增强。"""

    def __init__(self, cfg):
        self.intensity_lo, self.intensity_hi = cfg.aug_intensity_scale
        self.noise_std = cfg.aug_noise_std
        self.mask_ratio = cfg.aug_mask_ratio
        self.rt_shift = cfg.aug_rt_shift_max
        self.mz_shift = cfg.aug_mz_shift_max

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

        return x


class GCMSDataset(Dataset):
    """
    加载预处理好的 RT × m/z 张量。

    metadata_csv 必须包含列:
        tensor_path, product_fine, product_coarse, batch_idx, is_special
    """

    def __init__(self, metadata_csv, product_col="product_fine",
                 augmentation=None, exclude_blanks=True,
                 exclude_special=True, indices=None):
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row["tensor_path"])
        x = data["tensor"].astype(np.float32)

        if self.aug is not None:
            x = self.aug(x)

        return {
            "input": torch.from_numpy(x),
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
                          exclude_blanks=True, exclude_special=True, seed=42):
    """
    从未知类样本中划分 N-shot 注册集和测试集。
    返回 dict[int, dict]:  n_shot -> {"ref_idx": [...], "test_idx": [...]}
    """
    rng = np.random.RandomState(seed)
    df = _load_and_filter(metadata_csv, exclude_blanks, exclude_special)
    unknown_df = df.iloc[unknown_idx]

    # 建立原始行号 → unknown_ds 内部位置的映射，
    # 因为 GCMSDataset(indices=unknown_idx) 会 reset_index 到 0..N-1
    orig_to_local = {orig: local for local, orig in enumerate(unknown_idx)}

    results = {}
    for n_shot in n_shot_values:
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
    return results