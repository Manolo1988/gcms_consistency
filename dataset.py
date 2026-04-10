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


def open_set_splits(metadata_csv, product_col="product_fine",
                    num_test_classes=2, num_val_classes=1,
                    exclude_blanks=True, exclude_special=True, seed=42):
    """
    开集: 留出部分类不参与训练。
    返回 dict:
        train_classes: list[str]  — 训练用类名
        val_classes:   list[str]  — 验证用类名 (未知类)
        test_classes:  list[str]  — 测试用类名 (未知类)
        train_idx:     list[int]  — 训练样本索引
        val_known_idx: list[int]  — 已知类验证样本 (从训练类中按批次留出)
        val_open_idx:  list[int]  — 未知类验证样本
        test_idx:      list[int]  — 测试样本 (未知类)
    """
    rng = np.random.RandomState(seed)
    df = _load_and_filter(metadata_csv, exclude_blanks, exclude_special)

    all_classes = sorted(df[product_col].unique())
    n_total = len(all_classes)
    n_hold = num_test_classes + num_val_classes

    if n_hold >= n_total:
        raise ValueError(
            f"类别数不足: 共 {n_total} 类, 需留出 {n_hold} 类"
        )

    perm = rng.permutation(n_total)
    test_classes = [all_classes[i] for i in perm[:num_test_classes]]
    val_classes = [all_classes[i] for i in perm[num_test_classes:n_hold]]
    train_classes = [all_classes[i] for i in perm[n_hold:]]

    # 训练类样本
    train_mask = df[product_col].isin(train_classes)
    train_df = df[train_mask]

    # 从训练类中按批次留一个做 val_known
    train_batches = sorted(train_df["batch_idx"].unique())
    if len(train_batches) > 1:
        val_batch = rng.choice(train_batches)
        val_known_idx = train_df[train_df["batch_idx"] == val_batch].index.tolist()
        train_idx = train_df[train_df["batch_idx"] != val_batch].index.tolist()
    else:
        # 只有一个批次: 随机 80/20 split
        n = len(train_df)
        perm_t = rng.permutation(n)
        split = int(0.8 * n)
        train_idx = train_df.index[perm_t[:split]].tolist()
        val_known_idx = train_df.index[perm_t[split:]].tolist()

    # 未知类样本
    val_open_idx = df[df[product_col].isin(val_classes)].index.tolist()
    test_idx = df[df[product_col].isin(test_classes)].index.tolist()

    return {
        "train_classes": train_classes,
        "val_classes": val_classes,
        "test_classes": test_classes,
        "train_idx": train_idx,
        "val_known_idx": val_known_idx,
        "val_open_idx": val_open_idx,
        "test_idx": test_idx,
    }


def few_shot_splits(metadata_csv, product_col="product_fine",
                    n_shot_values=(1, 3, 5, 10),
                    exclude_blanks=True, exclude_special=True, seed=42):
    """
    少样本: 全类训练，测试时模拟 N-shot 注册。
    对每个 N-shot，随机选 N 个样本作注册参考，其余作测试。

    返回 dict:
        base_splits: list — leave-one-batch-out splits (用于训练)
        fewshot_evals: dict[int, list] — N -> list of (ref_idx, test_idx) per batch
    """
    rng = np.random.RandomState(seed)
    df = _load_and_filter(metadata_csv, exclude_blanks, exclude_special)

    # 训练仍按 leave-one-batch-out
    base_splits = leave_one_batch_out_splits(
        metadata_csv, product_col, exclude_blanks, exclude_special
    )

    # 对每个测试批次，模拟 N-shot 注册
    fewshot_evals = {}
    for n_shot in n_shot_values:
        evals = []
        for train_idx, test_idx, bname in base_splits:
            test_df = df.iloc[test_idx]
            ref_idx_list = []
            remaining_idx_list = []

            for cls in test_df[product_col].unique():
                cls_indices = test_df[test_df[product_col] == cls].index.tolist()
                if len(cls_indices) <= n_shot:
                    ref_idx_list.extend(cls_indices)
                    continue
                perm = rng.permutation(len(cls_indices))
                ref_idx_list.extend([cls_indices[i] for i in perm[:n_shot]])
                remaining_idx_list.extend([cls_indices[i] for i in perm[n_shot:]])

            evals.append({
                "batch_name": bname,
                "ref_idx": ref_idx_list,
                "test_idx": remaining_idx_list,
            })
        fewshot_evals[n_shot] = evals

    return {
        "base_splits": base_splits,
        "fewshot_evals": fewshot_evals,
    }