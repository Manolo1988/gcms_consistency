"""
PyTorch Dataset + 数据增强 + 批次感知切分。
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


def leave_one_batch_out_splits(metadata_csv, product_col="product_fine",
                               exclude_blanks=True, exclude_special=True):
    """
    生成 leave-one-batch-out 切分。
    返回 list of (train_indices, val_indices, test_batch_name)。
    """
    df = pd.read_csv(metadata_csv)
    if exclude_blanks:
        df = df[df["product_fine"] != "BLANK"]
    if exclude_special:
        df = df[~df["is_special"]]
    df = df.reset_index(drop=True)

    batches = sorted(df["batch_idx"].unique())
    splits = []
    for test_batch in batches:
        test_idx = df[df["batch_idx"] == test_batch].index.tolist()
        train_idx = df[df["batch_idx"] != test_batch].index.tolist()
        batch_name = df[df["batch_idx"] == test_batch]["batch_name"].iloc[0]
        splits.append((train_idx, test_idx, batch_name))
    return splits