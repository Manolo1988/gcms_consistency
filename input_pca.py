"""Input PCA utilities for deep_consistency pipeline.

Apply PCA along the m/z axis for each sample tensor channel:
  (C, H, W=256) -> (C, H, n_components)
This keeps RT structure while reducing m/z dimensionality.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from sklearn.decomposition import IncrementalPCA

from dataset import _load_and_filter


class RtAxisPcaTransform:
    """Callable transform: apply a fitted RT-axis PCA to input tensors."""

    def __init__(self, pca_model: IncrementalPCA):
        self.pca = pca_model

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError(f"期望输入形状 (C,H,W), 实际为 {x.shape}")
        out = []
        for c in range(x.shape[0]):
            out.append(self.pca.transform(x[c]))
        return np.stack(out, axis=0).astype(np.float32)


def _iter_tensors(metadata_csv: str, indices: Iterable[int]):
    df = _load_and_filter(metadata_csv)
    if indices is not None:
        df = df.iloc[list(indices)]
    for p in df["tensor_path"].tolist():
        arr = np.load(p)["tensor"].astype(np.float32)
        yield arr


def fit_rt_axis_pca(
    metadata_csv: str,
    indices: Iterable[int],
    n_components: int,
) -> Tuple[IncrementalPCA, dict]:
    """Fit IncrementalPCA using training tensors along m/z axis."""
    n_components = int(n_components)
    if n_components <= 0:
        raise ValueError("n_components 必须为正整数")

    ipca = IncrementalPCA(n_components=n_components)

    n_tensors = 0
    n_rows = 0
    in_width = None

    for tensor in _iter_tensors(metadata_csv, indices):
        n_tensors += 1
        if in_width is None:
            in_width = int(tensor.shape[2])
        for c in range(tensor.shape[0]):
            # 每次用一个通道的 (H, W) 矩阵做 partial_fit。
            ipca.partial_fit(tensor[c])
            n_rows += int(tensor.shape[1])

    if n_tensors == 0:
        raise RuntimeError("PCA 拟合失败: 训练样本为空")

    meta = {
        "n_tensors": int(n_tensors),
        "n_rows": int(n_rows),
        "input_width": int(in_width or 0),
        "n_components": int(n_components),
    }
    return ipca, meta


def save_rt_axis_pca(pca_model: IncrementalPCA, save_path: Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(pca_model, f)


def load_rt_axis_pca(load_path: Path) -> IncrementalPCA:
    with open(load_path, "rb") as f:
        return pickle.load(f)


def _cache_key(metadata_csv: str, indices: Iterable[int], n_components: int) -> str:
    metadata_path = Path(metadata_csv).resolve()
    st = metadata_path.stat()
    idx = [int(i) for i in (indices or [])]
    payload = {
        "metadata_csv": str(metadata_path),
        "metadata_mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "metadata_size": int(st.st_size),
        "n_components": int(n_components),
        "n_indices": int(len(idx)),
        "indices_sha1": hashlib.sha1(np.asarray(idx, dtype=np.int32).tobytes()).hexdigest(),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def load_or_fit_rt_axis_pca(
    metadata_csv: str,
    indices: Iterable[int],
    n_components: int,
    cache_root: Path,
    timeout_seconds: int = 7200,
) -> Tuple[IncrementalPCA, dict, bool, Path]:
    """Load cached PCA if available, otherwise fit once without file lock.

    Returns:
      pca_model, pca_meta, cache_hit, cache_model_path
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    key = _cache_key(metadata_csv, indices, n_components)
    model_path = cache_root / f"rt_axis_pca_{key}.pkl"
    meta_path = cache_root / f"rt_axis_pca_{key}.json"

    if model_path.exists():
        model = load_rt_axis_pca(model_path)
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        meta.setdefault("cache_key", key)
        return model, meta, True, model_path

    model, pca_meta = fit_rt_axis_pca(
        metadata_csv=metadata_csv,
        indices=indices,
        n_components=n_components,
    )
    pca_meta = dict(pca_meta)
    pca_meta["cache_key"] = key
    pca_meta["cache_model_path"] = str(model_path)
    pca_meta["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    save_rt_axis_pca(model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(pca_meta, f, ensure_ascii=False, indent=2)

    return model, pca_meta, False, model_path
