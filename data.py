"""Tensor loading shared by classical and deep close-and-few experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from protocol import ProtocolSpec


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_tensor_path(
    row: pd.Series,
    tensor_root: str | Path | None,
    spec: ProtocolSpec,
) -> Path:
    original = Path(str(row["tensor_path"]))
    if original.is_file():
        return original
    if not original.is_absolute():
        candidate = PROJECT_ROOT / original
        if candidate.is_file():
            return candidate
    if tensor_root is None:
        raise FileNotFoundError(
            f"tensor does not exist: {original}; pass --tensor-root to relocate old paths"
        )
    root = Path(tensor_root)
    parts = original.parts
    if "tensors" in parts:
        relative = Path(*parts[parts.index("tensors") + 1 :])
        candidate = root / relative
        if candidate.is_file():
            return candidate
    candidates = [
        root / str(row[spec.batch_col]) / str(row[spec.product_col]) / original.name,
        root / original.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot relocate tensor {original.name} below {root}")


def load_tensor(
    row: pd.Series,
    tensor_root: str | Path | None,
    spec: ProtocolSpec,
) -> np.ndarray:
    path = resolve_tensor_path(row, tensor_root, spec)
    with np.load(path, allow_pickle=False) as data:
        if "tensor" not in data.files:
            raise ValueError(f"NPZ has no tensor array: {path}")
        value = np.asarray(data["tensor"], dtype=np.float32)
    if value.ndim != 3:
        raise ValueError(f"expected CxRTxm/z tensor, got {value.shape}: {path}")
    return value


def extract_tic_matrix(
    df: pd.DataFrame,
    tensor_root: str | Path | None,
    spec: ProtocolSpec,
) -> np.ndarray:
    """Extract the first-channel total ion chromatogram for every sample."""
    rows = []
    expected_length = None
    for _, row in df.iterrows():
        tensor = load_tensor(row, tensor_root, spec)
        tic = tensor[0].sum(axis=1).astype(np.float32)
        low = float(np.min(tic)) if tic.size else 0.0
        high = float(np.max(tic)) if tic.size else 0.0
        tic = (tic - low) / max(high - low, 1e-8)
        if expected_length is None:
            expected_length = len(tic)
        if len(tic) != expected_length:
            raise ValueError("all TIC vectors must have the same RT length")
        rows.append(tic)
    if not rows:
        raise ValueError("metadata contains no samples")
    return np.stack(rows)
