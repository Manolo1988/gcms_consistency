"""Chemical evidence scores for GC-MS consistency evaluation."""
from __future__ import annotations

import numpy as np


def _as_vector(x):
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _safe_minmax01(x):
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def fourth_root_transform(x):
    """Variance-stabilizing transform used before spectral correlation scores."""
    arr = np.asarray(x, dtype=np.float32)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    return np.power(arr, 0.25, dtype=np.float32)


def cosine_similarity(a, b):
    av = _as_vector(a)
    bv = _as_vector(b)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def pearson_similarity(a, b):
    av = _as_vector(a)
    bv = _as_vector(b)
    av = av - float(av.mean())
    bv = bv - float(bv.mean())
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def covariance_mapping_similarity(a, b):
    """
    Covariance-map style score for two GC-MS intensity maps.

    This is a bounded normalized covariance score. It is intentionally simple
    and deterministic so it can serve as an interpretable evidence channel
    alongside the learned embedding score.
    """
    av = _as_vector(a)
    bv = _as_vector(b)
    av = av - float(av.mean())
    bv = bv - float(bv.mean())
    cov = float(np.dot(av, bv))
    energy = float(np.dot(av, av) + np.dot(bv, bv))
    if energy <= 1e-12:
        return 0.0
    return float((2.0 * cov) / energy)


def tic_from_tensor(x):
    """Collapse a GC-MS tensor into a retention-time TIC profile."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3:
        return arr.sum(axis=(0, 2))
    if arr.ndim == 2:
        return arr.sum(axis=1)
    return _as_vector(arr)


def mz_profile_from_tensor(x):
    """Collapse a GC-MS tensor into an m/z abundance profile."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3:
        return arr.sum(axis=(0, 1))
    if arr.ndim == 2:
        return arr.sum(axis=0)
    return _as_vector(arr)


def rt_peak_alignment_score(a, b):
    """Peak-position agreement score in RT space, scaled to [0, 1]."""
    ta = tic_from_tensor(a)
    tb = tic_from_tensor(b)
    if ta.size == 0 or tb.size == 0:
        return 0.0
    ia = int(np.argmax(ta))
    ib = int(np.argmax(tb))
    denom = max(max(len(ta), len(tb)) - 1, 1)
    return float(max(0.0, 1.0 - abs(ia - ib) / denom))


def tensor_chemical_scores(x, proto_x):
    """Return interpretable chemical similarity descriptors for two tensors."""
    x01 = _safe_minmax01(x)
    p01 = _safe_minmax01(proto_x)
    x4 = fourth_root_transform(x01)
    p4 = fourth_root_transform(p01)
    tic_x = tic_from_tensor(x01)
    tic_p = tic_from_tensor(p01)
    mz_x = mz_profile_from_tensor(x01)
    mz_p = mz_profile_from_tensor(p01)
    return {
        "chem_cosine": cosine_similarity(x4, p4),
        "chem_pearson": pearson_similarity(x4, p4),
        "chem_covmap": covariance_mapping_similarity(x4, p4),
        "chem_tic_cosine": cosine_similarity(tic_x, tic_p),
        "chem_mz_cosine": cosine_similarity(mz_x, mz_p),
        "chem_rt_peak": rt_peak_alignment_score(x01, p01),
    }


def tic_feature_scores(tic, proto_tic):
    """Return TIC-PCA feature similarity descriptors."""
    if tic is None or proto_tic is None:
        return {}
    t = _as_vector(tic)
    p = _as_vector(proto_tic)
    return {
        "chem_tic_pca_cosine": cosine_similarity(t, p),
        "chem_tic_pca_pearson": pearson_similarity(t, p),
    }


def aggregate_chemical_score(scores):
    """Aggregate chemical descriptors into one evidence score in [0, 1]."""
    weights = {
        "chem_covmap": 0.30,
        "chem_cosine": 0.20,
        "chem_pearson": 0.15,
        "chem_tic_cosine": 0.15,
        "chem_mz_cosine": 0.10,
        "chem_rt_peak": 0.05,
        "chem_tic_pca_cosine": 0.05,
    }
    total = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        if key not in scores:
            continue
        val = float(scores[key])
        if not np.isfinite(val):
            continue
        if key in {"chem_pearson", "chem_covmap", "chem_tic_pca_pearson"}:
            val = 0.5 * (val + 1.0)
        val = min(max(val, 0.0), 1.0)
        total += weight * val
        weight_sum += weight
    if weight_sum <= 1e-12:
        return float("nan")
    return float(total / weight_sum)


class ChemicalEvidenceStore:
    """Class-wise raw tensor and TIC prototypes for chemical evidence scoring."""

    def __init__(self):
        self.tensor_prototypes = {}
        self.tic_prototypes = {}

    @classmethod
    def from_loader(cls, loader, max_samples_per_class=128, use_tic_pca=False):
        store = cls()
        tensor_sums = {}
        tic_sums = {}
        counts = {}
        for batch in loader:
            x = batch["input"].detach().cpu().numpy()
            labels = batch["product"].detach().cpu().numpy()
            tic = batch.get("tic") if use_tic_pca else None
            tic_np = tic.detach().cpu().numpy() if tic is not None else None
            for i, label in enumerate(labels):
                key = int(label)
                n = counts.get(key, 0)
                if n >= int(max_samples_per_class):
                    continue
                xi = np.asarray(x[i], dtype=np.float32)
                if key not in tensor_sums:
                    tensor_sums[key] = np.zeros_like(xi, dtype=np.float32)
                tensor_sums[key] += xi
                if tic_np is not None:
                    ti = np.asarray(tic_np[i], dtype=np.float32)
                    if key not in tic_sums:
                        tic_sums[key] = np.zeros_like(ti, dtype=np.float32)
                    tic_sums[key] += ti
                counts[key] = n + 1

        for key, summed in tensor_sums.items():
            store.tensor_prototypes[key] = summed / max(float(counts.get(key, 1)), 1.0)
        for key, summed in tic_sums.items():
            store.tic_prototypes[key] = summed / max(float(counts.get(key, 1)), 1.0)
        return store

    def score(self, x, pred_label, tic=None, use_tic_pca=False):
        pred_label = int(pred_label)
        scores = {}
        proto_x = self.tensor_prototypes.get(pred_label)
        if proto_x is not None:
            scores.update(tensor_chemical_scores(x, proto_x))
        proto_tic = self.tic_prototypes.get(pred_label) if use_tic_pca else None
        if proto_tic is not None:
            scores.update(tic_feature_scores(tic, proto_tic))
        if scores:
            scores["chemical_score"] = aggregate_chemical_score(scores)
        return scores
