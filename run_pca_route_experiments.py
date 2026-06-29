"""Run two PCA routes and compare against TIC+PCA+MLP baseline.

Route 1: raw RT x m/z first, then PCA+MLP.
Route 2: binned (1024x256) RT x m/z summary, then PCA+MLP.

Outputs:
  - JSON metrics summary
  - Markdown report with baseline deltas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data_reader import read_gcms_data


class TICPcaMLPBaseline:
    """Lightweight local copy to avoid importing full DL baseline modules."""

    def __init__(
        self,
        n_components=64,
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=42,
    ):
        self.n_components = int(n_components)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)

        self.scaler = StandardScaler()
        self.pca = None
        self.mlp = None
        self._single_class = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) < 2:
            self._single_class = classes[0] if len(classes) == 1 else None
            self.pca = None
            self.mlp = None
            return

        _, counts = np.unique(y, return_counts=True)
        min_count = int(counts.min()) if len(counts) else 0
        use_early_stopping = min_count >= 3 and len(y) >= 30

        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        n_comp = max(1, int(n_comp))
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)

        Xs = self.scaler.fit_transform(X)
        Xp = self.pca.fit_transform(Xs)

        self.mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=self.max_iter,
            early_stopping=use_early_stopping,
            n_iter_no_change=12,
            random_state=self.random_state,
        )
        self.mlp.fit(Xp, y)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self._single_class is not None:
            preds = np.full(X.shape[0], self._single_class, dtype=object)
            scores = np.ones(X.shape[0], dtype=np.float32)
            return preds, scores

        Xs = self.scaler.transform(X)
        Xp = self.pca.transform(Xs)
        probs = self.mlp.predict_proba(Xp)
        idx = probs.argmax(axis=1)
        preds = self.mlp.classes_[idx]
        scores = probs[np.arange(len(idx)), idx]
        return preds, scores


def _load_and_filter(metadata_csv: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    df = df[(df["product_fine"] != "BLANK") & (~df["is_special"])].reset_index(drop=True)
    return df


def few_shot_from_unknown(
    unknown_idx: List[int],
    metadata_csv: str,
    product_col: str = "product_fine",
    n_shot_values: Tuple[int, ...] = (1, 3, 5, 10),
    seed: int = 42,
):
    """Local copy from dataset.py to avoid importing torch-heavy module."""
    rng = np.random.RandomState(seed)
    df = _load_and_filter(metadata_csv)
    unknown_df = df.iloc[unknown_idx]

    orig_to_local = {orig: local for local, orig in enumerate(unknown_idx)}

    results = {}
    for n_shot in n_shot_values:
        ref_idx_list = []
        test_idx_list = []
        for cls in unknown_df[product_col].unique():
            cls_indices = unknown_df[unknown_df[product_col] == cls].index.tolist()
            if len(cls_indices) <= n_shot:
                ref_idx_list.extend([orig_to_local[i] for i in cls_indices])
                continue
            perm = rng.permutation(len(cls_indices))
            ref_idx_list.extend([orig_to_local[cls_indices[j]] for j in perm[:n_shot]])
            test_idx_list.extend([orig_to_local[cls_indices[j]] for j in perm[n_shot:]])
        results[n_shot] = {"ref_idx": ref_idx_list, "test_idx": test_idx_list}
    return results


def _normalize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sd


def _interp_to_len(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if x.size == 1:
        return np.full(target_len, float(x[0]), dtype=np.float32)
    src = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


def _safe_matrix_from_raw(mode: str, rts, a, b) -> Tuple[np.ndarray, np.ndarray]:
    """Return (tic_raw, mean_spectrum_raw) from raw reader output."""
    if mode == "matrix":
        mz_vals = np.asarray(a, dtype=np.float32)
        mat = np.asarray(b, dtype=np.float32)
        if mat.shape == (len(mz_vals), len(rts)):
            mat = mat.T
        if mat.shape != (len(rts), len(mz_vals)):
            raise ValueError(
                f"matrix shape mismatch: mat={mat.shape}, rts={len(rts)}, mz={len(mz_vals)}"
            )
        tic = np.maximum(mat, 0.0).sum(axis=1)
        mean_spec = np.maximum(mat, 0.0).mean(axis=0)
        return tic.astype(np.float32), mean_spec.astype(np.float32)

    if mode == "spectra":
        spectra = a
        tic = []
        mz_min = np.inf
        mz_max = -np.inf
        for mzs, ints in spectra:
            mzs = np.asarray(mzs, dtype=np.float32)
            ints = np.asarray(ints, dtype=np.float32)
            if mzs.size:
                mz_min = min(mz_min, float(np.min(mzs)))
                mz_max = max(mz_max, float(np.max(mzs)))
            tic.append(float(np.sum(np.maximum(ints, 0.0))))

        tic = np.asarray(tic, dtype=np.float32)
        if not np.isfinite(mz_min) or not np.isfinite(mz_max) or mz_max <= mz_min:
            return tic, np.zeros(256, dtype=np.float32)

        bins = np.linspace(mz_min, mz_max, 257, dtype=np.float32)
        spec_hist = np.zeros(256, dtype=np.float32)
        for mzs, ints in spectra:
            mzs = np.asarray(mzs, dtype=np.float32)
            ints = np.asarray(ints, dtype=np.float32)
            if mzs.size == 0:
                continue
            h, _ = np.histogram(mzs, bins=bins, weights=np.maximum(ints, 0.0))
            spec_hist += h.astype(np.float32)
        spec_hist /= max(len(spectra), 1)
        return tic, spec_hist

    if mode == "tic":
        tic = np.asarray(a, dtype=np.float32)
        return tic, np.zeros(256, dtype=np.float32)

    raise ValueError(f"unsupported mode: {mode}")


def _raw_feature_from_d_path(d_path: str) -> np.ndarray:
    mode, rts, a, b = read_gcms_data(d_path, backend="auto")
    tic_raw, spec_raw = _safe_matrix_from_raw(mode, rts, a, b)

    tic_vec = _interp_to_len(np.log1p(np.maximum(tic_raw, 0.0)), 1024)
    spec_vec = _interp_to_len(np.log1p(np.maximum(spec_raw, 0.0)), 256)

    tic_vec = _normalize_1d(tic_vec)
    spec_vec = _normalize_1d(spec_vec)
    return np.concatenate([tic_vec, spec_vec], axis=0).astype(np.float32)


def _binned_feature_from_tensor(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as data:
        if "grid" in data:
            grid = np.asarray(data["grid"], dtype=np.float32)
        elif "tensor" in data:
            tensor = np.asarray(data["tensor"], dtype=np.float32)
            grid = tensor[0] if tensor.ndim == 3 else np.zeros((1024, 256), dtype=np.float32)
        else:
            grid = np.zeros((1024, 256), dtype=np.float32)

    if grid.ndim != 2:
        grid = np.zeros((1024, 256), dtype=np.float32)

    tic = np.maximum(grid, 0.0).sum(axis=1)
    spec = np.maximum(grid, 0.0).mean(axis=0)

    tic_vec = _normalize_1d(np.log1p(np.maximum(tic, 0.0)))
    spec_vec = _normalize_1d(np.log1p(np.maximum(spec, 0.0)))
    route2_vec = np.concatenate([tic_vec, spec_vec], axis=0).astype(np.float32)
    tic_only_vec = tic_vec.astype(np.float32)
    return route2_vec, tic_only_vec


def _fpr_at_95_tpr(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = int(np.searchsorted(tpr, 0.95))
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def _run_eval(
    X: np.ndarray,
    y: np.ndarray,
    split: Dict,
    metadata_csv: str,
    n_components: int,
    seed: int,
) -> Dict:
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_batch_idx = split["test_batch_idx"]
    test_unknown_idx = split["test_unknown_idx"]

    model = TICPcaMLPBaseline(n_components=n_components, random_state=seed)
    model.fit(X[train_idx], y[train_idx])

    # Setting A
    pred_a, _ = model.predict(X[test_batch_idx])
    y_a = y[test_batch_idx]
    setting_a = {
        "accuracy": float(accuracy_score(y_a, pred_a)),
        "macro_f1": float(f1_score(y_a, pred_a, average="macro", zero_division=0)),
    }

    # Setting B
    known_idx = sorted(set(val_idx) | set(test_batch_idx))
    _, score_known = model.predict(X[known_idx])
    _, score_unknown = model.predict(X[test_unknown_idx])
    y_open = np.concatenate(
        [np.ones(len(score_known), dtype=np.int32), np.zeros(len(score_unknown), dtype=np.int32)]
    )
    s_open = np.concatenate([score_known, score_unknown]).astype(np.float32)
    setting_b = {
        "open_set_AUROC": float(roc_auc_score(y_open, s_open)),
        "FPR_at_95TPR": _fpr_at_95_tpr(y_open, s_open),
    }

    # Setting C (3-shot)
    fs = few_shot_from_unknown(
        test_unknown_idx,
        metadata_csv,
        product_col="product_fine",
        n_shot_values=(3,),
        seed=seed,
    )
    shot = fs.get(3, {"ref_idx": [], "test_idx": []})
    ref_global = [test_unknown_idx[i] for i in shot["ref_idx"]]
    test_global = [test_unknown_idx[i] for i in shot["test_idx"]]

    if ref_global and test_global:
        fs_model = TICPcaMLPBaseline(n_components=n_components, random_state=seed)
        fs_model.fit(X[ref_global], y[ref_global])
        pred_c, _ = fs_model.predict(X[test_global])
        y_c = y[test_global]
        setting_c3 = {
            "accuracy": float(accuracy_score(y_c, pred_c)),
            "macro_f1": float(f1_score(y_c, pred_c, average="macro", zero_division=0)),
            "n_ref": int(len(ref_global)),
            "n_test": int(len(test_global)),
        }
    else:
        setting_c3 = {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "n_ref": int(len(ref_global)),
            "n_test": int(len(test_global)),
        }

    return {
        "setting_a": setting_a,
        "setting_b": setting_b,
        "setting_c": {"3": setting_c3},
    }


def _build_markdown(results: Dict, baseline_key: str) -> str:
    baseline = results[baseline_key]
    b_auroc = baseline["setting_b"]["open_set_AUROC"]
    b_fpr95 = baseline["setting_b"]["FPR_at_95TPR"]
    b_s3 = baseline["setting_c"]["3"]["accuracy"]

    lines = []
    lines.append("# PCA Route Experiments")
    lines.append("")
    lines.append("| method | settingA_acc | settingB_auroc | settingB_fpr95 | settingC_3shot_acc | d_auroc_vs_baseline | d_fpr95_vs_baseline | d_3shot_vs_baseline |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for key, item in results.items():
        a = item["setting_a"]["accuracy"]
        auroc = item["setting_b"]["open_set_AUROC"]
        fpr95 = item["setting_b"]["FPR_at_95TPR"]
        s3 = item["setting_c"]["3"]["accuracy"]
        lines.append(
            "| {k} | {a:.4f} | {u:.4f} | {f:.4f} | {s:.4f} | {du:.4f} | {df:.4f} | {ds:.4f} |".format(
                k=key,
                a=a,
                u=auroc,
                f=fpr95,
                s=s3,
                du=auroc - b_auroc,
                df=fpr95 - b_fpr95,
                ds=s3 - b_s3,
            )
        )

    return "\n".join(lines) + "\n"


def _parse_dims(text: str) -> List[int]:
    vals = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals


def main():
    parser = argparse.ArgumentParser(description="Run PCA route experiments")
    parser.add_argument("--metadata", default="prepared_data/metadata.csv")
    parser.add_argument("--split", default="prepared_data/split.json")
    parser.add_argument("--output_dir", default="outputs/pca_route_experiments")
    parser.add_argument("--bins_dims", default="64,128,256")
    parser.add_argument("--raw_dims", default="64,128,256")
    parser.add_argument("--baseline_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca_components", type=int, default=None, help="Number of PCA components to retain")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = Path(args.metadata)
    split_json = Path(args.split)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata not found: {metadata_csv}")
    if not split_json.exists():
        raise FileNotFoundError(f"split not found: {split_json}")

    df = pd.read_csv(metadata_csv)
    df = df[(df["product_fine"] != "BLANK") & (~df["is_special"])].reset_index(drop=True)
    y = df["product_fine"].to_numpy()

    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)

    needed_idx = sorted(
        set(split["train_idx"]) |
        set(split["val_idx"]) |
        set(split["test_batch_idx"]) |
        set(split["test_unknown_idx"])
    )
    _ = needed_idx  # kept for explicitness; all rows are extracted below for caching.

    # Route 1 raw feature cache
    raw_cache = out_dir / "raw_route_features.npz"
    if raw_cache.exists():
        raw_data = np.load(raw_cache)
        X_raw = np.asarray(raw_data["X_raw"], dtype=np.float32)
        if X_raw.shape[0] != len(df):
            X_raw = None
    else:
        X_raw = None

    if X_raw is None:
        feats = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="extract raw route features"):
            try:
                feats.append(_raw_feature_from_d_path(str(row["d_path"])))
            except Exception:
                feats.append(np.zeros(1280, dtype=np.float32))
        X_raw = np.stack(feats, axis=0)
        np.savez_compressed(raw_cache, X_raw=X_raw)

    # Route 2 (binned) + baseline TIC feature cache
    binned_cache = out_dir / "binned_route_features.npz"
    if binned_cache.exists():
        data = np.load(binned_cache)
        X_binned = np.asarray(data["X_binned"], dtype=np.float32)
        X_tic = np.asarray(data["X_tic"], dtype=np.float32)
        if X_binned.shape[0] != len(df) or X_tic.shape[0] != len(df):
            X_binned = None
            X_tic = None
    else:
        X_binned = None
        X_tic = None

    if X_binned is None or X_tic is None:
        feat_b = []
        feat_t = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="extract binned route features"):
            try:
                vb, vt = _binned_feature_from_tensor(str(row["tensor_path"]))
            except Exception:
                vb = np.zeros(1280, dtype=np.float32)
                vt = np.zeros(1024, dtype=np.float32)
            feat_b.append(vb)
            feat_t.append(vt)
        X_binned = np.stack(feat_b, axis=0)
        X_tic = np.stack(feat_t, axis=0)
        np.savez_compressed(binned_cache, X_binned=X_binned, X_tic=X_tic)

    # Initialize results dictionary
    results = {}

    # Ensure baseline is initialized before any usage
    baseline = TICPcaMLPBaseline(n_components=args.pca_components or args.baseline_dim, random_state=args.seed)

    # Baseline evaluation
    baseline_key = f"baseline_tic_pca_mlp_d{baseline.n_components}"
    results[baseline_key] = _run_eval(
        X=X_tic,
        y=y,
        split=split,
        metadata_csv=str(metadata_csv),
        n_components=baseline.n_components,
        seed=args.seed,
    )

    # Route 2: bins -> PCA
    for d in _parse_dims(args.bins_dims):
        key = f"route2_bins_then_pca_d{d}"
        results[key] = _run_eval(
            X=X_binned,
            y=y,
            split=split,
            metadata_csv=str(metadata_csv),
            n_components=d,
            seed=args.seed,
        )

    # Route 1: raw -> PCA
    for d in _parse_dims(args.raw_dims):
        key = f"route1_raw_then_pca_d{d}"
        results[key] = _run_eval(
            X=X_raw,
            y=y,
            split=split,
            metadata_csv=str(metadata_csv),
            n_components=d,
            seed=args.seed,
        )

    summary = {
        "metadata": str(metadata_csv),
        "split": str(split_json),
        "n_samples": int(len(df)),
        "results": results,
    }

    json_path = out_dir / "pca_route_results.json"
    md_path = out_dir / "PCA_ROUTE_REPORT.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = _build_markdown(results, baseline_key=baseline_key)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"saved: {json_path}")
    print(f"saved: {md_path}")


if __name__ == "__main__":
    main()
