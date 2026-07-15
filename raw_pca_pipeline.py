from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from data_reader import read_gcms_data
from dataset import few_shot_from_unknown, load_data_split


def _progress_enabled() -> bool:
    return os.environ.get("GCMS_SHOW_PROGRESS", "0") == "1"


def _load_and_filter(metadata_csv: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    df = df[(df["product_fine"] != "BLANK") & (~df["is_special"])].reset_index(drop=True)
    return df


def _normalize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mean = float(np.mean(x))
    std = float(np.std(x))
    if not np.isfinite(std) or std < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mean) / std).astype(np.float32)


def _interp_to_len(x: np.ndarray, out_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = int(x.shape[0])
    if n == out_len:
        return x
    if n <= 1:
        return np.zeros(out_len, dtype=np.float32)
    old_pos = np.linspace(0.0, 1.0, n, dtype=np.float32)
    new_pos = np.linspace(0.0, 1.0, out_len, dtype=np.float32)
    out = np.interp(new_pos, old_pos, x).astype(np.float32)
    return out


def _safe_matrix_from_raw(mode: str, rts, a, b) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "matrix":
        mat = np.asarray(b, dtype=np.float32)
        if mat.ndim != 2:
            return np.zeros(1, dtype=np.float32), np.zeros(256, dtype=np.float32)
        tic = np.maximum(mat, 0.0).sum(axis=1)
        spec = np.maximum(mat, 0.0).mean(axis=0)
        return tic.astype(np.float32), spec.astype(np.float32)

    if mode == "spectra":
        spectra = a
        if spectra is None:
            return np.zeros(1, dtype=np.float32), np.zeros(256, dtype=np.float32)

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


@dataclass
class RawPcaModelConfig:
    n_components: int = 128
    hidden_layer_sizes: Tuple[int, int] = (128, 64)
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3
    max_iter: int = 300
    random_state: int = 42
    score_blend: float = 1.0
    distance_percentile: float = 95.0


class RawPcaHybridClassifier:
    def __init__(self, cfg: RawPcaModelConfig):
        self.cfg = cfg

        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.mlp: Optional[MLPClassifier] = None

        self._single_class = None
        self.class_centroids: Dict[int, np.ndarray] = {}
        self.class_radii: Dict[int, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        classes = np.unique(y)
        if len(classes) < 2:
            self._single_class = int(classes[0]) if len(classes) == 1 else None
            self.pca = None
            self.mlp = None
            return

        _, counts = np.unique(y, return_counts=True)
        min_count = int(counts.min()) if len(counts) else 0
        use_early_stopping = min_count >= 3 and len(y) >= 30

        n_comp = min(int(self.cfg.n_components), X.shape[0] - 1, X.shape[1])
        n_comp = max(1, int(n_comp))

        self.pca = PCA(n_components=n_comp, random_state=int(self.cfg.random_state))
        Xs = self.scaler.fit_transform(X)
        Xp = self.pca.fit_transform(Xs)

        self.mlp = MLPClassifier(
            hidden_layer_sizes=tuple(int(v) for v in self.cfg.hidden_layer_sizes),
            activation="relu",
            solver="adam",
            alpha=float(self.cfg.alpha),
            learning_rate_init=float(self.cfg.learning_rate_init),
            max_iter=int(self.cfg.max_iter),
            early_stopping=use_early_stopping,
            n_iter_no_change=12,
            random_state=int(self.cfg.random_state),
        )
        self.mlp.fit(Xp, y)

        # Build class-wise centroid and robust radius in PCA space.
        self.class_centroids = {}
        self.class_radii = {}
        for cls in np.unique(y):
            mask = y == int(cls)
            Xi = Xp[mask]
            if Xi.shape[0] == 0:
                continue
            center = np.mean(Xi, axis=0)
            d = np.linalg.norm(Xi - center, axis=1)
            radius = float(np.percentile(d, float(self.cfg.distance_percentile))) if d.size else 1.0
            radius = max(radius, 1e-6)
            self.class_centroids[int(cls)] = center.astype(np.float32)
            self.class_radii[int(cls)] = radius

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler.transform(X)
        return self.pca.transform(Xs)

    def _distance_knownness(self, Xp: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dists = np.zeros(Xp.shape[0], dtype=np.float32)
        knownness = np.zeros(Xp.shape[0], dtype=np.float32)

        for i, cls in enumerate(preds.astype(np.int32).tolist()):
            center = self.class_centroids.get(cls)
            radius = self.class_radii.get(cls, 1.0)
            if center is None:
                dists[i] = 0.0
                knownness[i] = 1.0
                continue
            dist = float(np.linalg.norm(Xp[i] - center))
            dists[i] = dist
            knownness[i] = float(np.exp(-dist / max(radius, 1e-6)))

        return dists, knownness

    def predict_with_details(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)

        if self._single_class is not None:
            n = X.shape[0]
            preds = np.full(n, int(self._single_class), dtype=np.int32)
            score = np.ones(n, dtype=np.float32)
            return {
                "pred": preds,
                "open_score": score,
                "max_prob": score,
                "knownness": score,
                "min_dist": np.zeros(n, dtype=np.float32),
                "embedding": X,
            }

        Xp = self._transform(X)
        probs = self.mlp.predict_proba(Xp)
        idx = probs.argmax(axis=1)
        preds = self.mlp.classes_[idx].astype(np.int32)
        max_prob = probs[np.arange(len(idx)), idx].astype(np.float32)

        min_dist, knownness = self._distance_knownness(Xp, preds)

        blend = float(self.cfg.score_blend)
        blend = min(max(blend, 0.0), 1.0)
        open_score = blend * max_prob + (1.0 - blend) * knownness

        return {
            "pred": preds,
            "open_score": open_score.astype(np.float32),
            "max_prob": max_prob.astype(np.float32),
            "knownness": knownness.astype(np.float32),
            "min_dist": min_dist.astype(np.float32),
            "embedding": Xp.astype(np.float32),
        }


def _product_identification_metrics(records: List[Dict], train_records: Optional[List[Dict]] = None) -> Dict:
    y_true = np.asarray([r["true_product"] for r in records], dtype=np.int32)
    y_pred = np.asarray([r["pred_product"] for r in records], dtype=np.int32)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if train_records is not None and len(train_records) > 0:
        y_true_t = np.asarray([r["true_product"] for r in train_records], dtype=np.int32)
        y_pred_t = np.asarray([r["pred_product"] for r in train_records], dtype=np.int32)
        acc_same = float(accuracy_score(y_true_t, y_pred_t))
        out["cross_batch_gap"] = float(acc_same - out["accuracy"])

    return out


def _consistency_scoring_metrics(records: List[Dict]) -> Dict:
    correct = np.asarray([bool(r["correct"]) for r in records], dtype=bool)
    scores = np.asarray([float(r["consistency_score"]) for r in records], dtype=np.float32)

    result = {
        "accept_rate": float(np.mean([bool(r.get("is_known", True)) for r in records])) if records else float("nan")
    }

    if len(np.unique(correct.astype(np.int32))) > 1:
        result["AUROC_correct"] = float(roc_auc_score(correct.astype(np.int32), scores))

        s_correct = scores[correct]
        s_wrong = scores[~correct]
        pooled_std = np.sqrt(
            (s_correct.var() * max(len(s_correct) - 1, 0)
             + s_wrong.var() * max(len(s_wrong) - 1, 0))
            / max(len(s_correct) + len(s_wrong) - 2, 1)
        )
        result["cohens_d"] = float((s_correct.mean() - s_wrong.mean()) / max(pooled_std, 1e-8))
    else:
        result["AUROC_correct"] = float("nan")
        result["cohens_d"] = float("nan")

    return result


def _batch_robustness_metrics(records: List[Dict]) -> Dict:
    zs = np.stack([r["z"] for r in records], axis=0)
    product_labels = np.asarray([r["true_product"] for r in records], dtype=np.int32)
    batch_labels = np.asarray([r["batch_label"] for r in records], dtype=np.int32)

    result: Dict[str, float] = {}

    if len(np.unique(product_labels)) > 1 and len(zs) > len(np.unique(product_labels)):
        result["silhouette_product"] = float(
            silhouette_score(zs, product_labels, sample_size=min(len(zs), 2000))
        )
    else:
        result["silhouette_product"] = float("nan")

    if len(np.unique(batch_labels)) > 1 and len(zs) > len(np.unique(batch_labels)):
        result["silhouette_batch"] = float(
            silhouette_score(zs, batch_labels, sample_size=min(len(zs), 2000))
        )
    else:
        result["silhouette_batch"] = float("nan")

    if len(np.unique(batch_labels)) > 1 and len(zs) > 10:
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(zs, batch_labels)
        result["batch_predictability"] = float(clf.score(zs, batch_labels))
    else:
        result["batch_predictability"] = float("nan")

    return result


def _fpr_at_95_tpr(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = int(np.searchsorted(tpr, 0.95))
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def _open_set_metrics(known_records: List[Dict], unknown_records: List[Dict]) -> Dict:
    known_scores = np.asarray([r["open_set_score"] for r in known_records], dtype=np.float32)
    unknown_scores = np.asarray([r["open_set_score"] for r in unknown_records], dtype=np.float32)

    y_open = np.concatenate([
        np.ones(len(known_scores), dtype=np.int32),
        np.zeros(len(unknown_scores), dtype=np.int32),
    ])
    s_open = np.concatenate([known_scores, unknown_scores]).astype(np.float32)

    out = {
        "open_set_AUROC": float(roc_auc_score(y_open, s_open)) if len(np.unique(y_open)) > 1 else float("nan"),
        "FPR_at_95TPR": _fpr_at_95_tpr(y_open, s_open) if len(np.unique(y_open)) > 1 else float("nan"),
        "known_score_mean": float(np.mean(known_scores)) if known_scores.size else float("nan"),
        "unknown_score_mean": float(np.mean(unknown_scores)) if unknown_scores.size else float("nan"),
    }
    return out


def _records_from_details(
    details: Dict[str, np.ndarray],
    y_true: np.ndarray,
    batch_labels: np.ndarray,
    sample_indices: np.ndarray,
) -> List[Dict]:
    out = []
    preds = details["pred"]
    scores = details["open_score"]
    emb = details["embedding"]
    dists = details["min_dist"]

    for i in range(len(y_true)):
        out.append(
            {
                "sample_id": int(sample_indices[i]),
                "true_product": int(y_true[i]),
                "pred_product": int(preds[i]),
                "pred_class": str(int(preds[i])),
                "consistency_score": float(scores[i]),
                "open_set_score": float(scores[i]),
                "margin_score": float(scores[i]),
                "min_dist": float(dists[i]),
                "is_known": True,
                "correct": bool(int(preds[i]) == int(y_true[i])),
                "z": np.asarray(emb[i], dtype=np.float32),
                "batch_label": int(batch_labels[i]),
            }
        )
    return out


def _parse_hidden_sizes(hidden_text: str) -> Tuple[int, int]:
    vals = []
    for p in str(hidden_text).split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        return (256, 128)
    if len(vals) == 1:
        return (vals[0], max(32, vals[0] // 2))
    return (vals[0], vals[1])


def _raw_cache_path(cfg) -> Path:
    return Path(cfg.prepared_dir) / "raw_route_features.npz"


def _extract_or_load_raw_features(df: pd.DataFrame, cfg) -> np.ndarray:
    cache_path = _raw_cache_path(cfg)
    if cache_path.exists():
        with np.load(cache_path) as data:
            X = np.asarray(data["X_raw"], dtype=np.float32)
        if X.shape[0] == len(df) and X.shape[1] == 1280:
            return X

    feats = []
    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="extract raw route features",
        disable=not _progress_enabled(),
    ):
        try:
            feats.append(_raw_feature_from_d_path(str(row["d_path"])))
        except Exception:
            feats.append(np.zeros(1280, dtype=np.float32))

    X = np.stack(feats, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X_raw=X)
    return X


def _build_model_cfg_from_runtime(cfg) -> RawPcaModelConfig:
    hidden_sizes = _parse_hidden_sizes(getattr(cfg, "raw_pca_hidden", "256,128"))

    return RawPcaModelConfig(
        n_components=int(getattr(cfg, "raw_pca_components", 128) or 128),
        hidden_layer_sizes=hidden_sizes,
        alpha=float(getattr(cfg, "raw_pca_alpha", 1e-4) or 1e-4),
        learning_rate_init=float(getattr(cfg, "raw_pca_lr_init", 1e-3) or 1e-3),
        max_iter=int(getattr(cfg, "raw_pca_max_iter", 300) or 300),
        random_state=int(getattr(cfg, "seed", 42) or 42),
        score_blend=float(getattr(cfg, "raw_open_score_blend", 1.0) or 1.0),
        distance_percentile=float(getattr(cfg, "raw_distance_percentile", 95.0) or 95.0),
    )


def _few_shot_centroid_raw(X_ref: np.ndarray, y_ref: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    classes = np.unique(y_ref)
    centers = []
    for cls in classes:
        centers.append(np.mean(X_ref[y_ref == cls], axis=0))
    centers = np.stack(centers, axis=0)
    d = ((X_test[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return classes[d.argmin(axis=1)]


def _few_shot_svm_pca(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
    c_value: float,
) -> np.ndarray:
    sc = StandardScaler()
    X_ref_s = sc.fit_transform(X_ref)
    X_test_s = sc.transform(X_test)

    n_comp = max(1, min(128, X_ref_s.shape[0] - 1, X_ref_s.shape[1]))
    pca = PCA(n_components=n_comp, random_state=int(random_state))
    Z_ref = pca.fit_transform(X_ref_s)
    Z_test = pca.transform(X_test_s)

    clf = SVC(
        kernel="rbf",
        C=float(c_value),
        probability=False,
        random_state=int(random_state),
    )
    clf.fit(Z_ref, y_ref)
    return clf.predict(Z_test)


def train_single_model_raw_pca(cfg):
    split = load_data_split(cfg)
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = "product_fine" if cfg.product_granularity == "fine" else "product_coarse"

    df = _load_and_filter(metadata_csv)
    y_text = df[product_col].to_numpy()
    X = _extract_or_load_raw_features(df, cfg)

    train_idx = np.asarray(split["train_idx"], dtype=np.int32)
    y_train_text = y_text[train_idx]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_text).astype(np.int32)

    model_cfg = _build_model_cfg_from_runtime(cfg)
    model = RawPcaHybridClassifier(model_cfg)
    model.fit(X[train_idx], y_train)

    model_dir = Path(cfg.output_dir) / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "label_encoder": le,
        "model_cfg": asdict(model_cfg),
        "product_col": product_col,
    }
    with open(model_dir / "raw_pca_model.pkl", "wb") as f:
        pickle.dump(payload, f)

    with open(model_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "raw_pca_mlp",
                "n_components": int(model_cfg.n_components),
                "n_train": int(len(train_idx)),
                "n_features": int(X.shape[1]),
                "n_classes": int(len(le.classes_)),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n模型已保存到", model_dir)
    print("  model_type: raw_pca_mlp")
    print("  pca_components:", model_cfg.n_components)
    print("  classes:", len(le.classes_))

    return model


def evaluate_single_model_raw_pca(
    cfg,
    baseline_eval_fn: Callable,
    print_summary_fn: Callable,
    save_summary_fn: Callable,
):
    from dataset import GCMSDataset
    import torch

    split = load_data_split(cfg)
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = "product_fine" if cfg.product_granularity == "fine" else "product_coarse"
    out_dir = Path(cfg.output_dir)

    df = _load_and_filter(metadata_csv)
    y_text = df[product_col].to_numpy()
    batch_text = df["batch_idx"].astype(str).to_numpy()

    X = _extract_or_load_raw_features(df, cfg)

    model_path = out_dir / "final_model" / "raw_pca_model.pkl"
    if not model_path.exists():
        print("未找到 final_model/raw_pca_model.pkl, 请先运行 python main.py train")
        return None, None, None

    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    model: RawPcaHybridClassifier = payload["model"]
    le_train: LabelEncoder = payload["label_encoder"]

    global_product_enc = LabelEncoder().fit(sorted(np.unique(y_text).tolist()))
    global_batch_enc = LabelEncoder().fit(sorted(np.unique(batch_text).tolist()))

    y_global = global_product_enc.transform(y_text).astype(np.int32)
    b_global = global_batch_enc.transform(batch_text).astype(np.int32)

    train_known_mask = np.isin(y_text, le_train.classes_)

    def _predict_known(idx: np.ndarray) -> Dict[str, np.ndarray]:
        raw = model.predict_with_details(X[idx])
        pred_train_ids = raw["pred"]
        pred_train_names = le_train.inverse_transform(pred_train_ids)
        pred_global = global_product_enc.transform(pred_train_names).astype(np.int32)

        return {
            "pred": pred_global,
            "open_score": raw["open_score"],
            "embedding": raw["embedding"],
            "min_dist": raw["min_dist"],
        }

    # Setting A
    result_a = None
    if split.get("test_batch_idx"):
        train_idx = np.asarray(split["train_idx"], dtype=np.int32)
        test_idx = np.asarray(split["test_batch_idx"], dtype=np.int32)

        pred_train = _predict_known(train_idx)
        pred_test = _predict_known(test_idx)

        train_records = _records_from_details(
            pred_train,
            y_global[train_idx],
            b_global[train_idx],
            train_idx,
        )
        test_records = _records_from_details(
            pred_test,
            y_global[test_idx],
            b_global[test_idx],
            test_idx,
        )

        pid = _product_identification_metrics(test_records, train_records)
        con = _consistency_scoring_metrics(test_records)
        rob = _batch_robustness_metrics(test_records)

        result_a = {
            "product_identification": pid,
            "consistency_scoring": con,
            "batch_robustness": rob,
            "records": test_records,
        }

    # Setting B
    result_b = None
    if split.get("test_unknown_idx"):
        known_idx = np.asarray(sorted(set(split.get("val_idx", [])) | set(split.get("test_batch_idx", []))), dtype=np.int32)
        unknown_idx = np.asarray(split["test_unknown_idx"], dtype=np.int32)

        known_idx = known_idx[np.isin(known_idx, np.where(train_known_mask)[0])]

        pred_known = _predict_known(known_idx)

        # Unknown classes are out of train label space. We project by nearest known centroid only for score continuity.
        raw_unknown = model.predict_with_details(X[unknown_idx])
        # Convert unknown predictions to global id if possible, otherwise keep nearest known mapped id.
        pred_unknown_names = le_train.inverse_transform(raw_unknown["pred"])
        pred_unknown_global = global_product_enc.transform(pred_unknown_names).astype(np.int32)
        pred_unknown = {
            "pred": pred_unknown_global,
            "open_score": raw_unknown["open_score"],
            "embedding": raw_unknown["embedding"],
            "min_dist": raw_unknown["min_dist"],
        }

        known_records = _records_from_details(
            pred_known,
            y_global[known_idx],
            b_global[known_idx],
            known_idx,
        )
        unknown_records = _records_from_details(
            pred_unknown,
            y_global[unknown_idx],
            b_global[unknown_idx],
            unknown_idx,
        )

        result_b = {
            "open_set": _open_set_metrics(known_records, unknown_records)
        }

    # Setting C
    result_c = None
    if split.get("test_unknown_idx"):
        result_c = {"few_shot": {}}
        unknown_idx = split["test_unknown_idx"]
        fs = few_shot_from_unknown(
            unknown_idx,
            metadata_csv,
            product_col=product_col,
            n_shot_values=tuple(cfg.n_shot_values),
            seed=int(cfg.seed),
        )

        for n_shot in cfg.n_shot_values:
            shot = fs.get(n_shot, {"ref_idx": [], "test_idx": []})
            ref_global = [unknown_idx[i] for i in shot.get("ref_idx", [])]
            test_global = [unknown_idx[i] for i in shot.get("test_idx", [])]

            if not ref_global or not test_global:
                result_c["few_shot"][n_shot] = {
                    "accuracy": float("nan"),
                    "macro_f1": float("nan"),
                    "n_ref": int(len(ref_global)),
                    "n_test": int(len(test_global)),
                }
                continue

            ref_idx = np.asarray(ref_global, dtype=np.int32)
            test_idx = np.asarray(test_global, dtype=np.int32)

            le_fs = LabelEncoder()
            y_ref = le_fs.fit_transform(y_text[ref_idx]).astype(np.int32)
            y_test = le_fs.transform(y_text[test_idx]).astype(np.int32)

            if int(n_shot) <= 1:
                fs_pred = _few_shot_centroid_raw(X[ref_idx], y_ref, X[test_idx])
            else:
                c_value = float(getattr(cfg, "raw_fewshot_c_3shot", 2.0) or 2.0) if int(n_shot) == 3 else 1.0
                fs_pred = _few_shot_svm_pca(
                    X[ref_idx],
                    y_ref,
                    X[test_idx],
                    random_state=int(getattr(cfg, "seed", 42) or 42),
                    c_value=float(c_value),
                )

            result_c["few_shot"][n_shot] = {
                "accuracy": float(accuracy_score(y_test, fs_pred)),
                "macro_f1": float(f1_score(y_test, fs_pred, average="macro", zero_division=0)),
                "n_ref": int(len(ref_idx)),
                "n_test": int(len(test_idx)),
            }

    # README baselines keep old path for fair comparison.
    def _make_loader(indices):
        ds = GCMSDataset(metadata_csv, product_col=product_col, augmentation=None, indices=indices)
        ds.product_enc = global_product_enc
        ds.batch_enc = global_batch_enc
        ds.df["product_label"] = global_product_enc.transform(ds.df[product_col])
        ds.df["batch_label"] = global_batch_enc.transform(ds.df["batch_idx"].astype(str))
        ds.num_products = len(global_product_enc.classes_)
        ds.num_batches = len(global_batch_enc.classes_)
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
        return ds, loader

    try:
        baselines_readme = baseline_eval_fn(split, cfg, _make_loader, metadata_csv, product_col)
    except Exception as e:
        baselines_readme = {}
        print(f"\n[README Baselines] 评估失败: {e}")

    print_summary_fn(result_a, result_b, result_c, cfg, baselines_readme=baselines_readme)
    save_summary_fn(
        result_a,
        result_b,
        result_c,
        split,
        out_dir,
        baselines_readme=baselines_readme,
        cfg=cfg,
    )

    return result_a, result_b, result_c
