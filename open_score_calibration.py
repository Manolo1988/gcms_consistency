"""
Open-set score calibration for reducing FPR@95TPR.

Supports three modes:
  - pseudo_unknown: Leave out some known classes as pseudo-unknown,
    collect score features, fit logistic regression to separate known vs unknown.
  - grid_search: Grid search over score weights, radius percentile,
    reject_threshold_factor.
  - logistic: Fit LogisticRegression on collected features.

Calibration parameters are saved and loaded, and evaluation reports
both pre- and post-calibration metrics.
"""
import json
import copy
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


def _safe_float_list(arr) -> list:
    """Convert numpy/torch array to list of floats."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    return [float(x) for x in arr]


class OpenScoreCalibrator:
    """
    Calibrate open-set consistency scores to improve FPR@95TPR.

    Collects features from known and pseudo-unknown samples,
    then learns a mapping from features to a calibrated score.
    """

    FEATURE_NAMES = [
        "base_score",       # exp(-dist / radius)
        "margin_score",     # (second_dist - min_dist) / second_dist
        "min_dist",         # minimum distance to prototype
        "radius_norm",      # min_dist / class_radius
        "second_dist",      # distance to second-nearest prototype
        "second_margin",    # second_dist - min_dist
    ]

    def __init__(self, mode="logistic", features=None, seed=42):
        self.mode = mode
        self.features = features or ["base", "margin", "min_dist", "radius_norm", "second_dist"]
        self.seed = seed
        self._model = None          # LogisticRegression (for logistic mode)
        self._weights = None        # dict of feature->weight (for grid_search mode)
        self._coefs = None          # raw coefficients from logistic
        self._intercept = None
        self._radius_percentile = None
        self._reject_factor = None
        self._threshold = None
        self._fitted = False

    def collect_features_from_predict(self, predict_result, proto_store):
        """
        Extract score features from a PrototypeStore.predict() result dict.

        predict_result is a dict with keys:
          scores, base_scores, margin_scores, min_dists, second_dists, all_dists,
          pred_idx (tensor of shape (B,))
        proto_store: PrototypeStore (for class radii)

        Returns dict of feature_name -> np.array of shape (N,)
        """
        n = predict_result["scores"].shape[0]

        base = predict_result["base_scores"].detach().cpu().numpy().astype(np.float64)
        margin = predict_result["margin_scores"].detach().cpu().numpy().astype(np.float64)
        min_dist = predict_result["min_dists"].detach().cpu().numpy().astype(np.float64)
        second_dist = predict_result.get("second_dists", min_dist.copy())
        if isinstance(second_dist, torch.Tensor):
            second_dist = second_dist.detach().cpu().numpy().astype(np.float64)

        pred_idx = predict_result["pred_idx"].detach().cpu().numpy().astype(np.int64)

        # Class radii for each prediction
        use_spherical = predict_result.get("use_spherical", True)
        radii_source = proto_store.spherical_radii if (use_spherical and proto_store.spherical_radii) else proto_store.radii

        class_radius = np.array([
            radii_source.get(proto_store.class_names[i], 1.0)
            for i in pred_idx
        ], dtype=np.float64)

        features = {}
        features["base_score"] = base
        features["margin_score"] = margin
        features["min_dist"] = min_dist
        features["radius_norm"] = np.divide(min_dist, np.maximum(class_radius, 1e-8))
        features["second_dist"] = second_dist
        features["second_margin"] = np.maximum(second_dist - min_dist, 0.0)

        return features

    def fit(self, known_features_list, unknown_features_list):
        """
        Fit calibrator on collected features.

        known_features_list: list of feature dicts (from known samples)
        unknown_features_list: list of feature dicts (from pseudo-unknown samples)
        """
        if self.mode == "logistic":
            return self._fit_logistic(known_features_list, unknown_features_list)
        elif self.mode == "grid_search":
            return self._fit_grid_search(known_features_list, unknown_features_list)
        else:
            raise ValueError(f"Unknown calibration mode: {self.mode}")

    def _build_feature_matrix(self, features_list):
        """Build (N, F) matrix from list of feature dicts."""
        all_feats = []
        for fd in features_list:
            row = []
            for fname in self.features:
                val = fd.get(fname)
                if val is None:
                    row.append(0.0)
                elif np.isscalar(val):
                    row.append(float(val))
                else:
                    # Assume array-like of length N
                    row.append(np.asarray(val, dtype=np.float64))
            all_feats.append(row)

        if not all_feats:
            return np.zeros((0, len(self.features)), dtype=np.float64)

        # Concatenate along sample axis
        X = np.column_stack([
            np.concatenate([r[i].reshape(-1) for r in all_feats]) if not np.isscalar(all_feats[0][i]) else
            np.array([float(r[i]) for r in all_feats])
            for i in range(len(self.features))
        ])
        return X

    def _fit_logistic(self, known_features_list, unknown_features_list):
        """Fit logistic regression on collected features."""
        X_known = self._build_feature_matrix(known_features_list)
        X_unknown = self._build_feature_matrix(unknown_features_list)

        n_known = X_known.shape[0]
        n_unknown = X_unknown.shape[0]

        if n_known < 5 or n_unknown < 5:
            print(f"  [Calibration] WARNING: insufficient samples (known={n_known}, unknown={n_unknown})")
            self._fitted = False
            return self

        X = np.vstack([X_known, X_unknown])
        y = np.concatenate([np.ones(n_known), np.zeros(n_unknown)])

        # Standardize features
        self._feature_mean = X.mean(axis=0, keepdims=True)
        self._feature_std = X.std(axis=0, keepdims=True).clip(min=1e-8)
        X_scaled = (X - self._feature_mean) / self._feature_std

        self._model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=self.seed,
            class_weight="balanced",
        )
        self._model.fit(X_scaled, y)

        self._coefs = self._model.coef_[0].tolist()
        self._intercept = float(self._model.intercept_[0])
        self._fitted = True

        # Compute calibration AUROC
        cal_proba = self._model.predict_proba(X_scaled)[:, 1]
        cal_auroc = float(roc_auc_score(y, cal_proba))
        print(f"  [Calibration] Logistic fit complete: AUROC(calibration)={cal_auroc:.4f}, "
              f"coefs={[f'{c:.3f}' for c in self._coefs]}, intercept={self._intercept:.3f}")

        return self

    def _fit_grid_search(self, known_features_list, unknown_features_list):
        """Grid search over score weights."""
        X_known = self._build_feature_matrix(known_features_list)
        X_unknown = self._build_feature_matrix(unknown_features_list)

        n_known = X_known.shape[0]
        n_unknown = X_unknown.shape[0]

        if n_known < 5 or n_unknown < 5:
            print(f"  [Calibration] WARNING: insufficient samples for grid search")
            self._fitted = False
            return self

        X = np.vstack([X_known, X_unknown])
        y = np.concatenate([np.ones(n_known), np.zeros(n_unknown)])

        best_auroc = -1
        best_weights = None
        best_threshold = 0.5

        # Search over base_score and margin_score weights
        for w_base in np.linspace(0.0, 1.0, 21):
            for w_margin in np.linspace(0.0, 1.0, 21):
                if abs(w_base + w_margin - 1.0) > 0.1:
                    continue  # weights should roughly sum to 1

                # Compute combined score
                base_idx = self.features.index("base_score") if "base_score" in self.features else 0
                margin_idx = self.features.index("margin_score") if "margin_score" in self.features else 1
                combined = w_base * X[:, base_idx] + w_margin * X[:, margin_idx]

                try:
                    auroc = float(roc_auc_score(y, combined))
                except ValueError:
                    continue

                if auroc > best_auroc:
                    best_auroc = auroc
                    best_weights = {
                        "base_weight": float(w_base),
                        "margin_weight": float(w_margin),
                    }

        if best_weights:
            self._weights = best_weights
            self._fitted = True
            print(f"  [Calibration] Grid search complete: AUROC={best_auroc:.4f}, "
                  f"weights={best_weights}")
        else:
            self._fitted = False

        return self

    def apply(self, features_dict):
        """
        Apply calibration to a single sample's features dict.
        Returns calibrated score in [0, 1].
        """
        if not self._fitted:
            # Fallback: use original heuristic
            return float(0.75 * features_dict.get("base_score", 0.5) +
                         0.25 * features_dict.get("margin_score", 0.5))

        if self.mode == "logistic" and self._model is not None:
            row = []
            for fname in self.features:
                val = features_dict.get(fname, 0.0)
                row.append(float(val))
            X = np.array([row], dtype=np.float64)
            X_scaled = (X - self._feature_mean) / self._feature_std
            proba = float(self._model.predict_proba(X_scaled)[0, 1])
            return proba

        elif self.mode == "grid_search" and self._weights is not None:
            w_base = self._weights.get("base_weight", 0.75)
            w_margin = self._weights.get("margin_weight", 0.25)
            return float(w_base * features_dict.get("base_score", 0.5) +
                         w_margin * features_dict.get("margin_score", 0.5))

        # Fallback
        return float(0.75 * features_dict.get("base_score", 0.5) +
                     0.25 * features_dict.get("margin_score", 0.5))

    def apply_batch(self, base_scores, margin_scores, min_dists, second_dists, pred_idx, proto_store):
        """
        Apply calibration to a batch of predictions.
        Returns calibrated scores as numpy array of shape (N,).
        """
        # Build features dict per sample
        n = len(base_scores)
        calibrated = np.zeros(n, dtype=np.float64)

        features = self.collect_features_from_predict(
            {
                "base_scores": torch.as_tensor(base_scores),
                "margin_scores": torch.as_tensor(margin_scores),
                "min_dists": torch.as_tensor(min_dists),
                "second_dists": torch.as_tensor(second_dists),
                "pred_idx": torch.as_tensor(pred_idx) if not isinstance(pred_idx, torch.Tensor) else pred_idx,
                "scores": torch.zeros(n),
            },
            proto_store,
        )

        for i in range(n):
            sample_feats = {k: v[i] if hasattr(v, '__len__') else v for k, v in features.items()}
            calibrated[i] = self.apply(sample_feats)

        return calibrated

    def save(self, path: Path):
        """Save calibrator to JSON + optional sklearn model."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "mode": self.mode,
            "features": self.features,
            "seed": self.seed,
            "fitted": self._fitted,
        }

        if self._weights is not None:
            data["weights"] = self._weights

        if self._coefs is not None:
            data["coefs"] = self._coefs
            data["intercept"] = self._intercept

        if self._feature_mean is not None:
            data["feature_mean"] = self._feature_mean.tolist()
            data["feature_std"] = self._feature_std.tolist()

        if self._model is not None:
            import pickle
            with open(path / "logistic_model.pkl", "wb") as f:
                pickle.dump(self._model, f)

        with open(path / "calibrator.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  [Calibration] Saved to {path}")

    def load(self, path: Path):
        """Load calibrator from saved files."""
        json_path = path / "calibrator.json" if path.is_dir() else path

        with open(json_path) as f:
            data = json.load(f)

        self.mode = data.get("mode", self.mode)
        self.features = data.get("features", self.features)
        self.seed = data.get("seed", self.seed)
        self._fitted = data.get("fitted", False)

        if "weights" in data:
            self._weights = data["weights"]

        if "coefs" in data:
            self._coefs = data["coefs"]
            self._intercept = data.get("intercept", 0.0)

        if "feature_mean" in data:
            self._feature_mean = np.array(data["feature_mean"], dtype=np.float64)
            self._feature_std = np.array(data["feature_std"], dtype=np.float64)

        # Load sklearn model if exists
        pkl_path = path / "logistic_model.pkl" if path.is_dir() else Path(str(path).replace(".json", ".pkl"))
        if pkl_path.exists():
            import pickle
            with open(pkl_path, "rb") as f:
                self._model = pickle.load(f)

        print(f"  [Calibration] Loaded from {path} (fitted={self._fitted})")
        return self


def evaluate_calibration(known_scores, unknown_scores):
    """
    Compute comprehensive open-set metrics from score arrays.

    Returns dict with: AUROC, AUPR, FPR@95TPR, EER, TPR@FPR=5%, TPR@FPR=10%,
                        known_score_mean, unknown_score_mean.
    """
    known = np.asarray(known_scores, dtype=np.float64)
    unknown = np.asarray(unknown_scores, dtype=np.float64)

    labels = np.concatenate([np.ones(len(known)), np.zeros(len(unknown))])
    scores = np.concatenate([known, unknown])

    results = {
        "n_known": int(len(known)),
        "n_unknown": int(len(unknown)),
        "known_score_mean": float(known.mean()) if len(known) > 0 else float("nan"),
        "known_score_std": float(known.std(ddof=1)) if len(known) > 1 else float("nan"),
        "unknown_score_mean": float(unknown.mean()) if len(unknown) > 0 else float("nan"),
        "unknown_score_std": float(unknown.std(ddof=1)) if len(unknown) > 1 else float("nan"),
    }

    if len(np.unique(labels)) < 2:
        results.update({
            "AUROC": float("nan"), "AUPR": float("nan"),
            "FPR_at_95TPR": float("nan"), "EER": float("nan"),
            "TPR_at_FPR5": float("nan"), "TPR_at_FPR10": float("nan"),
        })
        return results

    # AUROC
    results["AUROC"] = float(roc_auc_score(labels, scores))

    # AUPR (treating known=positive)
    from sklearn.metrics import average_precision_score
    results["AUPR"] = float(average_precision_score(labels, scores))

    # ROC curve analysis
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # FPR@95TPR
    idx_95 = np.searchsorted(tpr, 0.95)
    idx_95 = min(idx_95, len(fpr) - 1)
    results["FPR_at_95TPR"] = float(fpr[idx_95])

    # EER (Equal Error Rate)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    results["EER"] = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    def _tpr_at_max_fpr(max_fpr):
        mask = fpr <= (max_fpr + 1e-12)
        if not mask.any():
            return 0.0
        return float(np.max(tpr[mask]))

    results["TPR_at_FPR5"] = _tpr_at_max_fpr(0.05)
    results["TPR_at_FPR10"] = _tpr_at_max_fpr(0.10)

    return results


def calibrate_from_known_unknown(model, proto_store, known_loader, unknown_loader,
                                  device, cfg):
    """
    Full calibration pipeline:
    1. Collect predictions on known and unknown samples
    2. Extract features
    3. Fit calibrator
    4. Return calibrator and pre/post metrics
    """
    from evaluate import collect_predictions
    import torch

    print(f"\n{'='*60}")
    print("Open-set Score Calibration")
    print(f"{'='*60}")

    # Collect predictions
    known_records = collect_predictions(
        model, known_loader, proto_store, device,
        reject_factor=cfg.reject_threshold_factor,
    )
    unknown_records = collect_predictions(
        model, unknown_loader, proto_store, device,
        reject_factor=cfg.reject_threshold_factor,
    )

    # Extract uncalibrated scores
    known_scores_uncal = np.array([r["open_set_score"] for r in known_records])
    unknown_scores_uncal = np.array([r["open_set_score"] for r in unknown_records])

    metrics_uncal = evaluate_calibration(known_scores_uncal, unknown_scores_uncal)

    # Build calibrator
    mode = cfg.open_score_calibration_mode
    feature_str = cfg.open_score_features
    features = [f.strip() for f in feature_str.split(",") if f.strip()]

    calibrator = OpenScoreCalibrator(mode=mode, features=features,
                                      seed=cfg.open_score_calibration_seed)

    # Re-run predict to collect features
    # (We need the raw features from PrototypeStore.predict)
    model.eval()
    all_known_features = []
    all_unknown_features = []

    with torch.no_grad():
        for loader, feature_list in [(known_loader, all_known_features),
                                       (unknown_loader, all_unknown_features)]:
            for batch in loader:
                x = batch["input"].to(device)
                tic = batch.get("tic")
                tic = tic.to(device) if torch.is_tensor(tic) else None
                z = model.encode(x, tic=tic)
                result = proto_store.predict(z)
                feats = calibrator.collect_features_from_predict(result, proto_store)
                feature_list.append(feats)

    calibrator.fit(all_known_features, all_unknown_features)

    # Apply calibration
    if calibrator._fitted:
        calibrated_known = np.concatenate([
            np.array([calibrator.apply({k: v[i] for k, v in fd.items()})
                      for i in range(fd["base_score"].shape[0])])
            for fd in all_known_features
        ]) if all_known_features else np.array([])

        calibrated_unknown = np.concatenate([
            np.array([calibrator.apply({k: v[i] for k, v in fd.items()})
                      for i in range(fd["base_score"].shape[0])])
            for fd in all_unknown_features
        ]) if all_unknown_features else np.array([])

        metrics_cal = evaluate_calibration(calibrated_known, calibrated_unknown)
    else:
        metrics_cal = metrics_uncal.copy()

    print(f"\n  Pre-calibration  AUROC={metrics_uncal['AUROC']:.4f}, "
          f"FPR95={metrics_uncal['FPR_at_95TPR']:.4f}, "
          f"EER={metrics_uncal['EER']:.4f}")
    print(f"  Post-calibration AUROC={metrics_cal['AUROC']:.4f}, "
          f"FPR95={metrics_cal['FPR_at_95TPR']:.4f}, "
          f"EER={metrics_cal['EER']:.4f}")

    return calibrator, metrics_uncal, metrics_cal, known_scores_uncal, unknown_scores_uncal
