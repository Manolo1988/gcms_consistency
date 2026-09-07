"""Leakage-controlled traditional baselines for the paper experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import extract_tic_matrix
from evaluation import classification_metrics, save_feature_table
from protocol import ProtocolSpec, metadata_fingerprint


def _require_sklearn():
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.covariance import LedoitWolf
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise RuntimeError(
            "traditional baselines require scikit-learn; install requirements.txt"
        ) from exc
    return StandardScaler, PCA, LedoitWolf, PLSRegression, SVC


def _mahalanobis_embedding(
    train: np.ndarray,
    all_values: np.ndarray,
    labels: np.ndarray,
    ledoit_wolf,
) -> np.ndarray:
    residuals = []
    for label in np.unique(labels):
        block = train[labels == label]
        residuals.append(block - block.mean(axis=0, keepdims=True))
    covariance = ledoit_wolf().fit(np.concatenate(residuals, axis=0)).covariance_
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues.max()) * 1e-6, 1e-8)
    whitening = eigenvectors @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, floor)))
    center = train.mean(axis=0, keepdims=True)
    return (all_values - center) @ whitening


def run_traditional_baselines(
    df: pd.DataFrame,
    manifest: dict,
    tensor_root: str | Path | None,
    output_dir: str | Path,
    spec: ProtocolSpec,
    pca_components: int = 64,
    seed: int = 42,
) -> dict[str, Path]:
    """Fit all preprocessing on training rows and export reusable feature spaces."""
    StandardScaler, PCA, LedoitWolf, PLSRegression, SVC = _require_sklearn()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_idx = np.asarray(manifest["train_idx"], dtype=int)
    val_idx = np.asarray(manifest["val_idx"], dtype=int)
    test_idx = np.asarray(manifest["test_batch_idx"], dtype=int)
    labels = df[spec.product_col].astype(str).to_numpy()
    fingerprint = metadata_fingerprint(df, spec)

    tic = extract_tic_matrix(df, tensor_root, spec)
    scaler = StandardScaler().fit(tic[train_idx])
    scaled = scaler.transform(tic)
    n_components = min(pca_components, len(train_idx) - 1, scaled.shape[1])
    if n_components < 1:
        raise ValueError("not enough training samples for PCA")
    pca = PCA(n_components=n_components, random_state=seed).fit(scaled[train_idx])
    pca_values = pca.transform(scaled)

    feature_paths: dict[str, Path] = {}
    feature_paths["tic_pca_proto"] = output / "tic_pca_proto.npz"
    save_feature_table(
        feature_paths["tic_pca_proto"], df["_sample_key"], pca_values,
        fingerprint, metric="euclidean",
    )

    mahalanobis = _mahalanobis_embedding(
        pca_values[train_idx], pca_values, labels[train_idx], LedoitWolf
    )
    feature_paths["tic_pca_mahalanobis"] = output / "tic_pca_mahalanobis.npz"
    save_feature_table(
        feature_paths["tic_pca_mahalanobis"], df["_sample_key"], mahalanobis,
        fingerprint, metric="euclidean",
    )

    classes = np.asarray(manifest["known_products"], dtype=str)
    class_to_index = {label: i for i, label in enumerate(classes)}
    y_train = np.asarray([class_to_index[label] for label in labels[train_idx]])
    one_hot = np.eye(len(classes), dtype=np.float64)[y_train]
    pls_components = min(max(len(classes) - 1, 1), 16, scaled.shape[1], len(train_idx) - 1)
    pls = PLSRegression(n_components=pls_components, scale=False).fit(
        scaled[train_idx], one_hot
    )
    pls_values = pls.transform(scaled)
    feature_paths["tic_plsda_latent"] = output / "tic_plsda_latent.npz"
    save_feature_table(
        feature_paths["tic_plsda_latent"], df["_sample_key"], pls_values,
        fingerprint, metric="euclidean",
    )

    svm = SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=seed)
    svm.fit(pca_values[train_idx], labels[train_idx])
    native = {}
    for name, indices in (("validation", val_idx), ("closed_test", test_idx)):
        svm_pred = svm.predict(pca_values[indices])
        pls_pred = classes[np.argmax(pls.predict(scaled[indices]), axis=1)]
        native[name] = {
            "tic_pca_svm": classification_metrics(
                labels[indices], svm_pred, labels=classes
            ),
            "tic_plsda": classification_metrics(
                labels[indices], pls_pred, labels=classes
            ),
        }
    provenance = {
        "fit_rows": train_idx.astype(int).tolist(),
        "fit_partition": "train_idx only",
        "pca_components": int(n_components),
        "pca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "pls_components": int(pls_components),
        "seed": int(seed),
        "native_closed_classifiers": native,
    }
    (output / "baseline_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return feature_paths
