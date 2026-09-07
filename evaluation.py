"""Common evaluation for closed-set recognition and incremental registration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from protocol import ProtocolSpec, make_fewshot_episode, metadata_fingerprint


@dataclass
class FeatureTable:
    sample_ids: np.ndarray
    values: np.ndarray
    metadata_sha256: str | None = None
    metric: str = "cosine"

    def __post_init__(self) -> None:
        self.sample_ids = np.asarray(self.sample_ids).astype(str)
        self.values = np.asarray(self.values, dtype=np.float32)
        if self.values.ndim != 2 or len(self.sample_ids) != len(self.values):
            raise ValueError("features must be a 2-D array aligned with sample_ids")
        if len(set(self.sample_ids.tolist())) != len(self.sample_ids):
            raise ValueError("feature sample_ids must be unique")
        if self.metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be cosine or euclidean")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureTable":
        with np.load(path, allow_pickle=False) as data:
            key = "features" if "features" in data.files else "embeddings"
            id_key = "sample_key" if "sample_key" in data.files else "sample_id"
            if key not in data.files or id_key not in data.files:
                raise ValueError("feature npz requires sample_key and features/embeddings")
            fingerprint = None
            if "metadata_sha256" in data.files:
                fingerprint = str(np.asarray(data["metadata_sha256"]).item())
            metric = str(np.asarray(data["metric"]).item()) if "metric" in data.files else "cosine"
            return cls(data[id_key], data[key], fingerprint, metric)

    def align(self, df: pd.DataFrame, sample_col: str = "_sample_key") -> np.ndarray:
        positions = {sample_id: i for i, sample_id in enumerate(self.sample_ids)}
        key_col = "_sample_key" if "_sample_key" in df else sample_col
        missing = [s for s in df[key_col].astype(str) if s not in positions]
        if missing:
            raise ValueError(f"features missing {len(missing)} metadata samples; first={missing[0]}")
        order = [positions[s] for s in df[key_col].astype(str)]
        return self.values[order]


def save_feature_table(
    path: str | Path,
    sample_ids: Iterable[str],
    features: np.ndarray,
    metadata_sha256: str,
    metric: str = "cosine",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_key=np.asarray(list(sample_ids), dtype=str),
        features=np.asarray(features, dtype=np.float32),
        metadata_sha256=np.asarray(metadata_sha256),
        metric=np.asarray(metric),
    )


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fit_mean_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(str)
    classes = np.asarray(sorted(np.unique(labels)), dtype=str)
    values = _normalize(features) if metric == "cosine" else np.asarray(features, dtype=np.float64)
    prototypes = np.stack([values[labels == cls].mean(axis=0) for cls in classes])
    if metric == "cosine":
        prototypes = _normalize(prototypes)
    return classes, prototypes


def predict_prototypes(
    features: np.ndarray,
    classes: np.ndarray,
    prototypes: np.ndarray,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    if metric == "cosine":
        scores = _normalize(features) @ _normalize(prototypes).T
    elif metric == "euclidean":
        delta = np.asarray(features, dtype=np.float64)[:, None, :] - prototypes[None, :, :]
        scores = -np.sum(delta * delta, axis=2)
    else:
        raise ValueError(f"unsupported prototype metric: {metric}")
    indices = np.argmax(scores, axis=1)
    return classes[indices], scores[np.arange(len(indices)), indices]


def _confusion_matrix(truth: np.ndarray, predicted: np.ndarray, labels: list[str]) -> np.ndarray:
    positions = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        if actual in positions and guess in positions:
            matrix[positions[actual], positions[guess]] += 1
    return matrix


def classification_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    labels: Iterable[str] | None = None,
) -> dict:
    truth = np.asarray(truth).astype(str)
    predicted = np.asarray(predicted).astype(str)
    labels = [str(v) for v in labels] if labels is not None else sorted(set(truth) | set(predicted))
    matrix = _confusion_matrix(truth, predicted, labels)
    support = np.asarray([(truth == label).sum() for label in labels], dtype=float)
    predicted_count = np.asarray([(predicted == label).sum() for label in labels], dtype=float)
    true_positive = np.asarray([
        ((truth == label) & (predicted == label)).sum() for label in labels
    ], dtype=float)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    precision = np.divide(
        true_positive, predicted_count, out=np.zeros_like(true_positive), where=predicted_count > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    present = support > 0
    return {
        "accuracy": float(np.mean(truth == predicted)) if len(truth) else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "per_class_recall": {str(k): float(v) for k, v in zip(labels, recall)},
        "labels": labels,
        "confusion_matrix": matrix.astype(int).tolist(),
        "n": int(len(truth)),
    }


def _metric_only(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    labels = sorted(set(np.asarray(truth).astype(str)))
    metrics = classification_metrics(truth, predicted, labels=labels)
    return {key: metrics[key] for key in ("accuracy", "macro_f1", "balanced_accuracy")}


def grouped_bootstrap_ci(
    truth: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    repeats: int = 2000,
    seed: int = 42,
    labels: Iterable[str] | None = None,
) -> dict:
    """Bootstrap independent lots/groups, not individual repeated measurements."""
    truth = np.asarray(truth).astype(str)
    predicted = np.asarray(predicted).astype(str)
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or repeats < 2:
        return {}
    group_rows = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.RandomState(seed)
    values = {key: [] for key in ("accuracy", "macro_f1", "balanced_accuracy")}
    for _ in range(repeats):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled])
        metrics = classification_metrics(truth[rows], predicted[rows], labels=labels)
        for key in values:
            values[key].append(metrics[key])
    return {
        key: {
            "low": float(np.percentile(vals, 2.5)),
            "high": float(np.percentile(vals, 97.5)),
            "repeats": int(repeats),
            "unit": "independent_group",
        }
        for key, vals in values.items()
    }


def evaluate_closed_set(
    df: pd.DataFrame,
    features: np.ndarray,
    manifest: dict,
    spec: ProtocolSpec,
    bootstrap_repeats: int = 2000,
    metric: str = "cosine",
) -> tuple[dict, np.ndarray]:
    train_idx = np.asarray(manifest["train_idx"], dtype=int)
    test_idx = np.asarray(manifest["test_batch_idx"], dtype=int)
    classes, prototypes = fit_mean_prototypes(
        features[train_idx], df.iloc[train_idx][spec.product_col].to_numpy(), metric
    )
    truth = df.iloc[test_idx][spec.product_col].astype(str).to_numpy()
    predicted, _ = predict_prototypes(features[test_idx], classes, prototypes, metric)
    metrics = classification_metrics(truth, predicted, labels=manifest["known_products"])
    metrics["ci95"] = grouped_bootstrap_ci(
        truth,
        predicted,
        df.iloc[test_idx]["_group_id"].to_numpy(),
        repeats=bootstrap_repeats,
        seed=spec.seed,
        labels=manifest["known_products"],
    )
    metrics["test_batches"] = manifest["holdout_batches"]
    return metrics, predicted


def _harmonic_mean(a: float, b: float) -> float:
    return float(2 * a * b / (a + b)) if a + b > 0 else 0.0


def evaluate_registration_episode(
    df: pd.DataFrame,
    features: np.ndarray,
    manifest: dict,
    episode: dict,
    spec: ProtocolSpec,
    metric: str = "cosine",
) -> dict:
    base_train = np.asarray(manifest["train_idx"], dtype=int)
    support = np.asarray(episode["support_idx"], dtype=int)
    base_query = np.asarray(manifest["test_batch_idx"], dtype=int)
    novel_query = np.asarray(episode["novel_query_idx"], dtype=int)

    start = time.perf_counter()
    base_classes, base_prototypes = fit_mean_prototypes(
        features[base_train], df.iloc[base_train][spec.product_col].to_numpy(), metric
    )
    novel_classes, novel_prototypes = fit_mean_prototypes(
        features[support], df.iloc[support][spec.product_col].to_numpy(), metric
    )
    classes = np.concatenate([base_classes, novel_classes])
    prototypes = np.concatenate([base_prototypes, novel_prototypes], axis=0)
    registration_ms = (time.perf_counter() - start) * 1000.0

    query = np.concatenate([base_query, novel_query])
    truth = df.iloc[query][spec.product_col].astype(str).to_numpy()
    predicted, _ = predict_prototypes(features[query], classes, prototypes, metric)
    is_base = np.isin(truth, base_classes)
    base_metrics = _metric_only(truth[is_base], predicted[is_base])
    novel_metrics = _metric_only(truth[~is_base], predicted[~is_base])
    all_metrics = classification_metrics(truth, predicted, labels=classes.tolist())
    old_to_new = float(np.mean(np.isin(predicted[is_base], novel_classes))) if is_base.any() else np.nan
    new_to_old = float(np.mean(np.isin(predicted[~is_base], base_classes))) if (~is_base).any() else np.nan
    return {
        "episode_seed": episode["seed"],
        "shot": episode["shot"],
        "base_macro_f1": base_metrics["macro_f1"],
        "novel_macro_f1": novel_metrics["macro_f1"],
        "all_macro_f1": all_metrics["macro_f1"],
        "base_balanced_accuracy": base_metrics["balanced_accuracy"],
        "novel_balanced_accuracy": novel_metrics["balanced_accuracy"],
        "all_balanced_accuracy": all_metrics["balanced_accuracy"],
        "harmonic_macro_f1": _harmonic_mean(base_metrics["macro_f1"], novel_metrics["macro_f1"]),
        "old_to_new_error_rate": old_to_new,
        "new_to_old_error_rate": new_to_old,
        "registration_ms": float(registration_ms),
        "n_support": int(len(support)),
        "n_base_query": int(len(base_query)),
        "n_novel_query": int(len(novel_query)),
        "details": episode["details"],
    }


def summarize_episodes(rows: list[dict]) -> dict:
    scalar_keys = [
        "base_macro_f1", "novel_macro_f1", "all_macro_f1",
        "base_balanced_accuracy", "novel_balanced_accuracy", "all_balanced_accuracy",
        "harmonic_macro_f1", "old_to_new_error_rate", "new_to_old_error_rate",
        "registration_ms",
    ]
    summary = {"episodes": len(rows)}
    for key in scalar_keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
            "ci95_low": float(np.nanpercentile(values, 2.5)),
            "ci95_high": float(np.nanpercentile(values, 97.5)),
        }
    return summary


def evaluate_method(
    df: pd.DataFrame,
    feature_table: FeatureTable,
    manifest: dict,
    spec: ProtocolSpec,
    method_name: str,
    bootstrap_repeats: int = 2000,
) -> tuple[dict, pd.DataFrame]:
    fingerprint = metadata_fingerprint(df, spec)
    if feature_table.metadata_sha256 and feature_table.metadata_sha256 != fingerprint:
        raise ValueError("feature file was produced from different metadata")
    if manifest.get("metadata_sha256") != fingerprint:
        raise ValueError("manifest was produced from different metadata")
    features = feature_table.align(df)
    closed, _ = evaluate_closed_set(
        df, features, manifest, spec, bootstrap_repeats, feature_table.metric
    )
    episode_rows = []
    fewshot = {}
    for shot in spec.shots:
        shot_rows = []
        for episode_index in range(spec.episodes):
            seed = spec.seed + episode_index + int(shot) * 100_000
            episode = make_fewshot_episode(df, manifest, shot, seed, spec)
            row = evaluate_registration_episode(
                df, features, manifest, episode, spec, feature_table.metric
            )
            row["method"] = method_name
            row["fold_id"] = manifest["fold_id"]
            row["episode"] = episode_index
            shot_rows.append(row)
            episode_rows.append({k: v for k, v in row.items() if k != "details"})
        fewshot[str(shot)] = summarize_episodes(shot_rows)
    result = {
        "method": method_name,
        "fold_id": manifest["fold_id"],
        "closed_set": closed,
        "fewshot_registration": fewshot,
        "protocol": manifest["fewshot"],
        "prototype_metric": feature_table.metric,
    }
    return result, pd.DataFrame(episode_rows)


def save_evaluation(
    result: dict,
    episodes: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    episodes.to_csv(output / "episodes.csv", index=False)


def paired_method_differences(episodes: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Paired episode deltas; pairing prevents support-draw luck from driving claims."""
    keys = ["fold_id", "shot", "episode", "episode_seed"]
    if "training_seed" in episodes.columns:
        keys.append("training_seed")
    metrics = ["base_macro_f1", "novel_macro_f1", "all_macro_f1", "harmonic_macro_f1"]
    reference_rows = episodes[episodes["method"] == reference][keys + metrics]
    if reference_rows.empty:
        raise ValueError(f"reference method not found: {reference}")
    output = []
    for method in sorted(set(episodes["method"]) - {reference}):
        candidate = episodes[episodes["method"] == method][keys + metrics]
        paired = candidate.merge(reference_rows, on=keys, suffixes=("", "_reference"))
        for shot, block in paired.groupby("shot"):
            for metric in metrics:
                delta = block[metric] - block[f"{metric}_reference"]
                output.append({
                    "method": method,
                    "reference": reference,
                    "shot": int(shot),
                    "metric": metric,
                    "pairs": int(len(delta)),
                    "delta_mean": float(delta.mean()),
                    "delta_ci95_low": float(delta.quantile(0.025)),
                    "delta_ci95_high": float(delta.quantile(0.975)),
                })
    return pd.DataFrame(output)
