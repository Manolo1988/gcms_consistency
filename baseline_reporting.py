"""Saved diagnostics for the classification baseline (no training dependencies)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, log_loss,
)

from paper_protocol import save_json, write_rows_csv


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_classification_diagnostics(output_dir, role, metadata, sample_ids,
                                    targets, probabilities, label_names):
    """Export all rows in loader order, including per-class probabilities."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(label_names)
    labels = np.arange(len(names))
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not len(targets) or probabilities.shape != (len(targets), len(names)):
        raise ValueError("Empty predictions or probability/label shape mismatch")
    if len(sample_ids) != len(targets) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Prediction sample IDs must be unique and match targets")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probability rows must sum to one")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    if not np.isin(targets, labels).all():
        raise ValueError("Target outside the trained label mapping")
    if metadata["sample_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate sample IDs")
    frame = metadata.set_index("sample_id").loc[list(sample_ids)].reset_index()
    actual_names = np.asarray(names)[targets]
    if not np.array_equal(frame["product_fine"].astype(str).to_numpy(), actual_names):
        raise ValueError("Prediction targets disagree with sample metadata")

    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        targets, predictions, labels=labels, target_names=names,
        output_dict=True, zero_division=0,
    )
    # Only classes actually present in this split contribute to balanced accuracy.
    present = [name for name in names if report[name]["support"] > 0]
    metrics = {
        "n_samples": int(len(targets)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_acc": float(np.mean([report[name]["recall"] for name in present])),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "log_loss": float(log_loss(targets, probabilities, labels=labels)),
    }
    save_json(metrics, output_dir / f"{role}_metrics.json")
    save_json(report, output_dir / f"{role}_classification_report.json")
    write_rows_csv([
        {"product": name, "precision": report[name]["precision"],
         "recall": report[name]["recall"], "f1": report[name]["f1-score"],
         "support": int(report[name]["support"])} for name in names
    ], output_dir / f"{role}_per_class_metrics.csv")

    columns = [column for column in (
        "sample_id", "metadata_index", "record_id", "source_relpath", "batch_idx",
        "batch_name", "label_source", "label_independently_verified",
        "preprocessing_quality_flags",
    ) if column in frame.columns]
    rows = frame[columns].copy()
    rows["true_product"] = actual_names
    rows["predicted_product"] = np.asarray(names)[predictions]
    rows["correct"] = predictions == targets
    rows["confidence"] = probabilities.max(axis=1)
    for index, name in enumerate(names):
        rows[f"probability_{name}"] = probabilities[:, index]
    rows.to_csv(output_dir / f"{role}_predictions.csv", index=False, encoding="utf-8-sig")
    rows.loc[~rows["correct"]].to_csv(
        output_dir / f"{role}_errors.csv", index=False, encoding="utf-8-sig",
    )
    if "batch_name" in rows:
        write_rows_csv([
            {"batch_name": str(batch), "n_samples": int(len(group)),
             "accuracy": float(group["correct"].mean())}
            for batch, group in rows.groupby("batch_name", sort=True)
        ], output_dir / f"{role}_per_batch_metrics.csv")

    matrix = confusion_matrix(targets, predictions, labels=labels)
    normalized = np.divide(
        matrix, matrix.sum(axis=1, keepdims=True),
        out=np.zeros(matrix.shape, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) != 0,
    )
    for suffix, values in (("", matrix), ("_normalized", normalized)):
        table = pd.DataFrame(values, index=names, columns=names)
        table.index.name = "true_product"
        table.to_csv(output_dir / f"{role}_confusion_matrix{suffix}.csv", encoding="utf-8-sig")
    _plot_confusion(output_dir / f"{role}_confusion_matrix.png", matrix, normalized, names)
    return metrics


def _plot_confusion(path, matrix, normalized, names):
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for axis, values, title in zip(axes, (matrix, normalized), ("Sample counts", "Recall by true class")):
        image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if values is normalized else None)
        axis.set(xticks=range(len(names)), yticks=range(len(names)),
                 xticklabels=names, yticklabels=names, xlabel="Predicted product",
                 ylabel="True product", title=title)
        for i in range(len(names)):
            for j in range(len(names)):
                text = f"{values[i, j]:.1%}" if values is normalized else str(values[i, j])
                axis.text(j, i, text, ha="center", va="center", fontsize=8,
                          color="white" if values[i, j] > values.max() / 2 else "black")
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_training_curves(history, output_dir):
    """Validation points are saved at the original checkpoint cadence."""
    output_dir = Path(output_dir)
    save_json(history, output_dir / "training_history.json")
    write_rows_csv(history, output_dir / "training_history.csv")
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    for axis, key, label in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        train_rows = [row for row in history if row.get(f"train_{key}") is not None]
        val_rows = [row for row in history if row.get(f"val_{key}") is not None]
        axis.plot([r["epoch"] for r in train_rows], [r[f"train_{key}"] for r in train_rows], label="Train (sampled, dropout on)")
        axis.plot([r["epoch"] for r in val_rows], [r[f"val_{key}"] for r in val_rows], marker="o", label="Validation")
        axis.set(xlabel="Epoch", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    if epochs:
        figure.suptitle(f"Best checkpoint: epoch {history[-1]['best_epoch']} (validation accuracy)")
    figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)


def summarize_fewshot(rows, output_dir):
    """Episode variation is descriptive; episodes reuse samples and one model."""
    frame = pd.DataFrame(rows)
    summary = []
    for (shot, scope, product), group in frame.groupby(["shot", "scope", "product"], sort=True):
        summary.append({
            "shot": int(shot), "scope": scope, "product": product,
            "episodes": int(len(group)), "mean_accuracy": float(group["accuracy"].mean()),
            "episode_std": float(group["accuracy"].std(ddof=0)),
            "min_accuracy": float(group["accuracy"].min()),
            "max_accuracy": float(group["accuracy"].max()),
        })
    write_rows_csv(summary, Path(output_dir) / "fewshot_summary.csv")
    return summary


def save_comparison(rows, output_dir):
    output_dir = Path(output_dir)
    write_rows_csv(rows, output_dir / "comparison_metrics.csv")
    save_json(rows, output_dir / "comparison_metrics.json")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    positions = np.arange(len(rows))
    for offset, (key, title) in enumerate((
        ("accuracy", "Accuracy"), ("balanced_acc", "Balanced accuracy"), ("macro_f1", "Macro F1"),
    )):
        bars = axis.bar(positions + (offset - 1) * 0.25,
                        [row[key] for row in rows], width=0.25, label=title)
        axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    axis.set(xticks=positions, xticklabels=[row["representation"] for row in rows],
             ylim=(0, 1.12), ylabel="Known test score", title="Preprocessing comparison (no augmentation)")
    axis.legend(loc="upper center", ncol=3, fontsize=8)
    figure.savefig(output_dir / "comparison_metrics.png", dpi=150)
    plt.close(figure)
