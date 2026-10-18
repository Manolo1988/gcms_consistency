"""Small synthetic checks for saved metrics and sample/split alignment; no torch."""

import importlib.util
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import shutil
import unittest
from uuid import uuid4

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_reporting import (
    save_classification_diagnostics, save_comparison, save_training_curves,
)

spec = importlib.util.spec_from_file_location("comparison_runner", ROOT / "scripts/run_preprocessing_comparison.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@contextmanager
def temporary_directory():
    # Inherit workspace permissions on Windows instead of tempfile's mode=0o700.
    path = ROOT / f".test_baseline_{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent != ROOT.resolve() or not resolved.name.startswith(".test_baseline_"):
            raise RuntimeError("Refusing to remove a directory outside this test workspace")
        shutil.rmtree(resolved)


class ClassificationReportingTest(unittest.TestCase):
    def test_imbalanced_metrics_prediction_alignment_and_confusion(self):
        ids = [f"S{i}" for i in range(6)]
        frame = pd.DataFrame({
            "sample_id": ids, "product_fine": ["A"] * 5 + ["B"],
            "batch_name": ["batch1"] * 3 + ["batch2"] * 3,
            "preprocessing_quality_flags": [""] * 5 + ["label_ambiguous"],
        }).iloc[::-1]  # Metadata order must not silently dictate prediction order.
        probabilities = np.tile([0.9, 0.1], (6, 1))
        with temporary_directory() as directory:
            path = Path(directory)
            metrics = save_classification_diagnostics(
                path, "test", frame, ids, [0] * 5 + [1], probabilities, ["A", "B"],
            )
            self.assertAlmostEqual(metrics["accuracy"], 5 / 6)
            self.assertAlmostEqual(metrics["balanced_acc"], 0.5)
            self.assertAlmostEqual(metrics["macro_f1"], 5 / 11)
            predictions = pd.read_csv(path / "test_predictions.csv")
            self.assertEqual(predictions["sample_id"].tolist(), ids)
            errors = pd.read_csv(path / "test_errors.csv")
            self.assertEqual(errors["sample_id"].tolist(), ["S5"])
            self.assertEqual(errors.iloc[0]["preprocessing_quality_flags"], "label_ambiguous")
            per_class = pd.read_csv(path / "test_per_class_metrics.csv").set_index("product")
            self.assertEqual(per_class.loc["B", "support"], 1)
            self.assertEqual(per_class.loc["B", "precision"], 0)
            matrix = pd.read_csv(path / "test_confusion_matrix.csv", index_col=0)
            self.assertEqual(matrix.to_numpy().tolist(), [[5, 0], [1, 0]])
            normalized = pd.read_csv(path / "test_confusion_matrix_normalized.csv", index_col=0)
            np.testing.assert_allclose(normalized.sum(axis=1), 1)
            self.assertTrue((path / "test_confusion_matrix.png").is_file())

    def test_wrong_sample_labels_fail_instead_of_saving_plausible_scores(self):
        frame = pd.DataFrame({"sample_id": ["S0"], "product_fine": ["B"]})
        with temporary_directory() as directory:
            with self.assertRaisesRegex(ValueError, "disagree"):
                save_classification_diagnostics(
                    directory, "test", frame, ["S0"], [0], [[0.9, 0.1]], ["A", "B"],
                )

    def test_curves_support_sparse_validation_points_and_comparison_plot(self):
        history = [{
            "epoch": epoch, "train_loss": 1 / epoch, "train_accuracy": 0.5 + epoch / 50,
            "val_loss": 0.7 if epoch == 10 else None,
            "val_accuracy": 0.6 if epoch == 10 else None,
            "best_epoch": 10 if epoch == 10 else None,
        } for epoch in range(1, 11)]
        with temporary_directory() as directory:
            path = Path(directory)
            save_training_curves(history, path)
            saved = json.loads((path / "training_history.json").read_text())
            self.assertIsNone(saved[0]["val_accuracy"])
            self.assertEqual(saved[-1]["val_accuracy"], 0.6)
            self.assertTrue((path / "training_curves.png").is_file())
            save_comparison([{"representation": "example", "accuracy": 0.7,
                              "balanced_acc": 0.6, "macro_f1": 0.5}], path)
            self.assertTrue((path / "comparison_metrics.png").is_file())


class FrozenSplitTest(unittest.TestCase):
    def test_same_fingerprint_cannot_hide_changed_labels(self):
        frame = pd.DataFrame({
            "sample_id": ["S0", "S1", "S2", "S3"],
            "product_fine": ["A", "A", "A", "B"],
            "batch_idx": [0, 1, 2, 3], "is_special": [False] * 4,
        })
        split = {key: [i] for i, key in enumerate(runner.ROLE_KEYS)}
        split.update(manifest_fingerprint="fixture", fixed_split=True, stage="prepared")
        with temporary_directory() as directory:
            root = Path(directory)
            (root / "split.json").write_text(json.dumps(split))
            frame.to_csv(root / "metadata.csv", index=False)
            for name in runner.VARIANTS:
                variant = root / "representations" / name
                variant.mkdir(parents=True)
                (variant / "split.json").write_text(json.dumps(split))
                (variant / "grid_info.json").write_text(json.dumps({
                    "input_representation": name, "source_manifest_fingerprint": "fixture",
                }))
                frame.to_csv(variant / "metadata.csv", index=False)
            runner.validate_inputs(root)
            changed = frame.copy()
            changed.loc[0, "product_fine"] = "B"
            changed.to_csv(root / "representations/dual_channel_pca/metadata.csv", index=False)
            with self.assertRaisesRegex(ValueError, "labels differ"):
                runner.validate_inputs(root)


if __name__ == "__main__":
    unittest.main()
