import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from closeandfew.evaluation import FeatureTable, evaluate_method
from closeandfew.protocol import (
    ProtocolSpec,
    build_protocol_manifests,
    load_metadata,
    make_fewshot_episode,
)


def synthetic_metadata(path: Path) -> pd.DataFrame:
    rows = []
    counter = 0
    for batch_number in range(1, 7):
        batch = f"20260{batch_number:02d}01"
        for product in ("A", "B", "C", "N1", "N2"):
            for replicate in range(6):
                counter += 1
                rows.append({
                    "sample_id": f"S{counter:04d}",
                    "product_fine": product,
                    "batch_name": batch,
                    "batch_idx": batch,
                    "lot_id": f"{product}-{batch}-{replicate}",
                    "is_special": False,
                    "tensor_path": f"missing/{counter}.npz",
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.metadata = root / "metadata.csv"
        synthetic_metadata(self.metadata)
        self.output = root / "manifests"
        self.spec = ProtocolSpec(
            max_product_folds=1,
            min_samples_per_product=1,
            min_batches_per_product=3,
            min_closed_test_classes=2,
            min_closed_test_samples=1,
            closed_test_batch_ratio=0.2,
            validation_batch_ratio=0.2,
            shots=(1, 3, 5),
            episodes=4,
            min_novel_query_per_class=5,
            seed=17,
        )
        self.manifest = build_protocol_manifests(
            self.metadata, self.output, self.spec,
            novel_product_sets=[("N1", "N2")],
        )[0]
        self.df = load_metadata(self.metadata, self.spec)

    def tearDown(self):
        self.temporary.cleanup()

    def test_closed_partitions_are_temporal_batch_and_group_disjoint(self):
        parts = {
            name: set(self.manifest[key])
            for name, key in {
                "train": "train_idx",
                "validation": "val_idx",
                "test": "test_batch_idx",
            }.items()
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(parts[left] & parts[right])
            left_batches = set(self.df.iloc[sorted(parts[left])].batch_name)
            right_batches = set(self.df.iloc[sorted(parts[right])].batch_name)
            self.assertFalse(left_batches & right_batches)
            left_groups = set(self.df.iloc[sorted(parts[left])]._group_id)
            right_groups = set(self.df.iloc[sorted(parts[right])]._group_id)
            self.assertFalse(left_groups & right_groups)
        self.assertLess(
            max(self.df.iloc[self.manifest["train_idx"]].batch_name),
            min(self.df.iloc[self.manifest["val_idx"]].batch_name),
        )
        self.assertLess(
            max(self.df.iloc[self.manifest["val_idx"]].batch_name),
            min(self.df.iloc[self.manifest["test_batch_idx"]].batch_name),
        )
        train_products = set(self.df.iloc[self.manifest["train_idx"]].product_fine)
        self.assertFalse(train_products & {"N1", "N2"})

    def test_fewshot_episodes_have_exact_independent_support(self):
        for shot in (1, 3, 5):
            first = make_fewshot_episode(self.df, self.manifest, shot, 101, self.spec)
            second = make_fewshot_episode(self.df, self.manifest, shot, 101, self.spec)
            self.assertEqual(first, second)
            support = self.df.iloc[first["support_idx"]]
            query = self.df.iloc[first["novel_query_idx"]]
            self.assertFalse(set(support.batch_name) & set(query.batch_name))
            self.assertFalse(set(support._group_id) & set(query._group_id))
            counts = support.groupby("product_fine")._group_id.nunique().to_dict()
            self.assertEqual(counts, {"N1": shot, "N2": shot})

    def test_joint_base_and_novel_evaluator(self):
        labels = sorted(self.df.product_fine.unique())
        label_to_index = {label: i for i, label in enumerate(labels)}
        features = np.zeros((len(self.df), len(labels)), dtype=np.float32)
        for index, label in enumerate(self.df.product_fine):
            features[index, label_to_index[label]] = 1.0
        table = FeatureTable(
            self.df._sample_key.to_numpy(), features,
            self.manifest["metadata_sha256"], "cosine",
        )
        result, episodes = evaluate_method(
            self.df, table, self.manifest, self.spec, "perfect", bootstrap_repeats=10
        )
        self.assertEqual(result["closed_set"]["accuracy"], 1.0)
        self.assertTrue((episodes.old_to_new_error_rate == 0.0).all())
        self.assertTrue((episodes.new_to_old_error_rate == 0.0).all())
        self.assertTrue((episodes.harmonic_macro_f1 == 1.0).all())

    def test_feature_fingerprint_mismatch_is_rejected(self):
        features = np.ones((len(self.df), 2), dtype=np.float32)
        table = FeatureTable(self.df._sample_key.to_numpy(), features, "wrong")
        with self.assertRaisesRegex(ValueError, "different metadata"):
            evaluate_method(
                self.df, table, self.manifest, self.spec, "wrong", bootstrap_repeats=0
            )

    def test_base_metrics_count_predictions_to_novel_as_errors(self):
        from closeandfew.evaluation import classification_metrics

        metrics = classification_metrics(
            np.asarray(["A", "A", "B", "B"]),
            np.asarray(["N1", "A", "B", "N1"]),
            labels=["A", "B"],
        )
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["per_class_recall"], {"A": 0.5, "B": 0.5})

    def test_manifest_and_audit_are_serializable(self):
        audit = json.loads((self.output / "data_audit.json").read_text())
        self.assertEqual(audit["manifests"], [self.manifest["fold_id"]])
        json.dumps(self.manifest, allow_nan=False)

    def test_automatic_novel_folds_are_product_disjoint(self):
        manifests = build_protocol_manifests(
            self.metadata,
            Path(self.temporary.name) / "automatic",
            replace(self.spec, max_product_folds=2),
        )
        products = [
            product
            for manifest in manifests
            for product in manifest["holdout_products"]
        ]
        self.assertEqual(len(products), len(set(products)))


if __name__ == "__main__":
    unittest.main()
