"""Macro-F1 model selection contract for the unified training entry point."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from sklearn.metrics import f1_score
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config, restore_saved_config
from light_cnn_protocol import build_attention_ablation_config, build_config, frozen_setting_b


def load_function(filename, name, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace[name]


class MacroF1SelectionTest(unittest.TestCase):
    def test_macro_f1_includes_all_declared_classes(self):
        score_fn = load_function(
            "train.py", "_macro_f1_score", {"f1_score": f1_score, "np": np}
        )
        score = score_fn([0, 0, 1], [0, 1, 1], 3)
        expected = f1_score(
            [0, 0, 1], [0, 1, 1], labels=[0, 1, 2],
            average="macro", zero_division=0,
        )
        self.assertAlmostEqual(score, expected)
        self.assertAlmostEqual(score, 4 / 9)

    def test_macro_f1_can_select_a_checkpoint(self):
        select_fn = load_function("train.py", "_select_checkpoint_score", {"np": np})
        score, key = select_fn(
            {"macro_f1": 0.72}, SimpleNamespace(model_select_metric="macro_f1")
        )
        self.assertEqual(key, "macro_f1")
        self.assertEqual(score, 0.72)

    def test_actual_training_selection_breaks_ties_by_balanced_accuracy(self):
        improved = load_function("train.py", "_checkpoint_improved", {})
        cfg = SimpleNamespace(evaluation_protocol="light_cnn_v1")
        for metrics, expected in [
            ({"macro_f1": .7, "balanced_accuracy": 1.}, False),
            ({"macro_f1": .8, "balanced_accuracy": .7}, True),
            ({"macro_f1": .8, "balanced_accuracy": .6}, False),
        ]:
            self.assertEqual(improved(metrics, cfg, metrics["macro_f1"], .8, .6, .1), expected)
        standard = SimpleNamespace(evaluation_protocol="standard")
        self.assertFalse(improved({}, standard, .85, .8, 0., .1))


class UnifiedAttentionExperimentTest(unittest.TestCase):
    def test_preset_changes_only_depth_and_reporting_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            old = build_config(directory, directory, "A_original_2loss")
            middle = build_attention_ablation_config(directory, directory)
        changed = {k for k, v in vars(old).items() if v != getattr(middle, k)}
        self.assertEqual(changed, {
            "blocks_per_stage", "experiment_preset", "evaluation_protocol",
            "open_score_base_weight", "open_score_margin_weight", "evaluate_save_visualizations",
        })
        self.assertEqual((middle.main_backbone, middle.blocks_per_stage), ("gcms", 1))

    def test_restore_prefers_saved_model_and_preserves_relocated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "final_model"
            model_dir.mkdir()
            (root / "run_config.json").write_text(json.dumps({"config": {
                "blocks_per_stage": 2, "main_backbone": "plain_cnn",
            }}), encoding="utf-8")
            meta = model_dir / "train_meta.json"
            meta.write_text(json.dumps({"config": {
                "blocks_per_stage": 1, "main_backbone": "gcms", "output_dir": "/old/server/path",
                "_private": "do not restore", "evaluation_protocol": "light_cnn_v1",
            }}), encoding="utf-8")
            cfg = Config(output_dir=str(model_dir))
            self.assertEqual(restore_saved_config(cfg), str(meta))
            self.assertEqual(cfg.output_dir, str(root))
            self.assertEqual((cfg.main_backbone, cfg.blocks_per_stage), ("gcms", 1))
            self.assertFalse(hasattr(cfg, "_private"))
            self.assertEqual(cfg.evaluation_protocol, "light_cnn_v1")
            meta.write_text('{"num_products": 7}', encoding="utf-8")
            self.assertEqual(restore_saved_config(cfg), str(root / "run_config.json"))
            self.assertEqual((cfg.main_backbone, cfg.blocks_per_stage), ("plain_cnn", 2))

    def test_frozen_b_reports_validation_decision_separately_from_radius_rule(self):
        record = lambda r, p: {"radius_norm": r, "true_class": p, "is_known": True}
        validation = [record(1., "A"), record(3., "A")]
        report = frozen_setting_b([record(.5, "A")], [record(10., "HMD")], validation)
        metrics = report["open_set"]
        self.assertEqual((metrics["known_n"], metrics["unknown_n"]), (1, 1))
        self.assertAlmostEqual(metrics["validation_accept_threshold"], -2.9)
        self.assertEqual(metrics["HMD_false_acceptance"], 0.)
        self.assertEqual(metrics["open_set_AUROC"], 1.)
        unknown = report["prediction_records"][1]
        self.assertTrue(unknown["prototype_accept"])
        self.assertFalse(unknown["is_known"])
        changed = frozen_setting_b([record(100., "A")], [record(.1, "HMD")], validation)
        self.assertEqual(changed["open_set"]["validation_accept_threshold"],
                         metrics["validation_accept_threshold"])

    def test_frozen_c_exports_cross_batch_episodes_and_f1_without_a_model(self):
        records, samples = [], []
        for product, embedding in (("HMD", [1., 0.]), ("XCJ", [0., 1.])):
            for batch in (1, 2):
                for sample in range(6):
                    samples.append({"sample_id": f"{product}_{batch}_{sample}",
                                    "product_fine": product, "batch_idx": batch,
                                    "is_special": False})
                    records.append({"z": np.asarray(embedding)})
        evaluate_c = load_function("evaluate.py", "evaluate_setting_c_frozen", {
            "np": np, "Path": Path, "collect_embeddings": lambda *_: records,
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.csv"
            pd.DataFrame(samples).to_csv(metadata, index=False)
            cfg = SimpleNamespace(n_shot_values=(1, 3, 5), fewshot_repeats=2,
                                  fewshot_seed_start=42000, output_dir=directory,
                                  experiment_preset="test", seed=42)
            result = evaluate_c(None, None, metadata,
                                {"test_unknown_idx": list(range(len(samples)))}, None, cfg)
            for shot in cfg.n_shot_values:
                metrics = result["few_shot"][shot]
                self.assertEqual((metrics["accuracy"], metrics["macro_f1"]), (1., 1.))
                self.assertEqual((metrics["repeats"], metrics["n_ref"], metrics["n_test"]),
                                 (2, 2 * shot, 12.))
            self.assertEqual(len(pd.read_csv(root / "setting_c_episodes.csv")), 18)
            self.assertEqual(json.loads((root / "fewshot_episodes.json").read_text(
                encoding="utf-8"))["episode_count"], 2)


if __name__ == "__main__":
    unittest.main()
