"""Synthetic protocol checks; no torch, training, or model evaluation required."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from light_cnn_protocol import (
    classification_metrics, fit_prototypes, open_set_metrics,
    prototype_predictions, selection_key, validation_threshold,
)
from paper_protocol import evaluate_embedding_episodes


def load_method(filename, class_name, method_name, namespace):
    """Execute the actual edited method on spies without needing torch installed."""
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace[method_name]


def forbidden(*_args, **_kwargs):
    raise AssertionError("A disabled loss/head was called")


class ProtocolTest(unittest.TestCase):
    def test_selection_macro_first_balanced_tie_then_earlier(self):
        rows = [(10, {"macro_f1": .8, "balanced_accuracy": .6}),
                (20, {"macro_f1": .7, "balanced_accuracy": 1}),
                (30, {"macro_f1": .8, "balanced_accuracy": .7}),
                (40, {"macro_f1": .8, "balanced_accuracy": .7})]
        best, epoch = None, None
        for candidate_epoch, metrics in rows:
            key = selection_key(metrics)
            if best is None or key > best:
                best, epoch = key, candidate_epoch
        self.assertEqual(epoch, 30)
        metrics = classification_metrics([0] * 9 + [1], [0] * 10, classes=("A", "B", "C"))
        self.assertAlmostEqual(metrics["accuracy"], .9)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], (18 / 19) / 3)
        self.assertEqual(metrics["per_class_recall"]["B"], 0)

    def test_raw_means_radii_and_score_direction(self):
        embeddings = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        prototypes, radii = fit_prototypes(embeddings, [0, 0, 1, 1], classes=("A", "B"))
        np.testing.assert_allclose(prototypes, [[.5, .5], [-.5, -.5]])
        np.testing.assert_allclose(radii, np.sqrt(.5))
        prediction = prototype_predictions(np.array([[.5, .5], [2., 2.]]), prototypes, radii)
        self.assertGreater(prediction["score"][0], prediction["score"][1])
        np.testing.assert_array_equal(prediction["prediction"], [0, 0])
        _, tiny = fit_prototypes([[1, 0], [-1, 0]], [0, 1], classes=("A", "B"))
        np.testing.assert_array_equal(tiny, [1e-6, 1e-6])

    def test_threshold_uses_validation_only_and_fpr95_is_diagnostic(self):
        validation = np.array([-3., -2., -1., 0.])
        threshold = validation_threshold(validation)
        self.assertAlmostEqual(threshold, -2.85)
        report = open_set_metrics(np.array([-.1, -.2]), np.array([-10., -20.]), threshold)
        self.assertEqual(report["auroc"], 1.)
        self.assertEqual(report["fpr95"], 0.)
        self.assertEqual(report["unknown_false_acceptance"], 0.)
        self.assertEqual(report["validation_accept_threshold"], threshold)
        changed = open_set_metrics(np.array([-100., -200.]), np.array([100., 200.]), threshold)
        self.assertEqual(changed["validation_accept_threshold"], threshold)
        self.assertNotEqual(report["fpr95_diagnostic_threshold"], threshold)

    def test_euclidean_episode_does_not_normalize_prototype_means(self):
        # A points in query direction but has a short mean; B is farther by angle.
        embeddings = np.array([[.05, np.sqrt(.9975)], [.05, -np.sqrt(.9975)],
                               [.6, .8], [.6, .8], [1., 0.], [.6, .8]])
        manifest = {"shots": [2], "episodes": [{"episode_index": 0, "episode_seed": 42000,
            "products": {"A": {"shots": {"2": {"ref_local_indices": [0, 1]}}, "query_local_indices": [4]},
                         "B": {"shots": {"2": {"ref_local_indices": [2, 3]}}, "query_local_indices": [5]}}}]}
        euclidean = evaluate_embedding_episodes(embeddings, manifest, "test", 42, distance="euclidean_raw_mean")
        cosine = evaluate_embedding_episodes(embeddings, manifest, "test", 42)
        self.assertEqual(euclidean[0]["accuracy"], .5)
        self.assertEqual(cosine[0]["accuracy"], 1.)
        bad = embeddings.copy()
        bad[0] *= 2
        with self.assertRaisesRegex(ValueError, "unit embeddings"):
            evaluate_embedding_episodes(bad, manifest, "test", 42, distance="euclidean_raw_mean")

    def test_zero_weight_losses_never_execute(self):
        forward = load_method("losses.py", "UnifiedLoss", "forward", {})
        spy = SimpleNamespace(lam_supcon=1., lam_cls=.4, lam_adv=0., lam_proto=0.,
                              lam_hard_pair=0., lam_recon=0., supcon_loss=lambda *_: 2.,
                              cls_loss=lambda *_: 3., domain_loss=forbidden,
                              proto_loss=forbidden, hard_pair_loss=forbidden, recon_loss=forbidden)
        out = {"z": SimpleNamespace(new_zeros=lambda _: 0.), "proj": object(), "cls_logits": object()}
        result = forward(spy, out, {"product": object()})
        self.assertAlmostEqual(result["total"], 3.2)
        for key in ("adv", "proto", "hardpair", "recon"):
            self.assertEqual(result[key], 0.)

    def test_disabled_heads_never_execute(self):
        forward = load_method("models.py", "GCMSConsistencyNet", "forward",
                              {"F": SimpleNamespace(normalize=lambda value, dim: value)})
        model = SimpleNamespace(encoder=lambda _: ("features", "map"), embed_head=lambda _: "embedding",
                                embed_normalize=True,
                                active_heads={"proj": True, "cls_logits": True, "domain_logits": False, "recon": False},
                                proj_head=lambda _: "projection", cls_head=lambda _: "class_logits",
                                domain_head=forbidden, decoder=forbidden)
        result = forward(model, None)
        self.assertEqual(result["proj"], "projection")
        self.assertIsNone(result["domain_logits"])
        self.assertIsNone(result["recon"])
        self.assertEqual(result["cls_logits"], "class_logits")


if __name__ == "__main__":
    unittest.main()
