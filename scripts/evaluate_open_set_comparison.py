#!/usr/bin/env python3
"""Compare open-set metrics for already-trained GC-MS and CNN checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines import BaselineCNN, BaselineSupCon
from config import Config, get_device
from dataset import GCMSDataset, load_data_split
from models import GCMSConsistencyNet
from register import PrototypeStore, register_from_loader


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_dir(path):
    path = Path(path)
    if path.is_file() and path.name == "model.pt":
        return path.parent
    if (path / "final_model" / "model.pt").exists():
        return path / "final_model"
    if (path / "model.pt").exists():
        return path
    raise FileNotFoundError(f"model.pt not found under {path}")


def find_run_config(path):
    path = Path(path)
    for candidate in (path / "run_config.json", path.parent / "run_config.json"):
        if candidate.exists():
            return candidate
    return None


def apply_saved_config(cfg, path):
    if path is None:
        return cfg
    payload = load_json(path)
    saved = payload.get("config") or payload.get("args") or {}
    for key, value in saved.items():
        if hasattr(cfg, key) and value is not None:
            if isinstance(getattr(cfg, key), tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)
    return cfg


def aligned_dataset(metadata_csv, indices, source, input_transform=None):
    dataset = GCMSDataset(
        metadata_csv, product_col="product_fine", augmentation=None,
        indices=indices, input_transform=input_transform,
    )
    dataset.product_enc = source.product_enc
    dataset.batch_enc = source.batch_enc
    dataset.df["product_label"] = source.product_enc.transform(dataset.df["product_fine"])
    dataset.df["batch_label"] = source.batch_enc.transform(dataset.df["batch_idx"])
    dataset.num_products = source.num_products
    dataset.num_batches = source.num_batches
    return dataset


def make_datasets(prepared_dir, split, input_transform=None):
    metadata_csv = Path(prepared_dir) / "metadata.csv"
    source = GCMSDataset(metadata_csv, product_col="product_fine", augmentation=None)
    train = aligned_dataset(metadata_csv, split["train_idx"], source, input_transform)
    known = aligned_dataset(metadata_csv, split["test_batch_idx"], source, input_transform)
    unknown = aligned_dataset(metadata_csv, split["test_unknown_idx"], source, input_transform)
    return source, train, known, unknown


def cnn_dataset(metadata_csv, indices, train_source, batch_source, unknown=False):
    dataset = GCMSDataset(
        metadata_csv, product_col="product_fine", augmentation=None, indices=indices,
    )
    dataset.product_enc = train_source.product_enc
    dataset.batch_enc = batch_source.batch_enc
    if unknown:
        dataset.df["product_label"] = -1
    else:
        dataset.df["product_label"] = train_source.product_enc.transform(dataset.df["product_fine"])
    dataset.df["batch_label"] = batch_source.batch_enc.transform(dataset.df["batch_idx"])
    dataset.num_products = train_source.num_products
    dataset.num_batches = batch_source.num_batches
    return dataset

def metrics(method, known_scores, unknown_scores, known_correct=None):
    known_scores = np.asarray(known_scores, dtype=float)
    unknown_scores = np.asarray(unknown_scores, dtype=float)
    labels = np.r_[np.ones(len(known_scores), dtype=np.int64),
                   np.zeros(len(unknown_scores), dtype=np.int64)]
    scores = np.r_[known_scores, unknown_scores]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    index = min(int(np.searchsorted(tpr, 0.95, side="left")), len(fpr) - 1)
    result = {
        "method": method,
        "n_known": int(len(known_scores)),
        "n_unknown": int(len(unknown_scores)),
        "open_set_AUROC": float(roc_auc_score(labels, scores)),
        "FPR_at_95TPR": float(fpr[index]),
        "threshold_at_selected_95TPR": float(thresholds[index]),
        "known_score_mean": float(known_scores.mean()),
        "unknown_score_mean": float(unknown_scores.mean()),
    }
    if known_correct is not None:
        result["known_accuracy"] = float(np.mean(np.asarray(known_correct, dtype=bool)))
    return result


def sample_rows(method, dataset, scores, predictions=None, is_known=True):
    rows = []
    for index, score in enumerate(scores):
        row = dataset.df.iloc[index]
        item = {
            "method": method,
            "sample_id": row["sample_id"],
            "product": row["product_fine"],
            "batch": row["batch_idx"],
            "score": float(score),
            "is_known": bool(is_known),
        }
        if predictions is not None:
            item["prediction"] = str(predictions[index])
            item["correct"] = bool(item["prediction"] == item["product"])
        rows.append(item)
    return rows


@torch.no_grad()
def main_scores(model, store, loader, device):
    scores, predictions = [], []
    model.eval()
    for batch in loader:
        result = store.predict(model.encode(batch["input"].to(device)), use_spherical=True)
        scores.extend(result["scores"].cpu().tolist())
        predictions.extend(result["pred_class"])
    return np.asarray(scores), predictions


@torch.no_grad()
def cnn_scores(model, loader, device, method, store=None):
    scores, predictions = [], []
    model.eval()
    for batch in loader:
        x = batch["input"].to(device)
        if method == "cnn_ce":
            probabilities = torch.softmax(model(x)["logits"], dim=1)
            batch_scores, batch_predictions = probabilities.max(dim=1)
            predictions.extend(batch_predictions.cpu().tolist())
        else:
            result = store.predict(model.encode(x), use_spherical=False)
            batch_scores = result["scores"]
            predictions.extend(result["pred_idx"].cpu().tolist())
        scores.extend(batch_scores.cpu().tolist())
    return np.asarray(scores), predictions


def load_main(path, device):
    directory = model_dir(path)
    cfg = apply_saved_config(Config(), find_run_config(directory))
    meta = load_json(directory / "train_meta.json")
    input_transform = None
    pca_path = directory / "input_rt_pca.pkl"
    if pca_path.exists():
        from input_pca import RtAxisPcaTransform, load_rt_axis_pca
        pca = load_rt_axis_pca(pca_path)
        input_transform = RtAxisPcaTransform(pca)
        cfg.mz_bins = int(getattr(pca, "n_components_", cfg.mz_bins))
    model = GCMSConsistencyNet(
        int(meta["num_batches"]), cfg, num_products=meta.get("num_products")
    ).to(device)
    state = torch.load(directory / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    store = PrototypeStore()
    if (directory / "prototypes").exists():
        store.load(directory / "prototypes")
    return model.eval(), store, input_transform


def load_cnn(path, method, train_source, device):
    directory = model_dir(path)
    cfg = apply_saved_config(Config(), find_run_config(directory))
    if method == "cnn_ce":
        model = BaselineCNN(train_source.num_products, cfg, embed_normalize=False)
    else:
        model = BaselineSupCon(cfg)
    state = torch.load(directory / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def build_store(model, loader, label_source, device, use_spherical=False):
    cfg = Config()
    cfg.open_score_base_weight = 0.0
    cfg.open_score_margin_weight = 1.0
    store, _, _ = register_from_loader(
        model, loader, label_source.get_label_name_map(), device,
        percentile=95.0, use_spherical=use_spherical, cfg=cfg,
    )
    return store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="new_prepared_data_relabel_v1")
    parser.add_argument("--main-run-dir", required=True)
    parser.add_argument("--cnn-ce-dir", required=True)
    parser.add_argument("--cnn-supcon-dir", required=True)
    parser.add_argument("--output-dir", default="result/open_set_comparison")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split = load_data_split(type("SplitConfig", (), {"prepared_dir": str(prepared_dir)})())
    device = get_device()

    main_model, main_store, main_transform = load_main(args.main_run_dir, device)
    source, train_ds, known_ds, unknown_ds = make_datasets(prepared_dir, split, main_transform)
    expected_known_ids = list(known_ds.df["sample_id"])
    expected_unknown_ids = list(unknown_ds.df["sample_id"])
    main_loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=False),
        "known": DataLoader(known_ds, batch_size=args.batch_size, shuffle=False),
        "unknown": DataLoader(unknown_ds, batch_size=args.batch_size, shuffle=False),
    }
    if not main_store.class_names:
        main_store = build_store(main_model, main_loaders["train"], source, device, use_spherical=True)
    known_scores, main_predictions = main_scores(main_model, main_store, main_loaders["known"], device)
    unknown_scores, _ = main_scores(main_model, main_store, main_loaders["unknown"], device)
    main_correct = [str(pred) == str(truth) for pred, truth in zip(main_predictions, known_ds.df["product_fine"])]

    results = [metrics("main", known_scores, unknown_scores, main_correct)]
    rows = sample_rows("main", known_ds, known_scores, main_predictions)
    rows += sample_rows("main", unknown_ds, unknown_scores, is_known=False)

    metadata_csv = prepared_dir / "metadata.csv"
    for method, path in (("cnn_ce", args.cnn_ce_dir), ("cnn_supcon", args.cnn_supcon_dir)):
        train = GCMSDataset(
            metadata_csv, product_col="product_fine", augmentation=None,
            indices=split["train_idx"],
        )
        known = cnn_dataset(metadata_csv, split["test_batch_idx"], train, source)
        unknown = cnn_dataset(
            metadata_csv, split["test_unknown_idx"], train, source, unknown=True,
        )
        model = load_cnn(path, method, train, device)
        if list(known.df["sample_id"]) != expected_known_ids:
            raise RuntimeError(f"{method}: known sample IDs do not match the main model")
        if list(unknown.df["sample_id"]) != expected_unknown_ids:
            raise RuntimeError(f"{method}: unknown sample IDs do not match the main model")
        loaders = {
            "train": DataLoader(train, batch_size=args.batch_size, shuffle=False),
            "known": DataLoader(known, batch_size=args.batch_size, shuffle=False),
            "unknown": DataLoader(unknown, batch_size=args.batch_size, shuffle=False),
        }
        store = build_store(model, loaders["train"], train, device) if method == "cnn_supcon" else None
        known_scores, predictions = cnn_scores(model, loaders["known"], device, method, store)
        unknown_scores, _ = cnn_scores(model, loaders["unknown"], device, method, store)
        if method == "cnn_ce":
            prediction_names = [train.get_product_names()[int(index)] for index in predictions]
        else:
            prediction_names = [train.get_product_names()[int(index)] for index in predictions]
        correct = [pred == truth for pred, truth in zip(prediction_names, known.df["product_fine"])]
        results.append(metrics(method, known_scores, unknown_scores, correct))
        rows += sample_rows(method, known, known_scores, prediction_names)
        rows += sample_rows(method, unknown, unknown_scores, is_known=False)

    summary = {
        "protocol": "same_split_setting_b_v1",
        "prepared_dir": str(prepared_dir),
        "holdout_products": split.get("holdout_products", []),
        "known_test_count": len(split["test_batch_idx"]),
        "unknown_test_count": len(split["test_unknown_idx"]),
        "sample_alignment_check": "passed: identical known/unknown sample_id order across methods",
        "main_run_dir": str(Path(args.main_run_dir).resolve()),
        "cnn_ce_dir": str(Path(args.cnn_ce_dir).resolve()),
        "cnn_supcon_dir": str(Path(args.cnn_supcon_dir).resolve()),
        "score_definition": {
            "main": "saved PrototypeStore score with spherical prototypes",
            "cnn_ce": "maximum softmax probability",
            "cnn_supcon": "frozen embedding PrototypeStore score without spherical redistribution",
        },
        "results": results,
    }
    pd.DataFrame(results).to_csv(output_dir / "open_set_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "open_set_sample_scores.csv", index=False)
    (output_dir / "open_set_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()



