#!/usr/bin/env python3
"""Diagnose frozen paper checkpoints without retraining deep models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines import BaselineCNN, BaselineSupCon
from config import Config, get_device
from dataset import GCMSDataset, load_data_split
from models import GCMSConsistencyNet
from paper_protocol import load_filtered_metadata, metadata_fingerprint

METHODS = ("main", "plain_cnn_ce", "plain_cnn_supcon")
SPLIT_KEYS = {
    "train": "train_idx", "val": "val_idx",
    "closed": "test_batch_idx", "unknown": "test_unknown_idx",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(payload, path):
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def apply_saved_config(cfg, path):
    path = Path(path)
    if not path.exists():
        return cfg
    payload = load_json(path)
    values = payload.get("config") or payload.get("args") or payload
    for key, value in values.items():
        if value is None or not hasattr(cfg, key):
            continue
        if isinstance(getattr(cfg, key), tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)
    return cfg


def find_main_run(checkpoint_root, seed):
    root = Path(checkpoint_root) / f"main_s{seed}"
    candidates = sorted(root.glob("run_*/final_model/model.pt"))
    if not candidates:
        candidates = sorted(root.glob("**/final_model/model.pt"))
    if not candidates:
        raise FileNotFoundError(f"main seed {seed}: no model.pt under {root}")
    return candidates[-1].parent.parent


def find_baseline_dir(checkpoint_root, method, seed):
    root = Path(checkpoint_root) / f"{method}_s{seed}"
    if (root / "model.pt").exists():
        return root
    candidates = sorted(root.glob("**/model.pt"))
    if not candidates:
        raise FileNotFoundError(f"{method} seed {seed}: no model.pt under {root}")
    return candidates[-1].parent


def load_input_transform(run_dir, cfg):
    pca_path = Path(run_dir) / "final_model" / "input_rt_pca.pkl"
    if not pca_path.exists():
        return None
    from input_pca import RtAxisPcaTransform, load_rt_axis_pca
    pca = load_rt_axis_pca(pca_path)
    cfg.mz_bins = int(getattr(pca, "n_components_", cfg.mz_bins))
    return RtAxisPcaTransform(pca)


def load_model(method, checkpoint_root, seed, num_products, device):
    cfg = Config()
    if method == "main":
        run_dir = find_main_run(checkpoint_root, seed)
        apply_saved_config(cfg, run_dir / "run_config.json")
        meta = load_json(run_dir / "final_model" / "train_meta.json")
        cfg.feature_dim = int(meta.get("feature_dim", cfg.feature_dim))
        cfg.proj_dim = int(meta.get("proj_dim", cfg.proj_dim))
        cfg.rt_bins = int(meta.get("input_raw_pca_rt_bins", cfg.rt_bins))
        transform = load_input_transform(run_dir, cfg)
        if bool(meta.get("input_raw_pca_enabled", False)) and transform is None:
            cfg.mz_bins = int(meta.get("input_raw_pca_components", cfg.mz_bins))
        model = GCMSConsistencyNet(
            int(meta["num_batches"]), cfg,
            num_products=int(meta.get("num_products", num_products)),
        )
        weight_path = run_dir / "final_model" / "model.pt"
        source = run_dir
    else:
        model_dir = find_baseline_dir(checkpoint_root, method, seed)
        apply_saved_config(cfg, model_dir / "run_config.json")
        transform = None
        model = (BaselineCNN(num_products, cfg, embed_normalize=False)
                 if method == "plain_cnn_ce" else BaselineSupCon(cfg))
        weight_path = model_dir / "model.pt"
        source = model_dir
    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), transform, str(source)


def make_dataset(metadata_csv, indices, transform):
    return GCMSDataset(
        metadata_csv, product_col="product_fine", augmentation=None,
        indices=[int(value) for value in indices], input_transform=transform,
    )


@torch.no_grad()
def extract_embeddings(model, dataset, batch_size, device, num_workers):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    chunks = []
    for batch in loader:
        chunks.append(model.encode(
            batch["input"].to(device, non_blocking=True)
        ).detach().cpu().numpy())
    values = np.concatenate(chunks).astype(np.float64, copy=False)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def probe_pipeline(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
    )


def cross_validated_probe(values, labels, seed, requested_folds):
    labels = np.asarray(labels)
    counts = pd.Series(labels).value_counts()
    folds = int(min(requested_folds, counts.min())) if len(counts) >= 2 else 0
    if folds < 2:
        return None
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_validate(
        probe_pipeline(seed), values, labels, cv=cv,
        scoring={"accuracy": "accuracy", "balanced": "balanced_accuracy"},
        n_jobs=1,
    )
    return {
        "n": len(labels), "n_classes": len(counts), "folds": folds,
        "chance_accuracy": 1.0 / len(counts),
        "accuracy_mean": np.mean(scores["test_accuracy"]),
        "accuracy_std": np.std(scores["test_accuracy"], ddof=1),
        "balanced_accuracy_mean": np.mean(scores["test_balanced"]),
        "balanced_accuracy_std": np.std(scores["test_balanced"], ddof=1),
    }


def product_probe_rows(method, seed, embeddings, frames, folds):
    train_labels = frames["train"]["product_fine"].astype(str).to_numpy()
    rows = []
    result = cross_validated_probe(embeddings["train"], train_labels, seed, folds)
    if result:
        rows.append({"method": method, "seed": seed,
                     "evaluation": "train_stratified_cv", **result})
    probe = probe_pipeline(seed).fit(embeddings["train"], train_labels)
    known_products = set(train_labels)
    for split_name in ("val", "closed"):
        labels = frames[split_name]["product_fine"].astype(str)
        mask = labels.isin(known_products).to_numpy()
        if not mask.any():
            continue
        truth = labels.to_numpy()[mask]
        predictions = probe.predict(embeddings[split_name][mask])
        rows.append({
            "method": method, "seed": seed, "evaluation": split_name,
            "n": len(truth), "n_classes": pd.Series(truth).nunique(), "folds": 0,
            "chance_accuracy": 1.0 / len(known_products),
            "accuracy_mean": accuracy_score(truth, predictions), "accuracy_std": 0.0,
            "balanced_accuracy_mean": balanced_accuracy_score(truth, predictions),
            "balanced_accuracy_std": 0.0,
        })
    return rows


def batch_probe_rows(method, seed, values, frame, folds):
    rows = []
    batch_labels = frame["batch_idx"].astype(str).to_numpy()
    result = cross_validated_probe(values, batch_labels, seed, folds)
    if result:
        rows.append({"method": method, "seed": seed, "scope": "overall",
                     "product": "__all__", "excluded_samples": 0,
                     "excluded_classes": 0, **result})
    for product, positions in frame.groupby("product_fine").indices.items():
        positions = np.asarray(positions, dtype=int)
        product_batches = batch_labels[positions]
        counts = pd.Series(product_batches).value_counts()
        eligible = set(counts[counts >= 2].index)
        keep = np.asarray([batch in eligible for batch in product_batches])
        result = cross_validated_probe(
            values[positions][keep], product_batches[keep], seed, folds
        )
        if result:
            rows.append({"method": method, "seed": seed,
                         "scope": "conditional_product",
                         "product": str(product),
                         "excluded_samples": int((~keep).sum()),
                         "excluded_classes": int((counts < 2).sum()), **result})
    return rows


def distance_rows(method, seed, split_name, values, frame, max_pairs, rng):
    left, right = np.triu_indices(len(frame), k=1)
    products = frame["product_fine"].astype(str).to_numpy()
    batches = frame["batch_idx"].astype(str).to_numpy()
    same_product = products[left] == products[right]
    same_batch = batches[left] == batches[right]
    categories = {
        "same_product_same_batch": same_product & same_batch,
        "same_product_cross_batch": same_product & ~same_batch,
        "different_product_same_batch": ~same_product & same_batch,
        "different_product_cross_batch": ~same_product & ~same_batch,
    }
    rows = []
    for pair_type, mask in categories.items():
        selected = np.flatnonzero(mask)
        if len(selected) > max_pairs:
            selected = rng.choice(selected, max_pairs, replace=False)
        if not len(selected):
            continue
        similarities = np.sum(values[left[selected]] * values[right[selected]], axis=1)
        distances = 1.0 - similarities
        rows.append({
            "method": method, "seed": seed, "split": split_name,
            "pair_type": pair_type, "n_pairs": len(selected),
            "cosine_similarity_mean": similarities.mean(),
            "cosine_similarity_std": similarities.std(ddof=1),
            "cosine_distance_mean": distances.mean(),
            "cosine_distance_std": distances.std(ddof=1),
        })
    return rows


def transfer_rows(method, seed, values, unknown_frame, manifest):
    rows = []
    batches = unknown_frame["batch_idx"].astype(str).to_numpy()
    for episode in manifest["episodes"]:
        for shot_value in manifest["shots"]:
            shot = int(shot_value)
            prototypes, classes = [], []
            for product, block in episode["products"].items():
                indices = np.asarray(
                    block["shots"][str(shot)]["ref_local_indices"], dtype=int
                )
                prototype = values[indices].mean(axis=0)
                prototypes.append(prototype / max(np.linalg.norm(prototype), 1e-12))
                classes.append(str(product))
            prototype_matrix = np.stack(prototypes)
            class_array = np.asarray(classes)
            for product, block in episode["products"].items():
                indices = np.asarray(block["query_local_indices"], dtype=int)
                query_batches = batches[indices]
                predictions = class_array[
                    np.argmax(values[indices] @ prototype_matrix.T, axis=1)
                ]
                for query_batch in sorted(set(query_batches)):
                    mask = query_batches == query_batch
                    correct = int(np.sum(predictions[mask] == str(product)))
                    rows.append({
                        "method": method, "seed": seed,
                        "episode_index": int(episode["episode_index"]),
                        "episode_seed": int(episode["episode_seed"]),
                        "shot": shot, "product": str(product),
                        "reference_batch": str(block["reference_batch"]),
                        "query_batch": str(query_batch),
                        "n_query": int(mask.sum()), "n_correct": correct,
                        "accuracy": correct / int(mask.sum()),
                    })
    return rows


def aggregate_transfer(frame):
    columns = ["method", "seed", "shot", "product", "reference_batch", "query_batch"]
    result = frame.groupby(columns, as_index=False).agg(
        episode_count=("episode_index", "nunique"), n_query=("n_query", "sum"),
        n_correct=("n_correct", "sum"), episode_accuracy_mean=("accuracy", "mean"),
        episode_accuracy_std=("accuracy", "std"),
    )
    result["weighted_accuracy"] = result["n_correct"] / result["n_query"]
    return result


def summarize(frame, groups, values):
    result = frame.groupby(groups)[values].agg(["mean", "std"]).reset_index()
    result.columns = [
        column if isinstance(column, str) else "_".join(
            str(part) for part in column if str(part)
        )
        for column in result.columns
    ]
    return result


def process_seed(args, seed, split, filtered_df, manifest, device):
    frames = {
        name: filtered_df.iloc[split[key]].reset_index(drop=True)
        for name, key in SPLIT_KEYS.items()
    }
    num_products = frames["train"]["product_fine"].nunique()
    seed_dir = Path(args.output_dir) / f"s{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    collected = {"product": [], "batch": [], "distance": [], "transfer": []}
    status = {"seed": seed, "methods": {}}
    metadata_csv = Path(args.prepared_dir) / "metadata.csv"

    for method in METHODS:
        print(f"[seed={seed}] extracting {method}", flush=True)
        model, transform, source = load_model(
            method, args.checkpoint_root, seed, num_products, device
        )
        embeddings = {}
        for split_name, key in SPLIT_KEYS.items():
            dataset = make_dataset(metadata_csv, split[key], transform)
            embeddings[split_name] = extract_embeddings(
                model, dataset, args.batch_size, device, args.num_workers
            )
            if len(embeddings[split_name]) != len(frames[split_name]):
                raise RuntimeError(f"{method} {split_name}: metadata length mismatch")
        collected["product"] += product_probe_rows(
            method, seed, embeddings, frames, args.probe_folds
        )
        collected["batch"] += batch_probe_rows(
            method, seed, embeddings["train"], frames["train"], args.probe_folds
        )
        rng = np.random.RandomState(args.analysis_seed + seed)
        for split_name in ("train", "closed", "unknown"):
            collected["distance"] += distance_rows(
                method, seed, split_name, embeddings[split_name], frames[split_name],
                args.max_pairs_per_group, rng,
            )
        collected["transfer"] += transfer_rows(
            method, seed, embeddings["unknown"], frames["unknown"], manifest
        )
        if args.save_embeddings:
            np.savez_compressed(seed_dir / f"{method}_embeddings.npz", **embeddings)
        status["methods"][method] = {
            "checkpoint": source,
            "embedding_shapes": {
                name: list(value.shape) for name, value in embeddings.items()
            },
        }
        del model, embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()

    product = pd.DataFrame(collected["product"])
    batch = pd.DataFrame(collected["batch"])
    distance = pd.DataFrame(collected["distance"])
    transfer_raw = pd.DataFrame(collected["transfer"])
    transfer = aggregate_transfer(transfer_raw)
    product.to_csv(seed_dir / "product_linear_probe.csv", index=False)
    batch.to_csv(seed_dir / "batch_linear_probe.csv", index=False)
    distance.to_csv(seed_dir / "embedding_distance_summary.csv", index=False)
    transfer_raw.to_csv(seed_dir / "batch_transfer_episode_rows.csv", index=False)
    transfer.to_csv(seed_dir / "batch_transfer_matrix.csv", index=False)
    save_json(status, seed_dir / "diagnostics_status.json")
    return product, batch, distance, transfer, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", default="new_prepared_data_relabel_v1")
    parser.add_argument("--checkpoint-root", default="new_outputs/paper_gate")
    parser.add_argument("--episode-manifest", default="result/paper_gate/fewshot_episodes.json")
    parser.add_argument("--output-dir", default="result/paper_gate/diagnostics_round1_embeddings")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--probe-folds", type=int, default=5)
    parser.add_argument("--max-pairs-per-group", type=int, default=200000)
    parser.add_argument("--analysis-seed", type=int, default=20260915)
    parser.add_argument("--save-embeddings", action="store_true")
    args = parser.parse_args()

    args.prepared_dir = str(Path(args.prepared_dir).resolve())
    args.checkpoint_root = str(Path(args.checkpoint_root).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    split = load_data_split(type("SplitConfig", (), {"prepared_dir": args.prepared_dir})())
    filtered_df = load_filtered_metadata(Path(args.prepared_dir) / "metadata.csv")
    manifest = load_json(args.episode_manifest)
    fingerprint = metadata_fingerprint(filtered_df)
    if manifest.get("metadata_fingerprint") != fingerprint:
        raise ValueError("episode manifest metadata fingerprint mismatch")
    if [int(value) for value in manifest.get("unknown_indices", [])] != [
        int(value) for value in split["test_unknown_idx"]
    ]:
        raise ValueError("episode manifest unknown indices do not match split.json")

    device = get_device()
    print(f"device={device}; output={args.output_dir}", flush=True)
    collected = {"product": [], "batch": [], "distance": [], "transfer": []}
    status = {
        "protocol": "frozen_checkpoint_diagnostics_v1",
        "metadata_fingerprint": fingerprint,
        "prepared_dir": args.prepared_dir,
        "checkpoint_root": args.checkpoint_root,
        "episode_manifest": str(Path(args.episode_manifest).resolve()),
        "seeds": [int(seed) for seed in args.seeds],
        "deep_models_trained": False,
        "notes": {
            "batch_probe": "Training-set stratified CV; product-conditional probes reduce product-batch confounding.",
            "product_probe": "Fit on train embeddings and evaluate validation/closed batches.",
            "transfer": "Fixed nested episodes grouped by true-product reference/query batch.",
        },
        "seed_results": [],
    }
    for seed in args.seeds:
        outputs = process_seed(args, int(seed), split, filtered_df, manifest, device)
        for key, frame in zip(collected, outputs[:4]):
            collected[key].append(frame)
        status["seed_results"].append(outputs[4])

    product = pd.concat(collected["product"], ignore_index=True)
    batch = pd.concat(collected["batch"], ignore_index=True)
    distance = pd.concat(collected["distance"], ignore_index=True)
    transfer = pd.concat(collected["transfer"], ignore_index=True)
    output_dir = Path(args.output_dir)
    product.to_csv(output_dir / "product_linear_probe_all_seeds.csv", index=False)
    batch.to_csv(output_dir / "batch_linear_probe_all_seeds.csv", index=False)
    distance.to_csv(output_dir / "embedding_distance_all_seeds.csv", index=False)
    transfer.to_csv(output_dir / "batch_transfer_all_seeds.csv", index=False)
    summarize(product, ["method", "evaluation"],
              ["accuracy_mean", "balanced_accuracy_mean"]).to_csv(
        output_dir / "product_linear_probe_summary.csv", index=False
    )
    summarize(batch[batch["scope"] == "overall"], ["method", "scope"],
              ["accuracy_mean", "balanced_accuracy_mean"]).to_csv(
        output_dir / "batch_linear_probe_summary.csv", index=False
    )
    summarize(batch[batch["scope"] == "conditional_product"],
              ["method", "scope", "product"],
              ["accuracy_mean", "balanced_accuracy_mean"]).to_csv(
        output_dir / "batch_linear_probe_conditional_summary.csv", index=False
    )
    summarize(distance, ["method", "split", "pair_type"],
              ["cosine_similarity_mean", "cosine_distance_mean"]).to_csv(
        output_dir / "embedding_distance_summary.csv", index=False
    )
    summarize(transfer,
              ["method", "shot", "product", "reference_batch", "query_batch"],
              ["weighted_accuracy", "episode_accuracy_mean"]).to_csv(
        output_dir / "batch_transfer_summary.csv", index=False
    )
    save_json(status, output_dir / "diagnostics_status.json")
    print(f"diagnostics saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
