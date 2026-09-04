"""Paper-grade closed-set and few-shot experiment utilities."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_SHOTS = (1, 3, 5)


def load_filtered_metadata(metadata_csv, exclude_blanks=True, exclude_special=True):
    df = pd.read_csv(metadata_csv)
    if exclude_blanks:
        df = df[df["product_fine"] != "BLANK"]
    if exclude_special:
        df = df[~df["is_special"]]
    return df.reset_index(drop=True)


def metadata_fingerprint(df):
    columns = [
        column for column in
        ("sample_id", "tensor_path", "product_fine", "product_coarse", "batch_idx")
        if column in df.columns
    ]
    payload = df[columns].fillna("").astype(str).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_index_set(split, key):
    return {int(value) for value in split.get(key, [])}


def audit_split(metadata_csv, split_path, expected_holdout_products=("HMD", "XCJ")):
    metadata_csv = Path(metadata_csv)
    split_path = Path(split_path)
    df = load_filtered_metadata(metadata_csv)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    named_sets = {
        "train": _as_index_set(split, "train_idx"),
        "val": _as_index_set(split, "val_idx"),
        "test_batch": _as_index_set(split, "test_batch_idx"),
        "test_unknown": _as_index_set(split, "test_unknown_idx"),
    }
    all_indices = set().union(*named_sets.values())
    invalid_indices = sorted(index for index in all_indices if index < 0 or index >= len(df))
    overlaps = {}
    names = list(named_sets)
    for left_pos, left in enumerate(names):
        for right in names[left_pos + 1:]:
            intersection = sorted(named_sets[left] & named_sets[right])
            if intersection:
                overlaps[f"{left}__{right}"] = intersection

    products = {}
    batches = {}
    for name, indices in named_sets.items():
        valid = sorted(index for index in indices if 0 <= index < len(df))
        subset = df.iloc[valid]
        products[name] = sorted(subset["product_fine"].astype(str).unique().tolist())
        batches[name] = sorted(subset["batch_idx"].astype(str).unique().tolist())

    declared_holdout = sorted(str(value) for value in split.get("holdout_products", []))
    expected_holdout = sorted(str(value) for value in expected_holdout_products)
    train_holdout_leakage = sorted(set(products["train"]) & set(declared_holdout))
    closed_unknown_leakage = sorted(set(products["test_batch"]) & set(declared_holdout))
    unknown_non_holdout = sorted(set(products["test_unknown"]) - set(declared_holdout))
    unknown_missing = sorted(set(declared_holdout) - set(products["test_unknown"]))
    checks = {
        "indices_in_range": not invalid_indices,
        "partitions_disjoint": not overlaps,
        "declared_holdout_matches_expected": declared_holdout == expected_holdout,
        "train_excludes_holdout_products": not train_holdout_leakage,
        "closed_test_excludes_holdout_products": not closed_unknown_leakage,
        "unknown_test_contains_only_holdout_products": not unknown_non_holdout,
        "unknown_test_contains_all_holdout_products": not unknown_missing,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "metadata_csv": str(metadata_csv.resolve()),
        "split_path": str(split_path.resolve()),
        "metadata_rows_filtered": int(len(df)),
        "metadata_fingerprint": metadata_fingerprint(df),
        "checks": checks,
        "declared_holdout_products": declared_holdout,
        "products": products,
        "batches": batches,
        "counts": {name: len(indices) for name, indices in named_sets.items()},
        "invalid_indices": invalid_indices,
        "overlaps": overlaps,
        "train_holdout_leakage": train_holdout_leakage,
        "closed_unknown_leakage": closed_unknown_leakage,
        "unknown_non_holdout": unknown_non_holdout,
        "unknown_missing": unknown_missing,
    }


def build_cross_batch_episodes(
    metadata_csv,
    unknown_indices,
    shots=PAPER_SHOTS,
    episode_count=100,
    seed_start=42000,
    product_col="product_fine",
):
    shots = tuple(sorted({int(shot) for shot in shots}))
    if not shots or min(shots) < 1:
        raise ValueError("shots must contain positive integers")

    df = load_filtered_metadata(metadata_csv)
    unknown_indices = [int(index) for index in unknown_indices]
    unknown_df = df.iloc[unknown_indices].copy()
    unknown_df["global_index"] = unknown_indices
    unknown_df["local_index"] = np.arange(len(unknown_df), dtype=int)
    max_shot = max(shots)
    product_groups = {}
    for product, product_df in unknown_df.groupby(product_col, sort=True):
        batch_groups = {
            str(batch): group.copy()
            for batch, group in product_df.groupby("batch_idx", sort=True)
        }
        eligible_batches = sorted(
            batch for batch, group in batch_groups.items()
            if len(group) >= max_shot and len(product_df) > len(group)
        )
        if not eligible_batches:
            raise ValueError(
                f"Product {product!r} has no batch with at least {max_shot} "
                "references and query samples in other batches"
            )
        product_groups[str(product)] = (batch_groups, eligible_batches)

    episodes = []
    for episode_index in range(int(episode_count)):
        episode_seed = int(seed_start) + episode_index
        rng = np.random.RandomState(episode_seed)
        products = {}
        for product, (batch_groups, eligible_batches) in product_groups.items():
            ref_batch = str(rng.choice(eligible_batches))
            ref_candidates = batch_groups[ref_batch]
            selected = ref_candidates.iloc[rng.permutation(len(ref_candidates))[:max_shot]]
            query = unknown_df[
                (unknown_df[product_col].astype(str) == product)
                & (unknown_df["batch_idx"].astype(str) != ref_batch)
            ]
            ref_local_all = selected["local_index"].astype(int).tolist()
            ref_global_all = selected["global_index"].astype(int).tolist()
            ref_samples_all = selected["sample_id"].astype(str).tolist()
            shot_blocks = {}
            for shot in shots:
                shot_blocks[str(shot)] = {
                    "ref_local_indices": ref_local_all[:shot],
                    "ref_global_indices": ref_global_all[:shot],
                    "ref_sample_ids": ref_samples_all[:shot],
                }
            products[product] = {
                "reference_batch": ref_batch,
                "shots": shot_blocks,
                "query_local_indices": query["local_index"].astype(int).tolist(),
                "query_global_indices": query["global_index"].astype(int).tolist(),
                "query_sample_ids": query["sample_id"].astype(str).tolist(),
                "query_batches": sorted(query["batch_idx"].astype(str).unique().tolist()),
            }
        episodes.append({
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "products": products,
        })

    return {
        "protocol": "cross_batch_nested_reference_v1",
        "metadata_fingerprint": metadata_fingerprint(df),
        "shots": list(shots),
        "episode_count": int(episode_count),
        "seed_start": int(seed_start),
        "unknown_indices": unknown_indices,
        "episodes": episodes,
    }


def validate_episode_manifest(manifest):
    shots = [int(value) for value in manifest["shots"]]
    errors = []
    for episode in manifest["episodes"]:
        for product, block in episode["products"].items():
            query_ids = set(block["query_sample_ids"])
            query_batches = set(str(value) for value in block["query_batches"])
            if str(block["reference_batch"]) in query_batches:
                errors.append(f"episode={episode['episode_index']} product={product}: batch leakage")
            for shot in shots:
                ref_ids = block["shots"][str(shot)]["ref_sample_ids"]
                if len(ref_ids) != shot:
                    errors.append(
                        f"episode={episode['episode_index']} product={product}: "
                        f"expected {shot} refs, got {len(ref_ids)}"
                    )
                if set(ref_ids) & query_ids:
                    errors.append(
                        f"episode={episode['episode_index']} product={product}: sample leakage"
                    )
    return {"status": "pass" if not errors else "fail", "errors": errors}


def _normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def evaluate_embedding_episodes(embeddings, manifest, method_name, train_seed):
    embeddings = _normalize_rows(embeddings)
    rows = []
    for episode in manifest["episodes"]:
        for shot_value in manifest["shots"]:
            shot = int(shot_value)
            prototypes = []
            classes = []
            query_indices = []
            query_true = []
            for product, block in episode["products"].items():
                ref_indices = block["shots"][str(shot)]["ref_local_indices"]
                prototype = embeddings[ref_indices].mean(axis=0, keepdims=True)
                prototypes.append(_normalize_rows(prototype)[0])
                classes.append(product)
                product_queries = [int(value) for value in block["query_local_indices"]]
                query_indices.extend(product_queries)
                query_true.extend([product] * len(product_queries))
            prototype_matrix = np.stack(prototypes)
            query_true = np.asarray(query_true)
            predictions = np.asarray(classes)[
                np.argmax(embeddings[query_indices] @ prototype_matrix.T, axis=1)
            ]
            base = {
                "method": str(method_name),
                "train_seed": int(train_seed),
                "episode_index": int(episode["episode_index"]),
                "episode_seed": int(episode["episode_seed"]),
                "shot": shot,
                "scope": "pooled",
                "product": "__pooled__",
                "accuracy": float(np.mean(predictions == query_true)),
                "n_query": int(len(query_true)),
            }
            rows.append(base)
            for product in classes:
                mask = query_true == product
                rows.append({
                    **base,
                    "scope": "product",
                    "product": product,
                    "accuracy": float(np.mean(predictions[mask] == query_true[mask])),
                    "n_query": int(mask.sum()),
                })
    return rows


def percentile_ci(values, confidence=0.95):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    alpha = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def paired_bootstrap_ci(main_values, baseline_values, iterations=10000, seed=20260904):
    main_values = np.asarray(main_values, dtype=np.float64)
    baseline_values = np.asarray(baseline_values, dtype=np.float64)
    if main_values.shape != baseline_values.shape:
        raise ValueError("paired arrays must have identical shapes")
    differences = main_values - baseline_values
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return {"mean_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.RandomState(int(seed))
    samples = rng.choice(differences, size=(int(iterations), len(differences)), replace=True)
    means = samples.mean(axis=1)
    low, high = percentile_ci(means)
    return {"mean_delta": float(differences.mean()), "ci95_low": low, "ci95_high": high}


def write_rows_csv(rows, path):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
