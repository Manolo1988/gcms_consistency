"""Dataset audit and immutable split manifests for the paper experiments."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProtocolSpec:
    product_col: str = "product_fine"
    batch_col: str = "batch_name"
    lot_col: str = "lot_id"
    sample_col: str = "sample_id"
    novel_classes_per_fold: int = 2
    max_product_folds: int = 4
    min_samples_per_product: int = 20
    min_batches_per_product: int = 3
    min_closed_test_classes: int = 3
    min_closed_test_samples: int = 20
    closed_test_batch_ratio: float = 0.10
    validation_batch_ratio: float = 0.10
    shots: tuple[int, ...] = (1, 3, 5)
    episodes: int = 100
    min_novel_query_per_class: int = 5
    future_query_only: bool = True
    seed: int = 42


def load_metadata(path: str | Path, spec: ProtocolSpec | None = None) -> pd.DataFrame:
    """Load metadata in the same filtered coordinate system as GCMSDataset."""
    spec = spec or ProtocolSpec()
    df = pd.read_csv(path)
    required = {spec.product_col, spec.batch_col, spec.sample_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"metadata missing required columns: {missing}")
    if "product_fine" in df:
        df = df[df["product_fine"].astype(str) != "BLANK"]
    if "is_special" in df:
        values = df["is_special"].fillna(False)
        if values.dtype == object:
            special = values.astype(str).str.strip().str.lower().isin(
                {"1", "true", "yes", "y"}
            )
        else:
            special = values.astype(bool)
        df = df[~special]
    df = df.reset_index(drop=True)
    df["_row_index"] = np.arange(len(df), dtype=np.int64)
    df[spec.product_col] = df[spec.product_col].astype(str)
    df[spec.batch_col] = df[spec.batch_col].astype(str)
    sample_ids = df[spec.sample_col].astype(str)
    occurrence = sample_ids.groupby(sample_ids, sort=False).cumcount().astype(str)
    duplicated = sample_ids.duplicated(keep=False)
    df["_sample_key"] = sample_ids
    df.loc[duplicated, "_sample_key"] = (
        sample_ids.loc[duplicated] + "#occurrence=" + occurrence.loc[duplicated]
    )
    df["_group_id"] = _independent_group_ids(df, spec)
    return df


def _independent_group_ids(df: pd.DataFrame, spec: ProtocolSpec) -> pd.Series:
    """Use product+lot when available and a unique sample fallback otherwise."""
    product = df[spec.product_col].astype(str)
    fallback = "sample:" + df["_sample_key"].astype(str)
    if spec.lot_col not in df:
        return product + "|" + fallback
    lot = df[spec.lot_col]
    valid = lot.notna() & lot.astype(str).str.strip().ne("")
    valid &= lot.astype(str).str.lower().ne("nan")
    group = fallback.copy()
    group.loc[valid] = "lot:" + lot.loc[valid].astype(str).str.strip()
    return product + "|" + group


def metadata_fingerprint(df: pd.DataFrame, spec: ProtocolSpec) -> str:
    cols = ["_sample_key", spec.product_col, spec.batch_col, "_group_id"]
    payload = df[cols].sort_values("_sample_key").to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_metadata(df: pd.DataFrame, spec: ProtocolSpec) -> dict:
    by_product = []
    for product, part in df.groupby(spec.product_col, sort=True):
        by_product.append({
            "product": str(product),
            "samples": int(len(part)),
            "batches": int(part[spec.batch_col].nunique()),
            "independent_groups": int(part["_group_id"].nunique()),
        })
    cross_batch_groups = (
        df.groupby("_group_id")[spec.batch_col].nunique().gt(1)
    )
    return {
        "samples": int(len(df)),
        "products": int(df[spec.product_col].nunique()),
        "batches": int(df[spec.batch_col].nunique()),
        "independent_groups": int(df["_group_id"].nunique()),
        "duplicate_sample_id_rows": int(
            df[spec.sample_col].astype(str).duplicated(keep=False).sum()
        ),
        "duplicate_sample_ids": int(
            df.loc[
                df[spec.sample_col].astype(str).duplicated(keep=False), spec.sample_col
            ].astype(str).nunique()
        ),
        "groups_spanning_batches": int(cross_batch_groups.sum()),
        "metadata_sha256": metadata_fingerprint(df, spec),
        "by_product": by_product,
    }


def eligible_products(df: pd.DataFrame, spec: ProtocolSpec) -> list[str]:
    stats = df.groupby(spec.product_col).agg(
        samples=(spec.sample_col, "size"),
        batches=(spec.batch_col, "nunique"),
    )
    mask = (
        (stats["samples"] >= spec.min_samples_per_product)
        & (stats["batches"] >= spec.min_batches_per_product)
    )
    return sorted(stats.index[mask].astype(str).tolist())


def _batch_order(df: pd.DataFrame, spec: ProtocolSpec) -> dict[str, int]:
    """Return the declared chronological order for date-sortable batch names."""
    batches = sorted(df[spec.batch_col].astype(str).unique().tolist())
    return {batch: rank for rank, batch in enumerate(batches)}


def _fewshot_cutoffs(
    df: pd.DataFrame,
    products: Sequence[str],
    shot: int,
    spec: ProtocolSpec,
) -> list[tuple[int, dict[str, list[str]]]]:
    """Find global cutoffs with support before and query after the cutoff."""
    order = _batch_order(df, spec)
    candidates = []
    for cutoff in range(max(len(order) - 1, 0)):
        support_batches: dict[str, list[str]] = {}
        feasible = True
        for product in products:
            part = df[df[spec.product_col] == product]
            before = part[
                part[spec.batch_col].astype(str).map(order).le(cutoff)
            ]
            after = part[
                part[spec.batch_col].astype(str).map(order).gt(cutoff)
            ]
            usable = [
                str(batch)
                for batch, block in before.groupby(spec.batch_col)
                if block["_group_id"].nunique() >= shot
            ]
            if not usable or after["_group_id"].nunique() < spec.min_novel_query_per_class:
                feasible = False
                break
            support_batches[str(product)] = sorted(usable)
        if feasible:
            candidates.append((cutoff, support_batches))
    return candidates


def _select_product_folds(
    df: pd.DataFrame,
    products: Sequence[str],
    spec: ProtocolSpec,
) -> list[tuple[str, ...]]:
    combinations = list(itertools.combinations(sorted(products), spec.novel_classes_per_fold))
    max_shot = max(spec.shots) if spec.shots else 1
    combinations = [
        combo for combo in combinations
        if _fewshot_cutoffs(df, combo, max_shot, spec)
    ]
    if not combinations:
        raise ValueError("not enough eligible products to create a novel-product fold")
    if spec.max_product_folds <= 0:
        return combinations
    rng = np.random.RandomState(spec.seed)
    ordered = [combinations[i] for i in rng.permutation(len(combinations))]
    target = min(
        spec.max_product_folds,
        len(products) // max(spec.novel_classes_per_fold, 1),
    )
    best: list[tuple[str, ...]] = []

    def search(start: int, chosen: list[tuple[str, ...]], used: set[str]) -> bool:
        nonlocal best
        if len(chosen) > len(best):
            best = chosen.copy()
        if len(chosen) >= target:
            return True
        for index in range(start, len(ordered)):
            candidate = ordered[index]
            if used.isdisjoint(candidate):
                if search(
                    index + 1,
                    chosen + [candidate],
                    used | set(candidate),
                ):
                    return True
        return False

    search(0, [], set())
    return sorted(best)


def _candidate_batches(df: pd.DataFrame, known: Sequence[str], spec: ProtocolSpec) -> list[str]:
    known_df = df[df[spec.product_col].isin(known)]
    stats = known_df.groupby(spec.batch_col).agg(
        samples=(spec.sample_col, "size"),
        classes=(spec.product_col, "nunique"),
    )
    min_classes = min(spec.min_closed_test_classes, len(known))
    eligible = stats[
        (stats["samples"] >= spec.min_closed_test_samples)
        & (stats["classes"] >= min_classes)
    ].index.astype(str).tolist()
    if len(eligible) < 2:
        eligible = stats.sort_values(
            ["classes", "samples"], ascending=False
        ).index.astype(str).tolist()
    if len(eligible) < 2:
        raise ValueError("at least two batches are required for validation and testing")
    return sorted(eligible)


def _purge_group_overlap(
    df: pd.DataFrame,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
) -> tuple[list[int], list[int], list[int], dict]:
    """Give test > validation > train priority for duplicated lot groups."""
    test_groups = set(df.iloc[list(test_idx)]["_group_id"])
    val = df.iloc[list(val_idx)]
    val = val[~val["_group_id"].isin(test_groups)]
    val_groups = set(val["_group_id"])
    train = df.iloc[list(train_idx)]
    train = train[~train["_group_id"].isin(test_groups | val_groups)]
    removed = {
        "train": int(len(train_idx) - len(train)),
        "validation": int(len(val_idx) - len(val)),
        "test": 0,
    }
    return (
        train["_row_index"].astype(int).tolist(),
        val["_row_index"].astype(int).tolist(),
        [int(i) for i in test_idx],
        removed,
    )


def validate_manifest(df: pd.DataFrame, manifest: dict, spec: ProtocolSpec) -> None:
    parts = {
        "train": set(manifest["train_idx"]),
        "validation": set(manifest["val_idx"]),
        "closed_test": set(manifest["test_batch_idx"]),
        "novel": set(manifest["test_unknown_idx"]),
    }
    names = list(parts)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = parts[left] & parts[right]
            if overlap:
                raise ValueError(f"row leakage between {left} and {right}: {len(overlap)}")
    groups = {
        name: set(df.iloc[sorted(rows)]["_group_id"]) if rows else set()
        for name, rows in parts.items()
    }
    for left, right in (("train", "validation"), ("train", "closed_test"),
                        ("validation", "closed_test")):
        overlap = groups[left] & groups[right]
        if overlap:
            raise ValueError(f"lot/group leakage between {left} and {right}: {len(overlap)}")
    train_batches = set(df.iloc[manifest["train_idx"]][spec.batch_col])
    val_batches = set(df.iloc[manifest["val_idx"]][spec.batch_col])
    test_batches = set(df.iloc[manifest["test_batch_idx"]][spec.batch_col])
    if train_batches & val_batches or train_batches & test_batches or val_batches & test_batches:
        raise ValueError("batch leakage in closed-set partitions")
    if train_batches and val_batches and max(train_batches) >= min(val_batches):
        raise ValueError("training batches must precede validation batches")
    if val_batches and test_batches and max(val_batches) >= min(test_batches):
        raise ValueError("validation batches must precede closed-test batches")
    train_products = set(df.iloc[manifest["train_idx"]][spec.product_col])
    if set(manifest["known_products"]) - train_products:
        missing = sorted(set(manifest["known_products"]) - train_products)
        raise ValueError(f"known products absent from training after purge: {missing}")


def build_outer_manifest(
    df: pd.DataFrame,
    novel_products: Sequence[str],
    spec: ProtocolSpec,
    fold_index: int,
) -> dict:
    viable = eligible_products(df, spec)
    novel = sorted(map(str, novel_products))
    invalid_novel = sorted(set(novel) - set(viable))
    if invalid_novel:
        raise ValueError(f"novel products are not eligible: {invalid_novel}")
    max_shot = max(spec.shots) if spec.shots else 1
    if not _fewshot_cutoffs(df, novel, max_shot, spec):
        raise ValueError(f"novel products cannot support {max_shot}-shot temporal episodes: {novel}")
    known = sorted(set(viable) - set(novel))
    if not known:
        raise ValueError("fold has no known products")
    batches = _candidate_batches(df, known, spec)
    n_test = max(1, int(math.ceil(len(batches) * spec.closed_test_batch_ratio)))
    n_test = min(n_test, max(len(batches) - 2, 1))
    test_batches = batches[-n_test:]
    remaining = [b for b in batches if b not in test_batches]
    n_val = max(1, int(math.ceil(len(remaining) * spec.validation_batch_ratio)))
    n_val = min(n_val, max(len(remaining) - 1, 1))
    val_batches = remaining[-n_val:]

    is_known = df[spec.product_col].isin(known)
    first_validation_batch = min(val_batches)
    train_idx = df[
        is_known & df[spec.batch_col].astype(str).lt(first_validation_batch)
    ]["_row_index"].tolist()
    val_idx = df[is_known & df[spec.batch_col].isin(val_batches)]["_row_index"].tolist()
    test_idx = df[is_known & df[spec.batch_col].isin(test_batches)]["_row_index"].tolist()
    novel_idx = df[df[spec.product_col].isin(novel)]["_row_index"].astype(int).tolist()
    train_idx, val_idx, test_idx, purged = _purge_group_overlap(
        df, train_idx, val_idx, test_idx
    )
    assigned_known = set(train_idx) | set(val_idx) | set(test_idx)
    unused_known_idx = df[
        is_known & ~df["_row_index"].isin(assigned_known)
    ]["_row_index"].astype(int).tolist()

    excluded = sorted(set(df[spec.product_col].unique()) - set(viable))
    manifest = {
        "protocol_version": 1,
        "fold_id": f"products_{fold_index:02d}_" + "-".join(novel),
        "known_products": known,
        "holdout_products": novel,
        "excluded_products": excluded,
        "train_batches": sorted(set(df.iloc[train_idx][spec.batch_col])),
        "model_train_batches": sorted(set(df.iloc[train_idx][spec.batch_col])),
        "model_select_holdout_batches": sorted(set(df.iloc[val_idx][spec.batch_col])),
        "holdout_batches": sorted(set(df.iloc[test_idx][spec.batch_col])),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_batch_idx": test_idx,
        "test_unknown_idx": novel_idx,
        "unused_known_idx": unused_known_idx,
        "seed": spec.seed,
        "split_seed": spec.seed,
        "metadata_sha256": metadata_fingerprint(df, spec),
        "independent_group_definition": f"{spec.product_col}+{spec.lot_col}; sample fallback",
        "purged_cross_partition_rows": purged,
        "fewshot": {
            "shots": list(spec.shots),
            "episodes": spec.episodes,
            "support_query_batch_disjoint": True,
            "support_query_group_disjoint": True,
            "future_query_only": spec.future_query_only,
            "min_query_per_class": spec.min_novel_query_per_class,
        },
        "stats": {
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test_batch": len(test_idx),
            "n_test_unknown": len(novel_idx),
            "n_unused_known": len(unused_known_idx),
            "n_excluded": int(df[spec.product_col].isin(excluded).sum()),
        },
    }
    validate_manifest(df, manifest, spec)
    return manifest


def build_protocol_manifests(
    metadata_path: str | Path,
    output_dir: str | Path,
    spec: ProtocolSpec | None = None,
    novel_product_sets: Iterable[Sequence[str]] | None = None,
) -> list[dict]:
    spec = spec or ProtocolSpec()
    df = load_metadata(metadata_path, spec)
    viable = eligible_products(df, spec)
    folds = (
        list(novel_product_sets)
        if novel_product_sets is not None
        else _select_product_folds(df, viable, spec)
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifests = []
    for fold_index, products in enumerate(folds):
        manifest = build_outer_manifest(df, products, spec, fold_index)
        manifests.append(manifest)
        path = output / f"{manifest['fold_id']}.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = audit_metadata(df, spec)
    audit["protocol_spec"] = asdict(spec)
    audit["eligible_products"] = viable
    audit["manifests"] = [m["fold_id"] for m in manifests]
    (output / "data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifests


def make_fewshot_episode(
    df: pd.DataFrame,
    manifest: dict,
    shot: int,
    seed: int,
    spec: ProtocolSpec,
) -> dict:
    """Create a reproducible, batch- and lot-disjoint registration episode."""
    rng = np.random.RandomState(seed)
    support_idx: list[int] = []
    query_idx: list[int] = []
    details = {}
    novel_df = df.iloc[manifest["test_unknown_idx"]]
    order = _batch_order(novel_df, spec)
    cutoff_candidates = _fewshot_cutoffs(
        novel_df, manifest["holdout_products"], shot, spec
    )
    if not cutoff_candidates:
        raise ValueError(
            f"no global temporal cutoff supports {shot}-shot registration for "
            f"{manifest['holdout_products']}"
        )
    cutoff, available_support_batches = cutoff_candidates[
        rng.randint(len(cutoff_candidates))
    ]
    cutoff_batch = next(batch for batch, rank in order.items() if rank == cutoff)
    for product in manifest["holdout_products"]:
        part = novel_df[novel_df[spec.product_col] == product]
        product_batches = available_support_batches[str(product)]
        batch = product_batches[rng.randint(len(product_batches))]
        support_pool = part[part[spec.batch_col].astype(str) == batch]
        if spec.future_query_only:
            query_pool = part[
                part[spec.batch_col].astype(str).map(order).gt(cutoff)
            ]
        else:
            query_pool = part[part[spec.batch_col].astype(str) != batch]
        group_names = support_pool["_group_id"].unique().tolist()
        selected_groups = rng.choice(group_names, size=shot, replace=False).tolist()
        selected_rows = []
        for group in selected_groups:
            rows = support_pool[support_pool["_group_id"] == group]["_row_index"].tolist()
            selected_rows.append(int(rows[rng.randint(len(rows))]))
        support_groups = set(df.iloc[selected_rows]["_group_id"])
        query_pool = query_pool[~query_pool["_group_id"].isin(support_groups)]
        if query_pool["_group_id"].nunique() < spec.min_novel_query_per_class:
            raise ValueError(f"{product}: too few query samples after lot purge")
        product_query = query_pool["_row_index"].astype(int).tolist()
        support_idx.extend(selected_rows)
        query_idx.extend(product_query)
        details[product] = {
            "support_batch": batch,
            "query_batches": sorted(query_pool[spec.batch_col].astype(str).unique().tolist()),
            "support_groups": sorted(support_groups),
            "n_query": len(product_query),
        }
    support_batches = set(df.iloc[support_idx][spec.batch_col])
    query_batches = set(df.iloc[query_idx][spec.batch_col])
    support_groups = set(df.iloc[support_idx]["_group_id"])
    query_groups = set(df.iloc[query_idx]["_group_id"])
    if support_batches & query_batches:
        raise AssertionError("support/query batch leakage")
    if support_groups & query_groups:
        raise AssertionError("support/query lot leakage")
    return {
        "shot": int(shot),
        "seed": int(seed),
        "support_idx": support_idx,
        "novel_query_idx": query_idx,
        "temporal_cutoff_batch": cutoff_batch,
        "details": details,
    }
