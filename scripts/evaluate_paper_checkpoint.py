#!/usr/bin/env python3
"""Evaluate one trained checkpoint with shared cross-batch few-shot episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines import extract_tic_features
from config import Config, get_device
from dataset import GCMSDataset
from models import GCMSConsistencyNet
from paper_protocol import (
    build_cross_batch_episodes,
    evaluate_embedding_episodes,
    load_filtered_metadata,
    load_json,
    metadata_fingerprint,
    save_json,
    validate_episode_manifest,
    write_rows_csv,
)


def _apply_run_config(cfg, run_dir):
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return
    payload = load_json(config_path)
    values = payload.get("args", payload)
    for key, value in values.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)


def _load_model(run_dir, cfg, device):
    model_dir = run_dir / "final_model"
    train_meta = load_json(model_dir / "train_meta.json")
    cfg.feature_dim = int(train_meta.get("feature_dim", cfg.feature_dim))
    cfg.proj_dim = int(train_meta.get("proj_dim", cfg.proj_dim))
    cfg.rt_bins = int(train_meta.get("input_raw_pca_rt_bins", cfg.rt_bins))
    if bool(train_meta.get("input_raw_pca_enabled", False)):
        cfg.mz_bins = int(train_meta.get("input_raw_pca_components", cfg.mz_bins))
    model = GCMSConsistencyNet(
        int(train_meta["num_batches"]),
        cfg,
        num_products=int(train_meta.get("num_products", 0)) or None,
    ).to(device)
    state = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _input_transform(run_dir, cfg):
    pca_path = run_dir / "final_model" / "input_rt_pca.pkl"
    if not pca_path.exists():
        return None
    from input_pca import RtAxisPcaTransform, load_rt_axis_pca

    model = load_rt_axis_pca(pca_path)
    cfg.mz_bins = int(getattr(model, "n_components_", cfg.mz_bins))
    return RtAxisPcaTransform(model)


@torch.no_grad()
def _extract_embeddings(model, loader, device):
    values = []
    for batch in loader:
        values.append(model.encode(batch["input"].to(device)).cpu().numpy())
    return np.concatenate(values, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prepared-dir")
    parser.add_argument("--method-name", default="main")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-seed-start", type=int, default=42000)
    parser.add_argument("--episode-manifest")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg = Config()
    _apply_run_config(cfg, run_dir)
    if args.prepared_dir:
        cfg.prepared_dir = str(Path(args.prepared_dir).resolve())
    prepared_dir = Path(cfg.prepared_dir)
    split = load_json(run_dir / "final_model" / "split.json")
    metadata_csv = prepared_dir / "metadata.csv"
    filtered_df = load_filtered_metadata(metadata_csv)

    if args.episode_manifest:
        manifest = load_json(args.episode_manifest)
        if manifest["metadata_fingerprint"] != metadata_fingerprint(filtered_df):
            raise ValueError("episode manifest metadata fingerprint does not match prepared data")
    else:
        manifest = build_cross_batch_episodes(
            metadata_csv,
            split["test_unknown_idx"],
            shots=args.shots,
            episode_count=args.episodes,
            seed_start=args.episode_seed_start,
        )
        manifest["validation"] = validate_episode_manifest(manifest)
        save_json(manifest, run_dir / "paper_fewshot_episodes.json")

    input_transform = _input_transform(run_dir, cfg)
    unknown_idx = [int(value) for value in split["test_unknown_idx"]]
    dataset = GCMSDataset(
        metadata_csv,
        product_col="product_fine",
        augmentation=None,
        indices=unknown_idx,
        input_transform=input_transform,
    )
    global_products = LabelEncoder().fit(sorted(filtered_df["product_fine"].unique()))
    dataset.product_enc = global_products
    dataset.df["product_label"] = global_products.transform(dataset.df["product_fine"])
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    device = get_device()
    model = _load_model(run_dir, cfg, device)
    embeddings = _extract_embeddings(model, loader, device)
    main_rows = evaluate_embedding_episodes(
        embeddings,
        manifest,
        method_name=args.method_name,
        train_seed=int(getattr(cfg, "seed", 0)),
    )
    tic_features, _, _ = extract_tic_features(loader)
    tic_rows = evaluate_embedding_episodes(
        tic_features,
        manifest,
        method_name="tic_cosine_prototype",
        train_seed=int(getattr(cfg, "seed", 0)),
    )
    train_dataset = GCMSDataset(
        metadata_csv,
        product_col="product_fine",
        augmentation=None,
        indices=split["train_idx"],
        input_transform=input_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    train_tic, _, _ = extract_tic_features(train_loader)
    scaler = StandardScaler()
    train_tic_scaled = scaler.fit_transform(train_tic)
    unknown_tic_scaled = scaler.transform(tic_features)
    component_count = max(1, min(32, len(train_tic_scaled) - 1, train_tic_scaled.shape[1]))
    pca = PCA(n_components=component_count, random_state=int(getattr(cfg, "seed", 0)))
    pca.fit(train_tic_scaled)
    tic_pca_rows = evaluate_embedding_episodes(
        pca.transform(unknown_tic_scaled),
        manifest,
        method_name="tic_pca_prototype",
        train_seed=int(getattr(cfg, "seed", 0)),
    )
    rows = main_rows + tic_rows + tic_pca_rows
    write_rows_csv(rows, run_dir / "paper_fewshot_episode_results.csv")

    summary_path = run_dir / "evaluation_summary.json"
    closed_set = load_json(summary_path).get("setting_a", {}) if summary_path.exists() else {}
    payload = {
        "method": args.method_name,
        "train_seed": int(getattr(cfg, "seed", 0)),
        "split_id": metadata_fingerprint(filtered_df)[:16],
        "checkpoint_selection": load_json(run_dir / "final_model" / "train_meta.json").get(
            "model_select_metric"
        ),
        "closed_set": closed_set,
        "fewshot_protocol": {
            "name": manifest["protocol"],
            "shots": manifest["shots"],
            "episodes": manifest["episode_count"],
            "episode_seed_start": manifest["seed_start"],
        },
        "fewshot_results_csv": "paper_fewshot_episode_results.csv",
    }
    save_json(payload, run_dir / "paper_gate_result.json")
    print(f"saved paper gate results to {run_dir / 'paper_gate_result.json'}")


if __name__ == "__main__":
    main()
