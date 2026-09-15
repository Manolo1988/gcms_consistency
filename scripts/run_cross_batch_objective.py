#!/usr/bin/env python3
"""Train one backbone with an explicit cross-batch episodic prototype loss."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Sampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines import PlainEncoder
from config import Config, get_device
from dataset import GCMSAugmentation, GCMSDataset, load_data_split
from models import GCMSEncoder
from paper_protocol import (
    evaluate_embedding_episodes,
    load_filtered_metadata,
    load_json,
    metadata_fingerprint,
    save_json,
    write_rows_csv,
)
from train import set_seed


class CrossBatchEpisodeSampler(Sampler):
    """Yield product-balanced episodes with distinct support/query batches."""

    def __init__(self, frame, ways, support, query, steps, seed):
        self.ways = int(ways)
        self.support = int(support)
        self.query = int(query)
        self.steps = int(steps)
        self.seed = int(seed)
        self.epoch = 0
        self.groups = {}
        for product, product_frame in frame.groupby("product_label"):
            batches = {
                str(batch): group.index.to_numpy(dtype=np.int64)
                for batch, group in product_frame.groupby("batch_idx")
            }
            batches = {key: value for key, value in batches.items() if len(value)}
            if len(batches) >= 2:
                self.groups[int(product)] = batches
        if len(self.groups) < self.ways:
            raise ValueError(
                f"Only {len(self.groups)} products have >=2 batches; ways={self.ways}"
            )

    def __len__(self):
        return self.steps

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch * 100003)
        products = np.asarray(sorted(self.groups), dtype=np.int64)
        for _ in range(self.steps):
            chosen_products = rng.choice(products, self.ways, replace=False)
            episode = []
            for product in chosen_products:
                groups = self.groups[int(product)]
                batch_names = np.asarray(sorted(groups), dtype=object)
                support_batch, query_batch = rng.choice(batch_names, 2, replace=False)
                support_pool = groups[str(support_batch)]
                query_pool = groups[str(query_batch)]
                episode.extend(rng.choice(
                    support_pool, self.support,
                    replace=len(support_pool) < self.support,
                ).tolist())
                episode.extend(rng.choice(
                    query_pool, self.query,
                    replace=len(query_pool) < self.query,
                ).tolist())
            yield episode


def configure_grid(cfg):
    path = Path(cfg.prepared_dir) / "grid_info.json"
    if not path.exists():
        return
    grid = json.loads(path.read_text(encoding="utf-8"))
    cfg.rt_bins = int(grid.get("input_pca_rt_bins", grid.get("rt_bins", cfg.rt_bins)))
    cfg.mz_bins = int(grid.get("input_pca_components", grid.get("mz_bins", cfg.mz_bins)))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def aligned_dataset(metadata_csv, indices, source, augmentation=None):
    dataset = GCMSDataset(
        metadata_csv, product_col="product_fine", augmentation=augmentation,
        indices=indices,
    )
    dataset.product_enc = source.product_enc
    dataset.df["product_label"] = source.product_enc.transform(
        dataset.df["product_fine"]
    )
    return dataset


def make_datasets(metadata_csv, split, cfg):
    train = GCMSDataset(
        metadata_csv, product_col="product_fine", augmentation=GCMSAugmentation(cfg),
        indices=split["train_idx"],
    )
    train_noaug = aligned_dataset(metadata_csv, split["train_idx"], train)
    val = aligned_dataset(metadata_csv, split["val_idx"], train)
    closed = aligned_dataset(metadata_csv, split["test_batch_idx"], train)
    unknown = GCMSDataset(
        metadata_csv, product_col="product_fine", augmentation=None,
        indices=split["test_unknown_idx"],
    )
    return train, train_noaug, val, closed, unknown


class CrossBatchModel(nn.Module):
    def __init__(self, method, num_products, cfg):
        super().__init__()
        if method == "plain_cnn_crossbatch":
            self.encoder = PlainEncoder(
                in_channels=cfg.in_channels, channels=cfg.encoder_channels,
                dropout=cfg.dropout, blocks_per_stage=cfg.blocks_per_stage,
            )
        else:
            self.encoder = GCMSEncoder(
                in_channels=cfg.in_channels, channels=cfg.encoder_channels,
                num_heads=cfg.num_axial_heads, dropout=cfg.dropout,
                blocks_per_stage=cfg.blocks_per_stage,
            )
        self.classifier = nn.Linear(self.encoder.out_dim, num_products)

    def forward(self, inputs):
        raw, _ = self.encoder(inputs)
        return F.normalize(raw, dim=1), self.classifier(raw)

    def encode(self, inputs):
        raw, _ = self.encoder(inputs)
        return F.normalize(raw, dim=1)


def build_model(method, num_products, cfg, device):
    return CrossBatchModel(method, num_products, cfg).to(device)


def embedding_logits(model, inputs, method):
    del method
    return model(inputs)


def episodic_loss(embeddings, ways, support, query, temperature):
    block = support + query
    shaped = embeddings.reshape(ways, block, -1)
    prototypes = F.normalize(shaped[:, :support].mean(dim=1), dim=1)
    queries = F.normalize(shaped[:, support:].reshape(ways * query, -1), dim=1)
    targets = torch.arange(ways, device=embeddings.device).repeat_interleave(query)
    return F.cross_entropy(queries @ prototypes.T / temperature, targets)


def train_epoch(model, loader, sampler, optimizer, method, args, device, epoch):
    sampler.set_epoch(epoch)
    model.train()
    totals = {"total": 0.0, "ce": 0.0, "cross_batch_proto": 0.0}
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        labels = batch["product"].to(device, non_blocking=True)
        embeddings, logits = embedding_logits(model, inputs, method)
        ce = F.cross_entropy(logits, labels)
        proto = episodic_loss(
            embeddings, args.ways, args.support, args.query, args.temperature
        )
        total = args.lambda_ce * ce + args.lambda_proto * proto
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        totals["total"] += total.item()
        totals["ce"] += ce.item()
        totals["cross_batch_proto"] += proto.item()
    return {key: value / max(len(loader), 1) for key, value in totals.items()}


@torch.no_grad()
def extract_embeddings(model, loader, method, device):
    model.eval()
    chunks = []
    for batch in loader:
        embeddings, _ = embedding_logits(
            model, batch["input"].to(device, non_blocking=True), method
        )
        chunks.append(embeddings.cpu().numpy())
    return np.concatenate(chunks)


@torch.no_grad()
def prototype_metrics(model, reference_loader, query_loader, method, device):
    reference = extract_embeddings(model, reference_loader, method, device)
    reference_labels = np.concatenate([
        batch["product"].numpy() for batch in reference_loader
    ])
    query = extract_embeddings(model, query_loader, method, device)
    truth = np.concatenate([batch["product"].numpy() for batch in query_loader])
    classes = np.unique(reference_labels)
    prototypes = np.stack([
        reference[reference_labels == label].mean(axis=0) for label in classes
    ])
    prototypes /= np.clip(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12, None)
    predictions = classes[np.argmax(query @ prototypes.T, axis=1)]
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_acc": float(balanced_accuracy_score(truth, predictions)),
    }


def embedding_geometry(values, frame):
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
    result = {}
    for name, mask in categories.items():
        similarities = np.sum(values[left[mask]] * values[right[mask]], axis=1)
        if len(similarities):
            result[name] = {
                "n_pairs": int(len(similarities)),
                "cosine_similarity_mean": float(similarities.mean()),
                "cosine_distance_mean": float((1.0 - similarities).mean()),
            }
    same = result.get("same_product_same_batch", {}).get("cosine_distance_mean")
    cross = result.get("same_product_cross_batch", {}).get("cosine_distance_mean")
    result["same_product_batch_gap"] = (
        float(cross - same) if same is not None and cross is not None else None
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=["plain_cnn_crossbatch", "dual_axis_crossbatch"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--prepared-dir", default="new_prepared_data_relabel_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-manifest", default="result/paper_gate/fewshot_episodes.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.00026)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--ways", type=int, default=4)
    parser.add_argument("--support", type=int, default=2)
    parser.add_argument("--query", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument("--lambda-proto", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed, deterministic=True)
    cfg = Config()
    cfg.seed = args.seed
    cfg.prepared_dir = str(Path(args.prepared_dir).resolve())
    cfg.output_dir = str(Path(args.output_dir).resolve())
    configure_grid(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = load_data_split(cfg)
    metadata_csv = Path(cfg.prepared_dir) / "metadata.csv"
    train, train_noaug, val, closed, unknown = make_datasets(
        metadata_csv, split, cfg
    )
    device = get_device()
    model = build_model(args.method, train.num_products, cfg, device)

    episode_size = args.ways * (args.support + args.query)
    steps = args.steps_per_epoch or math.ceil(len(train) / episode_size)
    sampler = CrossBatchEpisodeSampler(
        train.df, args.ways, args.support, args.query, steps, args.seed
    )
    train_loader = DataLoader(
        train, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
    )
    eval_kwargs = {
        "batch_size": args.eval_batch_size, "shuffle": False,
        "num_workers": args.num_workers, "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_eval_loader = DataLoader(train_noaug, **eval_kwargs)
    val_loader = DataLoader(val, **eval_kwargs)
    closed_loader = DataLoader(closed, **eval_kwargs)
    unknown_loader = DataLoader(unknown, **eval_kwargs)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    best_score = -1.0
    best_state = None
    history = []
    for epoch in range(args.epochs):
        losses = train_epoch(
            model, train_loader, sampler, optimizer, args.method, args, device, epoch
        )
        scheduler.step()
        record = {"epoch": epoch + 1, **losses}
        if (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs:
            validation = prototype_metrics(
                model, train_eval_loader, val_loader, args.method, device
            )
            record.update({f"val_{key}": value for key, value in validation.items()})
            print(
                f"{args.method} epoch={epoch + 1}/{args.epochs} "
                f"loss={losses['total']:.4f} "
                f"val_proto_acc={validation['accuracy']:.4f}",
                flush=True,
            )
            if validation["accuracy"] > best_score:
                best_score = validation["accuracy"]
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    torch.save(model.state_dict(), output_dir / "model.pt")
    closed_metrics = prototype_metrics(
        model, train_eval_loader, closed_loader, args.method, device
    )
    manifest = load_json(args.episode_manifest)
    filtered_df = load_filtered_metadata(metadata_csv)
    if manifest["metadata_fingerprint"] != metadata_fingerprint(filtered_df):
        raise ValueError("episode manifest metadata fingerprint mismatch")
    unknown_embeddings = extract_embeddings(
        model, unknown_loader, args.method, device
    )
    train_embeddings = extract_embeddings(
        model, train_eval_loader, args.method, device
    )
    closed_embeddings = extract_embeddings(
        model, closed_loader, args.method, device
    )
    geometry = {
        "train": embedding_geometry(train_embeddings, train_noaug.df),
        "closed": embedding_geometry(closed_embeddings, closed.df),
        "unknown": embedding_geometry(unknown_embeddings, unknown.df),
    }
    fewshot_rows = evaluate_embedding_episodes(
        unknown_embeddings, manifest, args.method, args.seed
    )
    write_rows_csv(fewshot_rows, output_dir / "paper_fewshot_episode_results.csv")
    save_json(history, output_dir / "training_history.json")
    save_json(geometry, output_dir / "embedding_geometry.json")
    save_json(split, output_dir / "split.json")
    save_json(vars(args), output_dir / "run_config.json")
    payload = {
        "method": args.method,
        "train_seed": args.seed,
        "split_id": metadata_fingerprint(filtered_df)[:16],
        "checkpoint_selection": "validation_cross_batch_prototype_accuracy",
        "best_val_accuracy": best_score,
        "closed_set": closed_metrics,
        "embedding_geometry": geometry,
        "fewshot_protocol": {
            "name": manifest["protocol"], "shots": manifest["shots"],
            "episodes": manifest["episode_count"],
            "episode_seed_start": manifest["seed_start"],
        },
        "training_objective": {
            "lambda_ce": args.lambda_ce, "lambda_proto": args.lambda_proto,
            "ways": args.ways, "support": args.support, "query": args.query,
            "temperature": args.temperature,
            "support_query_batches_distinct": True,
        },
        "fewshot_results_csv": "paper_fewshot_episode_results.csv",
    }
    save_json(payload, output_dir / "paper_gate_result.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
