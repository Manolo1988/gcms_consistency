#!/usr/bin/env python3
"""Train the full GCMS model with an explicit cross-batch prototype objective."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Sampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config, get_device
from dataset import GCMSAugmentation, GCMSDataset, load_data_split
from losses import UnifiedLoss
from models import GCMSConsistencyNet
from paper_protocol import load_json, save_json
from train import (
    _attach_focal_class_weights,
    _build_input_transform_for_training,
    _channels_last_enabled,
    _enable_cuda_fast_paths,
    _loader_runtime_kwargs,
    _model_state_dict_for_save,
    set_seed,
)


class CrossBatchEpisodeSampler(Sampler):
    """Build product-balanced episodes with distinct support/query batches."""

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
                f"Only {len(self.groups)} products have at least two batches; "
                f"ways={self.ways}"
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
    grid_path = Path(cfg.prepared_dir) / "grid_info.json"
    if not grid_path.exists():
        return
    grid = load_json(grid_path)
    cfg.rt_bins = int(grid.get("input_pca_rt_bins", grid.get("rt_bins", cfg.rt_bins)))
    cfg.mz_bins = int(grid.get("input_pca_components", grid.get("mz_bins", cfg.mz_bins)))


def align_dataset(metadata_csv, indices, source, augmentation=None, input_transform=None):
    dataset = GCMSDataset(
        metadata_csv,
        product_col="product_fine",
        augmentation=augmentation,
        indices=indices,
        input_transform=input_transform,
    )
    dataset.product_enc = source.product_enc
    dataset.batch_enc = source.batch_enc
    dataset.df["product_label"] = source.product_enc.transform(dataset.df["product_fine"])
    dataset.df["batch_label"] = source.batch_enc.transform(dataset.df["batch_idx"])
    dataset.num_products = source.num_products
    dataset.num_batches = source.num_batches
    return dataset


def make_datasets(metadata_csv, split, cfg, input_transform):
    train = GCMSDataset(
        metadata_csv,
        product_col="product_fine",
        augmentation=GCMSAugmentation(cfg),
        indices=split["train_idx"],
        input_transform=input_transform,
    )
    raw_val = GCMSDataset(
        metadata_csv,
        product_col="product_fine",
        indices=split["val_idx"],
        input_transform=input_transform,
    )
    raw_closed = GCMSDataset(
        metadata_csv,
        product_col="product_fine",
        indices=split["test_batch_idx"],
        input_transform=input_transform,
    )

    # LOBO/跨批次划分可能把验证批次或 Setting A 留出批次完全放在
    # train 之外。所有会被加载的数据集必须共用覆盖完整范围的编码器。
    shared_product_enc = LabelEncoder().fit(np.concatenate([
        train.df["product_fine"].to_numpy(),
        raw_val.df["product_fine"].to_numpy(),
        raw_closed.df["product_fine"].to_numpy(),
    ]))
    shared_batch_enc = LabelEncoder().fit(np.concatenate([
        train.df["batch_idx"].to_numpy(),
        raw_val.df["batch_idx"].to_numpy(),
        raw_closed.df["batch_idx"].to_numpy(),
    ]))
    train.product_enc = shared_product_enc
    train.batch_enc = shared_batch_enc
    train.df["product_label"] = shared_product_enc.transform(
        train.df["product_fine"]
    )
    train.df["batch_label"] = shared_batch_enc.transform(train.df["batch_idx"])
    train.num_products = len(shared_product_enc.classes_)
    train.num_batches = len(shared_batch_enc.classes_)

    train_noaug = align_dataset(
        metadata_csv, split["train_idx"], train, input_transform=input_transform
    )
    val = align_dataset(
        metadata_csv, split["val_idx"], train, input_transform=input_transform
    )
    closed = align_dataset(
        metadata_csv, split["test_batch_idx"], train, input_transform=input_transform
    )
    return train, train_noaug, val, closed


def cross_batch_prototype_loss(embeddings, ways, support, query, temperature):
    block = int(support) + int(query)
    expected = int(ways) * block
    if embeddings.shape[0] != expected:
        raise ValueError(f"Episode size mismatch: got {embeddings.shape[0]}, expected {expected}")
    shaped = embeddings.reshape(int(ways), block, -1)
    prototypes = F.normalize(shaped[:, :support].mean(dim=1), dim=1)
    queries = F.normalize(shaped[:, support:].reshape(int(ways) * int(query), -1), dim=1)
    targets = torch.arange(int(ways), device=embeddings.device).repeat_interleave(query)
    return F.cross_entropy((queries @ prototypes.T) / float(temperature), targets)


@torch.no_grad()
def extract_embeddings(model, loader, device, cfg):
    model.eval()
    values, labels = [], []
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        if _channels_last_enabled(cfg, device) and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        values.append(model.encode(x).cpu())
        labels.append(batch["product"].cpu())
    return torch.cat(values), torch.cat(labels)


@torch.no_grad()
def prototype_metrics(model, reference_loader, query_loader, device, cfg):
    reference, reference_labels = extract_embeddings(model, reference_loader, device, cfg)
    query, query_labels = extract_embeddings(model, query_loader, device, cfg)
    classes = torch.unique(reference_labels, sorted=True)
    prototypes = torch.stack([
        reference[reference_labels == label].mean(dim=0) for label in classes
    ])
    prototypes = F.normalize(prototypes, dim=1)
    predictions = classes[(F.normalize(query, dim=1) @ prototypes.T).argmax(dim=1)]
    return {
        "accuracy": float(accuracy_score(query_labels.numpy(), predictions.numpy())),
        "balanced_acc": float(balanced_accuracy_score(query_labels.numpy(), predictions.numpy())),
    }


def train_one_epoch(model, loader, sampler, criterion, optimizer, device, cfg, epoch):
    model.train()
    sampler.set_epoch(epoch)
    progress = {"total": 0.0, "cross_batch_proto": 0.0}
    alpha = 2.0 / (1.0 + np.exp(-10.0 * epoch / max(cfg.epochs, 1))) - 1.0
    model.domain_head.set_alpha(alpha)
    for batch in loader:
        batch_dev = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        output = model(batch_dev["input"])
        base_losses = criterion(output, batch_dev)
        cross_loss = cross_batch_prototype_loss(
            output["z"], cfg.cross_batch_ways, cfg.cross_batch_support,
            cfg.cross_batch_query, cfg.cross_batch_temperature,
        )
        total = base_losses["total"] + cfg.lambda_cross_batch_proto * cross_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        for key, value in base_losses.items():
            progress[key] = progress.get(key, 0.0) + float(value.detach().cpu())
        progress["cross_batch_proto"] += float(cross_loss.detach().cpu())
        progress["total"] += float(total.detach().cpu())
    count = max(len(loader), 1)
    return {key: value / count for key, value in progress.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prepared-dir", default="new_prepared_data_relabel_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.00026)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--ways", type=int, default=4)
    parser.add_argument("--support", type=int, default=2)
    parser.add_argument("--query", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lambda-cross-batch-proto", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed, deterministic=True)
    cfg = Config()
    cfg.seed = args.seed
    cfg.prepared_dir = str(Path(args.prepared_dir).resolve())
    cfg.output_dir = str(Path(args.output_dir).resolve())
    cfg.epochs = int(args.epochs)
    cfg.lr = float(args.lr)
    cfg.weight_decay = float(args.weight_decay)
    cfg.cross_batch_ways = int(args.ways)
    cfg.cross_batch_support = int(args.support)
    cfg.cross_batch_query = int(args.query)
    cfg.cross_batch_temperature = float(args.temperature)
    cfg.lambda_supcon = 1.0
    cfg.lambda_adv = 0.06
    cfg.lambda_proto = 0.75
    cfg.lambda_recon = 0.30
    cfg.lambda_cls = 0.25
    cfg.lambda_hard_pair = 0.05
    cfg.supcon_temperature = 0.07
    cfg.proto_margin = 1.5
    cfg.hard_pair_margin = 0.40
    cfg.focal_gamma = 0.0
    cfg.input_raw_pca_enabled = True
    cfg.input_raw_pca_components = 256
    cfg.rt_range = (3.17, 36.91)
    cfg.mz_range = (30.0, 200.0)
    cfg.lambda_cross_batch_proto = float(args.lambda_cross_batch_proto)
    cfg.dataloader_workers = int(args.num_workers)
    cfg.deterministic = True
    configure_grid(cfg)

    output_dir = Path(cfg.output_dir)
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    split = load_data_split(cfg)
    metadata_csv = Path(cfg.prepared_dir) / "metadata.csv"
    input_transform, _ = _build_input_transform_for_training(
        cfg, str(metadata_csv), split["train_idx"]
    )
    train, train_noaug, val, closed = make_datasets(
        str(metadata_csv), split, cfg, input_transform
    )
    device = get_device()
    _enable_cuda_fast_paths(cfg, device)
    loader_kwargs = _loader_runtime_kwargs(cfg, device)
    episode_size = cfg.cross_batch_ways * (cfg.cross_batch_support + cfg.cross_batch_query)
    sampler = CrossBatchEpisodeSampler(
        train.df, cfg.cross_batch_ways, cfg.cross_batch_support,
        cfg.cross_batch_query, math.ceil(len(train) / episode_size), args.seed,
    )
    loader_train = DataLoader(
        train, batch_sampler=sampler, **loader_kwargs,
    )
    loader_train_noaug = DataLoader(train_noaug, batch_size=64, shuffle=False, **loader_kwargs)
    loader_val = DataLoader(val, batch_size=64, shuffle=False, **loader_kwargs)
    loader_closed = DataLoader(closed, batch_size=64, shuffle=False, **loader_kwargs)

    _attach_focal_class_weights(cfg, train)
    model = GCMSConsistencyNet(train.num_batches, cfg, num_products=train.num_products).to(device)
    criterion = UnifiedLoss(cfg).to(device)
    criterion.set_label_names(train.get_label_name_map())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    best_score = -1.0
    best_state = None
    history = []
    for epoch in range(cfg.epochs):
        losses = train_one_epoch(model, loader_train, sampler, criterion, optimizer, device, cfg, epoch)
        scheduler.step()
        record = {"epoch": epoch + 1, **losses}
        if (epoch + 1) % args.eval_interval == 0 or epoch + 1 == cfg.epochs:
            validation = prototype_metrics(model, loader_train_noaug, loader_val, device, cfg)
            record.update({f"val_{key}": value for key, value in validation.items()})
            print(
                f"{epoch + 1}/{cfg.epochs} total={losses['total']:.4f} "
                f"cross_batch_proto={losses['cross_batch_proto']:.4f} "
                f"val_cross_batch_acc={validation['accuracy']:.4f}", flush=True,
            )
            if validation["accuracy"] > best_score:
                best_score = validation["accuracy"]
                best_state = copy.deepcopy(_model_state_dict_for_save(model))
        history.append(record)

    if best_state is None:
        best_state = _model_state_dict_for_save(model)
    model.load_state_dict(best_state)
    torch.save(best_state, final_dir / "model.pt")
    save_json(split, final_dir / "split.json")
    save_json(list(train.product_enc.classes_), final_dir / "product_classes.json")
    save_json(history, output_dir / "training_history.json")
    save_json({
        "command": "train",
        "args": vars(args),
        "config": {
            "main_backbone": cfg.main_backbone,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "lambda_supcon": cfg.lambda_supcon,
            "lambda_adv": cfg.lambda_adv,
            "lambda_proto": cfg.lambda_proto,
            "lambda_recon": cfg.lambda_recon,
            "lambda_cls": cfg.lambda_cls,
            "lambda_hard_pair": cfg.lambda_hard_pair,
            "lambda_cross_batch_proto": cfg.lambda_cross_batch_proto,
            "supcon_temperature": cfg.supcon_temperature,
            "proto_margin": cfg.proto_margin,
            "hard_pair_margin": cfg.hard_pair_margin,
            "input_raw_pca_enabled": cfg.input_raw_pca_enabled,
            "input_raw_pca_components": cfg.input_raw_pca_components,
            "rt_range": list(cfg.rt_range),
            "mz_range": list(cfg.mz_range),
            "cross_batch_ways": cfg.cross_batch_ways,
            "cross_batch_support": cfg.cross_batch_support,
            "cross_batch_query": cfg.cross_batch_query,
            "cross_batch_temperature": cfg.cross_batch_temperature,
        },
    }, output_dir / "run_config.json")
    save_json({
        "num_batches": train.num_batches,
        "num_products": train.num_products,
        "feature_dim": int(cfg.feature_dim),
        "proj_dim": int(cfg.proj_dim),
        "input_raw_pca_enabled": bool(cfg.input_raw_pca_enabled),
        "input_raw_pca_precomputed": bool(getattr(cfg, "_input_pca_precomputed_active", False)),
        "input_raw_pca_components": int(cfg.mz_bins),
        "input_raw_pca_rt_bins": int(cfg.rt_bins),
        "model_select_metric": "cross_batch_proto_acc",
    }, final_dir / "train_meta.json")
    closed_metrics = prototype_metrics(model, loader_train_noaug, loader_closed, device, cfg)
    save_json({
        "method": "full_main_cross_batch",
        "train_seed": args.seed,
        "checkpoint_selection": "validation_cross_batch_prototype_accuracy",
        "best_val_accuracy": best_score,
        "closed_set": closed_metrics,
        "training_objective": {
            "lambda_cross_batch_proto": cfg.lambda_cross_batch_proto,
            "ways": cfg.cross_batch_ways,
            "support": cfg.cross_batch_support,
            "query": cfg.cross_batch_query,
            "temperature": cfg.cross_batch_temperature,
            "support_query_batches_distinct": True,
        },
    }, output_dir / "paper_gate_result.json")
    print(json.dumps({
        "output_dir": str(output_dir),
        "best_val_cross_batch_accuracy": best_score,
        "closed_set": closed_metrics,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()









