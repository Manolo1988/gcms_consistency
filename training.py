"""Fixed-backbone ablation training for closed recognition and registration."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_tensor
from evaluation import (
    classification_metrics,
    fit_mean_prototypes,
    predict_prototypes,
    save_feature_table,
)
from protocol import ProtocolSpec, metadata_fingerprint


@dataclass(frozen=True)
class TrainingSpec:
    method: str = "cnn_supcon_batchadv"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    embedding_dim: int = 128
    dropout: float = 0.2
    ce_weight: float = 1.0
    supcon_weight: float = 1.0
    batch_adversarial_weight: float = 0.1
    temperature: float = 0.07
    eval_interval: int = 5
    early_stop_patience: int = 8
    samples_per_class: int = 2
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42


VALID_METHODS = {"cnn_ce", "cnn_supcon", "cnn_supcon_batchadv"}


def _log(message: str) -> None:
    print(f"[training] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


def _imports():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, Dataset, Sampler
    except ImportError as exc:
        raise RuntimeError("deep methods require PyTorch; install requirements.txt") from exc
    return torch, nn, functional, DataLoader, Dataset, Sampler


def _set_seed(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(name: str, torch):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalization_stats(
    df: pd.DataFrame,
    indices: np.ndarray,
    tensor_root: str | Path | None,
    protocol: ProtocolSpec,
) -> tuple[np.ndarray, np.ndarray]:
    total = None
    total_sq = None
    count = 0
    for index in indices:
        value = load_tensor(df.iloc[int(index)], tensor_root, protocol).astype(np.float64)
        channel_sum = value.sum(axis=(1, 2))
        channel_sq = np.square(value).sum(axis=(1, 2))
        total = channel_sum if total is None else total + channel_sum
        total_sq = channel_sq if total_sq is None else total_sq + channel_sq
        count += value.shape[1] * value.shape[2]
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _build_runtime(torch, nn, functional, Dataset, Sampler):
    class TensorDataset(Dataset):
        def __init__(self, df, indices, labels, batches, root, protocol, mean, std, augment):
            self.df = df
            self.indices = np.asarray(indices, dtype=int)
            self.labels = labels
            self.batches = batches
            self.root = root
            self.protocol = protocol
            self.mean = mean[:, None, None]
            self.std = std[:, None, None]
            self.augment = augment

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, item):
            row_index = int(self.indices[item])
            value = load_tensor(self.df.iloc[row_index], self.root, self.protocol)
            value = np.clip((value - self.mean) / self.std, -8.0, 8.0)
            if self.augment:
                value = value * np.random.uniform(0.9, 1.1)
                value = value + np.random.normal(0.0, 0.01, value.shape).astype(np.float32)
                value = np.roll(value, np.random.randint(-4, 5), axis=1)
            return (
                torch.from_numpy(np.asarray(value, dtype=np.float32)),
                torch.tensor(self.labels[row_index], dtype=torch.long),
                torch.tensor(self.batches[row_index], dtype=torch.long),
                row_index,
            )

    class BalancedBatchSampler(Sampler):
        def __init__(self, local_labels, batch_size, samples_per_class, seed):
            self.labels = np.asarray(local_labels, dtype=int)
            self.batch_size = int(batch_size)
            self.samples_per_class = int(samples_per_class)
            if self.samples_per_class < 2 or self.batch_size < self.samples_per_class * 2:
                raise ValueError("balanced batches need at least two classes and two samples per class")
            self.classes_per_batch = self.batch_size // self.samples_per_class
            self.by_class = {
                label: np.flatnonzero(self.labels == label)
                for label in np.unique(self.labels)
            }
            self.seed = int(seed)
            self.epoch = 0

        def __len__(self):
            return max(1, int(np.ceil(len(self.labels) / self.batch_size)))

        def __iter__(self):
            rng = np.random.RandomState(self.seed + self.epoch)
            self.epoch += 1
            classes = np.asarray(sorted(self.by_class), dtype=int)
            for _ in range(len(self)):
                selected = rng.choice(
                    classes,
                    size=self.classes_per_batch,
                    replace=len(classes) < self.classes_per_batch,
                )
                batch = []
                for label in selected:
                    pool = self.by_class[int(label)]
                    batch.extend(rng.choice(
                        pool, size=self.samples_per_class,
                        replace=len(pool) < self.samples_per_class,
                    ).tolist())
                rng.shuffle(batch)
                yield batch

    class GradientReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, alpha):
            ctx.alpha = alpha
            return value.view_as(value)

        @staticmethod
        def backward(ctx, gradient):
            return -ctx.alpha * gradient, None

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=2):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, value):
            return self.layers(value)

    class PlainGCMSCNN(nn.Module):
        def __init__(self, in_channels, products, batches, embedding_dim, dropout):
            super().__init__()
            self.backbone = nn.Sequential(
                ConvBlock(in_channels, 32, stride=2),
                ConvBlock(32, 64, stride=2),
                ConvBlock(64, 128, stride=2),
                ConvBlock(128, 192, stride=2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.embedding = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(192, embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim), nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, embedding_dim),
            )
            self.classifier = nn.Linear(embedding_dim, products)
            self.domain = nn.Linear(embedding_dim, batches)

        def forward(self, value, adversarial_alpha=1.0):
            raw = self.embedding(self.backbone(value))
            embedding = functional.normalize(raw, dim=1)
            projection = functional.normalize(self.projection(raw), dim=1)
            reversed_value = GradientReverse.apply(raw, adversarial_alpha)
            return embedding, projection, self.classifier(raw), self.domain(reversed_value)

        def encode(self, value):
            return functional.normalize(self.embedding(self.backbone(value)), dim=1)

    return TensorDataset, BalancedBatchSampler, PlainGCMSCNN


def _supcon_loss(features, labels, temperature, torch):
    if len(features) < 2:
        return features.sum() * 0.0
    logits = features @ features.T / temperature
    identity = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    logits = logits.masked_fill(identity, -1e9)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    counts = positive.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        return features.sum() * 0.0
    return -(log_probability * positive).sum(dim=1)[valid].div(counts[valid]).mean()


def _extract(model, loader, n_rows, dimension, device, torch) -> np.ndarray:
    values = np.zeros((n_rows, dimension), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for inputs, _, _, row_indices in loader:
            embeddings = model.encode(inputs.to(device)).cpu().numpy()
            values[np.asarray(row_indices, dtype=int)] = embeddings
    return values


def train_deep_method(
    df: pd.DataFrame,
    manifest: dict,
    tensor_root: str | Path | None,
    output_dir: str | Path,
    protocol: ProtocolSpec,
    training: TrainingSpec,
) -> Path:
    """Train one ablation using validation batches only for checkpoint selection."""
    if training.method not in VALID_METHODS:
        raise ValueError(f"method must be one of {sorted(VALID_METHODS)}")
    started = time.perf_counter()
    torch, nn, functional, DataLoader, Dataset, Sampler = _imports()
    TensorDataset, BalancedBatchSampler, PlainGCMSCNN = _build_runtime(
        torch, nn, functional, Dataset, Sampler
    )
    _set_seed(training.seed, torch)
    device = _resolve_device(training.device, torch)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_idx = np.asarray(manifest["train_idx"], dtype=int)
    val_idx = np.asarray(manifest["val_idx"], dtype=int)
    all_idx = np.arange(len(df), dtype=int)
    _log(
        f"start method={training.method} seed={training.seed} fold={manifest.get('fold_id', 'unknown')} "
        f"train={len(train_idx)} val={len(val_idx)} total={len(df)} device={device}"
    )
    product_names = sorted(df.iloc[train_idx][protocol.product_col].astype(str).unique())
    batch_names = sorted(df.iloc[train_idx][protocol.batch_col].astype(str).unique())
    product_map = {name: i for i, name in enumerate(product_names)}
    batch_map = {name: i for i, name in enumerate(batch_names)}
    product_labels = np.full(len(df), -1, dtype=int)
    batch_labels = np.full(len(df), -1, dtype=int)
    for index in train_idx:
        product_labels[index] = product_map[str(df.iloc[index][protocol.product_col])]
        batch_labels[index] = batch_map[str(df.iloc[index][protocol.batch_col])]

    _log("computing train-only normalization statistics")
    mean, std = _normalization_stats(df, train_idx, tensor_root, protocol)
    train_dataset = TensorDataset(
        df, train_idx, product_labels, batch_labels, tensor_root, protocol, mean, std, True
    )
    selection_idx = np.concatenate([train_idx, val_idx])
    selection_dataset = TensorDataset(
        df, selection_idx, product_labels, batch_labels, tensor_root, protocol, mean, std, False
    )
    evaluation_dataset = TensorDataset(
        df, all_idx, product_labels, batch_labels, tensor_root, protocol, mean, std, False
    )
    sampler = BalancedBatchSampler(
        product_labels[train_idx], training.batch_size,
        training.samples_per_class, training.seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=training.num_workers
    )
    selection_loader = DataLoader(
        selection_dataset, batch_size=training.batch_size, shuffle=False,
        num_workers=training.num_workers,
    )
    evaluation_loader = DataLoader(
        evaluation_dataset, batch_size=training.batch_size, shuffle=False,
        num_workers=training.num_workers,
    )
    sample_tensor = load_tensor(df.iloc[int(train_idx[0])], tensor_root, protocol)
    model = PlainGCMSCNN(
        sample_tensor.shape[0], len(product_names), len(batch_names),
        training.embedding_dim, training.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(training.epochs, 1)
    )
    _log(
        f"model ready products={len(product_names)} batches={len(batch_names)} "
        f"embedding_dim={training.embedding_dim} batch_size={training.batch_size} epochs={training.epochs}"
    )

    use_supcon = training.method in {"cnn_supcon", "cnn_supcon_batchadv"}
    use_adversarial = training.method == "cnn_supcon_batchadv"
    best_score = -1.0
    best_epoch = 0
    stale = 0
    history = []
    checkpoint_path = output / "best_checkpoint.pt"
    for epoch in range(1, training.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "supcon": 0.0, "batch_adv": 0.0}
        batches_seen = 0
        for inputs, products, batches, _ in train_loader:
            inputs = inputs.to(device)
            products = products.to(device)
            batches = batches.to(device)
            embedding, projection, logits, domain_logits = model(inputs)
            ce = functional.cross_entropy(logits, products)
            supcon = _supcon_loss(projection, products, training.temperature, torch)
            adversarial = functional.cross_entropy(domain_logits, batches)
            loss = training.ce_weight * ce
            if use_supcon:
                loss = loss + training.supcon_weight * supcon
            if use_adversarial:
                loss = loss + training.batch_adversarial_weight * adversarial
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["ce"] += float(ce.detach().cpu())
            totals["supcon"] += float(supcon.detach().cpu())
            totals["batch_adv"] += float(adversarial.detach().cpu())
            batches_seen += 1
        scheduler.step()

        should_validate = epoch % training.eval_interval == 0 or epoch == training.epochs
        row = {key: value / max(batches_seen, 1) for key, value in totals.items()}
        row.update({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]})
        if should_validate:
            embeddings = _extract(
                model, selection_loader, len(df), training.embedding_dim, device, torch
            )
            classes, prototypes = fit_mean_prototypes(
                embeddings[train_idx], df.iloc[train_idx][protocol.product_col].to_numpy()
            )
            truth = df.iloc[val_idx][protocol.product_col].astype(str).to_numpy()
            predicted, _ = predict_prototypes(embeddings[val_idx], classes, prototypes)
            val_metrics = classification_metrics(truth, predicted, labels=product_names)
            row["validation"] = val_metrics
            score = val_metrics["macro_f1"]
            if score > best_score + 1e-8:
                best_score = score
                best_epoch = epoch
                stale = 0
                torch.save({
                    "model": model.state_dict(),
                    "training_spec": asdict(training),
                    "protocol_fingerprint": manifest["metadata_sha256"],
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "product_names": product_names,
                    "batch_names": batch_names,
                    "best_validation_macro_f1": best_score,
                    "best_epoch": best_epoch,
                }, checkpoint_path)
            else:
                stale += 1
            _log(
                f"epoch {epoch:03d}/{training.epochs} "
                f"loss={row['loss']:.4f} ce={row['ce']:.4f} supcon={row['supcon']:.4f} "
                f"batch_adv={row['batch_adv']:.4f} val_macro_f1={score:.4f} "
                f"best={best_score:.4f}@{best_epoch} stale={stale} "
                f"time={time.perf_counter() - epoch_started:.1f}s"
            )
        else:
            _log(
                f"epoch {epoch:03d}/{training.epochs} "
                f"loss={row['loss']:.4f} ce={row['ce']:.4f} supcon={row['supcon']:.4f} "
                f"batch_adv={row['batch_adv']:.4f} lr={row['learning_rate']:.6g} "
                f"time={time.perf_counter() - epoch_started:.1f}s"
            )
        history.append(row)
        if should_validate and training.early_stop_patience > 0 and stale >= training.early_stop_patience:
            _log(f"early stop at epoch={epoch}; best_epoch={best_epoch} best_val_macro_f1={best_score:.4f}")
            break

    if not checkpoint_path.exists():
        raise RuntimeError("training did not produce a validation checkpoint")
    # The checkpoint is created by this run and contains NumPy normalization
    # arrays, so PyTorch 2.6 needs the trusted-file opt-out explicitly.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    _log(f"extracting frozen features from best checkpoint epoch={best_epoch}")
    features = _extract(
        model, evaluation_loader, len(df), training.embedding_dim, device, torch
    )
    feature_path = output / "features.npz"
    save_feature_table(
        feature_path, df["_sample_key"], features,
        metadata_fingerprint(df, protocol), metric="cosine",
    )
    run_record = {
        "method": training.method,
        "selection_partition": "val_idx closed-set batches only",
        "test_or_novel_used_for_selection": False,
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_validation_macro_f1": float(best_score),
        "normalization_fit_partition": "train_idx only",
        "training_spec": asdict(training),
        "history": history,
    }
    (output / "training.json").write_text(
        json.dumps(run_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _log(f"done feature={feature_path.resolve()} elapsed={time.perf_counter() - started:.1f}s")
    return feature_path
