#!/usr/bin/env python3
"""Train and evaluate a paper deep-learning baseline on the fixed split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines import train_dl_baseline_fold
from baseline_reporting import (
    save_classification_diagnostics, save_training_curves, summarize_fewshot,
)
from config import Config, apply_prepared_input_config, get_device
from dataset import GCMSDataset, load_data_split
from paper_protocol import (
    build_cross_batch_episodes,
    evaluate_embedding_episodes,
    load_json,
    metadata_fingerprint,
    save_json,
    validate_episode_manifest,
    write_rows_csv,
    load_filtered_metadata,
)
from train import set_seed


@torch.no_grad()
def _extract_embeddings(model, loader, device):
    values = []
    for batch in loader:
        values.append(model.encode(batch["input"].to(device)).cpu().numpy())
    return np.concatenate(values, axis=0)


@torch.no_grad()
def _closed_metrics(model, loader, device, output_dir, role):
    probabilities, targets, sample_ids = [], [], []
    for batch in loader:
        output = model(batch["input"].to(device))
        probabilities.append(torch.softmax(output["logits"], dim=1).cpu().numpy())
        targets.extend(batch["product"].numpy().tolist())
        sample_ids.extend(batch["sample_id"])
    return save_classification_diagnostics(
        output_dir, role, loader.dataset.df, sample_ids, targets,
        np.concatenate(probabilities), loader.dataset.get_product_names(),
    )


def _prototype_closed_metrics(model, train_loader, test_loader, label_names, device, cfg):
    from evaluate import collect_predictions, product_identification_metrics
    from register import register_from_loader

    store, _, _ = register_from_loader(
        model,
        train_loader,
        label_names,
        device,
        percentile=cfg.accept_percentile,
        cfg=cfg,
    )
    records = collect_predictions(
        model,
        test_loader,
        store,
        device,
        reject_factor=cfg.reject_threshold_factor,
    )
    metrics = product_identification_metrics(records)
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_acc": float(metrics["balanced_acc"]),
    }


def _aligned_dataset(metadata_csv, indices, encoder_source):
    dataset = GCMSDataset(metadata_csv, augmentation=None, indices=indices)
    dataset.product_enc = encoder_source.product_enc
    dataset.batch_enc = encoder_source.batch_enc
    dataset.df["product_label"] = dataset.product_enc.transform(dataset.df["product_fine"])
    known_batches = set(dataset.batch_enc.classes_)
    if set(dataset.df["batch_idx"]).issubset(known_batches):
        dataset.df["batch_label"] = dataset.batch_enc.transform(dataset.df["batch_idx"])
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["plain_cnn_ce", "plain_cnn_supcon"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prepared-dir", default=Config().prepared_dir,
                        help="Prepared input directory (default: C, dual_channel_no_pca)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.00026)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-seed-start", type=int, default=42000)
    parser.add_argument("--episode-manifest")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable all input augmentation; keep the network and sampler unchanged")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.episodes < 1 or args.num_workers < 0:
        parser.error("epochs/batch-size/lr/episodes must be positive; num-workers must be nonnegative")

    cfg = Config()
    cfg.seed = args.seed
    cfg.prepared_dir = str(Path(args.prepared_dir).resolve())
    cfg.output_dir = str(Path(args.output_dir).resolve())
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.dataloader_workers = args.num_workers
    cfg.deterministic = True
    input_info = apply_prepared_input_config(cfg)
    set_seed(cfg.seed, deterministic=True)

    output_dir = Path(cfg.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split = load_data_split(cfg)
    metadata_csv = Path(cfg.prepared_dir) / "metadata.csv"
    save_json(vars(args), output_dir / "run_config.json")
    save_json({
        **asdict(cfg), "augmentation_enabled": not args.no_augmentation,
        "input_contract": input_info, "checkpoint_selection": "validation_accuracy",
        "validation_interval": 10, "method": args.method,
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__,
                        "cuda": torch.version.cuda},
    }, output_dir / "effective_config.json")
    save_json(split, output_dir / "split.json")
    save_json({"status": "running"}, output_dir / "run_status.json")
    print(f"Input: {cfg.prepared_dir}; augmentation={not args.no_augmentation}", flush=True)
    history = []

    def record_epoch(row, best_state):
        history.append(row)
        save_json(history, output_dir / "training_history.json")
        write_rows_csv(history, output_dir / "training_history.csv")
        if best_state is not None:
            torch.save(best_state, output_dir / "model.pt")
        save_json({
            "checkpoint_selection": "validation_accuracy", "best_epoch": row["best_epoch"],
            "best_val_accuracy": row["best_val_accuracy"], "last_finished_epoch": row["epoch"],
        }, output_dir / "training_summary.json")
        val_text = "" if row["val_accuracy"] is None else f" val_acc={row['val_accuracy']:.4f}"
        print(f"Epoch {row['epoch']}/{cfg.epochs} train_loss={row['train_loss']:.5f}{val_text}", flush=True)

    started = time.perf_counter()
    baseline_name = "ResNet-CE" if args.method == "plain_cnn_ce" else "ResNet-SupCon"
    model, train_dataset, _, _ = train_dl_baseline_fold(
        baseline_name,
        split["train_idx"],
        split["val_idx"],
        ",".join(str(value) for value in split.get("model_select_holdout_batches", [])),
        str(metadata_csv),
        cfg,
        augmentation_enabled=not args.no_augmentation,
        epoch_callback=record_epoch,
    )
    training_seconds = time.perf_counter() - started
    save_training_curves(history, output_dir)
    device = get_device()
    model = model.to(device).eval()
    torch.save(model.state_dict(), output_dir / "model.pt")

    train_eval = _aligned_dataset(metadata_csv, split["train_idx"], train_dataset)
    val_dataset = _aligned_dataset(metadata_csv, split["val_idx"], train_dataset)
    test_dataset = _aligned_dataset(metadata_csv, split["test_batch_idx"], train_dataset)
    train_loader = DataLoader(train_eval, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    validation_metrics = None
    if args.method == "plain_cnn_ce":
        _closed_metrics(model, train_loader, device, output_dir, "train")
        validation_metrics = _closed_metrics(
            model, DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False),
            device, output_dir, "val",
        )
        closed_set = _closed_metrics(model, test_loader, device, output_dir, "test")
    else:
        closed_set = _prototype_closed_metrics(
            model,
            train_loader,
            test_loader,
            train_dataset.get_label_name_map(),
            device,
            cfg,
        )

    if args.episode_manifest:
        manifest = load_json(args.episode_manifest)
    else:
        manifest = build_cross_batch_episodes(
            metadata_csv,
            split["test_unknown_idx"],
            shots=(1, 3, 5),
            episode_count=args.episodes,
            seed_start=args.episode_seed_start,
        )
        manifest["validation"] = validate_episode_manifest(manifest)

    validation = validate_episode_manifest(manifest)
    if validation["status"] != "pass":
        raise ValueError(f"Few-shot manifest validation failed: {validation}")
    manifest["validation"] = validation
    save_json(manifest, output_dir / "fewshot_episodes.json")

    unknown_dataset = GCMSDataset(
        metadata_csv,
        augmentation=None,
        indices=split["test_unknown_idx"],
    )
    unknown_loader = DataLoader(unknown_dataset, batch_size=cfg.batch_size, shuffle=False)
    embeddings = _extract_embeddings(model, unknown_loader, device)
    rows = evaluate_embedding_episodes(embeddings, manifest, args.method, cfg.seed)
    write_rows_csv(rows, output_dir / "paper_fewshot_episode_results.csv")

    fewshot_summary = summarize_fewshot(rows, output_dir)

    filtered_df = load_filtered_metadata(metadata_csv)
    payload = {
        "method": args.method,
        "train_seed": cfg.seed,
        "split_id": metadata_fingerprint(filtered_df)[:16],
        "checkpoint_selection": "validation_accuracy",
        "closed_set": closed_set,
        "validation": validation_metrics,
        "augmentation_enabled": not args.no_augmentation,
        "training_seconds": training_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": history[-1]["best_epoch"],
        "fewshot_protocol": {
            "name": manifest["protocol"],
            "shots": manifest["shots"],
            "episodes": manifest["episode_count"],
            "episode_seed_start": manifest["seed_start"],
        },
        "fewshot_results_csv": "paper_fewshot_episode_results.csv",
        "fewshot_summary": fewshot_summary,
    }
    save_json(payload, output_dir / "paper_gate_result.json")
    save_json({"status": "complete"}, output_dir / "run_status.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
