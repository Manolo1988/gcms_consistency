#!/usr/bin/env python3
"""Run the approved two-loss backbone comparison sequentially; never refresh inputs.

All model artifacts go to --output-dir, all train/eval logs to --log-dir.
--check-only validates inputs and shared episodes without importing torch.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from light_cnn_protocol import (
    GROUPS, KNOWN, PROTOCOL, build_config, classification_metrics, fit_prototypes,
    open_set_metrics, prototype_predictions, selection_key, serializable,
    tensor_path, validate_inputs, validation_threshold, write_json,
)
from paper_protocol import (
    build_cross_batch_episodes, evaluate_embedding_episodes, load_json,
    metadata_fingerprint, validate_episode_manifest, write_rows_csv,
)


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return False


@contextlib.contextmanager
def phase_log(path):
    with Path(path).open("x", encoding="utf-8", buffering=1) as handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, handle)), contextlib.redirect_stderr(Tee(sys.stderr, handle)):
            yield


def pin_tensor_paths(dataset, prepared):
    # Resolve only within this representation, independent of the shell cwd.
    dataset.df["tensor_path"] = dataset.df["tensor_path"].map(lambda p: str(tensor_path(prepared, p)))


def make_loader(cfg, indices, device, product_encoder=None, salt=104):
    from dataset import GCMSDataset
    from torch.utils.data import DataLoader
    from train import _loader_runtime_kwargs, _make_generator
    dataset = GCMSDataset(Path(cfg.prepared_dir) / "metadata.csv", indices=indices, augmentation=None)
    pin_tensor_paths(dataset, cfg.prepared_dir)
    if product_encoder is not None:
        dataset.product_enc = product_encoder
        dataset.df["product_label"] = product_encoder.transform(dataset.df["product_fine"])
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                      **_loader_runtime_kwargs(cfg, device, generator=_make_generator(cfg, salt)))


def extract_embeddings(model, loader, device, cfg, expected_ids):
    import torch
    from train import _channels_last_enabled
    values, sample_ids = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)
            if _channels_last_enabled(cfg, device):
                x = x.contiguous(memory_format=torch.channels_last)
            values.append(model.encode(x).cpu().numpy())
            sample_ids.extend(str(s) for s in batch["sample_id"])
    if sample_ids != list(map(str, expected_ids)):
        raise ValueError("Embedding/sample order mismatch")
    result = np.concatenate(values)
    if not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-4):
        raise ValueError("Expected finite unit-length embeddings")
    return result


def true_labels(frame):
    class_to_id = {name: i for i, name in enumerate(KNOWN)}
    return frame["product_fine"].map(class_to_id).to_numpy(dtype=np.int64)


def train_group(cfg, frame, split, audit):
    import torch
    from config import get_device
    from losses import UnifiedLoss
    from models import GCMSConsistencyNet
    from train import (
        _attach_focal_class_weights, _enable_cuda_fast_paths, _maybe_optimize_model,
        _model_state_dict_for_save, build_loaders, set_seed, train_one_epoch,
    )
    set_seed(cfg.seed, deterministic=True)
    device = get_device()
    if device.type != "cuda":
        raise RuntimeError("This approved full-resolution training requires a CUDA server; refusing a local/CPU run")
    _enable_cuda_fast_paths(cfg, device)
    started = time.perf_counter()
    output = Path(cfg.output_dir)
    final = output / "final_model"
    final.mkdir()
    ds_train, ds_val, loader_train, loader_val = build_loaders(
        Path(cfg.prepared_dir) / "metadata.csv", split["train_idx"], split["val_idx"],
        cfg, "product_fine", device=device)
    for dataset in (ds_train, ds_val):
        pin_tensor_paths(dataset, cfg.prepared_dir)
    if tuple(ds_train.product_enc.classes_) != KNOWN:
        raise ValueError("Training label encoding differs from the approved classes")
    loader_reference = make_loader(cfg, split["train_idx"], device, ds_train.product_enc)
    model = GCMSConsistencyNet(ds_train.num_batches, cfg, num_products=len(KNOWN)).to(device)
    model = _maybe_optimize_model(model, cfg, device)
    # Retain disabled modules in state_dict, but neither execute nor optimize them.
    for module in (model.domain_head, model.decoder):
        module.requires_grad_(False)
    _attach_focal_class_weights(cfg, ds_train)
    criterion = UnifiedLoss(cfg).to(device)
    criterion.set_label_names(ds_train.get_label_name_map())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    counts = np.bincount(ds_train.df["product_label"], minlength=len(KNOWN))
    metadata = {
        "protocol": PROTOCOL, "config": vars(cfg), "classes": list(KNOWN),
        "num_products": len(KNOWN), "num_batches": ds_train.num_batches,
        "actual_training_batches": 24, "input_audit": audit,
        "class_counts": dict(zip(KNOWN, counts.tolist())),
        "class_weights": dict(zip(KNOWN, cfg.focal_alpha.tolist())),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "active_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "inference_parameters": sum(p.numel() for module in (model.encoder, model.embed_head) for p in module.parameters()),
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "lr": cfg.lr,
                      "weight_decay": cfg.weight_decay},
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 100, "eta_min": 0, "step": "after_epoch"},
        "sampler": {"name": "WeightedRandomSampler", "replacement": True, "draws": len(ds_train), "drop_last": True},
        "gradient_clip_norm": 5.0,
        "evaluation_protocol": {
            "prototype": "unnormalized class mean of unchanged model.encode unit embeddings",
            "matching": "Euclidean in A/B/C", "train_radius_percentile": 95,
            "radius_floor": 1e-6, "known_score": "-d_min/r_pred",
            "threshold": "5th percentile of validation-known scores only",
            "setting_a_known_n": 468, "setting_b_known_n": 468, "setting_b_unknown_n": 133,
            "setting_c": "100 shared cross-batch nested 1/3/5-shot episodes, seeds 42000..42099",
            "calibration": False, "auto_score_blending": False,
        },
        "checkpoint_rule": "max validation (macro_f1, balanced_accuracy); exact ties keep earlier epoch",
        "classification_loss_reduction": "mean of per-sample weighted CE, preserving existing FocalLoss(gamma=0)",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "device": str(device),
                        "gpu": torch.cuda.get_device_name(device), "hash_seed": os.environ.get("PYTHONHASHSEED")},
    }
    write_json(metadata, output / "run_config.json")
    write_json(split, final / "split.json")
    write_json({str(i): name for i, name in enumerate(KNOWN)}, final / "product_classes.json")
    write_json({"num_batches": ds_train.num_batches, "num_products": len(KNOWN), "config_file": "../run_config.json",
                "input_representation": "dual_channel_no_pca", "source_manifest_fingerprint": audit["manifest_fingerprint"]},
               final / "train_meta.json")
    training_rows, validation_rows = [], []
    best, best_epoch = None, None
    reference_rows = frame.iloc[split["train_idx"]]
    val_rows = frame.iloc[split["val_idx"]]
    print(f"Device={device}; backbone={cfg.main_backbone}; blocks/stage={cfg.blocks_per_stage}; "
          f"active parameters={metadata['active_trainable_parameters']:,}", flush=True)
    for epoch in range(1, 101):
        epoch_started = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        losses = train_one_epoch(model, loader_train, criterion, optimizer, device,
                                 epoch - 1, 100, cfg=cfg, scaler=None)
        if not all(np.isfinite(list(losses.values()))):
            raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
        scheduler.step()
        training_rows.append({"epoch": epoch, "lr": learning_rate, "next_lr": optimizer.param_groups[0]["lr"],
                              "seconds": time.perf_counter() - epoch_started, **losses})
        print(f"Epoch {epoch}/100 lr={learning_rate:.7f} "
              f"supcon={losses['supcon']:.5f} cls={losses['cls']:.5f} total={losses['total']:.5f}", flush=True)
        write_rows_csv(training_rows, output / "training_history.csv")
        if epoch % 10:
            continue
        reference = extract_embeddings(model, loader_reference, device, cfg, reference_rows["sample_id"])
        prototypes, radii = fit_prototypes(reference, true_labels(reference_rows))
        validation = extract_embeddings(model, loader_val, device, cfg, val_rows["sample_id"])
        predicted = prototype_predictions(validation, prototypes, radii)
        metrics = classification_metrics(true_labels(val_rows), predicted["prediction"])
        key = selection_key(metrics)
        improved = best is None or key > best
        if improved:
            best, best_epoch = key, epoch
            state = {k: v.detach().cpu().clone() for k, v in _model_state_dict_for_save(model).items()}
            torch.save(state, final / "model.pt")
            del state
            write_json({"epoch": epoch, "selection_key": key, "validation": metrics}, output / "best_selection.json")
        validation_rows.append({"epoch": epoch, "prototype_samples": len(reference), "selected": improved,
                                "best_epoch": best_epoch, **metrics})
        write_json(validation_rows, output / "validation_history.json")
        print(f"Validation: Macro-F1={key[0]:.5f} balanced_accuracy={key[1]:.5f} "
              f"accuracy={metrics['accuracy']:.5f} best_epoch={best_epoch}", flush=True)
        print(f"Per-class recall: {metrics['per_class_recall']}", flush=True)
    metadata.update(best_epoch=best_epoch, best_validation_key=best, training_seconds=time.perf_counter() - started)
    write_json(metadata, output / "run_config.json")
    return serializable(metadata)


def load_v1_model(directory, device):
    """Reconstruct the exact saved architecture; do not use legacy CLI defaults."""
    import torch
    from config import Config
    from models import GCMSConsistencyNet
    from train import _maybe_optimize_model
    payload = load_json(Path(directory) / "run_config.json")
    if payload["protocol"] != PROTOCOL:
        raise ValueError("Unexpected model protocol")
    cfg = Config()
    for key, value in payload["config"].items():
        setattr(cfg, key, value)
    model = GCMSConsistencyNet(payload["num_batches"], cfg, num_products=payload["num_products"]).to(device)
    model.load_state_dict(torch.load(Path(directory) / "final_model/model.pt", map_location=device, weights_only=True), strict=True)
    model = _maybe_optimize_model(model, cfg, device)
    return model.eval(), cfg


def prediction_rows(frame, result, role, threshold):
    rows = []
    for position, (_, sample) in enumerate(frame.iterrows()):
        pred = KNOWN[int(result["prediction"][position])]
        truth = str(sample["product_fine"])
        rows.append({"sample_id": str(sample["sample_id"]), "record_id": str(sample["record_id"]),
                     "metadata_index": int(sample["metadata_index"]), "role": role,
                     "batch_name": str(sample["batch_name"]), "true_product": truth,
                     "predicted_product": pred, "is_known": role != "test_unknown",
                     "correct_known_class": truth == pred, "score": float(result["score"][position]),
                     "distance": float(result["distance"][position]), "radius": float(result["radius"][position]),
                     "second_distance": float(result["second_distance"][position]),
                     "margin_diagnostic": float(result["margin"][position]),
                     "accepted": bool(result["score"][position] >= threshold), "threshold": threshold})
    return rows


def evaluate_group(directory, frame, split, episodes, metadata):
    from config import get_device
    from train import _enable_cuda_fast_paths, set_seed
    from baseline_reporting import summarize_fewshot
    set_seed(42, deterministic=True)
    device = get_device()
    model, cfg = load_v1_model(directory, device)
    _enable_cuda_fast_paths(cfg, device)
    directory = Path(directory)
    embedded = {}
    role_frames = {}
    for salt, key in enumerate(("train_idx", "val_idx", "test_batch_idx", "test_unknown_idx"), 200):
        rows = frame.iloc[split[key]]
        print(f"Extracting {key}: {len(rows)} unaugmented samples", flush=True)
        loader = make_loader(cfg, split[key], device, salt=salt)
        embedded[key] = extract_embeddings(model, loader, device, cfg, rows["sample_id"])
        role_frames[key] = rows
        np.savez_compressed(directory / f"{key}_embeddings.npz", embeddings=embedded[key],
                            sample_ids=rows["sample_id"].astype(str).to_numpy(dtype=str))
        del loader
    prototypes, radii = fit_prototypes(embedded["train_idx"], true_labels(role_frames["train_idx"]))
    np.savez_compressed(directory / "registered_prototypes.npz", prototypes=prototypes, radii=radii,
                        classes=np.asarray(KNOWN))
    predictions = {key: prototype_predictions(value, prototypes, radii) for key, value in embedded.items()}
    val_metrics = classification_metrics(true_labels(role_frames["val_idx"]), predictions["val_idx"]["prediction"])
    if not np.allclose(selection_key(val_metrics), metadata["best_validation_key"], atol=1e-12, rtol=0):
        raise RuntimeError("Reloaded checkpoint does not reproduce selected validation metrics")
    threshold = validation_threshold(predictions["val_idx"]["score"])
    write_json({"threshold": threshold, "source": "143 validation-known scores only", "percentile": 5,
                "comparison": ">= accepts", "score": "-d_min/r_pred", "radius_floor": 1e-6,
                "prototype": "raw mean of unit embeddings; no mean renormalization; Euclidean",
                "validation_acceptance": float(np.mean(predictions["val_idx"]["score"] >= threshold))},
               directory / "acceptance_rule.json")
    for key, role in (("val_idx", "validation"), ("test_batch_idx", "test_known"), ("test_unknown_idx", "test_unknown")):
        write_rows_csv(prediction_rows(role_frames[key], predictions[key], role, threshold),
                       directory / f"{role}_predictions.csv")
    test = role_frames["test_batch_idx"]
    known_pred = predictions["test_batch_idx"]["prediction"]
    closed = classification_metrics(true_labels(test), known_pred)
    by_batch = []
    for batch in sorted(test["batch_name"].astype(str).unique()):
        mask = test["batch_name"].astype(str).eq(batch).to_numpy()
        by_batch.append({"batch_name": batch, **classification_metrics(true_labels(test)[mask], known_pred[mask])})
    write_json({"overall": closed, "per_batch": by_batch}, directory / "setting_a_metrics.json")
    write_rows_csv([{k: row[k] for k in ("batch_name", "n", "accuracy", "macro_f1", "balanced_accuracy")} for row in by_batch],
                   directory / "setting_a_per_batch.csv")
    write_rows_csv([{"product": name, "n": closed["per_class_count"][name], "recall": closed["per_class_recall"][name]}
                    for name in KNOWN], directory / "setting_a_per_class.csv")
    write_rows_csv([{"true_product": name, **dict(zip(KNOWN, row))} for name, row in zip(KNOWN, closed["confusion_matrix"])],
                   directory / "setting_a_confusion.csv")
    opened = open_set_metrics(predictions["test_batch_idx"]["score"], predictions["test_unknown_idx"]["score"], threshold)
    for name in ("HMD", "XCJ"):
        mask = role_frames["test_unknown_idx"]["product_fine"].eq(name).to_numpy()
        opened[f"{name}_false_acceptance"] = float(np.mean(predictions["test_unknown_idx"]["score"][mask] >= threshold))
    write_json(opened, directory / "setting_b_metrics.json")
    write_rows_csv(prediction_rows(test, predictions["test_batch_idx"], "test_known", threshold) +
                   prediction_rows(role_frames["test_unknown_idx"], predictions["test_unknown_idx"], "test_unknown", threshold),
                   directory / "setting_b_predictions.csv")
    if episodes["metadata_fingerprint"] != metadata_fingerprint(frame) or episodes["unknown_indices"] != split["test_unknown_idx"]:
        raise ValueError("Shared episode identity/order mismatch")
    fewshot = evaluate_embedding_episodes(embedded["test_unknown_idx"], episodes, directory.name, 42,
                                          distance="euclidean_raw_mean")
    write_rows_csv(fewshot, directory / "setting_c_episodes.csv")
    fewshot_summary = summarize_fewshot(fewshot, directory)
    summary = {"protocol": PROTOCOL, "group": directory.name, "train_seed": 42, "best_epoch": metadata["best_epoch"],
               "training_seconds": metadata["training_seconds"], "parameter_count": metadata["parameter_count"],
               "active_trainable_parameters": metadata["active_trainable_parameters"],
               "inference_parameters": metadata["inference_parameters"], "validation": val_metrics,
               "setting_a": closed, "setting_b": opened, "setting_c": fewshot_summary,
               "fewshot_distance": "euclidean_raw_mean", "independent_training_repeats": 1,
               "interpretation": "Episode variation reuses one trained model and samples; not independent training repeats."}
    write_json(summary, directory / "evaluation_summary.json")
    print(f"Setting A accuracy={closed['accuracy']:.5f}, Macro-F1={closed['macro_f1']:.5f}; "
          f"Setting B AUROC={opened['auroc']:.5f}, FPR95={opened['fpr95']:.5f}", flush=True)
    return summary


def source_snapshot(shared):
    names = ("config.py", "dataset.py", "models.py", "baselines.py", "losses.py", "train.py", "paper_protocol.py",
             "baseline_reporting.py", "light_cnn_protocol.py", "scripts/run_light_cnn_v1.py")
    hashes = {}
    for name in names:
        data = (ROOT / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        target = shared / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    write_json(hashes, shared / "source_sha256.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True, help="Existing frozen dual_channel_no_pca directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output/ directory; must not exist")
    parser.add_argument("--log-dir", type=Path, required=True, help="Task logs/ directory, sibling of output/")
    parser.add_argument("--check-only", action="store_true", help="Input/header/episode checks only; no torch import")
    args = parser.parse_args(argv)
    prepared, output, logs = args.prepared_dir.resolve(), args.output_dir.resolve(), args.log_dir.resolve()
    if output.name != "output" or logs.name != "logs" or output.parent != logs.parent:
        parser.error("Use sibling <task>/output and <task>/logs directories")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite any existing output: {output}")
    if output.is_relative_to(prepared) or logs.is_relative_to(prepared):
        raise ValueError("Results/logs must not be written into prepared inputs")
    task_class = "checks" if args.check_only else "experiments"
    if not output.parent.is_relative_to(ROOT / "new_result" / task_class):
        parser.error(f"This task belongs under new_result/{task_class}/<unique_task>/")
    logs.mkdir(parents=True, exist_ok=True)
    for name in ("runner.pid", "runner.pgid", "failure.log"):
        if (logs / name).exists():
            raise FileExistsError(f"Refusing to reuse previous task logs: {logs / name}")
    (logs / "runner.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    if hasattr(os, "getpgrp"):
        (logs / "runner.pgid").write_text(str(os.getpgrp()) + "\n", encoding="utf-8")
    print("Validating frozen inputs and every tensor header ...", flush=True)
    frame, split, audit = validate_inputs(prepared)
    episodes = build_cross_batch_episodes(prepared / "metadata.csv", split["test_unknown_idx"],
                                           shots=(1, 3, 5), episode_count=100, seed_start=42000)
    episode_audit = validate_episode_manifest(episodes)
    if episode_audit["status"] != "pass":
        raise ValueError(episode_audit)
    if not args.check_only:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA training environment; use --check-only locally")
        if os.environ.get("PYTHONHASHSEED") != "42":
            raise RuntimeError("Start this experiment with PYTHONHASHSEED=42")
    output.mkdir(parents=True, exist_ok=False)
    shared = output / "shared"
    shared.mkdir()
    write_json(audit, shared / "input_audit.json")
    write_json(split, shared / "split.json")
    frame.to_csv(shared / "metadata_snapshot.csv", index=False, encoding="utf-8-sig")
    write_json(episodes, shared / "fewshot_episodes.json")
    write_json(episode_audit, shared / "episode_audit.json")
    write_json({"mode": task_class, "protocol": PROTOCOL, "arguments": vars(args), "argv": sys.argv,
                "pid": os.getpid(), "process_group": os.getpgrp() if hasattr(os, "getpgrp") else None,
                "approved_groups": GROUPS}, shared / "task_config.json")
    source_snapshot(shared)
    for group in GROUPS:
        write_json(vars(build_config(prepared, output / group, group)), shared / f"{group}_planned_config.json")
    print(f"Inputs verified: {audit['counts']}; 100 shared cross-batch episodes valid.", flush=True)
    if args.check_only:
        write_json({"status": "pass", "mode": "checks", "models_executed": False}, shared / "status.json")
        print(f"Check only: no model executed. Audit output: {output}", flush=True)
        return

    def stop(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    summaries = []
    write_json({"status": "running", "completed_groups": []}, shared / "status.json")
    try:
        for group in GROUPS:
            directory = output / group
            directory.mkdir(exist_ok=False)
            cfg = build_config(prepared, directory, group)
            write_json({"status": "running", "active_group": group, "completed_groups": [s["group"] for s in summaries]},
                       shared / "status.json")
            print(f"\nStarting {group}; training log: {logs / (group + '_train.log')}", flush=True)
            with phase_log(logs / f"{group}_train.log"):
                metadata = train_group(cfg, frame, split, audit)
            gc.collect()
            torch.cuda.empty_cache()
            with phase_log(logs / f"{group}_eval.log"):
                summary = evaluate_group(directory, frame, split, episodes, metadata)
            summaries.append(summary)
            compact = []
            for item in summaries:
                row = {"group": item["group"], "seed": 42, "best_epoch": item["best_epoch"],
                       "training_seconds": item["training_seconds"], "inference_parameters": item["inference_parameters"],
                       **{key: item["setting_a"][key] for key in ("accuracy", "macro_f1", "balanced_accuracy")},
                       **{key: item["setting_b"][key] for key in ("auroc", "fpr95", "known_acceptance", "unknown_false_acceptance")}}
                row.update({f"fewshot_{r['shot']}shot_accuracy": r["mean_accuracy"] for r in item["setting_c"] if r["scope"] == "pooled"})
                compact.append(row)
            write_rows_csv(compact, shared / "comparison_metrics.csv")
            write_json(summaries, shared / "comparison_metrics.json")
            gc.collect()
            torch.cuda.empty_cache()
    except BaseException as exc:
        write_json({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "completed_groups": [s["group"] for s in summaries], "active_group": group,
                    "error": str(exc)}, shared / "status.json")
        with (logs / "failure.log").open("x", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        raise
    write_json({"status": "complete", "completed_groups": list(GROUPS)}, shared / "status.json")
    print(f"Both groups complete. Results: {output}; logs: {logs}", flush=True)


if __name__ == "__main__":
    main()
