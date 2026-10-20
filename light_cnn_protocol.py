"""Frozen light-CNN v1 protocol. No torch import, model execution, or input writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

from config import Config, apply_prepared_input_config
from paper_protocol import load_filtered_metadata

MANIFEST_HASH = "31b181f56f21d7f4a97d931ef30cc10441b950c6378712cf5e1a235c7efbb3e5"
SPLIT_HASH = "0fc492c4c0b7240da2129afa68775cc8ad029cde89bc9753c9eff78c7dda925a"
ROLE_COUNTS = {"train_idx": 1999, "val_idx": 143, "test_batch_idx": 468, "test_unknown_idx": 133}
KNOWN = ("H88", "H99", "HRJ", "HRY", "HYZ", "RJD", "YJD")
GROUPS = {"A_original_2loss": ("gcms", 2), "B_light_2loss": ("plain_cnn", 1)}
SHAPE = (2, 2048, 503)
PROTOCOL = "light_cnn_v1_raw_mean_euclidean_20261020"


def hash_json(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def tensor_path(prepared, value):
    """The frozen representation uses portable paths under its own tensors/."""
    prepared = Path(prepared).resolve()
    relative = Path(str(value).replace("\\", "/"))
    candidate = (prepared / relative).resolve()
    if not candidate.is_relative_to(prepared / "tensors"):
        raise ValueError(f"Tensor path leaves the selected representation: {value}")
    return candidate


def validate_inputs(prepared, inspect_headers=True):
    """Verify actual identities, not only hash strings copied into JSON files."""
    import zipfile
    prepared = Path(prepared).resolve()
    frame = pd.read_csv(prepared / "metadata.csv")
    filtered = load_filtered_metadata(prepared / "metadata.csv")
    if len(frame) != len(filtered) or len(frame) != sum(ROLE_COUNTS.values()):
        raise ValueError("Prepared row coordinate system changed")
    if not frame["tensor_ready"].eq(True).all() or frame["sample_id"].duplicated().any():
        raise ValueError("Unready tensors or duplicate sample IDs")
    split = json.loads((prepared / "split.json").read_text(encoding="utf-8"))
    grid = json.loads((prepared / "grid_info.json").read_text(encoding="utf-8"))
    columns = ["record_id", "source_relpath", "product_fine", "batch_name", "group_id",
               "label_source", "split_role", "raw_relpath", "raw_size_bytes"]
    canonical = frame[columns].fillna("").copy()
    canonical["raw_size_bytes"] = canonical["raw_size_bytes"].map(int)
    manifest_hash = hash_json(canonical.astype(str).to_dict("records"))
    if manifest_hash != MANIFEST_HASH or split.get("manifest_fingerprint") != MANIFEST_HASH:
        raise ValueError("Approved manifest fingerprint mismatch")
    actual_partitions = {}
    seen = []
    for key, count in ROLE_COUNTS.items():
        indices = split[key]
        if len(indices) != count or any(type(i) is not int or not 0 <= i < len(frame) for i in indices):
            raise ValueError(f"Changed count or invalid indices: {key}")
        seen.extend(indices)
        actual_partitions[key] = frame.iloc[indices]["record_id"].astype(str).tolist()
        if actual_partitions[key] != split["partition_record_ids"][key]:
            raise ValueError(f"Record identity/order mismatch: {key}")
        role = {"train_idx": "train", "val_idx": "val", "test_batch_idx": "test_batch",
                "test_unknown_idx": "test_unknown"}[key]
        if not frame.iloc[indices]["split_role"].eq(role).all():
            raise ValueError(f"Metadata role mismatch: {key}")
    if sorted(seen) != list(range(len(frame))):
        raise ValueError("Split overlap, duplicates, or missing samples")
    split_hash = hash_json({"protocol": split["protocol"], "manifest_fingerprint": manifest_hash,
                            "partitions": actual_partitions})
    if split_hash != SPLIT_HASH or split.get("split_fingerprint") != SPLIT_HASH:
        raise ValueError("Approved split fingerprint mismatch")
    if tuple(split["known_products"]) != KNOWN or sorted(split["holdout_products"]) != ["HMD", "XCJ"]:
        raise ValueError("Product roles changed")
    batches = {}
    known_groups = []
    for key in ROLE_COUNTS:
        rows = frame.iloc[split[key]]
        products = sorted(rows["product_fine"].unique())
        if key != "test_unknown_idx" and products != list(KNOWN):
            raise ValueError(f"Missing or unexpected known classes: {key}")
        if key == "test_unknown_idx" and rows["product_fine"].value_counts().to_dict() != {"HMD": 68, "XCJ": 65}:
            raise ValueError("Unknown pool changed")
        batches[key] = sorted(rows["batch_name"].astype(str).unique())
        if key != "test_unknown_idx":
            known_groups.append(set(rows["group_id"].astype(str)))
    if any(a & b for i, a in enumerate(known_groups) for b in known_groups[i + 1:]):
        raise ValueError("Known product group leakage across train/val/test")
    if batches["val_idx"] != ["20240312", "20250815"] or batches["test_batch_idx"] != ["20240308", "20250905", "20250912", "20250920", "20260313"] or len(batches["train_idx"]) != 24:
        raise ValueError("Approved batch isolation changed")
    cfg = Config(prepared_dir=str(prepared))
    info = apply_prepared_input_config(cfg)
    if (info is None or grid.get("input_representation") != "dual_channel_no_pca"
            or grid.get("source_manifest_fingerprint") != MANIFEST_HASH
            or (cfg.in_channels, cfg.rt_bins, cfg.mz_bins) != SHAPE
            or tuple(cfg.rt_range) != (0.0, 40.0) or tuple(cfg.mz_range) != (30.0, 532.0)
            or not grid.get("preprocessing_frozen") or grid.get("runtime_input_pca_allowed") is not False):
        raise ValueError("Approved frozen input representation changed")
    paths = [tensor_path(prepared, value) for value in frame["tensor_path"]]
    if len(set(paths)) != len(frame):
        raise ValueError("Multiple samples reuse a tensor file")
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty prepared tensor: {path}")
        if inspect_headers:
            # Read only the .npy header, not the full 2x2048x503 array.
            with zipfile.ZipFile(path) as archive, archive.open("tensor.npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError(f"Unsupported tensor header {version}: {path}")
            if tuple(shape) != SHAPE or dtype != np.dtype("float32"):
                raise ValueError(f"Unexpected tensor shape/dtype: {path}: {shape}, {dtype}")
    audit = {"protocol": PROTOCOL, "status": "pass", "prepared_dir": str(prepared),
             "manifest_fingerprint": manifest_hash, "split_fingerprint": split_hash,
             "counts": ROLE_COUNTS, "batches": batches, "shape": SHAPE,
             "tensor_files_checked": len(paths), "all_tensor_headers_checked": inspect_headers}
    return frame, split, audit


def build_config(prepared, output, group):
    backbone, depth = GROUPS[group]
    cfg = Config(prepared_dir=str(Path(prepared).resolve()), output_dir=str(Path(output).resolve()))
    settings = dict(main_backbone=backbone, blocks_per_stage=depth, encoder_channels=(32, 64, 128, 256),
                    num_axial_heads=4, feature_dim=256, proj_dim=128, dropout=0.3, embed_normalize=True,
                    in_channels=2, rt_bins=2048, mz_bins=503, main_backbone_model="", pretrained_feature_model="",
                    seed=42, split_seed=42, epochs=100, batch_size=32, lr=3e-4, weight_decay=1e-4,
                    lambda_supcon=1.0, lambda_cls=0.40, lambda_adv=0.0, lambda_proto=0.0,
                    lambda_recon=0.0, lambda_hard_pair=0.0, lambda_open_set=0.0,
                    supcon_temperature=0.07, focal_gamma=0.0, focal_beta=0.999,
                    skip_disabled_loss_heads=True, eval_interval=10, eval_interval_search=10,
                    eval_interval_final=10, model_select_metric="macro_f1", model_select_min_epoch=0,
                    early_stop_patience=0, warmup_guard_enabled=False, swa_enabled=False,
                    proto_val_subset_ratio=1.0, proto_val_subset_max_samples=1999, proto_val_full_every=1,
                    deterministic=True, amp_enabled=False, allow_tf32=False, cuda_benchmark=False,
                    channels_last=True, torch_compile=False, dataloader_workers=4,
                    dataloader_pin_memory=True, dataloader_persistent_workers=True,
                    dataloader_prefetch_factor=2, dataset_cache_in_memory=False,
                    product_granularity="fine", auto_create_split_on_train=False, input_raw_pca_enabled=False,
                    use_spherical_prototypes=False, accept_percentile=95.0,
                    open_score_calibration_enabled=False, open_score_calibration_apply=False,
                    open_score_auto_blend=False, evaluate_readme_baselines=False, tic_branch_enabled=False,
                    aug_intensity_scale=(0.8, 1.2), aug_noise_std=0.05, aug_mask_ratio=0.15,
                    aug_rt_shift_max=8, aug_mz_shift_max=2, aug_baseline_wander_amp=0.03,
                    aug_baseline_wander_freq=3, aug_peak_broaden_sigma=1.5, aug_peak_broaden_prob=0.1,
                    aug_rt_warp_strength=0.02, aug_rt_warp_prob=0.2, aug_mz_channel_drop=0.05,
                    aug_tic_jitter=0.1, n_shot_values=(1, 3, 5), fewshot_repeats=100, fewshot_seed_start=42000)
    for key, value in settings.items():
        setattr(cfg, key, value)
    apply_prepared_input_config(cfg)
    return cfg


def classification_metrics(truth, prediction, classes=KNOWN):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if not len(truth) or len(truth) != len(prediction):
        raise ValueError("Classification arrays must be nonempty and aligned")
    labels = list(range(len(classes)))
    matrix = confusion_matrix(truth, prediction, labels=labels)
    counts = matrix.sum(axis=1)
    recall = np.divide(matrix.diagonal(), counts, out=np.zeros(len(classes)), where=counts > 0)
    return {"n": len(truth), "accuracy": float(np.mean(truth == prediction)),
            "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
            "balanced_accuracy": float(recall.mean()), "classes": list(classes),
            "per_class_recall": dict(zip(classes, recall.tolist())),
            "per_class_count": dict(zip(classes, counts.tolist())), "confusion_matrix": matrix.tolist(),
            "averaging": "all declared classes; absent/never-predicted classes contribute zero"}


def selection_key(metrics):
    key = (float(metrics["macro_f1"]), float(metrics["balanced_accuracy"]))
    if not np.isfinite(key).all():
        raise ValueError("Nonfinite validation selection metric")
    return key


def fit_prototypes(embeddings, labels, classes=KNOWN):
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    prototypes, radii = [], []
    for label in range(len(classes)):
        values = embeddings[labels == label]
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Invalid training embeddings for class {classes[label]}")
        prototype = values.mean(axis=0)  # Deliberately do not normalize the mean.
        radius = max(float(np.percentile(np.linalg.norm(values - prototype, axis=1), 95)), 1e-6)
        prototypes.append(prototype)
        radii.append(radius)
    return np.stack(prototypes), np.asarray(radii)


def prototype_predictions(embeddings, prototypes, radii):
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite embeddings")
    distances = np.linalg.norm(embeddings[:, None, :] - prototypes[None, :, :], axis=2)
    predicted = distances.argmin(axis=1)
    minimum = distances[np.arange(len(distances)), predicted]
    second = np.partition(distances, 1, axis=1)[:, 1]
    radius = radii[predicted]
    return {"prediction": predicted, "distance": minimum, "second_distance": second,
            "radius": radius, "score": -minimum / radius,
            "margin": (second - minimum) / np.maximum(second, 1e-12)}


def validation_threshold(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid validation scores")
    return float(np.percentile(scores, 5))


def open_set_metrics(known_scores, unknown_scores, threshold):
    known_scores, unknown_scores = np.asarray(known_scores), np.asarray(unknown_scores)
    targets = np.r_[np.ones(len(known_scores)), np.zeros(len(unknown_scores))]
    scores = np.r_[known_scores, unknown_scores]
    if not len(known_scores) or not len(unknown_scores) or not np.isfinite(scores).all():
        raise ValueError("Both test pools need finite scores")
    fpr, tpr, thresholds = roc_curve(targets, scores, drop_intermediate=False)
    index = int(np.flatnonzero(tpr >= 0.95)[0])
    return {"known_n": len(known_scores), "unknown_n": len(unknown_scores),
            "auroc": float(roc_auc_score(targets, scores)), "fpr95": float(fpr[index]),
            "fpr95_definition": "first empirical ROC point with known TPR >= 0.95",
            "fpr95_diagnostic_threshold": float(thresholds[index]),
            "validation_accept_threshold": float(threshold),
            "known_acceptance": float(np.mean(known_scores >= threshold)),
            "unknown_false_acceptance": float(np.mean(unknown_scores >= threshold)),
            "unknown_rejection": float(np.mean(unknown_scores < threshold)),
            "score_definition": "-nearest_raw_mean_euclidean_distance/predicted_class_train_radius95",
            "threshold_scope": "validation_known_only; >= threshold accepts"}
