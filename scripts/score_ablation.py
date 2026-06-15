#!/usr/bin/env python3
"""Score-only ablation for Setting B open-set detection.

This script reuses an existing trained run. It does not train or rewrite the
checkpoint. It evaluates multiple prototype score definitions on the same
known/unknown split and writes a table under the run directory.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config, get_device
from dataset import GCMSDataset, load_data_split
from models import GCMSConsistencyNet
from register import PrototypeStore, register_from_loader


def _load_cfg(run_dir: Path) -> Config:
    cfg = Config()
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "config" in data and isinstance(data["config"], dict):
            data = data["config"]
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    cfg.output_dir = str(run_dir)
    return cfg


def _metrics(known_scores, unknown_scores):
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    labels = np.concatenate([
        np.ones(len(known_scores), dtype=np.int64),
        np.zeros(len(unknown_scores), dtype=np.int64),
    ])
    scores = np.concatenate([known_scores, unknown_scores])
    out = {
        "known_mean": float(np.mean(known_scores)) if len(known_scores) else float("nan"),
        "unknown_mean": float(np.mean(unknown_scores)) if len(unknown_scores) else float("nan"),
        "gap": float(np.mean(known_scores) - np.mean(unknown_scores))
        if len(known_scores) and len(unknown_scores) else float("nan"),
    }
    if len(np.unique(labels)) < 2:
        out.update({
            "AUROC": float("nan"),
            "AUPR": float("nan"),
            "FPR95": float("nan"),
            "EER": float("nan"),
            "TPR_FPR5": float("nan"),
            "TPR_FPR10": float("nan"),
        })
        return out

    fpr, tpr, _ = roc_curve(labels, scores)
    idx95 = np.searchsorted(tpr, 0.95, side="left")
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))

    def _tpr_at(max_fpr):
        valid = np.where(fpr <= max_fpr)[0]
        return float(np.max(tpr[valid])) if len(valid) else 0.0

    out.update({
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPR": float(average_precision_score(labels, scores)),
        "FPR95": float(fpr[min(idx95, len(fpr) - 1)]),
        "EER": float((fpr[eer_idx] + fnr[eer_idx]) / 2.0),
        "TPR_FPR5": _tpr_at(0.05),
        "TPR_FPR10": _tpr_at(0.10),
    })
    return out


def _batch_tic(batch, device):
    tic = batch.get("tic") if isinstance(batch, dict) else None
    return tic.to(device, non_blocking=True) if torch.is_tensor(tic) else None


def _make_dataset(metadata_csv, product_col, indices, cfg, product_enc, batch_enc):
    ds = GCMSDataset(
        metadata_csv,
        product_col=product_col,
        augmentation=None,
        indices=indices,
        cfg=cfg,
    )
    ds.product_enc = product_enc
    ds.batch_enc = batch_enc
    ds.df["product_label"] = product_enc.transform(ds.df[product_col])
    ds.df["batch_label"] = batch_enc.transform(ds.df["batch_idx"])
    ds.num_products = len(product_enc.classes_)
    ds.num_batches = len(batch_enc.classes_)
    return ds


def _make_loader(metadata_csv, product_col, indices, cfg, product_enc, batch_enc):
    ds = _make_dataset(metadata_csv, product_col, indices, cfg, product_enc, batch_enc)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    return ds, loader


def _label_name_map(dataset):
    return {int(i): str(name) for i, name in enumerate(dataset.product_enc.classes_)}


def _collect_score_parts(model, proto_store, loader, device, use_spherical):
    model.eval()
    score_parts = {}
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        with torch.no_grad():
            z = model.encode(x, tic=_batch_tic(batch, device))
            pred = proto_store.predict(z, use_spherical=use_spherical)

        pred_idx = pred["pred_idx"].detach().cpu().numpy().astype(int)
        min_dist = pred["min_dists"].detach().cpu().numpy()
        second_dist = pred["second_dists"].detach().cpu().numpy()
        base = pred["base_scores"].detach().cpu().numpy()
        margin = pred["margin_scores"].detach().cpu().numpy()
        blend = pred["scores"].detach().cpu().numpy()

        if use_spherical and proto_store.spherical_radii:
            radii_source = proto_store.spherical_radii
            prefix = "spherical"
        else:
            radii_source = proto_store.radii
            prefix = "raw"
        radii = np.asarray(
            [radii_source[proto_store.class_names[i]] for i in pred_idx],
            dtype=np.float64,
        )
        radius_norm = min_dist / np.clip(radii, 1e-8, None)
        gap = second_dist - min_dist

        batch_scores = {
            f"{prefix}_blend": blend,
            f"{prefix}_base": base,
            f"{prefix}_margin": margin,
            f"{prefix}_neg_min_dist": -min_dist,
            f"{prefix}_neg_radius_norm": -radius_norm,
            f"{prefix}_gap": gap,
            f"{prefix}_neg_second_dist": -second_dist,
        }
        for key, values in batch_scores.items():
            score_parts.setdefault(key, []).append(np.asarray(values))

    return {k: np.concatenate(v) if v else np.asarray([]) for k, v in score_parts.items()}


def _write_tables(rows, out_csv: Path, out_md: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "score", "AUROC", "AUPR", "FPR95", "EER", "TPR_FPR5", "TPR_FPR10",
        "known_mean", "unknown_mean", "gap",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines = [
        "# Setting B Score Ablation",
        "",
        "| Score | AUROC | FPR95 | EER | TPR@FPR5 | Known Mean | Unknown Mean | Gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['score']} | {row['AUROC']:.4f} | {row['FPR95']:.4f} | "
            f"{row['EER']:.4f} | {row['TPR_FPR5']:.4f} | "
            f"{row['known_mean']:.4f} | {row['unknown_mean']:.4f} | {row['gap']:.4f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate open-set score definitions on an existing run")
    parser.add_argument("--run_dir", required=True, help="Existing run directory, e.g. output_new/diag677_fast_v2_seed42")
    parser.add_argument("--output_csv", default="", help="Optional CSV output path")
    parser.add_argument("--output_md", default="", help="Optional Markdown output path")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--rebuild_prototypes",
        action="store_true",
        help="Recompute prototypes from train_idx instead of loading final_model/prototypes",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg = _load_cfg(run_dir)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.dataset_cache_in_memory = bool(getattr(cfg, "dataset_cache_in_memory", False))

    device = get_device()
    split = load_data_split(cfg)
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = "product_fine" if cfg.product_granularity == "fine" else "product_coarse"

    model_dir = run_dir / "final_model"
    meta_path = model_dir / "train_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        train_meta = json.load(f)
    num_batches_model = int(train_meta["num_batches"])
    for key in [
        "tic_branch_enabled", "tic_encoder", "tic_embed_dim",
        "tic_fusion_mode", "tic_fusion_output_dim",
    ]:
        if key in train_meta:
            setattr(cfg, key, train_meta[key])

    model = GCMSConsistencyNet(num_batches_model, cfg).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))

    import pandas as pd
    full_df = pd.read_csv(metadata_csv)
    full_df = full_df[(full_df["product_fine"] != "BLANK") & (~full_df["is_special"])].reset_index(drop=True)
    product_enc = LabelEncoder().fit(sorted(full_df[product_col].unique()))
    batch_enc = LabelEncoder().fit(sorted(full_df["batch_idx"].unique()))

    proto_store = None
    if not args.rebuild_prototypes:
        try:
            proto_store = PrototypeStore()
            proto_store.load(model_dir / "prototypes")
            print("Loaded prototypes from final_model/prototypes")
        except Exception as exc:
            print(f"Prototype load failed, rebuilding from train split: {exc}")
            proto_store = None

    if proto_store is None:
        ds_train, loader_train = _make_loader(
            metadata_csv, product_col, split["train_idx"], cfg, product_enc, batch_enc
        )
        proto_store, _, _ = register_from_loader(
            model,
            loader_train,
            _label_name_map(ds_train),
            device,
            percentile=float(getattr(cfg, "accept_percentile", 95.0)),
        )
        print("Rebuilt prototypes from train split")

    known_idx = sorted(set(split["val_idx"]) | set(split["test_batch_idx"]))
    unknown_idx = split["test_unknown_idx"]
    _, loader_known = _make_loader(metadata_csv, product_col, known_idx, cfg, product_enc, batch_enc)
    _, loader_unknown = _make_loader(metadata_csv, product_col, unknown_idx, cfg, product_enc, batch_enc)

    known_scores = {}
    unknown_scores = {}
    for use_spherical in [True, False]:
        known_scores.update(_collect_score_parts(model, proto_store, loader_known, device, use_spherical))
        unknown_scores.update(_collect_score_parts(model, proto_store, loader_unknown, device, use_spherical))

    rows = []
    for score_name in sorted(known_scores.keys()):
        row = {"score": score_name}
        row.update(_metrics(known_scores[score_name], unknown_scores[score_name]))
        rows.append(row)
        inv = {"score": f"inv_{score_name}"}
        inv.update(_metrics(-known_scores[score_name], -unknown_scores[score_name]))
        rows.append(inv)

    rows.sort(key=lambda r: (-(r.get("AUROC", float("nan"))), r.get("FPR95", 1.0)))

    out_csv = Path(args.output_csv) if args.output_csv else run_dir / "score_ablation.csv"
    out_md = Path(args.output_md) if args.output_md else run_dir / "score_ablation.md"
    _write_tables(rows, out_csv, out_md)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print("\nTop scores:")
    for row in rows[:10]:
        print(
            f"{row['score']:28s} AUROC={row['AUROC']:.4f} "
            f"FPR95={row['FPR95']:.4f} gap={row['gap']:.4f}"
        )


if __name__ == "__main__":
    main()
