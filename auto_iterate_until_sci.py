"""Unified auto-iteration entry for SCI-oriented GCMS search.

This version is migrated to the current new_outputs + main.py pipeline.
It prioritizes Setting B FPR95 reduction while guarding Setting A and few-shot metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "new_outputs"
PROGRESS_LOG = OUTPUTS_DIR / "PROJECT_PROGRESS.md"
RESULTS_JSONL = OUTPUTS_DIR / "AUTO_SEARCH_RESULTS.jsonl"
PRACTICAL_RESULTS_JSONL = OUTPUTS_DIR / "AUTO_PRACTICAL_CANDIDATES.jsonl"
BEST_JSON = OUTPUTS_DIR / "AUTO_BEST_RUNS.json"
BEST_MD = OUTPUTS_DIR / "AUTO_BEST_RUNS.md"
PYTHON = sys.executable
SCRIPT_TAG = "AUTO3"

TARGETS = {
    "setting_a_accuracy_min": 0.90,
    "setting_a_balanced_acc_min": 0.75,
    "setting_a_macro_f1_min": 0.75,
    "setting_b_open_set_AUROC_min": 0.88,
    "setting_b_fpr95_max": 0.35,
    "setting_c_1shot_acc_min": 0.70,
    "setting_c_3shot_acc_min": 0.95,
    "known_unknown_gap_min": 0.20,
}

SEARCH_GUARDS = {
    "setting_a_accuracy_min": 0.84,
    "setting_a_balanced_acc_min": 0.66,
    "known_unknown_gap_min": 0.30,
}

FALLBACK_BASE = {
    "epochs": 100,
    "batch_size": 16,
    "lr": 2.0e-4,
    "weight_decay": 1.0e-4,
    "lambda_adv": 0.10,
    "lambda_proto": 0.85,
    "lambda_recon": 0.20,
    "lambda_cls": 0.50,
    "supcon_temperature": 0.07,
    "accept_percentile": 95.0,
    "reject_threshold_factor": 2.0,
    "eval_interval": 10,
    "eval_interval_search": 10,
    "eval_interval_final": 5,
    "eval_final_start_ratio": 0.7,
    "early_stop_patience": 12,
    "min_epochs_before_early_stop": 70,
    "min_epoch_ratio_before_early_stop": 0.7,
    "early_stop_min_lr_ratio": 0.1,
    "early_stop_min_delta": 3e-4,
    "proto_val_subset_ratio": 0.5,
    "proto_val_subset_min_samples": 256,
    "proto_val_subset_max_samples": 1024,
    "proto_val_full_every": 3,
    "dataloader_workers": 4,
    "dataloader_prefetch_factor": 2,
    "feature_dim": 256,
    "proj_dim": 128,
    "main_backbone": "gcms",
    "main_backbone_model": "",
    "main_feature_layers": "layer4",
    "main_feature_fuse": "concat",
    "transformer_patch_size": 16,
    "transformer_embed_dim": 256,
    "transformer_depth": 6,
    "transformer_num_heads": 8,
    "transformer_mlp_ratio": 4.0,
    "encoder_channels": (32, 64, 128, 256),
    "blocks_per_stage": 2,
    "num_axial_heads": 4,
    "dropout": 0.3,
    "primary_model": "deep_consistency",
    "prepared_dir": str(PROJECT_ROOT / "new_prepared_data"),
    "input_raw_pca_enabled": True,
    "input_raw_pca_components": 239,
    "rt_bins": 1024,
    "mz_bins": 239,
    "open_score_blend_objective": "fpr95",
    "open_score_auto_blend": True,
    "open_score_base_weight": 0.2,
    "open_score_margin_weight": 0.8,
    "aug_peak_broaden_prob": 0.1,
    "aug_rt_warp_prob": 0.2,
}

KEYS_TO_LOAD = [
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "lambda_adv",
    "lambda_proto",
    "lambda_recon",
    "lambda_cls",
    "supcon_temperature",
    "accept_percentile",
    "reject_threshold_factor",
    "eval_interval",
    "eval_interval_search",
    "eval_interval_final",
    "eval_final_start_ratio",
    "early_stop_patience",
    "min_epochs_before_early_stop",
    "min_epoch_ratio_before_early_stop",
    "early_stop_min_lr_ratio",
    "early_stop_min_delta",
    "proto_val_subset_ratio",
    "proto_val_subset_min_samples",
    "proto_val_subset_max_samples",
    "proto_val_full_every",
    "dataloader_workers",
    "dataloader_prefetch_factor",
    "feature_dim",
    "proj_dim",
    "main_backbone",
    "main_backbone_model",
    "main_feature_layers",
    "main_feature_fuse",
    "transformer_patch_size",
    "transformer_embed_dim",
    "transformer_depth",
    "transformer_num_heads",
    "transformer_mlp_ratio",
    "encoder_channels",
    "blocks_per_stage",
    "num_axial_heads",
    "dropout",
    "primary_model",
    "prepared_dir",
    "input_raw_pca_enabled",
    "input_raw_pca_components",
    "rt_bins",
    "mz_bins",
    "open_score_blend_objective",
    "open_score_base_weight",
    "open_score_margin_weight",
    "aug_peak_broaden_prob",
    "aug_rt_warp_prob",
]

MAX_TRIALS = int(os.environ.get("AUTO3_MAX_TRIALS", "80"))
POLL_SECONDS = int(os.environ.get("AUTO3_POLL_SECONDS", "15"))
GPU_IDS = [s.strip() for s in os.environ.get("AUTO3_GPU_IDS", "0").split(",") if s.strip()]
KEEP_ALL_RUN_DIRS = os.environ.get("AUTO_KEEP_ALL_RUN_DIRS", "0") != "0"
KEEP_TOP_N = int(os.environ.get("AUTO_KEEP_TOP_N", "5"))
WARMUP_GUARD_ENABLED = os.environ.get("AUTO3_WARMUP_GUARD", "1") != "0"
WARMUP_GUARD_EPOCH = int(os.environ.get("AUTO3_WARMUP_EPOCH", "10"))
WARMUP_GUARD_MIN_RATIO = float(os.environ.get("AUTO3_WARMUP_MIN_RATIO", "0.78"))
WARMUP_GUARD_COMPARE_BEST = os.environ.get("AUTO3_WARMUP_COMPARE_BEST", "0") == "1"


def parse_gpu_list(value: str | None) -> list[str | None]:
    if value is None:
        return [None]
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return parts or [None]


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_progress(lines: list[str]) -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write("\n")
        for line in lines:
            f.write(line + "\n")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except Exception:
        return default


def _normalize_encoder_channels(value: Any) -> tuple[int, ...]:
    if value is None:
        return tuple(FALLBACK_BASE["encoder_channels"])
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if parts:
            return tuple(int(v) for v in parts)
    return tuple(FALLBACK_BASE["encoder_channels"])


def has_active_training() -> bool:
    proc = subprocess.run(
        ["bash", "-lc", "ps -ef | rg 'main.py train' | rg -v rg"],
        capture_output=True,
        text=True,
    )
    return bool(proc.stdout.strip())


def extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    sa = summary.get("setting_a", {})
    sb = summary.get("setting_b", {})
    sc1 = summary.get("setting_c", {}).get("1", {})
    sc3 = summary.get("setting_c", {}).get("3", {})
    blend = summary.get("setting_b_score_blend_used", {}) or {}

    known_score_mean = sb.get("known_score_mean")
    unknown_score_mean = sb.get("unknown_score_mean")
    known_unknown_gap = None
    if known_score_mean is not None and unknown_score_mean is not None:
        try:
            known_unknown_gap = float(known_score_mean) - float(unknown_score_mean)
        except Exception:
            known_unknown_gap = None

    return {
        "setting_a_accuracy": sa.get("accuracy"),
        "setting_a_balanced_acc": sa.get("balanced_acc"),
        "setting_a_macro_f1": sa.get("macro_f1"),
        "open_set_AUROC": sb.get("open_set_AUROC"),
        "FPR_at_95TPR": sb.get("FPR_at_95TPR"),
        "known_unknown_gap": known_unknown_gap,
        "shot1_acc": sc1.get("accuracy"),
        "shot3_acc": sc3.get("accuracy"),
        "blend_base_weight": blend.get("base_weight"),
        "blend_margin_weight": blend.get("margin_weight"),
    }


def search_score(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    fpr95 = _safe_float(metrics.get("FPR_at_95TPR"), 1e9)
    auroc = _safe_float(metrics.get("open_set_AUROC"), -1.0)
    gap = _safe_float(metrics.get("known_unknown_gap"), -1.0)
    a_acc = _safe_float(metrics.get("setting_a_accuracy"), -1.0)
    a_bal = _safe_float(metrics.get("setting_a_balanced_acc"), -1.0)
    a_macro = _safe_float(metrics.get("setting_a_macro_f1"), -1.0)
    shot1 = _safe_float(metrics.get("shot1_acc"), -1.0)
    a_score = 0.45 * a_acc + 0.35 * a_bal + 0.20 * a_macro
    return (-fpr95, auroc, gap, a_score, shot1)


def collect_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in OUTPUTS_DIR.glob("**/evaluation_summary.json"):
        run_dir = summary_path.parent
        run_config_path = run_dir / "run_config.json"
        if not run_config_path.exists() and run_dir.name == "final_model":
            run_config_path = run_dir.parent / "run_config.json"
        summary = read_json(summary_path)
        if not summary:
            continue
        run_config = read_json(run_config_path) or {}
        cfg = run_config.get("config", {})
        # 兼容扁平结构：如果嵌套 config 为空且顶层有 feature_dim，直接读顶层
        if not cfg and "feature_dim" in run_config:
            _meta_keys = {"timestamp", "command", "args"}
            cfg = {k: v for k, v in run_config.items() if k not in _meta_keys}
        metrics = extract_metrics(summary)
        rows.append(
            {
                "run_dir": run_dir,
                "relative_name": str(run_dir.relative_to(OUTPUTS_DIR)),
                "summary": summary,
                "run_config": run_config,
                "config": cfg,
                "metrics": metrics,
            }
        )
    rows.sort(key=lambda row: search_score(row["metrics"]), reverse=True)
    return rows


def derive_base_config(best_run: dict[str, Any] | None) -> dict[str, Any]:
    base = dict(FALLBACK_BASE)
    if not best_run:
        return base
    cfg = best_run.get("config", {}) or {}
    for key in KEYS_TO_LOAD:
        if key in cfg and cfg[key] is not None:
            base[key] = cfg[key]
    base["encoder_channels"] = _normalize_encoder_channels(base.get("encoder_channels"))
    if isinstance(base.get("prepared_dir"), str) and not Path(base["prepared_dir"]).exists():
        base["prepared_dir"] = FALLBACK_BASE["prepared_dir"]
    return base


def next_auto_index() -> int:
    max_idx = 0
    for path in OUTPUTS_DIR.glob("**"):
        name = path.name
        if not name.startswith("iter_auto"):
            continue
        digits = []
        for ch in name[len("iter_auto"):]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            max_idx = max(max_idx, int("".join(digits)))
    return max_idx + 1


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def choose_directive(best_metrics: dict[str, Any]) -> str:
    fpr95 = _safe_float(best_metrics.get("FPR_at_95TPR"), 1.0)
    gap = _safe_float(best_metrics.get("known_unknown_gap"), 0.0)
    shot1 = _safe_float(best_metrics.get("shot1_acc"), 0.0)
    a_acc = _safe_float(best_metrics.get("setting_a_accuracy"), 0.0)
    a_bal = _safe_float(best_metrics.get("setting_a_balanced_acc"), 0.0)
    a_macro = _safe_float(best_metrics.get("setting_a_macro_f1"), 0.0)

    if fpr95 > 0.52:
        return "fpr95_crisis"
    if fpr95 > 0.45:
        return "fpr95_hard"
    if a_acc < 0.88 or a_bal < 0.72 or a_macro < 0.70:
        return "recover_setting_a"
    if fpr95 > 0.40 or gap < 0.33:
        return "fpr95_polish"
    if shot1 < 0.62:
        return "recover_shot1"
    if a_acc < 0.86:
        return "recover_setting_a"
    return "balanced_polish"


def choose_open_score_settings(directive: str, idx: int) -> tuple[bool, float | None, float | None, str]:
    if directive == "fpr95_crisis":
        pairs = [(False, 0.85, 0.15), (False, 0.90, 0.10), (True, None, None)]
        auto, base_w, margin_w = pairs[idx % len(pairs)]
        return auto, base_w, margin_w, "fpr95"
    if directive == "fpr95_hard":
        pairs = [(False, 0.80, 0.20), (False, 0.85, 0.15), (True, None, None)]
        auto, base_w, margin_w = pairs[idx % len(pairs)]
        return auto, base_w, margin_w, "fpr95"
    if directive == "fpr95_polish":
        pairs = [(True, None, None), (False, 0.75, 0.25), (False, 0.70, 0.30)]
        auto, base_w, margin_w = pairs[idx % len(pairs)]
        return auto, base_w, margin_w, "fpr95"
    if directive == "recover_shot1":
        return True, None, None, "balanced"
    if directive == "recover_setting_a":
        return True, None, None, "balanced"
    return True, None, None, "fpr95"


def build_candidate(base_cfg: dict[str, Any], idx: int, directive: str) -> dict[str, Any]:
    random.seed(20260702 + idx)
    presets = {
        "fpr95_crisis": {
            "lr_factors": [0.84, 0.88, 0.92],
            "adv_shifts": [-0.04, -0.03, -0.02],
            "proto_shifts": [-0.06, -0.04, -0.02],
            "temp_shifts": [-0.015, -0.01, -0.005],
            "accp_shifts": [-4.0, -3.0, -2.0],
            "reject_shifts": [-0.25, -0.20, -0.10],
            "batch_choices": [16, 16, 8],
            "deepen_prob": 0.55,
        },
        "fpr95_hard": {
            "lr_factors": [0.88, 0.92, 0.96],
            "adv_shifts": [-0.03, -0.02, -0.01],
            "proto_shifts": [-0.04, -0.02, 0.0],
            "temp_shifts": [-0.01, -0.005, 0.0],
            "accp_shifts": [-3.0, -2.0, -1.0],
            "reject_shifts": [-0.20, -0.10, 0.0],
            "batch_choices": [16, 16, 8],
            "deepen_prob": 0.45,
        },
        "fpr95_polish": {
            "lr_factors": [0.92, 0.96, 1.0],
            "adv_shifts": [-0.02, -0.01, 0.0],
            "proto_shifts": [-0.03, 0.0, 0.03],
            "temp_shifts": [-0.005, 0.0, 0.005],
            "accp_shifts": [-2.0, -1.0, 0.0],
            "reject_shifts": [-0.10, 0.0, 0.10],
            "batch_choices": [16, 16, 8],
            "deepen_prob": 0.35,
        },
        "recover_shot1": {
            "lr_factors": [0.94, 0.98, 1.02],
            "adv_shifts": [0.0, 0.01, 0.02],
            "proto_shifts": [0.02, 0.04, 0.06],
            "temp_shifts": [-0.005, 0.0],
            "accp_shifts": [0.0, 1.0, 2.0],
            "reject_shifts": [0.0, 0.10],
            "batch_choices": [16, 8],
            "deepen_prob": 0.15,
        },
        "recover_setting_a": {
            "lr_factors": [0.96, 1.0, 1.04],
            "adv_shifts": [0.01, 0.02, 0.03],
            "proto_shifts": [0.02, 0.04, 0.06],
            "temp_shifts": [0.0, 0.005],
            "accp_shifts": [0.0, 1.0],
            "reject_shifts": [0.0, 0.10],
            "batch_choices": [16, 8],
            "deepen_prob": 0.10,
        },
        "balanced_polish": {
            "lr_factors": [0.94, 0.98, 1.0],
            "adv_shifts": [-0.01, 0.0, 0.01],
            "proto_shifts": [0.0, 0.02, 0.04],
            "temp_shifts": [-0.005, 0.0, 0.005],
            "accp_shifts": [-1.0, 0.0, 1.0],
            "reject_shifts": [-0.10, 0.0, 0.10],
            "batch_choices": [16, 8],
            "deepen_prob": 0.10,
        },
    }

    preset = presets[directive]
    auto_blend, base_weight, margin_weight, blend_objective = choose_open_score_settings(directive, idx)

    lr = clip(float(base_cfg["lr"]) * random.choice(preset["lr_factors"]), 1.4e-4, 2.6e-4)
    lambda_adv = clip(float(base_cfg["lambda_adv"]) + random.choice(preset["adv_shifts"]), 0.04, 0.18)
    lambda_proto = clip(float(base_cfg["lambda_proto"]) + random.choice(preset["proto_shifts"]), 0.70, 0.96)
    supcon_temperature = clip(float(base_cfg["supcon_temperature"]) + random.choice(preset["temp_shifts"]), 0.05, 0.08)
    accept_percentile = clip(float(base_cfg["accept_percentile"]) + random.choice(preset["accp_shifts"]), 90.0, 98.5)
    reject_threshold_factor = clip(float(base_cfg["reject_threshold_factor"]) + random.choice(preset["reject_shifts"]), 1.65, 2.2)
    batch_size = int(random.choice(preset["batch_choices"]))

    feature_dim = int(random.choice([192, 256, 320]))
    proj_dim = int(random.choice([128, 192, 256]))
    input_raw_pca_components = int(random.choice([192, 239, 256]))
    main_backbone = str(base_cfg.get("main_backbone", "gcms") or "gcms")
    if directive == "recover_setting_a" and random.random() < 0.25:
        main_backbone = random.choice(["gcms", "resnet18"])
    else:
        main_backbone = random.choice([main_backbone, "gcms"])

    blocks = int(base_cfg.get("blocks_per_stage", 2))
    channels = tuple(base_cfg.get("encoder_channels", FALLBACK_BASE["encoder_channels"]))
    dropout = float(base_cfg.get("dropout", 0.3))
    deepen = random.random() < float(preset["deepen_prob"]) and blocks < 3
    if deepen:
        blocks = 3
        channels = random.choice([(32, 64, 128, 320), (32, 64, 160, 320)])
        dropout = clip(dropout + 0.05, 0.25, 0.5)

    warmup_name = f"{directive}_d{blocks}" if deepen else directive
    name = (
        f"iter_auto{idx:03d}_bs{batch_size}_lr{int(round(lr * 1e6))}"
        f"_a{int(round(lambda_adv * 100))}_p{int(round(lambda_proto * 100))}"
        f"_e{feature_dim}_q{proj_dim}_pca{input_raw_pca_components}_{main_backbone}_{warmup_name}"
    )

    return {
        **base_cfg,
        "name": name,
        "epochs": max(int(base_cfg.get("epochs", 100)), 100),
        "batch_size": batch_size,
        "lr": lr,
        "lambda_adv": lambda_adv,
        "lambda_proto": lambda_proto,
        "feature_dim": feature_dim,
        "proj_dim": proj_dim,
        "supcon_temperature": supcon_temperature,
        "accept_percentile": accept_percentile,
        "reject_threshold_factor": reject_threshold_factor,
        "open_score_auto_blend": auto_blend,
        "open_score_base_weight": base_weight,
        "open_score_margin_weight": margin_weight,
        "open_score_blend_objective": blend_objective,
        "eval_interval": 10,
        "eval_interval_search": 10,
        "eval_interval_final": 5,
        "early_stop_patience": max(int(base_cfg.get("early_stop_patience", 12)), 12),
        "min_epochs_before_early_stop": max(int(base_cfg.get("min_epochs_before_early_stop", 70)), 70),
        "min_epoch_ratio_before_early_stop": max(float(base_cfg.get("min_epoch_ratio_before_early_stop", 0.7)), 0.7),
        "early_stop_min_lr_ratio": min(float(base_cfg.get("early_stop_min_lr_ratio", 0.1)), 0.15),
        "encoder_channels": channels,
        "blocks_per_stage": blocks,
        "dropout": dropout,
        "input_raw_pca_components": input_raw_pca_components,
        "mz_bins": input_raw_pca_components,
        "main_backbone": main_backbone,
        "search_directive": directive,
    }


def build_candidate_batch(base_cfg: dict[str, Any], best_metrics: dict[str, Any], count: int) -> list[dict[str, Any]]:
    directive = choose_directive(best_metrics)
    idx = next_auto_index()
    candidates = [build_candidate(base_cfg, idx, directive)]

    a_acc = _safe_float(best_metrics.get("setting_a_accuracy"), 0.0)
    a_bal = _safe_float(best_metrics.get("setting_a_balanced_acc"), 0.0)
    a_macro = _safe_float(best_metrics.get("setting_a_macro_f1"), 0.0)
    needs_a_push = a_acc < TARGETS["setting_a_accuracy_min"] or a_bal < TARGETS["setting_a_balanced_acc_min"] or a_macro < TARGETS["setting_a_macro_f1_min"]
    if needs_a_push and directive != "recover_setting_a" and len(candidates) < count:
        candidates.append(build_candidate(base_cfg, idx + len(candidates), "recover_setting_a"))

    while len(candidates) < count:
        candidates.append(build_candidate(base_cfg, idx + len(candidates), directive))
    return candidates


def summarize_top_runs(runs: list[dict[str, Any]], limit: int = 5) -> list[str]:
    lines: list[str] = []
    for idx, row in enumerate(runs[:limit], start=1):
        m = row["metrics"]
        lines.append(
            f"{idx}. {row['relative_name']}: "
            f"A_acc={_safe_float(m.get('setting_a_accuracy'), float('nan')):.4f}, "
            f"A_bal={_safe_float(m.get('setting_a_balanced_acc'), float('nan')):.4f}, "
            f"A_f1={_safe_float(m.get('setting_a_macro_f1'), float('nan')):.4f}, "
            f"AUROC={_safe_float(m.get('open_set_AUROC'), float('nan')):.4f}, "
            f"FPR95={_safe_float(m.get('FPR_at_95TPR'), float('nan')):.4f}, "
            f"gap={_safe_float(m.get('known_unknown_gap'), float('nan')):.4f}, "
            f"1shot={_safe_float(m.get('shot1_acc'), float('nan')):.4f}, "
            f"3shot={_safe_float(m.get('shot3_acc'), float('nan')):.4f}"
        )
    return lines


def _compact_config(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "search_directive", "seed", "epochs", "batch_size", "lr", "weight_decay",
        "lambda_adv", "lambda_proto", "lambda_recon", "lambda_cls",
        "supcon_temperature", "accept_percentile", "reject_threshold_factor",
        "feature_dim", "proj_dim", "input_raw_pca_components",
        "main_backbone", "encoder_channels", "blocks_per_stage", "dropout",
        "open_score_auto_blend", "open_score_base_weight",
        "open_score_margin_weight", "open_score_blend_objective",
    ]
    return {k: cfg.get(k) for k in keys if k in cfg}


def write_best_snapshot(runs: list[dict[str, Any]], limit: int = 20) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    top = runs[:max(int(limit), 1)]
    payload = []
    for rank, row in enumerate(top, start=1):
        payload.append({
            "rank": rank,
            "run": row["relative_name"],
            "run_dir": str(row["run_dir"]),
            "metrics": row["metrics"],
            "config": _compact_config(row.get("config", {}) or {}),
        })

    with open(BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = [
        "# AUTO Best Runs",
        "",
        f"Updated: {ts()}",
        "",
        "| Rank | Run | A_acc | A_bal | A_f1 | B_AUROC | B_FPR95 | Gap | 1-shot | 3-shot | Key params |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload:
        m = row["metrics"]
        c = row["config"]
        params = (
            f"bb={c.get('main_backbone')}, e={c.get('feature_dim')}, "
            f"q={c.get('proj_dim')}, pca={c.get('input_raw_pca_components')}, "
            f"bs={c.get('batch_size')}, lr={c.get('lr')}, "
            f"adv={c.get('lambda_adv')}, proto={c.get('lambda_proto')}, "
            f"blend={c.get('open_score_base_weight')}/{c.get('open_score_margin_weight')}"
        )
        lines.append(
            f"| {row['rank']} | {row['run']} | "
            f"{_safe_float(m.get('setting_a_accuracy'), float('nan')):.4f} | "
            f"{_safe_float(m.get('setting_a_balanced_acc'), float('nan')):.4f} | "
            f"{_safe_float(m.get('setting_a_macro_f1'), float('nan')):.4f} | "
            f"{_safe_float(m.get('open_set_AUROC'), float('nan')):.4f} | "
            f"{_safe_float(m.get('FPR_at_95TPR'), float('nan')):.4f} | "
            f"{_safe_float(m.get('known_unknown_gap'), float('nan')):.4f} | "
            f"{_safe_float(m.get('shot1_acc'), float('nan')):.4f} | "
            f"{_safe_float(m.get('shot3_acc'), float('nan')):.4f} | "
            f"{params} |"
        )
    BEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_command(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        return proc.wait()


def _parse_warmup_val_acc(log_path: Path, epoch: int) -> float | None:
    if not log_path.exists():
        return None
    epoch_pat = re.compile(rf"\bEpoch\s+{int(epoch)}/")
    val_pat = re.compile(r"->\s+val_acc=([0-9.]+)")
    in_epoch = False
    try:
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if epoch_pat.search(line):
                in_epoch = True
                continue
            if in_epoch:
                m = val_pat.search(line)
                if m:
                    return float(m.group(1))
                if "Epoch " in line:
                    in_epoch = False
    except Exception:
        return None
    return None


def _best_warmup_reference(best_run: dict[str, Any] | None) -> float | None:
    if not best_run:
        return None

    run_config = best_run.get("run_config", {}) or {}
    cfg = run_config.get("config", {}) or {}
    ref = cfg.get("warmup_guard_best_at_epoch")
    if ref not in (None, 0, "0"):
        try:
            return float(ref)
        except Exception:
            pass

    run_dir = Path(best_run.get("run_dir", ""))
    for name in ("train.log", "iter_train.log"):
        parsed = _parse_warmup_val_acc(run_dir / name, WARMUP_GUARD_EPOCH)
        if parsed is not None:
            return parsed
    return None


def _warmup_guard_args(best_run: dict[str, Any] | None) -> list[str]:
    if not WARMUP_GUARD_ENABLED or not best_run:
        return []
    ref = _best_warmup_reference(best_run)
    if ref is None or ref <= 0:
        return []

    args = [
        "--warmup_guard_enabled",
        "--warmup_guard_epoch",
        str(WARMUP_GUARD_EPOCH),
        "--warmup_guard_best_at_epoch",
        str(ref),
        "--warmup_guard_min_ratio",
        str(WARMUP_GUARD_MIN_RATIO),
    ]
    if WARMUP_GUARD_COMPARE_BEST:
        args.append("--warmup_guard_compare_best")
    else:
        args.append("--warmup_guard_no_compare_best")
    return args


def launch_candidate(cfg: dict[str, Any], best_run: dict[str, Any] | None, gpu: str | None) -> dict[str, Any]:
    run_dir = OUTPUTS_DIR / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["GCMS_SHOW_PROGRESS"] = "0"
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    train_cmd = [
        PYTHON,
        "main.py",
        "train",
        "--output_dir",
        str(run_dir),
        "--prepared_dir",
        str(cfg["prepared_dir"]),
        "--epochs",
        str(cfg["epochs"]),
        "--batch_size",
        str(cfg["batch_size"]),
        "--lr",
        str(cfg["lr"]),
        "--weight_decay",
        str(cfg["weight_decay"]),
        "--lambda_adv",
        str(cfg["lambda_adv"]),
        "--lambda_proto",
        str(cfg["lambda_proto"]),
        "--lambda_recon",
        str(cfg["lambda_recon"]),
        "--lambda_cls",
        str(cfg["lambda_cls"]),
        "--supcon_temperature",
        str(cfg["supcon_temperature"]),
        "--accept_percentile",
        str(cfg["accept_percentile"]),
        "--reject_threshold_factor",
        str(cfg["reject_threshold_factor"]),
        "--eval_interval",
        str(cfg["eval_interval"]),
        "--eval_interval_search",
        str(cfg["eval_interval_search"]),
        "--eval_interval_final",
        str(cfg["eval_interval_final"]),
        "--eval_final_start_ratio",
        str(cfg["eval_final_start_ratio"]),
        "--early_stop_patience",
        str(cfg["early_stop_patience"]),
        "--min_epochs_before_early_stop",
        str(cfg["min_epochs_before_early_stop"]),
        "--min_epoch_ratio_before_early_stop",
        str(cfg["min_epoch_ratio_before_early_stop"]),
        "--early_stop_min_lr_ratio",
        str(cfg["early_stop_min_lr_ratio"]),
        "--early_stop_min_delta",
        str(cfg["early_stop_min_delta"]),
        "--proto_val_subset_ratio",
        str(cfg["proto_val_subset_ratio"]),
        "--proto_val_subset_min_samples",
        str(cfg["proto_val_subset_min_samples"]),
        "--proto_val_subset_max_samples",
        str(cfg["proto_val_subset_max_samples"]),
        "--proto_val_full_every",
        str(cfg["proto_val_full_every"]),
        "--dataloader_workers",
        str(cfg["dataloader_workers"]),
        "--dataloader_prefetch_factor",
        str(cfg["dataloader_prefetch_factor"]),
        "--feature_dim",
        str(cfg["feature_dim"]),
        "--proj_dim",
        str(cfg["proj_dim"]),
        "--main_backbone",
        str(cfg["main_backbone"]),
        "--main_backbone_model",
        str(cfg["main_backbone_model"]),
        "--main_feature_layers",
        str(cfg["main_feature_layers"]),
        "--main_feature_fuse",
        str(cfg["main_feature_fuse"]),
        "--transformer_patch_size",
        str(cfg["transformer_patch_size"]),
        "--transformer_embed_dim",
        str(cfg["transformer_embed_dim"]),
        "--transformer_depth",
        str(cfg["transformer_depth"]),
        "--transformer_num_heads",
        str(cfg["transformer_num_heads"]),
        "--transformer_mlp_ratio",
        str(cfg["transformer_mlp_ratio"]),
        "--encoder_channels",
        ",".join(str(v) for v in cfg["encoder_channels"]),
        "--blocks_per_stage",
        str(cfg["blocks_per_stage"]),
        "--num_axial_heads",
        str(cfg["num_axial_heads"]),
        "--dropout",
        str(cfg["dropout"]),
        "--primary_model",
        str(cfg["primary_model"]),
        "--input_raw_pca_components",
        str(cfg["input_raw_pca_components"]),
        "--rt_bins",
        str(cfg["rt_bins"]),
        "--mz_bins",
        str(cfg["mz_bins"]),
        "--aug_peak_broaden_prob",
        str(cfg["aug_peak_broaden_prob"]),
        "--aug_rt_warp_prob",
        str(cfg["aug_rt_warp_prob"]),
        "--open_score_blend_objective",
        str(cfg["open_score_blend_objective"]),
    ]
    if cfg.get("input_raw_pca_enabled", True):
        train_cmd.append("--enable_input_raw_pca")
    else:
        train_cmd.append("--disable_input_raw_pca")
    if not cfg.get("open_score_auto_blend", True):
        train_cmd.append("--no_auto_open_score_blend")
        train_cmd.extend(["--open_score_base_weight", str(cfg["open_score_base_weight"])])
        train_cmd.extend(["--open_score_margin_weight", str(cfg["open_score_margin_weight"])])
    train_cmd.extend(_warmup_guard_args(best_run))

    eval_cmd = [
        PYTHON,
        "main.py",
        "evaluate",
        "--output_dir",
        str(run_dir),
        "--prepared_dir",
        str(cfg["prepared_dir"]),
        "--batch_size",
        str(cfg["batch_size"]),
        "--feature_dim",
        str(cfg["feature_dim"]),
        "--proj_dim",
        str(cfg["proj_dim"]),
        "--open_score_blend_objective",
        str(cfg["open_score_blend_objective"]),
    ]
    if not cfg.get("open_score_auto_blend", True):
        eval_cmd.append("--no_auto_open_score_blend")
        eval_cmd.extend(["--open_score_base_weight", str(cfg["open_score_base_weight"])])
        eval_cmd.extend(["--open_score_margin_weight", str(cfg["open_score_margin_weight"])])

    train_exit = _run_command(train_cmd, env, run_dir / "iter_train.log")
    eval_exit = -1
    if train_exit == 0:
        eval_exit = _run_command(eval_cmd, env, run_dir / "iter_eval.log")

    return {"train_exit": train_exit, "eval_exit": eval_exit, "run_dir": str(run_dir)}


def meets_targets(metrics: dict[str, Any]) -> bool:
    return (
        _safe_float(metrics.get("setting_a_accuracy"), -1.0) >= TARGETS["setting_a_accuracy_min"]
        and _safe_float(metrics.get("setting_a_balanced_acc"), -1.0) >= TARGETS["setting_a_balanced_acc_min"]
        and _safe_float(metrics.get("setting_a_macro_f1"), -1.0) >= TARGETS["setting_a_macro_f1_min"]
        and _safe_float(metrics.get("open_set_AUROC"), -1.0) >= TARGETS["setting_b_open_set_AUROC_min"]
        and _safe_float(metrics.get("FPR_at_95TPR"), 1e9) <= TARGETS["setting_b_fpr95_max"]
        and _safe_float(metrics.get("shot1_acc"), -1.0) >= TARGETS["setting_c_1shot_acc_min"]
        and _safe_float(metrics.get("shot3_acc"), -1.0) >= TARGETS["setting_c_3shot_acc_min"]
        and _safe_float(metrics.get("known_unknown_gap"), -1.0) >= TARGETS["known_unknown_gap_min"]
    )


def is_practical_candidate(metrics: dict[str, Any]) -> bool:
    return (
        _safe_float(metrics.get("setting_a_accuracy"), -1.0) >= SEARCH_GUARDS["setting_a_accuracy_min"]
        and _safe_float(metrics.get("setting_a_balanced_acc"), -1.0) >= SEARCH_GUARDS["setting_a_balanced_acc_min"]
        and _safe_float(metrics.get("known_unknown_gap"), -1.0) >= SEARCH_GUARDS["known_unknown_gap_min"]
    )


def _slim_run_dir(path: Path) -> None:
    if not path.name.startswith("iter_auto"):
        return
    keep_files = {
        "run_config.json",
        "evaluation_summary.json",
        "train.log",
        "eval.log",
        "iter_train.log",
        "iter_eval.log",
    }
    for child in path.iterdir():
        if child.name in keep_files:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def prune_old_runs(keep_names: set[str]) -> None:
    if KEEP_ALL_RUN_DIRS:
        return
    for path in OUTPUTS_DIR.iterdir():
        if not path.is_dir():
            continue
        if path.name in keep_names:
            continue
        if path.name.startswith("run_") or path.name.startswith("run_seed"):
            continue
        _slim_run_dir(path)


def record_completed_candidate(cfg: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    summary = read_json(Path(result["run_dir"]) / "evaluation_summary.json")
    metrics = extract_metrics(summary) if summary else {}
    record = {
        "timestamp": ts(),
        "phase": SCRIPT_TAG,
        "name": cfg["name"],
        "status": "done" if result["train_exit"] == 0 and result["eval_exit"] == 0 else "failed",
        "config": cfg,
        "result": result,
        "metrics": metrics,
    }
    append_jsonl(RESULTS_JSONL, record)
    if is_practical_candidate(metrics):
        append_jsonl(PRACTICAL_RESULTS_JSONL, record)
    append_progress([
        f"- [{ts()}] {SCRIPT_TAG} DONE {cfg['name']}",
        f"  - train_exit={result['train_exit']}, eval_exit={result['eval_exit']}",
        f"  - setting_a_accuracy={metrics.get('setting_a_accuracy')}",
        f"  - setting_a_balanced_acc={metrics.get('setting_a_balanced_acc')}",
        f"  - setting_a_macro_f1={metrics.get('setting_a_macro_f1')}",
        f"  - open_set_AUROC={metrics.get('open_set_AUROC')}",
        f"  - FPR_at_95TPR={metrics.get('FPR_at_95TPR')}",
        f"  - known_unknown_gap={metrics.get('known_unknown_gap')}",
        f"  - shot1_acc={metrics.get('shot1_acc')}, shot3_acc={metrics.get('shot3_acc')}",
        f"  - meets_targets={meets_targets(metrics)}",
    ])
    return record


def log_candidate_start(cfg: dict[str, Any], gpu: str | None) -> None:
    append_progress([
        f"- [{ts()}] {SCRIPT_TAG} RUN {cfg['name']}",
        f"  - gpu={gpu}",
        f"  - search_directive={cfg['search_directive']}",
        f"  - epochs={cfg['epochs']}, batch_size={cfg['batch_size']}, lr={cfg['lr']}",
        f"  - lambda_adv={cfg['lambda_adv']}, lambda_proto={cfg['lambda_proto']}, lambda_recon={cfg['lambda_recon']}",
        f"  - accept_percentile={cfg['accept_percentile']}, reject_threshold_factor={cfg['reject_threshold_factor']}",
        f"  - auto_blend={cfg['open_score_auto_blend']}, base={cfg.get('open_score_base_weight')}, margin={cfg.get('open_score_margin_weight')}, objective={cfg['open_score_blend_objective']}",
        f"  - backbone={cfg['main_backbone']}, encoder_channels={cfg['encoder_channels']}, blocks={cfg['blocks_per_stage']}, dropout={cfg['dropout']}",
        f"  - feature_dim={cfg['feature_dim']}, proj_dim={cfg['proj_dim']}, input_raw_pca_components={cfg['input_raw_pca_components']}",
    ])


def run_candidate_job(cfg: dict[str, Any], best_run: dict[str, Any] | None, gpu: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    log_candidate_start(cfg, gpu)
    result = launch_candidate(cfg, best_run, gpu)
    return cfg, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified SCI auto-iteration on new_outputs")
    parser.add_argument("--count", type=int, default=4, help="Number of candidates per batch")
    parser.add_argument("--max_trials", type=int, default=MAX_TRIALS, help="Max launched trials")
    parser.add_argument("--gpu", type=str, default=(GPU_IDS[0] if GPU_IDS else None), help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--concurrent", type=int, default=int(os.environ.get("AUTO3_MAX_CONCURRENT", "1")),
                        help="Number of candidate runs to execute at the same time")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze recent results and print planned candidates")
    args = parser.parse_args()

    runs = collect_runs()
    write_best_snapshot(runs, limit=20)
    best_run = runs[0] if runs else None
    base_cfg = derive_base_config(best_run)
    best_metrics = best_run["metrics"] if best_run else extract_metrics({})
    candidates = build_candidate_batch(base_cfg, best_metrics, max(int(args.count), 1))
    directive = choose_directive(best_metrics)

    append_progress([
        f"- [{ts()}] {SCRIPT_TAG} START",
        f"  - outputs_dir={OUTPUTS_DIR}",
        f"  - discovered_runs={len(runs)}",
        f"  - best_run={(best_run['relative_name'] if best_run else 'fallback')}",
        f"  - directive={directive}",
        f"  - count={args.count}, max_trials={args.max_trials}, gpu={args.gpu}, concurrent={args.concurrent}",
        *[f"  - top: {line}" for line in summarize_top_runs(runs, limit=3)],
    ])

    print("=== Recent Top Runs ===")
    for line in summarize_top_runs(runs, limit=5):
        print(line)

    print("\n=== Planned Candidates ===")
    for cfg in candidates:
        print(
            f"{cfg['name']}: bs={cfg['batch_size']}, lr={cfg['lr']:.6f}, "
            f"adv={cfg['lambda_adv']:.3f}, proto={cfg['lambda_proto']:.3f}, "
            f"temp={cfg['supcon_temperature']:.3f}, accp={cfg['accept_percentile']:.1f}, "
            f"reject={cfg['reject_threshold_factor']:.2f}, blocks={cfg['blocks_per_stage']}, "
            f"embed={cfg['feature_dim']}, proj={cfg['proj_dim']}, "
            f"pca={cfg['input_raw_pca_components']}, backbone={cfg['main_backbone']}, "
            f"dropout={cfg['dropout']:.2f}, auto_blend={cfg['open_score_auto_blend']}, "
            f"blend_obj={cfg['open_score_blend_objective']}"
        )

    if args.analyze_only:
        return 0

    max_trials = max(int(args.max_trials), 1)
    concurrent = max(int(args.concurrent or 1), 1)
    to_run = candidates[:max_trials]
    gpu_list = parse_gpu_list(args.gpu)

    if concurrent <= 1:
        for cfg in to_run:
            while has_active_training():
                append_progress([
                    f"- [{ts()}] {SCRIPT_TAG} WAIT",
                    "  - detected active main.py train process, waiting before next candidate",
                ])
                import time
                time.sleep(POLL_SECONDS)

            _cfg, result = run_candidate_job(cfg, best_run, gpu_list[0])
            record = record_completed_candidate(_cfg, result)
            runs = collect_runs()
            write_best_snapshot(runs, limit=20)
            best_run = runs[0] if runs else best_run
            if best_run:
                best_metrics = best_run["metrics"]
            if meets_targets(record["metrics"]):
                append_progress([
                    f"- [{ts()}] {SCRIPT_TAG} TARGET REACHED {_cfg['name']}",
                    f"  - best_run={(best_run['relative_name'] if best_run else _cfg['name'])}",
                ])
                break
    else:
        while has_active_training():
            append_progress([
                f"- [{ts()}] {SCRIPT_TAG} WAIT",
                "  - detected active main.py train process before concurrent batch",
            ])
            import time
            time.sleep(POLL_SECONDS)

        with ThreadPoolExecutor(max_workers=min(concurrent, len(to_run))) as pool:
            futures = []
            for i, cfg in enumerate(to_run):
                gpu = gpu_list[i % len(gpu_list)]
                futures.append(pool.submit(run_candidate_job, cfg, best_run, gpu))

            for fut in as_completed(futures):
                _cfg, result = fut.result()
                record = record_completed_candidate(_cfg, result)
                runs = collect_runs()
                write_best_snapshot(runs, limit=20)
                if meets_targets(record["metrics"]):
                    append_progress([
                        f"- [{ts()}] {SCRIPT_TAG} TARGET REACHED {_cfg['name']}",
                        "  - concurrent batch will finish already launched candidates",
                    ])

    runs = collect_runs()
    write_best_snapshot(runs, limit=20)
    keep_names = {row["run_dir"].name for row in runs[: max(KEEP_TOP_N, int(args.count), 3)]}
    prune_old_runs(keep_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
