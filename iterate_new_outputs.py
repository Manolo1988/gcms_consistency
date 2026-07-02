from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "new_outputs"
PROGRESS_LOG = OUTPUTS_DIR / "ITERATION_PROGRESS.md"
RESULTS_JSONL = OUTPUTS_DIR / "ITERATION_RESULTS.jsonl"
PYTHON = sys.executable

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
    "aug_peak_broaden_prob",
    "aug_rt_warp_prob",
]


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_progress(lines: list[str]) -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write("\n")
        for line in lines:
            f.write(line + "\n")


def append_result(obj: dict[str, Any]) -> None:
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


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


def extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    sa = summary.get("setting_a", {})
    sb = summary.get("setting_b", {})
    sc1 = summary.get("setting_c", {}).get("1", {})
    sc3 = summary.get("setting_c", {}).get("3", {})

    known_score_mean = sb.get("known_score_mean")
    unknown_score_mean = sb.get("unknown_score_mean")
    known_unknown_gap = None
    if known_score_mean is not None and unknown_score_mean is not None:
        try:
            known_unknown_gap = float(known_score_mean) - float(unknown_score_mean)
        except Exception:
            known_unknown_gap = None

    blend = summary.get("setting_b_score_blend_used", {}) or {}
    backbone = summary.get("main_model_backbone", {}) or {}

    return {
        "setting_a_accuracy": sa.get("accuracy"),
        "setting_a_balanced_acc": sa.get("balanced_acc"),
        "open_set_AUROC": sb.get("open_set_AUROC"),
        "FPR_at_95TPR": sb.get("FPR_at_95TPR"),
        "known_unknown_gap": known_unknown_gap,
        "shot1_acc": sc1.get("accuracy"),
        "shot3_acc": sc3.get("accuracy"),
        "blend_base_weight": blend.get("base_weight"),
        "blend_margin_weight": blend.get("margin_weight"),
        "main_model_backbone": backbone,
    }


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except Exception:
        return default


def rank_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    a_acc = _safe_float(metrics.get("setting_a_accuracy"), -1.0)
    a_bal = _safe_float(metrics.get("setting_a_balanced_acc"), -1.0)
    auroc = _safe_float(metrics.get("open_set_AUROC"), -1.0)
    fpr95 = _safe_float(metrics.get("FPR_at_95TPR"), 1e9)
    shot1 = _safe_float(metrics.get("shot1_acc"), -1.0)
    gap = _safe_float(metrics.get("known_unknown_gap"), -1.0)
    return (-(fpr95), auroc, gap, a_acc + 0.35 * a_bal, shot1)


def collect_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
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
        metrics = extract_metrics(summary)
        runs.append(
            {
                "run_dir": run_dir,
                "relative_name": str(run_dir.relative_to(OUTPUTS_DIR)),
                "summary": summary,
                "run_config": run_config,
                "config": cfg,
                "metrics": metrics,
            }
        )
    runs.sort(key=lambda item: rank_key(item["metrics"]), reverse=True)
    return runs


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
    for run_dir in OUTPUTS_DIR.glob("**/run_*"):
        name = run_dir.name
        if name.startswith("run_"):
            continue
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

    if fpr95 > 0.48:
        return "tighten_fpr_hard"
    if fpr95 > 0.42 or gap < 0.33:
        return "tighten_fpr"
    if shot1 < 0.62:
        return "raise_shot1"
    if a_acc < 0.86:
        return "recover_setting_a"
    return "balanced_polish"


def build_candidate(base_cfg: dict[str, Any], idx: int, directive: str) -> dict[str, Any]:
    random.seed(20260702 + idx)

    presets = {
        "tighten_fpr_hard": {
            "lr_factors": [0.88, 0.92, 0.96],
            "adv_shifts": [-0.03, -0.02, -0.01],
            "proto_shifts": [-0.04, -0.02, 0.0],
            "temp_shifts": [-0.01, -0.005, 0.0],
            "accp_shifts": [-3.0, -2.0, -1.0],
            "reject_shifts": [-0.20, -0.10, 0.0],
            "batch_choices": [16, 16, 8],
            "target_blend": "fpr95",
        },
        "tighten_fpr": {
            "lr_factors": [0.92, 0.96, 1.0],
            "adv_shifts": [-0.02, -0.01, 0.0],
            "proto_shifts": [-0.03, 0.0, 0.03],
            "temp_shifts": [-0.005, 0.0, 0.005],
            "accp_shifts": [-2.0, -1.0, 0.0],
            "reject_shifts": [-0.10, 0.0, 0.10],
            "batch_choices": [16, 16, 8],
            "target_blend": "fpr95",
        },
        "raise_shot1": {
            "lr_factors": [0.94, 0.98, 1.02],
            "adv_shifts": [0.0, 0.01, 0.02],
            "proto_shifts": [0.02, 0.04, 0.06],
            "temp_shifts": [-0.005, 0.0],
            "accp_shifts": [0.0, 1.0, 2.0],
            "reject_shifts": [0.0, 0.10],
            "batch_choices": [16, 8],
            "target_blend": "fpr95",
        },
        "recover_setting_a": {
            "lr_factors": [0.96, 1.0, 1.04],
            "adv_shifts": [0.01, 0.02, 0.03],
            "proto_shifts": [0.02, 0.04, 0.06],
            "temp_shifts": [0.0, 0.005],
            "accp_shifts": [0.0, 1.0],
            "reject_shifts": [0.0, 0.10],
            "batch_choices": [16, 8],
            "target_blend": "balanced",
        },
        "balanced_polish": {
            "lr_factors": [0.94, 0.98, 1.0],
            "adv_shifts": [-0.01, 0.0, 0.01],
            "proto_shifts": [0.0, 0.02, 0.04],
            "temp_shifts": [-0.005, 0.0, 0.005],
            "accp_shifts": [-1.0, 0.0, 1.0],
            "reject_shifts": [-0.10, 0.0, 0.10],
            "batch_choices": [16, 8],
            "target_blend": "fpr95",
        },
    }

    preset = presets[directive]
    lr = clip(float(base_cfg["lr"]) * random.choice(preset["lr_factors"]), 1.5e-4, 2.6e-4)
    lambda_adv = clip(float(base_cfg["lambda_adv"]) + random.choice(preset["adv_shifts"]), 0.05, 0.18)
    lambda_proto = clip(float(base_cfg["lambda_proto"]) + random.choice(preset["proto_shifts"]), 0.72, 0.96)
    supcon_temperature = clip(float(base_cfg["supcon_temperature"]) + random.choice(preset["temp_shifts"]), 0.055, 0.08)
    accept_percentile = clip(float(base_cfg["accept_percentile"]) + random.choice(preset["accp_shifts"]), 91.0, 98.5)
    reject_threshold_factor = clip(float(base_cfg["reject_threshold_factor"]) + random.choice(preset["reject_shifts"]), 1.7, 2.2)
    batch_size = int(random.choice(preset["batch_choices"]))

    blocks = int(base_cfg.get("blocks_per_stage", 2))
    channels = tuple(base_cfg.get("encoder_channels", FALLBACK_BASE["encoder_channels"]))
    dropout = float(base_cfg.get("dropout", 0.3))
    deepen = directive in {"tighten_fpr_hard", "tighten_fpr"} and random.random() < 0.35 and blocks < 3
    if deepen:
        blocks = 3
        channels = random.choice([(32, 64, 128, 320), (32, 64, 160, 320)])
        dropout = clip(dropout + 0.05, 0.25, 0.5)

    name = (
        f"iter_auto{idx:03d}_bs{batch_size}_lr{int(round(lr * 1e6))}"
        f"_a{int(round(lambda_adv * 100))}_p{int(round(lambda_proto * 100))}_{directive}"
    )

    return {
        **base_cfg,
        "name": name,
        "epochs": max(int(base_cfg.get("epochs", 100)), 100),
        "batch_size": batch_size,
        "lr": lr,
        "lambda_adv": lambda_adv,
        "lambda_proto": lambda_proto,
        "supcon_temperature": supcon_temperature,
        "accept_percentile": accept_percentile,
        "reject_threshold_factor": reject_threshold_factor,
        "open_score_blend_objective": preset["target_blend"],
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
        "search_directive": directive,
    }


def build_candidate_batch(base_cfg: dict[str, Any], best_metrics: dict[str, Any], count: int) -> list[dict[str, Any]]:
    directive = choose_directive(best_metrics)
    idx = next_auto_index()
    candidates = []
    for _ in range(count):
        candidates.append(build_candidate(base_cfg, idx, directive))
        idx += 1
    return candidates


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


def launch_candidate(cfg: dict[str, Any], gpu: str | None) -> dict[str, Any]:
    run_dir = OUTPUTS_DIR / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["GCMS_SHOW_PROGRESS"] = "0"
    if gpu is not None:
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
        "--open_score_blend_objective",
        str(cfg["open_score_blend_objective"]),
    ]

    train_exit = _run_command(train_cmd, env, run_dir / "iter_train.log")
    eval_exit = -1
    if train_exit == 0:
        eval_exit = _run_command(eval_cmd, env, run_dir / "iter_eval.log")

    return {
        "train_exit": train_exit,
        "eval_exit": eval_exit,
        "run_dir": str(run_dir),
    }


def summarize_top_runs(runs: list[dict[str, Any]], limit: int = 5) -> list[str]:
    lines: list[str] = []
    for idx, item in enumerate(runs[:limit], start=1):
        m = item["metrics"]
        lines.append(
            f"{idx}. {item['relative_name']}: "
            f"A_acc={_safe_float(m.get('setting_a_accuracy'), float('nan')):.4f}, "
            f"A_bal={_safe_float(m.get('setting_a_balanced_acc'), float('nan')):.4f}, "
            f"AUROC={_safe_float(m.get('open_set_AUROC'), float('nan')):.4f}, "
            f"FPR95={_safe_float(m.get('FPR_at_95TPR'), float('nan')):.4f}, "
            f"gap={_safe_float(m.get('known_unknown_gap'), float('nan')):.4f}, "
            f"1shot={_safe_float(m.get('shot1_acc'), float('nan')):.4f}, "
            f"3shot={_safe_float(m.get('shot3_acc'), float('nan')):.4f}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Iterate GCMS experiments from new_outputs")
    parser.add_argument("--count", type=int, default=4, help="Number of candidate runs to generate")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES value to use")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze recent runs and print planned candidates")
    args = parser.parse_args()

    runs = collect_runs()
    best_run = runs[0] if runs else None
    base_cfg = derive_base_config(best_run)
    best_metrics = best_run["metrics"] if best_run else extract_metrics({})
    candidates = build_candidate_batch(base_cfg, best_metrics, max(int(args.count), 1))

    append_progress([
        f"- [{ts()}] ITERATE START",
        f"  - discovered_runs={len(runs)}",
        f"  - best_run={(best_run['relative_name'] if best_run else 'fallback')}",
        f"  - directive={choose_directive(best_metrics)}",
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
            f"dropout={cfg['dropout']:.2f}, blend_obj={cfg['open_score_blend_objective']}"
        )

    if args.analyze_only:
        return 0

    for cfg in candidates:
        append_progress([
            f"- [{ts()}] ITERATE RUN {cfg['name']}",
            f"  - search_directive={cfg['search_directive']}",
            f"  - epochs={cfg['epochs']}, batch_size={cfg['batch_size']}, lr={cfg['lr']}",
            f"  - lambda_adv={cfg['lambda_adv']}, lambda_proto={cfg['lambda_proto']}, lambda_recon={cfg['lambda_recon']}",
            f"  - accept_percentile={cfg['accept_percentile']}, reject_threshold_factor={cfg['reject_threshold_factor']}",
            f"  - backbone={cfg['main_backbone']}, encoder_channels={cfg['encoder_channels']}, blocks={cfg['blocks_per_stage']}, dropout={cfg['dropout']}",
        ])
        result = launch_candidate(cfg, args.gpu)
        summary = read_json(Path(result["run_dir"]) / "evaluation_summary.json")
        metrics = extract_metrics(summary) if summary else {}
        append_result(
            {
                "timestamp": ts(),
                "name": cfg["name"],
                "config": cfg,
                "result": result,
                "metrics": metrics,
            }
        )
        append_progress([
            f"- [{ts()}] ITERATE DONE {cfg['name']}",
            f"  - train_exit={result['train_exit']}, eval_exit={result['eval_exit']}",
            f"  - setting_a_accuracy={metrics.get('setting_a_accuracy')}",
            f"  - setting_a_balanced_acc={metrics.get('setting_a_balanced_acc')}",
            f"  - open_set_AUROC={metrics.get('open_set_AUROC')}",
            f"  - FPR_at_95TPR={metrics.get('FPR_at_95TPR')}",
            f"  - known_unknown_gap={metrics.get('known_unknown_gap')}",
            f"  - shot1_acc={metrics.get('shot1_acc')}, shot3_acc={metrics.get('shot3_acc')}",
        ])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
