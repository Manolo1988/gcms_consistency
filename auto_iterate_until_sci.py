"""Continuous auto-iteration until SCI-level targets are met.

Strategy:
- Always compare new strategy against current best completed run.
- Keep artifacts only for top-N runs by final metrics.
- Generate new candidate hyperparameters around current best region.
- Continue searching until targets are reached or max trials hit.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROGRESS_LOG = OUTPUTS_DIR / "PROJECT_PROGRESS.md"
RESULTS_JSONL = OUTPUTS_DIR / "AUTO_SEARCH_RESULTS.jsonl"
PRACTICAL_RESULTS_JSONL = OUTPUTS_DIR / "AUTO_PRACTICAL_CANDIDATES.jsonl"
PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")

TARGETS = {
    "setting_b_open_set_AUROC_min": 0.61,
    "setting_b_fpr95_max": 0.70,
    "setting_c_3shot_acc_min": 0.85,
}

BASELINE_DOMINANCE_MARGIN = {
    "setting_a_accuracy": 0.02,
    "setting_a_balanced_acc": 0.02,
    "open_set_AUROC": 0.02,
    "FPR_at_95TPR": 0.02,
    "shot1_acc": 0.03,
    "shot3_acc": 0.02,
}

INDUSTRIAL_TARGETS = {
    "setting_a_accuracy_min": 0.60,
    "setting_a_balanced_acc_min": 0.45,
    "setting_b_open_set_AUROC_min": 0.88,
    "setting_b_fpr95_max": 0.35,
    "setting_c_1shot_acc_min": 0.70,
    "setting_c_3shot_acc_min": 0.95,
    "known_unknown_gap_min": 0.20,
}

SEARCH_GUARDS = {
    "setting_a_accuracy_min": 0.18,
    "setting_a_balanced_acc_min": 0.18,
    "known_unknown_gap_min": 0.08,
}

# Path A: 约束式多目标搜索
# 先满足硬约束，再在可行解中优先比较 AUROC / A_bal。
CONSTRAINT_MODE = os.environ.get("AUTO3_CONSTRAINT_MODE", "1") != "0"
CONSTRAINT_ANCHOR_RUN = os.environ.get(
    "AUTO3_CONSTRAINT_ANCHOR_RUN",
    "iter_auto287_bs16_lr260_a12_p88_fer50",
)
CONSTRAINT_MAX_A_DROP = float(os.environ.get("AUTO3_CONSTRAINT_MAX_A_DROP", "0.01"))
CONSTRAINT_MAX_FPR_RISE = float(os.environ.get("AUTO3_CONSTRAINT_MAX_FPR_RISE", "0.02"))
CONSTRAINT_MAX_C1_DROP = float(os.environ.get("AUTO3_CONSTRAINT_MAX_C1_DROP", "0.05"))

# 路径A补充留档: 不要求全面碾压 anchor，只要主指标有提升且其余核心指标仍高于对比基线。
PRACTICAL_RECORD_ENABLED = os.environ.get("AUTO3_PRACTICAL_RECORD_ENABLED", "1") != "0"
PRACTICAL_MAIN_METRIC = os.environ.get("AUTO3_PRACTICAL_MAIN_METRIC", "setting_a_accuracy")
PRACTICAL_MAIN_GAIN_MIN = float(os.environ.get("AUTO3_PRACTICAL_MAIN_GAIN_MIN", "0.0"))
PRACTICAL_FPR_HEADROOM = float(os.environ.get("AUTO3_PRACTICAL_FPR_HEADROOM", "0.0"))
PRACTICAL_MAIN_WEIGHT = float(os.environ.get("AUTO3_PRACTICAL_MAIN_WEIGHT", "3.0"))
PRACTICAL_SOFT_WEIGHT = float(os.environ.get("AUTO3_PRACTICAL_SOFT_WEIGHT", "1.0"))
PRACTICAL_WEIGHTED_MIN = float(os.environ.get("AUTO3_PRACTICAL_WEIGHTED_MIN", "0.70"))

WAIT_SECONDS = 60
POLL_SECONDS = int(os.environ.get("AUTO3_POLL_SECONDS", "15"))
WARMUP_GUARD_EPOCH = int(os.environ.get("AUTO3_WARMUP_EPOCH", "10"))
WARMUP_GUARD_FALLBACK_EPOCH = int(os.environ.get("AUTO3_WARMUP_FALLBACK_EPOCH", "15"))
WARMUP_GUARD_COMPARE_BEST = os.environ.get("AUTO3_WARMUP_COMPARE_BEST", "0") == "1"
WARMUP_GUARD_MIN_RATIO = float(os.environ.get("AUTO3_WARMUP_MIN_RATIO", "0.75"))
KEEP_TOP_N = 1
MAX_TRIALS = int(os.environ.get("AUTO3_MAX_TRIALS", "80"))
KEEP_ALL_RUN_DIRS = os.environ.get("AUTO_KEEP_ALL_RUN_DIRS", "1") != "0"
MAX_CONCURRENT_TRAININGS = int(os.environ.get("AUTO3_MAX_CONCURRENT", "3"))
GPU_IDS = [s.strip() for s in os.environ.get("AUTO3_GPU_IDS", "0,1").split(",") if s.strip()]
SCRIPT_TAG = "AUTO3"

# 一次性插入输入格式横向对照: 与当前best同模型同核心超参，仅改输入格式
INSERT_INPUT_FORMAT_ABLATIONS = os.environ.get("AUTO3_INSERT_INPUT_FORMAT_ABLATIONS", "1") == "1"
INPUT_ABLATION_BINS_PREPARED_DIR = os.environ.get(
    "AUTO3_INPUT_ABLATION_BINS_PREPARED_DIR",
    str(PROJECT_ROOT / "prepared_data" / "bins_rt1152_mz288"),
)
INPUT_ABLATION_PCA_PREPARED_DIR = os.environ.get(
    "AUTO3_INPUT_ABLATION_PCA_PREPARED_DIR",
    str(PROJECT_ROOT / "prepared_data_pca171"),
)
INSERT_STRUCTURED_LOCAL_QUEUE = os.environ.get("AUTO3_INSERT_STRUCTURED_LOCAL_QUEUE", "1") == "1"
INSERT_MAIN_BACKBONE_ABLATIONS = os.environ.get("AUTO3_INSERT_MAIN_BACKBONE_ABLATIONS", "1") == "1"

_CONSTRAINT_ANCHOR_METRICS_CACHE: dict | None = None
_CONSTRAINT_ANCHOR_METRICS_LOADED = False


def effective_max_concurrent() -> int:
    gpu_count = max(len(GPU_IDS), 1)
    return max(1, min(MAX_CONCURRENT_TRAININGS, gpu_count * 2, 4))


EFFECTIVE_MAX_CONCURRENT = effective_max_concurrent()

FALLBACK_BASE = {
    "epochs": 200,
    "batch_size": 16,
    "lr": 2.0e-4,
    "lambda_adv": 0.20,
    "lambda_proto": 0.80,
    "lambda_recon": 0.20,
    "supcon_temperature": 0.07,
    "accept_percentile": 95.0,
    "reject_threshold_factor": 2.0,
    "raw_open_score_blend": 1.0,
    "raw_distance_percentile": 95.0,
    "eval_interval": 5,
    "early_stop_patience": 12,
    "min_epochs_before_early_stop": 120,
    "min_epoch_ratio_before_early_stop": 0.6,
    "early_stop_min_lr_ratio": 0.1,
    "early_stop_min_delta": 3e-4,
    "encoder_channels": (32, 64, 128, 256),
    "blocks_per_stage": 2,
    "num_axial_heads": 4,
    "dropout": 0.3,
    "main_backbone": "gcms",
    "main_backbone_model": "",
    "main_feature_layers": "layer4",
    "main_feature_fuse": "concat",
    "transformer_patch_size": 16,
    "transformer_embed_dim": 256,
    "transformer_depth": 6,
    "transformer_num_heads": 8,
    "transformer_mlp_ratio": 4.0,
}

PRETRAINED_MODEL_PATTERNS = [
    ("resnet18", "r18", "pretrained_models/timm/resnet18*.pth"),
    ("resnet50", "r50", "pretrained_models/timm/resnet50*.pth"),
    ("wide_resnet50_2", "wr50", "pretrained_models/timm/wide_resnet50*.pth"),
]


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_progress(lines: list[str]) -> None:
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write("\n")
        for line in lines:
            f.write(line + "\n")


def append_result_jsonl(obj: dict) -> None:
    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_practical_jsonl(obj: dict) -> None:
    with open(PRACTICAL_RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def has_active_training() -> bool:
    proc = subprocess.run(
        ["bash", "-lc", "ps -ef | rg 'run_experiment.py' | rg -v rg"],
        capture_output=True,
        text=True,
    )
    return bool(proc.stdout.strip())


def read_summary(run_name: str) -> dict | None:
    p = OUTPUTS_DIR / run_name / "evaluation_summary.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(summary: dict) -> dict:
    sa = summary.get("setting_a", {})
    sb = summary.get("setting_b", {})
    sc1 = summary.get("setting_c", {}).get("1", {})
    sc = summary.get("setting_c", {}).get("3", {})
    baseline = summary.get("baseline_tic_pca_mlp", {})
    baseline_sb = baseline.get("setting_b", {})
    baseline_sc3 = baseline.get("setting_c", {}).get("3", {})

    known_score_mean = sb.get("known_score_mean")
    unknown_score_mean = sb.get("unknown_score_mean")
    score_gap = None
    if known_score_mean is not None and unknown_score_mean is not None:
        score_gap = float(known_score_mean) - float(unknown_score_mean)

    m = {
        "setting_a_accuracy": sa.get("accuracy"),
        "setting_a_balanced_acc": sa.get("balanced_acc"),
        "open_set_AUROC": sb.get("open_set_AUROC"),
        "FPR_at_95TPR": sb.get("FPR_at_95TPR"),
        "known_score_mean": known_score_mean,
        "unknown_score_mean": unknown_score_mean,
        "known_unknown_gap": score_gap,
        "shot1_acc": sc1.get("accuracy"),
        "shot3_acc": sc.get("accuracy"),
        "baseline_open_set_AUROC": baseline_sb.get("open_set_AUROC"),
        "baseline_FPR_at_95TPR": baseline_sb.get("FPR_at_95TPR"),
        "baseline_shot3_acc": baseline_sc3.get("accuracy"),
    }

    if m["open_set_AUROC"] is not None and m["baseline_open_set_AUROC"] is not None:
        m["delta_vs_baseline_open_set_AUROC"] = (
            float(m["open_set_AUROC"]) - float(m["baseline_open_set_AUROC"]))
    if m["FPR_at_95TPR"] is not None and m["baseline_FPR_at_95TPR"] is not None:
        m["delta_vs_baseline_FPR_at_95TPR"] = (
            float(m["FPR_at_95TPR"]) - float(m["baseline_FPR_at_95TPR"]))
    if m["shot3_acc"] is not None and m["baseline_shot3_acc"] is not None:
        m["delta_vs_baseline_shot3_acc"] = (
            float(m["shot3_acc"]) - float(m["baseline_shot3_acc"]))

    baselines_readme = summary.get("baselines_readme", {})
    cmp_readme = summary.get("main_vs_readme_baselines", {})
    baseline_brief = {}
    delta_brief = {}
    baseline_best_core = {
        "setting_a_accuracy": None,
        "setting_a_balanced_acc": None,
        "open_set_AUROC": None,
        "FPR_at_95TPR": None,
        "shot1_acc": None,
        "shot3_acc": None,
    }
    for key in ["pca_mahalanobis", "pls_da", "svm_rbf", "tic_pca_mlp"]:
        item = baselines_readme.get(key) or {}
        sa_i = item.get("setting_a") or {}
        sb_i = item.get("setting_b") or {}
        sc1_i = (item.get("setting_c") or {}).get("1", {})
        sc3_i = (item.get("setting_c") or {}).get("3", {})
        baseline_brief[key] = {
            "name": item.get("name", key),
            "feature_mode": item.get("feature_mode"),
            "setting_a_accuracy": sa_i.get("accuracy"),
            "setting_a_balanced_acc": sa_i.get("balanced_acc"),
            "open_set_AUROC": sb_i.get("open_set_AUROC"),
            "FPR_at_95TPR": sb_i.get("FPR_at_95TPR"),
            "shot1_acc": sc1_i.get("accuracy"),
            "shot3_acc": sc3_i.get("accuracy"),
        }

        sa_acc_i = sa_i.get("accuracy")
        if sa_acc_i is not None:
            if baseline_best_core["setting_a_accuracy"] is None or float(sa_acc_i) > float(baseline_best_core["setting_a_accuracy"]):
                baseline_best_core["setting_a_accuracy"] = float(sa_acc_i)

        sa_bal_i = sa_i.get("balanced_acc")
        if sa_bal_i is not None:
            if baseline_best_core["setting_a_balanced_acc"] is None or float(sa_bal_i) > float(baseline_best_core["setting_a_balanced_acc"]):
                baseline_best_core["setting_a_balanced_acc"] = float(sa_bal_i)

        auroc_i = sb_i.get("open_set_AUROC")
        if auroc_i is not None:
            if baseline_best_core["open_set_AUROC"] is None or float(auroc_i) > float(baseline_best_core["open_set_AUROC"]):
                baseline_best_core["open_set_AUROC"] = float(auroc_i)

        fpr_i = sb_i.get("FPR_at_95TPR")
        if fpr_i is not None:
            if baseline_best_core["FPR_at_95TPR"] is None or float(fpr_i) < float(baseline_best_core["FPR_at_95TPR"]):
                baseline_best_core["FPR_at_95TPR"] = float(fpr_i)

        shot1_i = sc1_i.get("accuracy")
        if shot1_i is not None:
            if baseline_best_core["shot1_acc"] is None or float(shot1_i) > float(baseline_best_core["shot1_acc"]):
                baseline_best_core["shot1_acc"] = float(shot1_i)

        shot3_i = sc3_i.get("accuracy")
        if shot3_i is not None:
            if baseline_best_core["shot3_acc"] is None or float(shot3_i) > float(baseline_best_core["shot3_acc"]):
                baseline_best_core["shot3_acc"] = float(shot3_i)

        cmp_i = cmp_readme.get(key) or {}
        cmp_sb_i = cmp_i.get("setting_b") or {}
        cmp_sc3_i = (cmp_i.get("setting_c") or {}).get("3", {})
        delta_brief[key] = {
            "delta_open_set_AUROC": cmp_sb_i.get("open_set_AUROC"),
            "delta_FPR_at_95TPR": cmp_sb_i.get("FPR_at_95TPR"),
            "delta_shot3_acc": cmp_sc3_i.get("accuracy"),
        }

    m["readme_baselines"] = baseline_brief
    m["main_minus_readme_baselines"] = delta_brief
    m["baseline_best_core"] = baseline_best_core

    if m["setting_a_accuracy"] is not None and baseline_best_core["setting_a_accuracy"] is not None:
        m["delta_vs_best_readme_setting_a_accuracy"] = (
            float(m["setting_a_accuracy"]) - float(baseline_best_core["setting_a_accuracy"])
        )
    if m["setting_a_balanced_acc"] is not None and baseline_best_core["setting_a_balanced_acc"] is not None:
        m["delta_vs_best_readme_setting_a_balanced_acc"] = (
            float(m["setting_a_balanced_acc"]) - float(baseline_best_core["setting_a_balanced_acc"])
        )
    if m["open_set_AUROC"] is not None and baseline_best_core["open_set_AUROC"] is not None:
        m["delta_vs_best_readme_open_set_AUROC"] = (
            float(m["open_set_AUROC"]) - float(baseline_best_core["open_set_AUROC"])
        )
    if m["FPR_at_95TPR"] is not None and baseline_best_core["FPR_at_95TPR"] is not None:
        m["delta_vs_best_readme_FPR_at_95TPR"] = (
            float(m["FPR_at_95TPR"]) - float(baseline_best_core["FPR_at_95TPR"])
        )
    if m["shot1_acc"] is not None and baseline_best_core["shot1_acc"] is not None:
        m["delta_vs_best_readme_shot1_acc"] = (
            float(m["shot1_acc"]) - float(baseline_best_core["shot1_acc"])
        )
    if m["shot3_acc"] is not None and baseline_best_core["shot3_acc"] is not None:
        m["delta_vs_best_readme_shot3_acc"] = (
            float(m["shot3_acc"]) - float(baseline_best_core["shot3_acc"])
        )

    pretrained_info = summary.get("pretrained_feature_extractor") or {}
    m["pretrained_feature_extractor"] = {
        "enabled": pretrained_info.get("enabled"),
        "arch": pretrained_info.get("arch"),
        "model_path": pretrained_info.get("model_path"),
        "layers": pretrained_info.get("layers"),
        "fuse": pretrained_info.get("fuse"),
    }

    main_backbone = summary.get("main_model_backbone") or {}
    m["main_model_backbone"] = {
        "backbone": main_backbone.get("backbone"),
        "model_path": main_backbone.get("model_path"),
        "layers": main_backbone.get("layers"),
        "fuse": main_backbone.get("fuse"),
    }

    return m


def discover_pretrained_cycle() -> list[dict]:
    out = []
    for arch, tag, pattern in PRETRAINED_MODEL_PATTERNS:
        matches = sorted(PROJECT_ROOT.glob(pattern))
        if not matches:
            continue
        out.append({
            "arch": arch,
            "tag": tag,
            "path": str(matches[0]),
        })
    return out


def dominates_readme_baselines(m: dict) -> bool:
    baseline_best = m.get("baseline_best_core") or {}
    checks = [
        (
            m.get("setting_a_accuracy"),
            baseline_best.get("setting_a_accuracy"),
            BASELINE_DOMINANCE_MARGIN["setting_a_accuracy"],
            True,
        ),
        (
            m.get("setting_a_balanced_acc"),
            baseline_best.get("setting_a_balanced_acc"),
            BASELINE_DOMINANCE_MARGIN["setting_a_balanced_acc"],
            True,
        ),
        (
            m.get("open_set_AUROC"),
            baseline_best.get("open_set_AUROC"),
            BASELINE_DOMINANCE_MARGIN["open_set_AUROC"],
            True,
        ),
        (
            m.get("FPR_at_95TPR"),
            baseline_best.get("FPR_at_95TPR"),
            BASELINE_DOMINANCE_MARGIN["FPR_at_95TPR"],
            False,
        ),
        (
            m.get("shot1_acc"),
            baseline_best.get("shot1_acc"),
            BASELINE_DOMINANCE_MARGIN["shot1_acc"],
            True,
        ),
        (
            m.get("shot3_acc"),
            baseline_best.get("shot3_acc"),
            BASELINE_DOMINANCE_MARGIN["shot3_acc"],
            True,
        ),
    ]

    for ours, baseline, margin, higher_is_better in checks:
        if ours is None or baseline is None:
            return False
        if higher_is_better:
            if float(ours) < float(baseline) + float(margin):
                return False
        else:
            if float(ours) > float(baseline) - float(margin):
                return False
    return True


def meets_industrial_targets(m: dict) -> bool:
    return (
        (m.get("setting_a_accuracy") is not None)
        and (m.get("setting_a_balanced_acc") is not None)
        and (m.get("open_set_AUROC") is not None)
        and (m.get("FPR_at_95TPR") is not None)
        and (m.get("shot1_acc") is not None)
        and (m.get("shot3_acc") is not None)
        and (m.get("known_unknown_gap") is not None)
        and (m["setting_a_accuracy"] >= INDUSTRIAL_TARGETS["setting_a_accuracy_min"])
        and (m["setting_a_balanced_acc"] >= INDUSTRIAL_TARGETS["setting_a_balanced_acc_min"])
        and (m["open_set_AUROC"] >= INDUSTRIAL_TARGETS["setting_b_open_set_AUROC_min"])
        and (m["FPR_at_95TPR"] <= INDUSTRIAL_TARGETS["setting_b_fpr95_max"])
        and (m["shot1_acc"] >= INDUSTRIAL_TARGETS["setting_c_1shot_acc_min"])
        and (m["shot3_acc"] >= INDUSTRIAL_TARGETS["setting_c_3shot_acc_min"])
        and (m["known_unknown_gap"] >= INDUSTRIAL_TARGETS["known_unknown_gap_min"])
    )


def meets_targets(m: dict) -> bool:
    return dominates_readme_baselines(m) and meets_industrial_targets(m)


def _baseline_gaps(m: dict) -> tuple[float, float, float, float, float, float, float]:
    baseline_best = m.get("baseline_best_core") or {}

    acc = m.get("setting_a_accuracy")
    bal = m.get("setting_a_balanced_acc")
    auroc = m.get("open_set_AUROC")
    fpr95 = m.get("FPR_at_95TPR")
    shot1 = m.get("shot1_acc")
    shot3 = m.get("shot3_acc")

    acc_gap = 1.0 if acc is None or baseline_best.get("setting_a_accuracy") is None else max(
        0.0,
        float(baseline_best["setting_a_accuracy"]) + BASELINE_DOMINANCE_MARGIN["setting_a_accuracy"] - float(acc),
    )
    bal_gap = 1.0 if bal is None or baseline_best.get("setting_a_balanced_acc") is None else max(
        0.0,
        float(baseline_best["setting_a_balanced_acc"]) + BASELINE_DOMINANCE_MARGIN["setting_a_balanced_acc"] - float(bal),
    )
    auroc_gap = 1.0 if auroc is None or baseline_best.get("open_set_AUROC") is None else max(
        0.0,
        float(baseline_best["open_set_AUROC"]) + BASELINE_DOMINANCE_MARGIN["open_set_AUROC"] - float(auroc),
    )
    fpr_gap = 1.0 if fpr95 is None or baseline_best.get("FPR_at_95TPR") is None else max(
        0.0,
        float(fpr95) - (float(baseline_best["FPR_at_95TPR"]) - BASELINE_DOMINANCE_MARGIN["FPR_at_95TPR"]),
    )
    shot1_gap = 1.0 if shot1 is None or baseline_best.get("shot1_acc") is None else max(
        0.0,
        float(baseline_best["shot1_acc"]) + BASELINE_DOMINANCE_MARGIN["shot1_acc"] - float(shot1),
    )
    shot3_gap = 1.0 if shot3 is None or baseline_best.get("shot3_acc") is None else max(
        0.0,
        float(baseline_best["shot3_acc"]) + BASELINE_DOMINANCE_MARGIN["shot3_acc"] - float(shot3),
    )

    total_gap = acc_gap + bal_gap + auroc_gap + fpr_gap + shot1_gap + shot3_gap
    return total_gap, acc_gap, bal_gap, auroc_gap, fpr_gap, shot1_gap, shot3_gap


def _industrial_gaps(m: dict) -> tuple[float, float, float, float, float, float, float, float]:
    acc = m.get("setting_a_accuracy")
    bal = m.get("setting_a_balanced_acc")
    auroc = m.get("open_set_AUROC")
    fpr95 = m.get("FPR_at_95TPR")
    shot1 = m.get("shot1_acc")
    shot3 = m.get("shot3_acc")
    score_gap = m.get("known_unknown_gap")

    acc_gap = 1.0 if acc is None else max(0.0, INDUSTRIAL_TARGETS["setting_a_accuracy_min"] - float(acc))
    bal_gap = 1.0 if bal is None else max(0.0, INDUSTRIAL_TARGETS["setting_a_balanced_acc_min"] - float(bal))
    auroc_gap = 1.0 if auroc is None else max(0.0, INDUSTRIAL_TARGETS["setting_b_open_set_AUROC_min"] - float(auroc))
    fpr_gap = 1.0 if fpr95 is None else max(0.0, float(fpr95) - INDUSTRIAL_TARGETS["setting_b_fpr95_max"])
    shot1_gap = 1.0 if shot1 is None else max(0.0, INDUSTRIAL_TARGETS["setting_c_1shot_acc_min"] - float(shot1))
    shot3_gap = 1.0 if shot3 is None else max(0.0, INDUSTRIAL_TARGETS["setting_c_3shot_acc_min"] - float(shot3))
    sep_gap = 1.0 if score_gap is None else max(0.0, INDUSTRIAL_TARGETS["known_unknown_gap_min"] - float(score_gap))

    total_gap = acc_gap + bal_gap + auroc_gap + fpr_gap + shot1_gap + shot3_gap + sep_gap
    return total_gap, acc_gap, bal_gap, auroc_gap, fpr_gap, shot1_gap, shot3_gap, sep_gap


def _target_gaps(m: dict) -> tuple[float, float, float, float]:
    auroc = m.get("open_set_AUROC")
    fpr95 = m.get("FPR_at_95TPR")
    shot3 = m.get("shot3_acc")

    auroc_gap = 1.0 if auroc is None else max(
        0.0,
        TARGETS["setting_b_open_set_AUROC_min"] - float(auroc),
    )
    fpr_gap = 1.0 if fpr95 is None else max(
        0.0,
        float(fpr95) - TARGETS["setting_b_fpr95_max"],
    )
    shot3_gap = 1.0 if shot3 is None else max(
        0.0,
        TARGETS["setting_c_3shot_acc_min"] - float(shot3),
    )
    total_gap = auroc_gap + fpr_gap + shot3_gap
    return total_gap, auroc_gap, fpr_gap, shot3_gap


def _guard_gaps(m: dict) -> tuple[float, float, float, float]:
    acc = m.get("setting_a_accuracy")
    bal = m.get("setting_a_balanced_acc")
    score_gap = m.get("known_unknown_gap")

    acc_gap = 1.0 if acc is None else max(
        0.0,
        SEARCH_GUARDS["setting_a_accuracy_min"] - float(acc),
    )
    bal_gap = 1.0 if bal is None else max(
        0.0,
        SEARCH_GUARDS["setting_a_balanced_acc_min"] - float(bal),
    )
    sep_gap = 1.0 if score_gap is None else max(
        0.0,
        SEARCH_GUARDS["known_unknown_gap_min"] - float(score_gap),
    )
    total_gap = acc_gap + bal_gap + sep_gap
    return total_gap, acc_gap, bal_gap, sep_gap


def _load_constraint_anchor_metrics() -> dict | None:
    global _CONSTRAINT_ANCHOR_METRICS_CACHE
    global _CONSTRAINT_ANCHOR_METRICS_LOADED

    if _CONSTRAINT_ANCHOR_METRICS_LOADED:
        return _CONSTRAINT_ANCHOR_METRICS_CACHE

    _CONSTRAINT_ANCHOR_METRICS_LOADED = True
    if not CONSTRAINT_ANCHOR_RUN:
        return None

    p = OUTPUTS_DIR / CONSTRAINT_ANCHOR_RUN / "evaluation_summary.json"
    if not p.exists():
        return None

    try:
        with open(p, "r", encoding="utf-8") as f:
            summary = json.load(f)
        _CONSTRAINT_ANCHOR_METRICS_CACHE = extract_metrics(summary)
    except Exception:
        _CONSTRAINT_ANCHOR_METRICS_CACHE = None
    return _CONSTRAINT_ANCHOR_METRICS_CACHE


def _constraint_deltas(m: dict) -> dict | None:
    anchor = _load_constraint_anchor_metrics()
    if anchor is None:
        return None

    a = m.get("setting_a_accuracy")
    fpr = m.get("FPR_at_95TPR")
    c1 = m.get("shot1_acc")
    a_ref = anchor.get("setting_a_accuracy")
    fpr_ref = anchor.get("FPR_at_95TPR")
    c1_ref = anchor.get("shot1_acc")
    if (
        a is None or fpr is None or c1 is None
        or a_ref is None or fpr_ref is None or c1_ref is None
    ):
        return None

    return {
        "dA": float(a) - float(a_ref),
        "dFPR": float(fpr) - float(fpr_ref),
        "dC1": float(c1) - float(c1_ref),
    }


def _constraint_violations(m: dict) -> dict:
    if not CONSTRAINT_MODE:
        return {
            "ok": True,
            "total": 0.0,
            "a_violation": 0.0,
            "fpr_violation": 0.0,
            "c1_violation": 0.0,
            "deltas": None,
        }

    deltas = _constraint_deltas(m)
    if deltas is None:
        # anchor 不可用时不阻塞运行，但会在启动日志中提示状态。
        return {
            "ok": True,
            "total": 0.0,
            "a_violation": 0.0,
            "fpr_violation": 0.0,
            "c1_violation": 0.0,
            "deltas": None,
        }

    a_violation = max(0.0, (-deltas["dA"]) - CONSTRAINT_MAX_A_DROP)
    fpr_violation = max(0.0, deltas["dFPR"] - CONSTRAINT_MAX_FPR_RISE)
    c1_violation = max(0.0, (-deltas["dC1"]) - CONSTRAINT_MAX_C1_DROP)
    total = a_violation + fpr_violation + c1_violation
    return {
        "ok": total <= 1e-12,
        "total": total,
        "a_violation": a_violation,
        "fpr_violation": fpr_violation,
        "c1_violation": c1_violation,
        "deltas": deltas,
    }


def _practical_record_status(m: dict) -> dict:
    if not PRACTICAL_RECORD_ENABLED:
        return {
            "record": False,
            "reason": "disabled",
            "main_metric": PRACTICAL_MAIN_METRIC,
            "main_gain": None,
            "baseline_checks": {},
            "guard_ok": False,
            "required_ok": False,
            "weighted_score": None,
            "weighted_ok": False,
            "soft_pass_ratio": None,
        }

    anchor = _load_constraint_anchor_metrics()
    baseline_best = m.get("baseline_best_core") or {}
    metric = PRACTICAL_MAIN_METRIC

    main_val = m.get(metric)
    anchor_val = None if anchor is None else anchor.get(metric)
    main_gain = None
    if main_val is not None and anchor_val is not None:
        main_gain = float(main_val) - float(anchor_val)

    checks: dict[str, bool] = {}
    core = [
        ("setting_a_accuracy", True),
        ("setting_a_balanced_acc", True),
        ("open_set_AUROC", True),
        ("FPR_at_95TPR", False),
        ("shot1_acc", True),
        ("shot3_acc", True),
    ]
    higher_map = {k: higher for k, higher in core}
    for key, higher_is_better in core:
        ours = m.get(key)
        base = baseline_best.get(key)
        if ours is None or base is None:
            checks[key] = False
            continue
        if higher_is_better:
            checks[key] = float(ours) >= float(base)
        else:
            checks[key] = float(ours) <= float(base) + PRACTICAL_FPR_HEADROOM

    required_keys = ["open_set_AUROC", "FPR_at_95TPR"]
    if PRACTICAL_MAIN_METRIC in checks and PRACTICAL_MAIN_METRIC not in required_keys:
        required_keys.append(PRACTICAL_MAIN_METRIC)
    required_ok = all(checks.get(k, False) for k in required_keys)

    soft_keys = [k for k in checks.keys() if k not in required_keys]
    soft_pass = sum(1 for k in soft_keys if checks.get(k, False))
    soft_pass_ratio = float(soft_pass) / float(len(soft_keys)) if soft_keys else 1.0

    w_main = max(float(PRACTICAL_MAIN_WEIGHT), 0.0)
    w_soft = max(float(PRACTICAL_SOFT_WEIGHT), 0.0)
    w_total = w_main + w_soft if (w_main + w_soft) > 0 else 1.0

    guard_total, *_ = _guard_gaps(m)
    guard_ok = guard_total <= 1e-12
    gain_ok = (main_gain is not None) and (main_gain >= PRACTICAL_MAIN_GAIN_MIN)
    gain_score = 1.0 if gain_ok else 0.0
    weighted_score = (w_main * gain_score + w_soft * soft_pass_ratio) / w_total
    weighted_ok = weighted_score >= PRACTICAL_WEIGHTED_MIN
    record = bool(gain_ok and required_ok and guard_ok and weighted_ok)

    reason_parts = []
    if not gain_ok:
        reason_parts.append("main_metric_not_improved")
    if not required_ok:
        reason_parts.append("below_required_baselines")
    if not guard_ok:
        reason_parts.append("below_search_guards")
    if not weighted_ok:
        reason_parts.append("weighted_score_too_low")

    return {
        "record": record,
        "reason": "ok" if record else ",".join(reason_parts),
        "main_metric": metric,
        "main_gain": main_gain,
        "baseline_checks": checks,
        "guard_ok": guard_ok,
        "required_ok": required_ok,
        "required_keys": required_keys,
        "soft_keys": soft_keys,
        "soft_pass_ratio": soft_pass_ratio,
        "weighted_score": weighted_score,
        "weighted_ok": weighted_ok,
    }


def diagnose_search_direction(m: dict) -> tuple[str, str]:
    cons = _constraint_violations(m)
    deltas = cons.get("deltas")
    if CONSTRAINT_MODE and (not cons["ok"]) and deltas is not None:
        if cons["c1_violation"] >= max(cons["fpr_violation"], cons["a_violation"], 1e-12):
            return (
                "fewshot_recovery",
                (
                    "路径A约束触发：1-shot 跌幅超限 "
                    f"(dC1={deltas['dC1']:.3f} < -{CONSTRAINT_MAX_C1_DROP:.3f})"
                ),
            )
        if cons["fpr_violation"] >= max(cons["a_violation"], 1e-12):
            return (
                "tighten_fpr",
                (
                    "路径A约束触发：FPR 回升超限 "
                    f"(dFPR={deltas['dFPR']:.3f} > +{CONSTRAINT_MAX_FPR_RISE:.3f})"
                ),
            )
        return (
            "recover_setting_a",
            (
                "路径A约束触发：Setting A 跌幅超限 "
                f"(dA={deltas['dA']:.3f} < -{CONSTRAINT_MAX_A_DROP:.3f})"
            ),
        )

    baseline_total_gap, b_acc_gap, b_bal_gap, b_auroc_gap, b_fpr_gap, b_shot1_gap, b_shot3_gap = _baseline_gaps(m)
    industrial_total_gap, i_acc_gap, i_bal_gap, i_auroc_gap, i_fpr_gap, i_shot1_gap, i_shot3_gap, i_sep_gap = _industrial_gaps(m)
    total_gap, auroc_gap, fpr_gap, shot3_gap = _target_gaps(m)
    _guard_total, acc_gap, bal_gap, sep_gap = _guard_gaps(m)

    if baseline_total_gap > 0:
        if b_acc_gap > 0 or b_bal_gap > 0:
            return (
                "recover_setting_a",
                (
                    f"阶段1未完成：Setting A 仍落后对比算法(acc_gap={b_acc_gap:.3f}, bal_gap={b_bal_gap:.3f})，"
                    "优先补齐已知类跨批次识别"
                ),
            )
        if b_shot1_gap > max(b_auroc_gap, b_fpr_gap, 0.02):
            return (
                "fewshot_recovery",
                f"阶段1未完成：1-shot 仍未全面超过对比算法(shot1_gap={b_shot1_gap:.3f})",
            )
        if b_fpr_gap > max(b_auroc_gap, 0.02):
            return (
                "tighten_fpr",
                f"阶段1未完成：FPR 仍未明显优于最强基线(fpr_gap={b_fpr_gap:.3f})",
            )
        return (
            "separate_open_set",
            f"阶段1未完成：开放集/3-shot 仍需拉开对比优势(auroc_gap={b_auroc_gap:.3f}, shot3_gap={b_shot3_gap:.3f})",
        )

    if acc_gap > 0 or bal_gap > 0:
        return (
            "recover_setting_a",
            (
                f"setting_a低于护栏(acc_gap={acc_gap:.3f}, bal_gap={bal_gap:.3f})，"
                "先恢复已知类识别稳定性"
            ),
        )
    if industrial_total_gap > 0:
        if i_acc_gap > 0 or i_bal_gap > 0:
            return (
                "recover_setting_a",
                (
                    f"阶段2未完成：工业化 Setting A 仍不足(acc_gap={i_acc_gap:.3f}, bal_gap={i_bal_gap:.3f})，"
                    "继续抬升跨批次识别"
                ),
            )
        if i_shot1_gap > max(i_fpr_gap, i_auroc_gap, 0.02):
            return (
                "fewshot_recovery",
                f"阶段2未完成：工业化 1-shot 仍不足(shot1_gap={i_shot1_gap:.3f})",
            )
        if i_sep_gap > 0.02 or (i_auroc_gap > 0 and i_sep_gap > 0):
            return (
                "separate_open_set",
                (
                    f"阶段2未完成：工业化开放集分离仍不足(auroc_gap={i_auroc_gap:.3f}, sep_gap={i_sep_gap:.3f})，"
                    "优先拉开已知/未知分数"
                ),
            )
        if i_fpr_gap > 0.05:
            return (
                "tighten_fpr",
                f"阶段2未完成：工业化 FPR 缺口最大(fpr_gap={i_fpr_gap:.3f})",
            )
        return (
            "raise_auroc",
            f"阶段2未完成：继续抬升工业化绝对指标(auroc_gap={i_auroc_gap:.3f}, shot3_gap={i_shot3_gap:.3f})",
        )

    if sep_gap > 0.02 or (auroc_gap > 0 and sep_gap > 0):
        return (
            "separate_open_set",
            (
                f"已知/未知分数间隔不足(sep_gap={sep_gap:.3f})，"
                "优先拉开开放集分数分离"
            ),
        )
    if fpr_gap > 0.12:
        return (
            "tighten_fpr",
            f"FPR缺口最大(fpr_gap={fpr_gap:.3f})，优先压低误报",
        )
    if shot3_gap > max(auroc_gap, 0.02):
        return (
            "fewshot_recovery",
            f"3-shot仍落后(shot3_gap={shot3_gap:.3f})，优先恢复few-shot表现",
        )
    if total_gap > 0:
        return (
            "raise_auroc",
            f"主要剩余缺口来自AUROC(auroc_gap={auroc_gap:.3f})，继续提升排序能力",
        )
    return ("balanced_polish", "核心指标已接近目标，做小步均衡微调")


def _rank_metrics(m: dict) -> tuple:
    cons = _constraint_violations(m)
    baseline_total_gap, b_acc_gap, b_bal_gap, b_auroc_gap, b_fpr_gap, b_shot1_gap, b_shot3_gap = _baseline_gaps(m)
    industrial_total_gap, i_acc_gap, i_bal_gap, i_auroc_gap, i_fpr_gap, i_shot1_gap, i_shot3_gap, i_sep_gap = _industrial_gaps(m)
    total_gap, auroc_gap, fpr_gap, shot3_gap = _target_gaps(m)
    guard_total_gap, acc_gap, bal_gap, sep_gap = _guard_gaps(m)

    acc = m.get("setting_a_accuracy")
    bal = m.get("setting_a_balanced_acc")
    auroc = m.get("open_set_AUROC")
    fpr95 = m.get("FPR_at_95TPR")
    shot1 = m.get("shot1_acc")
    shot3 = m.get("shot3_acc")
    score_gap = m.get("known_unknown_gap")
    acc_v = -1.0 if acc is None else float(acc)
    bal_v = -1.0 if bal is None else float(bal)
    auroc_v = -1.0 if auroc is None else float(auroc)
    fpr_v = 1e9 if fpr95 is None else float(fpr95)
    shot1_v = -1.0 if shot1 is None else float(shot1)
    shot3_v = -1.0 if shot3 is None else float(shot3)
    score_gap_v = -1.0 if score_gap is None else float(score_gap)

    baseline_done = 1 if baseline_total_gap == 0 else 0
    industrial_done = 1 if industrial_total_gap == 0 else 0
    constraint_ok = 1 if cons["ok"] else 0

    return (
        # Path A: 先约束可行性，再比较 AUROC / A_bal
        constraint_ok,
        -cons["total"],
        -cons["c1_violation"],
        -cons["fpr_violation"],
        -cons["a_violation"],
        auroc_v,
        bal_v,
        acc_v,
        -fpr_v,
        shot1_v,
        shot3_v,
        score_gap_v,
        baseline_done,
        industrial_done,
        -baseline_total_gap,
        -b_acc_gap,
        -b_bal_gap,
        -b_fpr_gap,
        -b_shot1_gap,
        -b_auroc_gap,
        -b_shot3_gap,
        -industrial_total_gap,
        -i_acc_gap,
        -i_bal_gap,
        -i_fpr_gap,
        -i_shot1_gap,
        -i_auroc_gap,
        -i_shot3_gap,
        -i_sep_gap,
        -guard_total_gap,
        -acc_gap,
        -bal_gap,
        -sep_gap,
        -total_gap,
        -fpr_gap,
        -shot3_gap,
        -auroc_gap,
        acc_v,
        bal_v,
        auroc_v,
        -fpr_v,
        shot1_v,
        shot3_v,
        score_gap_v,
    )


def derive_current_best_run() -> tuple[str | None, dict | None]:
    ranked = []
    for summary_path in OUTPUTS_DIR.glob("iter*/evaluation_summary.json"):
        run_name = summary_path.parent.name
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        ranked.append((run_name, extract_metrics(summary)))

    if not ranked:
        return None, None
    best_run, best_metrics = max(ranked, key=lambda x: _rank_metrics(x[1]))
    return best_run, best_metrics


def derive_top_runs(top_n: int = KEEP_TOP_N) -> list[str]:
    ranked = []
    for summary_path in OUTPUTS_DIR.glob("iter*/evaluation_summary.json"):
        run_name = summary_path.parent.name
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        ranked.append((run_name, extract_metrics(summary)))

    ranked.sort(key=lambda x: _rank_metrics(x[1]), reverse=True)
    return [name for name, _ in ranked[: max(int(top_n), 1)]]


def cleanup_non_best_artifacts(keep_runs: list[str], active_runs: set[str] | None = None) -> list[str]:
    if KEEP_ALL_RUN_DIRS:
        return []

    keep_set = set(keep_runs)
    active_set = active_runs or set()
    if not keep_set:
        return []

    pruned_runs = []

    for run_dir in OUTPUTS_DIR.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("iter"):
            continue
        run_name = run_dir.name
        if run_name in keep_set or run_name in active_set:
            continue
        shutil.rmtree(run_dir, ignore_errors=True)
        pruned_runs.append(run_name)

    return pruned_runs


def read_val_acc_at_epoch(run_name: str, epoch: int = 10) -> float | None:
    log_path = OUTPUTS_DIR / run_name / "run.log"
    if not log_path.exists():
        return None

    p_epoch = re.compile(r"Epoch\s+(\d+)/")
    p_val = re.compile(r"->\s*val_acc=([0-9.]+)")
    p_inline = re.compile(r"Epoch\s+(\d+)/\d+.*val_acc=([0-9.]+)")

    current_epoch = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m_inline = p_inline.search(line)
            if m_inline and int(m_inline.group(1)) == epoch:
                return float(m_inline.group(2))

            m_epoch = p_epoch.search(line)
            if m_epoch:
                current_epoch = int(m_epoch.group(1))
            m_val = p_val.search(line)
            if m_val and current_epoch == epoch:
                return float(m_val.group(1))
    return None


def derive_warmup_guard_reference(epoch: int = 10) -> tuple[float | None, str | None]:
    best_run, _ = derive_current_best_run()
    if not best_run:
        return None, None

    ref = read_val_acc_at_epoch(best_run, epoch=epoch)
    if ref is not None:
        return ref, best_run

    refs = []
    for summary_path in OUTPUTS_DIR.glob("iter*/evaluation_summary.json"):
        run_name = summary_path.parent.name
        v = read_val_acc_at_epoch(run_name, epoch=epoch)
        if v is not None:
            refs.append((run_name, v))
    if not refs:
        return None, None
    run_name, v = max(refs, key=lambda x: x[1])
    return v, run_name


def next_auto_index() -> int:
    max_idx = 0
    p = re.compile(r"iter_auto(\d+)_")
    for d in OUTPUTS_DIR.glob("iter_auto*"):
        m = p.match(d.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def load_run_config(run_name: str) -> dict | None:
    cfg_path = OUTPUTS_DIR / run_name / "run_config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("config")


def parse_run_hints_from_name(run_name: str) -> dict | None:
    m = re.search(r"_bs(\d+)_lr(\d+)_a(\d+)_p(\d+)", run_name)
    if not m:
        return None
    return {
        "batch_size": int(m.group(1)),
        "lr": float(int(m.group(2)) / 1e6),
        "lambda_adv": float(int(m.group(3)) / 100.0),
        "lambda_proto": float(int(m.group(4)) / 100.0),
    }


def _normalize_encoder_channels(value) -> tuple:
    if value is None:
        return tuple(FALLBACK_BASE["encoder_channels"])
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            return tuple(FALLBACK_BASE["encoder_channels"])
        return tuple(int(v) for v in parts)
    return tuple(FALLBACK_BASE["encoder_channels"])


def normalize_base_config(cfg: dict, run_name: str) -> dict:
    base = {
        "epochs": int(cfg.get("epochs", FALLBACK_BASE["epochs"])),
        "batch_size": int(cfg.get("batch_size", FALLBACK_BASE["batch_size"])),
        "lr": float(cfg.get("lr", FALLBACK_BASE["lr"])),
        "lambda_adv": float(cfg.get("lambda_adv", FALLBACK_BASE["lambda_adv"])),
        "lambda_proto": float(cfg.get("lambda_proto", FALLBACK_BASE["lambda_proto"])),
        "lambda_recon": float(cfg.get("lambda_recon", FALLBACK_BASE["lambda_recon"])),
        "supcon_temperature": float(cfg.get("supcon_temperature", FALLBACK_BASE["supcon_temperature"])),
        "accept_percentile": float(cfg.get("accept_percentile", FALLBACK_BASE["accept_percentile"])),
        "reject_threshold_factor": float(cfg.get("reject_threshold_factor", FALLBACK_BASE["reject_threshold_factor"])),
        "raw_open_score_blend": float(cfg.get("raw_open_score_blend", FALLBACK_BASE["raw_open_score_blend"])),
        "raw_distance_percentile": float(cfg.get("raw_distance_percentile", FALLBACK_BASE["raw_distance_percentile"])),
        "eval_interval": int(cfg.get("eval_interval", FALLBACK_BASE["eval_interval"])),
        "early_stop_patience": int(cfg.get("early_stop_patience", FALLBACK_BASE["early_stop_patience"])),
        "min_epochs_before_early_stop": int(cfg.get("min_epochs_before_early_stop", FALLBACK_BASE["min_epochs_before_early_stop"])),
        "min_epoch_ratio_before_early_stop": float(cfg.get("min_epoch_ratio_before_early_stop", FALLBACK_BASE["min_epoch_ratio_before_early_stop"])),
        "early_stop_min_lr_ratio": float(cfg.get("early_stop_min_lr_ratio", FALLBACK_BASE["early_stop_min_lr_ratio"])),
        "early_stop_min_delta": float(cfg.get("early_stop_min_delta", FALLBACK_BASE["early_stop_min_delta"])),
        "encoder_channels": _normalize_encoder_channels(cfg.get("encoder_channels")),
        "blocks_per_stage": int(cfg.get("blocks_per_stage", FALLBACK_BASE["blocks_per_stage"])),
        "num_axial_heads": int(cfg.get("num_axial_heads", FALLBACK_BASE["num_axial_heads"])),
        "dropout": float(cfg.get("dropout", FALLBACK_BASE["dropout"])),
        "main_backbone": str(cfg.get("main_backbone", FALLBACK_BASE["main_backbone"])),
        "main_backbone_model": str(cfg.get("main_backbone_model", FALLBACK_BASE["main_backbone_model"])),
        "main_feature_layers": str(cfg.get("main_feature_layers", FALLBACK_BASE["main_feature_layers"])),
        "main_feature_fuse": str(cfg.get("main_feature_fuse", FALLBACK_BASE["main_feature_fuse"])),
        "transformer_patch_size": int(cfg.get("transformer_patch_size", FALLBACK_BASE["transformer_patch_size"])),
        "transformer_embed_dim": int(cfg.get("transformer_embed_dim", FALLBACK_BASE["transformer_embed_dim"])),
        "transformer_depth": int(cfg.get("transformer_depth", FALLBACK_BASE["transformer_depth"])),
        "transformer_num_heads": int(cfg.get("transformer_num_heads", FALLBACK_BASE["transformer_num_heads"])),
        "transformer_mlp_ratio": float(cfg.get("transformer_mlp_ratio", FALLBACK_BASE["transformer_mlp_ratio"])),
    }

    # 同名 run 可能被 skip_train 评估覆盖 run_config，优先用名称里的超参标签纠偏。
    hints = parse_run_hints_from_name(run_name)
    if hints:
        base["batch_size"] = hints["batch_size"]
        base["lr"] = hints["lr"]
        base["lambda_adv"] = hints["lambda_adv"]
        base["lambda_proto"] = hints["lambda_proto"]

    return base


def _load_grid_shape(prepared_dir: str) -> tuple[int | None, int | None]:
    grid_path = Path(prepared_dir) / "grid_info.json"
    if not grid_path.exists():
        return None, None
    try:
        obj = json.loads(grid_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    rt_bins = obj.get("rt_bins")
    mz_bins = obj.get("mz_bins")
    if rt_bins is None or mz_bins is None:
        return None, None
    return int(rt_bins), int(mz_bins)


def _is_valid_prepared_dir(prepared_dir: str) -> bool:
    p = Path(prepared_dir)
    return (p / "metadata.csv").exists() and (p / "split.json").exists()


def build_input_ablation_queue(start_idx: int, base_cfg: dict,
                               best_metrics: dict | None) -> tuple[list[dict[str, Any]], int, list[str]]:
    """构建一次性输入格式横向对照队列。

    三种输入格式中的“当前best格式”由 existing best run 表示；
    此处插入两条新迭代：
      1) bins 直入
      2) raw->bins->PCA（复用 prepared_data_pca171）
    """
    if not INSERT_INPUT_FORMAT_ABLATIONS:
        return [], start_idx, ["AUTO3 input ablation insertion disabled by env"]

    info_lines: list[str] = []
    queue: list[dict[str, Any]] = []
    idx = int(start_idx)

    pre = (best_metrics or {}).get("pretrained_feature_extractor") or {}
    pre_model = str(pre.get("model_path") or "")
    pre_arch = str(pre.get("arch") or "auto")

    common = {
        "search_directive": "input_format_ablation",
        "epochs": int(base_cfg["epochs"]),
        "batch_size": int(base_cfg["batch_size"]),
        "lr": float(base_cfg["lr"]),
        "lambda_adv": float(base_cfg["lambda_adv"]),
        "lambda_proto": float(base_cfg["lambda_proto"]),
        "lambda_recon": float(base_cfg["lambda_recon"]),
        "supcon_temperature": float(base_cfg["supcon_temperature"]),
        "accept_percentile": float(base_cfg["accept_percentile"]),
        "reject_threshold_factor": float(base_cfg["reject_threshold_factor"]),
        "raw_open_score_blend": float(base_cfg["raw_open_score_blend"]),
        "raw_distance_percentile": float(base_cfg["raw_distance_percentile"]),
        "eval_interval": int(base_cfg["eval_interval"]),
        "early_stop_patience": int(base_cfg["early_stop_patience"]),
        "min_epochs_before_early_stop": int(base_cfg["min_epochs_before_early_stop"]),
        "min_epoch_ratio_before_early_stop": float(base_cfg["min_epoch_ratio_before_early_stop"]),
        "early_stop_min_lr_ratio": float(base_cfg["early_stop_min_lr_ratio"]),
        "early_stop_min_delta": float(base_cfg["early_stop_min_delta"]),
        "encoder_channels": tuple(base_cfg["encoder_channels"]),
        "blocks_per_stage": int(base_cfg["blocks_per_stage"]),
        "num_axial_heads": int(base_cfg["num_axial_heads"]),
        "dropout": float(base_cfg["dropout"]),
        "pretrained_feature_model": pre_model,
        "pretrained_feature_arch": pre_arch,
        "main_backbone": str(base_cfg.get("main_backbone", "gcms")),
        "main_backbone_model": str(base_cfg.get("main_backbone_model", "")),
        "main_feature_layers": str(base_cfg.get("main_feature_layers", "layer4")),
        "main_feature_fuse": str(base_cfg.get("main_feature_fuse", "concat")),
        "transformer_patch_size": int(base_cfg.get("transformer_patch_size", 16)),
        "transformer_embed_dim": int(base_cfg.get("transformer_embed_dim", 256)),
        "transformer_depth": int(base_cfg.get("transformer_depth", 6)),
        "transformer_num_heads": int(base_cfg.get("transformer_num_heads", 8)),
        "transformer_mlp_ratio": float(base_cfg.get("transformer_mlp_ratio", 4.0)),
    }

    variants = [
        {
            "suffix": "fmtbins",
            "prepared_dir": INPUT_ABLATION_BINS_PREPARED_DIR,
            "input_raw_pca_enabled": False,
        },
        {
            "suffix": "fmtrawbinspca",
            "prepared_dir": INPUT_ABLATION_PCA_PREPARED_DIR,
            "input_raw_pca_enabled": True,
        },
    ]

    for var in variants:
        prepared_dir = str(var["prepared_dir"])
        if not _is_valid_prepared_dir(prepared_dir):
            info_lines.append(
                f"skip {var['suffix']}: invalid prepared_dir (need metadata.csv + split.json): {prepared_dir}"
            )
            continue

        rt_bins, mz_bins = _load_grid_shape(prepared_dir)
        name = (
            f"iter_auto{idx:03d}_bs{common['batch_size']}_lr{int(round(common['lr']*1e6))}"
            f"_a{int(round(common['lambda_adv']*100))}"
            f"_p{int(round(common['lambda_proto']*100))}_{var['suffix']}"
        )

        cfg = {
            **common,
            "name": name,
            "prepared_dir": prepared_dir,
            "input_raw_pca_enabled": bool(var["input_raw_pca_enabled"]),
            "rt_bins": rt_bins,
            "mz_bins": mz_bins,
        }

        if cfg["input_raw_pca_enabled"] and mz_bins is not None:
            cfg["input_raw_pca_components"] = int(mz_bins)

        queue.append(cfg)
        info_lines.append(
            f"queued {name}: prepared_dir={prepared_dir}, input_raw_pca_enabled={cfg['input_raw_pca_enabled']}, rt_bins={rt_bins}, mz_bins={mz_bins}"
        )
        idx += 1

    return queue, idx, info_lines


def build_main_backbone_ablation_queue(start_idx: int, base_cfg: dict,
                                       pretrained_cycle: list[dict] | None) -> tuple[list[dict[str, Any]], int, list[str]]:
    """构建主模型骨干插队队列：更换骨干 + resnet50 多层融合。"""
    if not INSERT_MAIN_BACKBONE_ABLATIONS:
        return [], start_idx, ["AUTO3 main backbone ablation insertion disabled by env"]

    info_lines: list[str] = []
    queue: list[dict[str, Any]] = []
    idx = int(start_idx)

    cycle_map = {str(item.get("arch")): str(item.get("path")) for item in (pretrained_cycle or [])}
    if not cycle_map:
        return [], start_idx, ["skip backbone ablation: pretrained_cycle empty"]

    pre_model = cycle_map.get(
        str(base_cfg.get("main_backbone") or "").lower(),
        cycle_map.get("resnet50", ""),
    )
    pre_arch = "resnet50" if pre_model else "auto"

    common = {
        "search_directive": "main_backbone_ablation",
        "epochs": int(base_cfg["epochs"]),
        "batch_size": int(base_cfg["batch_size"]),
        "lr": float(base_cfg["lr"]),
        "lambda_adv": float(base_cfg["lambda_adv"]),
        "lambda_proto": float(base_cfg["lambda_proto"]),
        "lambda_recon": float(base_cfg["lambda_recon"]),
        "supcon_temperature": float(base_cfg["supcon_temperature"]),
        "accept_percentile": float(base_cfg["accept_percentile"]),
        "reject_threshold_factor": float(base_cfg["reject_threshold_factor"]),
        "raw_open_score_blend": float(base_cfg["raw_open_score_blend"]),
        "raw_distance_percentile": float(base_cfg["raw_distance_percentile"]),
        "eval_interval": int(base_cfg["eval_interval"]),
        "early_stop_patience": int(base_cfg["early_stop_patience"]),
        "min_epochs_before_early_stop": int(base_cfg["min_epochs_before_early_stop"]),
        "min_epoch_ratio_before_early_stop": float(base_cfg["min_epoch_ratio_before_early_stop"]),
        "early_stop_min_lr_ratio": float(base_cfg["early_stop_min_lr_ratio"]),
        "early_stop_min_delta": float(base_cfg["early_stop_min_delta"]),
        "encoder_channels": tuple(base_cfg["encoder_channels"]),
        "blocks_per_stage": int(base_cfg["blocks_per_stage"]),
        "num_axial_heads": int(base_cfg["num_axial_heads"]),
        "dropout": float(base_cfg["dropout"]),
        "pretrained_feature_model": pre_model,
        "pretrained_feature_arch": pre_arch,
        "main_backbone": str(base_cfg.get("main_backbone", "gcms")),
        "main_backbone_model": str(base_cfg.get("main_backbone_model", "")),
        "main_feature_layers": str(base_cfg.get("main_feature_layers", "layer4")),
        "main_feature_fuse": str(base_cfg.get("main_feature_fuse", "concat")),
        "transformer_patch_size": int(base_cfg.get("transformer_patch_size", 16)),
        "transformer_embed_dim": int(base_cfg.get("transformer_embed_dim", 256)),
        "transformer_depth": int(base_cfg.get("transformer_depth", 6)),
        "transformer_num_heads": int(base_cfg.get("transformer_num_heads", 8)),
        "transformer_mlp_ratio": float(base_cfg.get("transformer_mlp_ratio", 4.0)),
    }

    variants = [
        {
            "suffix": "mb_r18_l4",
            "main_backbone": "resnet18",
            "main_backbone_model": cycle_map.get("resnet18", ""),
            "main_feature_layers": "layer4",
            "main_feature_fuse": "last",
            "pretrained_feature_arch": "resnet18",
            "pretrained_feature_model": cycle_map.get("resnet18", ""),
        },
        {
            "suffix": "mb_wr50_l4",
            "main_backbone": "wide_resnet50_2",
            "main_backbone_model": cycle_map.get("wide_resnet50_2", ""),
            "main_feature_layers": "layer4",
            "main_feature_fuse": "last",
            "pretrained_feature_arch": "wide_resnet50_2",
            "pretrained_feature_model": cycle_map.get("wide_resnet50_2", ""),
        },
        {
            "suffix": "mb_r50_l34c",
            "main_backbone": "resnet50",
            "main_backbone_model": cycle_map.get("resnet50", ""),
            "main_feature_layers": "layer3,layer4",
            "main_feature_fuse": "concat",
            "pretrained_feature_arch": "resnet50",
            "pretrained_feature_model": cycle_map.get("resnet50", ""),
        },
        {
            "suffix": "mb_r50_l234c",
            "main_backbone": "resnet50",
            "main_backbone_model": cycle_map.get("resnet50", ""),
            "main_feature_layers": "layer2,layer3,layer4",
            "main_feature_fuse": "concat",
            "pretrained_feature_arch": "resnet50",
            "pretrained_feature_model": cycle_map.get("resnet50", ""),
        },
        {
            "suffix": "mb_tf_s_l4",
            "main_backbone": "transformer",
            "main_backbone_model": "",
            "main_feature_layers": "layer4",
            "main_feature_fuse": "last",
            "pretrained_feature_arch": pre_arch,
            "pretrained_feature_model": pre_model,
            "transformer_patch_size": 16,
            "transformer_embed_dim": 192,
            "transformer_depth": 6,
            "transformer_num_heads": 6,
            "transformer_mlp_ratio": 4.0,
        },
        {
            "suffix": "mb_tf_m_l34c",
            "main_backbone": "transformer",
            "main_backbone_model": "",
            "main_feature_layers": "layer3,layer4",
            "main_feature_fuse": "concat",
            "pretrained_feature_arch": pre_arch,
            "pretrained_feature_model": pre_model,
            "transformer_patch_size": 16,
            "transformer_embed_dim": 256,
            "transformer_depth": 8,
            "transformer_num_heads": 8,
            "transformer_mlp_ratio": 4.0,
        },
        {
            "suffix": "mb_tf_l_l234c",
            "main_backbone": "transformer",
            "main_backbone_model": "",
            "main_feature_layers": "layer2,layer3,layer4",
            "main_feature_fuse": "concat",
            "pretrained_feature_arch": pre_arch,
            "pretrained_feature_model": pre_model,
            "transformer_patch_size": 14,
            "transformer_embed_dim": 320,
            "transformer_depth": 10,
            "transformer_num_heads": 10,
            "transformer_mlp_ratio": 4.0,
            "batch_size": 8,
        },
    ]

    for var in variants:
        if var["main_backbone"] in {"resnet18", "resnet50", "wide_resnet50_2"} and not var["main_backbone_model"]:
            info_lines.append(f"skip {var['suffix']}: missing local weight for {var['main_backbone']}")
            continue

        bs = int(var.get("batch_size", common["batch_size"]))

        name = (
            f"iter_auto{idx:03d}_bs{bs}_lr{int(round(common['lr']*1e6))}"
            f"_a{int(round(common['lambda_adv']*100))}"
            f"_p{int(round(common['lambda_proto']*100))}_{var['suffix']}"
        )

        cfg = {
            **common,
            "name": name,
            "batch_size": bs,
            "main_backbone": str(var["main_backbone"]),
            "main_backbone_model": str(var["main_backbone_model"]),
            "main_feature_layers": str(var["main_feature_layers"]),
            "main_feature_fuse": str(var["main_feature_fuse"]),
            "pretrained_feature_arch": str(var["pretrained_feature_arch"]),
            "pretrained_feature_model": str(var["pretrained_feature_model"]),
            "transformer_patch_size": int(var.get("transformer_patch_size", common["transformer_patch_size"])),
            "transformer_embed_dim": int(var.get("transformer_embed_dim", common["transformer_embed_dim"])),
            "transformer_depth": int(var.get("transformer_depth", common["transformer_depth"])),
            "transformer_num_heads": int(var.get("transformer_num_heads", common["transformer_num_heads"])),
            "transformer_mlp_ratio": float(var.get("transformer_mlp_ratio", common["transformer_mlp_ratio"])),
        }
        queue.append(cfg)
        info_lines.append(
            f"queued {name}: backbone={cfg['main_backbone']}, batch_size={cfg['batch_size']}, layers={cfg['main_feature_layers']}, fuse={cfg['main_feature_fuse']}, model={cfg['main_backbone_model']}, tf_patch={cfg['transformer_patch_size']}, tf_dim={cfg['transformer_embed_dim']}, tf_depth={cfg['transformer_depth']}, tf_heads={cfg['transformer_num_heads']}"
        )
        idx += 1

    return queue, idx, info_lines


def build_structured_local_queue(start_idx: int, base_cfg: dict,
                                 best_metrics: dict | None,
                                 pretrained_cycle: list[dict] | None) -> tuple[list[dict[str, Any]], int, list[str]]:
    """围绕当前best做结构化局部搜索: 开集阈值 + 损失权重 + 小幅加深。"""
    if not INSERT_STRUCTURED_LOCAL_QUEUE:
        return [], start_idx, ["AUTO3 structured local queue disabled by env"]

    info_lines: list[str] = []
    queue: list[dict[str, Any]] = []
    idx = int(start_idx)

    pre = (best_metrics or {}).get("pretrained_feature_extractor") or {}
    pre_model = str(pre.get("model_path") or "")
    pre_arch = str(pre.get("arch") or "auto")
    if not pre_model and pretrained_cycle:
        item = pretrained_cycle[0]
        pre_model = item["path"]
        pre_arch = item["arch"]

    base_lr = float(base_cfg["lr"])
    base_adv = float(base_cfg["lambda_adv"])
    base_proto = float(base_cfg["lambda_proto"])
    base_temp = float(base_cfg["supcon_temperature"])
    base_accp = float(base_cfg["accept_percentile"])
    base_reject = float(base_cfg["reject_threshold_factor"])
    base_blend = float(base_cfg["raw_open_score_blend"])
    base_dist = float(base_cfg["raw_distance_percentile"])
    base_dropout = float(base_cfg["dropout"])

    common = {
        "search_directive": "structured_local_287",
        "epochs": int(base_cfg["epochs"]),
        "batch_size": int(base_cfg["batch_size"]),
        "lr": base_lr,
        "lambda_adv": base_adv,
        "lambda_proto": base_proto,
        "lambda_recon": float(base_cfg["lambda_recon"]),
        "supcon_temperature": base_temp,
        "accept_percentile": base_accp,
        "reject_threshold_factor": base_reject,
        "raw_open_score_blend": base_blend,
        "raw_distance_percentile": base_dist,
        "eval_interval": int(base_cfg["eval_interval"]),
        "early_stop_patience": int(base_cfg["early_stop_patience"]),
        "min_epochs_before_early_stop": int(base_cfg["min_epochs_before_early_stop"]),
        "min_epoch_ratio_before_early_stop": float(base_cfg["min_epoch_ratio_before_early_stop"]),
        "early_stop_min_lr_ratio": float(base_cfg["early_stop_min_lr_ratio"]),
        "early_stop_min_delta": float(base_cfg["early_stop_min_delta"]),
        "encoder_channels": tuple(base_cfg["encoder_channels"]),
        "blocks_per_stage": int(base_cfg["blocks_per_stage"]),
        "num_axial_heads": int(base_cfg["num_axial_heads"]),
        "dropout": base_dropout,
        "pretrained_feature_model": pre_model,
        "pretrained_feature_arch": pre_arch,
        "main_backbone": str(base_cfg.get("main_backbone", "gcms")),
        "main_backbone_model": str(base_cfg.get("main_backbone_model", "")),
        "main_feature_layers": str(base_cfg.get("main_feature_layers", "layer4")),
        "main_feature_fuse": str(base_cfg.get("main_feature_fuse", "concat")),
        "transformer_patch_size": int(base_cfg.get("transformer_patch_size", 16)),
        "transformer_embed_dim": int(base_cfg.get("transformer_embed_dim", 256)),
        "transformer_depth": int(base_cfg.get("transformer_depth", 6)),
        "transformer_num_heads": int(base_cfg.get("transformer_num_heads", 8)),
        "transformer_mlp_ratio": float(base_cfg.get("transformer_mlp_ratio", 4.0)),
    }

    variants = [
        {
            "suffix": "s287_fpr1",
            "lr_factor": 0.90,
            "adv_shift": -0.02,
            "proto_shift": -0.02,
            "accp_shift": -1.0,
            "reject": 1.9,
            "blend": 0.85,
            "dist": 97.0,
        },
        {
            "suffix": "s287_fpr2",
            "lr_factor": 0.94,
            "adv_shift": -0.01,
            "proto_shift": 0.00,
            "accp_shift": 0.0,
            "reject": 1.8,
            "blend": 0.85,
            "dist": 97.0,
        },
        {
            "suffix": "s287_bal1",
            "lr_factor": 1.00,
            "adv_shift": 0.00,
            "proto_shift": 0.02,
            "accp_shift": 1.0,
            "reject": 2.0,
            "blend": 1.0,
            "dist": 95.0,
        },
        {
            "suffix": "s287_bal2",
            "lr_factor": 1.03,
            "adv_shift": 0.01,
            "proto_shift": 0.02,
            "accp_shift": 1.0,
            "reject": 2.1,
            "blend": 1.0,
            "dist": 95.0,
        },
        {
            "suffix": "s287_few1",
            "lr_factor": 0.96,
            "adv_shift": 0.02,
            "proto_shift": 0.02,
            "accp_shift": 1.0,
            "reject": 2.0,
            "blend": 1.0,
            "dist": 95.0,
        },
        {
            "suffix": "s287_few2",
            "lr_factor": 0.92,
            "adv_shift": 0.00,
            "proto_shift": 0.04,
            "accp_shift": 1.0,
            "reject": 2.1,
            "blend": 1.0,
            "dist": 95.0,
        },
        {
            "suffix": "s287_d3a",
            "lr_factor": 0.90,
            "adv_shift": 0.00,
            "proto_shift": 0.02,
            "accp_shift": 1.0,
            "reject": 2.0,
            "blend": 1.0,
            "dist": 95.0,
            "blocks": 3,
            "channels": (32, 64, 160, 320),
            "dropout": base_dropout + 0.05,
        },
        {
            "suffix": "s287_d3b",
            "lr_factor": 0.94,
            "adv_shift": -0.01,
            "proto_shift": 0.02,
            "accp_shift": 0.0,
            "reject": 1.9,
            "blend": 0.85,
            "dist": 97.0,
            "blocks": 3,
            "channels": (32, 64, 128, 320),
            "dropout": base_dropout + 0.05,
        },
    ]

    for var in variants:
        lr = float(clip(base_lr * float(var["lr_factor"]), 1.5e-4, 2.6e-4))
        adv = float(clip(base_adv + float(var["adv_shift"]), 0.07, 0.18))
        proto = float(clip(base_proto + float(var["proto_shift"]), 0.80, 0.95))
        accp = float(clip(base_accp + float(var["accp_shift"]), 94.0, 98.5))
        reject = float(clip(float(var["reject"]), 1.7, 2.2))
        blend = float(clip(float(var["blend"]), 0.6, 1.0))
        dist = float(clip(float(var["dist"]), 90.0, 99.0))

        channels = tuple(var.get("channels", common["encoder_channels"]))
        blocks = int(var.get("blocks", common["blocks_per_stage"]))
        dropout = float(clip(float(var.get("dropout", common["dropout"])), 0.2, 0.5))

        name = (
            f"iter_auto{idx:03d}_bs{common['batch_size']}_lr{int(round(lr*1e6))}"
            f"_a{int(round(adv*100))}_p{int(round(proto*100))}_{var['suffix']}"
        )

        cfg = {
            **common,
            "name": name,
            "lr": lr,
            "lambda_adv": adv,
            "lambda_proto": proto,
            "accept_percentile": accp,
            "reject_threshold_factor": reject,
            "raw_open_score_blend": blend,
            "raw_distance_percentile": dist,
            "encoder_channels": channels,
            "blocks_per_stage": blocks,
            "dropout": dropout,
            "search_directive": "structured_local_287_deepen" if blocks > common["blocks_per_stage"] else "structured_local_287",
        }

        queue.append(cfg)
        info_lines.append(
            f"queued {name}: lr={lr}, a={adv}, p={proto}, accp={accp}, reject={reject}, blend={blend}, dist={dist}, blocks={blocks}, channels={channels}, dropout={dropout}"
        )
        idx += 1

    return queue, idx, info_lines


def derive_base_config() -> tuple[dict, str]:
    best_run, _ = derive_current_best_run()
    if best_run:
        cfg = load_run_config(best_run)
        if cfg:
            return normalize_base_config(cfg, best_run), best_run

    cfg = load_run_config("iter2_tuned_e40_rerun")
    if cfg:
        return normalize_base_config(cfg, "iter2_tuned_e40_rerun"), "iter2_tuned_e40_rerun"

    return FALLBACK_BASE.copy(), "fallback"


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def generate_candidate(idx: int, base_cfg: dict,
                       pretrained_cycle: list[dict] | None = None,
                       directive: str = "balanced_polish") -> dict:
    random.seed(20260509 + idx)

    presets = {
        "recover_setting_a": {
            "lr_factors": [0.96, 1.00, 1.04],
            "adv_shifts": [0.02, 0.04, 0.06],
            "proto_shifts": [0.0, 0.04, 0.08],
            "temp_shifts": [0.0, 0.005],
            "accp_shifts": [0.0, 1.0, 2.0],
            "batch_choices": [16, 16, 16, 8],
            "adv_range": (0.09, 0.18),
            "proto_range": (0.88, 0.96),
            "temp_range": (0.070, 0.080),
            "accp_range": (94.0, 97.0),
        },
        "separate_open_set": {
            "lr_factors": [0.90, 0.96, 1.00, 1.04],
            "adv_shifts": [-0.02, -0.01, 0.0],
            "proto_shifts": [-0.05, 0.0, 0.05],
            "temp_shifts": [-0.01, -0.005, 0.0],
            "accp_shifts": [-3.0, -2.0, -1.0, 0.0],
            "batch_choices": [16, 16, 8],
            "adv_range": (0.07, 0.12),
            "proto_range": (0.75, 0.90),
            "temp_range": (0.055, 0.070),
            "accp_range": (91.0, 95.0),
        },
        "tighten_fpr": {
            "lr_factors": [0.90, 0.96, 1.00],
            "adv_shifts": [-0.02, -0.01, 0.0],
            "proto_shifts": [-0.05, 0.0, 0.05],
            "temp_shifts": [-0.01, -0.005, 0.0],
            "accp_shifts": [-3.0, -2.0, -1.0],
            "batch_choices": [16, 16, 8],
            "adv_range": (0.07, 0.12),
            "proto_range": (0.75, 0.90),
            "temp_range": (0.055, 0.070),
            "accp_range": (91.0, 95.0),
        },
        "fewshot_recovery": {
            "lr_factors": [0.94, 1.00, 1.06],
            "adv_shifts": [-0.01, 0.0, 0.01],
            "proto_shifts": [0.0, 0.05, 0.10],
            "temp_shifts": [-0.005, 0.0, 0.005],
            "accp_shifts": [-1.0, 0.0, 1.0],
            "batch_choices": [16, 8, 8],
            "adv_range": (0.07, 0.13),
            "proto_range": (0.82, 0.95),
            "temp_range": (0.060, 0.075),
            "accp_range": (93.0, 96.0),
        },
        "raise_auroc": {
            "lr_factors": [0.90, 0.96, 1.00, 1.06],
            "adv_shifts": [-0.015, 0.0, 0.015],
            "proto_shifts": [-0.05, 0.0, 0.05],
            "temp_shifts": [-0.01, -0.005, 0.0],
            "accp_shifts": [-2.0, -1.0, 0.0],
            "batch_choices": [16, 16, 8],
            "adv_range": (0.07, 0.12),
            "proto_range": (0.78, 0.90),
            "temp_range": (0.055, 0.070),
            "accp_range": (92.0, 95.0),
        },
        "balanced_polish": {
            "lr_factors": [0.92, 0.96, 1.00, 1.04],
            "adv_shifts": [-0.015, 0.0, 0.015],
            "proto_shifts": [-0.05, 0.0, 0.05],
            "temp_shifts": [-0.005, 0.0, 0.005],
            "accp_shifts": [-1.0, 0.0, 1.0],
            "batch_choices": [16, 16, 8],
            "adv_range": (0.07, 0.12),
            "proto_range": (0.75, 0.90),
            "temp_range": (0.060, 0.075),
            "accp_range": (92.0, 96.0),
        },
    }
    preset = presets.get(directive, presets["balanced_polish"])

    threshold_presets = {
        "recover_setting_a": {
            "reject_shifts": [0.0, 0.1],
            "raw_dist_shifts": [0.0, 1.0],
            "raw_blends": [0.9, 1.0],
            "deepen_prob": 0.15,
        },
        "separate_open_set": {
            "reject_shifts": [-0.2, -0.1, 0.0],
            "raw_dist_shifts": [1.0, 2.0],
            "raw_blends": [0.8, 0.85, 0.9],
            "deepen_prob": 0.25,
        },
        "tighten_fpr": {
            "reject_shifts": [-0.2, -0.1, 0.0],
            "raw_dist_shifts": [1.0, 2.0],
            "raw_blends": [0.8, 0.85, 0.9],
            "deepen_prob": 0.25,
        },
        "fewshot_recovery": {
            "reject_shifts": [0.0, 0.1],
            "raw_dist_shifts": [0.0, 1.0],
            "raw_blends": [0.9, 1.0],
            "deepen_prob": 0.20,
        },
        "raise_auroc": {
            "reject_shifts": [-0.1, 0.0],
            "raw_dist_shifts": [0.0, 1.0, 2.0],
            "raw_blends": [0.85, 0.9, 1.0],
            "deepen_prob": 0.25,
        },
        "balanced_polish": {
            "reject_shifts": [-0.1, 0.0, 0.1],
            "raw_dist_shifts": [0.0, 1.0],
            "raw_blends": [0.85, 0.9, 1.0],
            "deepen_prob": 0.20,
        },
    }
    threshold_preset = threshold_presets.get(directive, threshold_presets["balanced_polish"])

    lr_factor = random.choice(preset["lr_factors"])
    adv_shift = random.choice(preset["adv_shifts"])
    proto_shift = random.choice(preset["proto_shifts"])
    temp_shift = random.choice(preset["temp_shifts"])
    accp_shift = random.choice(preset["accp_shifts"])
    reject_shift = random.choice(threshold_preset["reject_shifts"])
    raw_dist_shift = random.choice(threshold_preset["raw_dist_shifts"])
    raw_blend = float(random.choice(threshold_preset["raw_blends"]))
    bs = int(random.choice(preset["batch_choices"]))

    if EFFECTIVE_MAX_CONCURRENT >= 4 and bs == 16:
        bs = int(random.choice([16, 8]))

    # 用户要求提高训练预算，统一把最大训练轮次设为 200，
    # 通过早停门槛控制无效训练时长。
    epochs = 200

    lr = float(clip(base_cfg["lr"] * lr_factor, 1.5e-4, 2.6e-4))
    lambda_adv = float(clip(
        base_cfg["lambda_adv"] + adv_shift,
        preset["adv_range"][0],
        preset["adv_range"][1],
    ))
    lambda_proto = float(clip(
        base_cfg["lambda_proto"] + proto_shift,
        preset["proto_range"][0],
        preset["proto_range"][1],
    ))
    supcon_temperature = float(clip(
        base_cfg["supcon_temperature"] + temp_shift,
        preset["temp_range"][0],
        preset["temp_range"][1],
    ))
    accept_percentile = float(clip(
        base_cfg["accept_percentile"] + accp_shift,
        preset["accp_range"][0],
        preset["accp_range"][1],
    ))
    reject_threshold_factor = float(clip(
        float(base_cfg["reject_threshold_factor"]) + reject_shift,
        1.7,
        2.2,
    ))
    raw_open_score_blend = float(clip(raw_blend, 0.7, 1.0))
    raw_distance_percentile = float(clip(
        float(base_cfg["raw_distance_percentile"]) + raw_dist_shift,
        93.0,
        97.0,
    ))

    eval_interval = 5
    early_stop_patience = int(max(10, min(int(base_cfg.get("early_stop_patience", 12)), 16)))
    min_epoch_ratio_before_early_stop = float(
        clip(float(base_cfg.get("min_epoch_ratio_before_early_stop", 0.6)), 0.55, 0.8)
    )
    min_epochs_before_early_stop = int(
        max(120, min(int(epochs * min_epoch_ratio_before_early_stop), epochs))
    )
    early_stop_min_lr_ratio = float(
        clip(float(base_cfg.get("early_stop_min_lr_ratio", 0.1)), 0.08, 0.25)
    )
    early_stop_min_delta = float(
        clip(float(base_cfg.get("early_stop_min_delta", 3e-4)), 1e-4, 5e-3)
    )

    pretrained_item = None
    if pretrained_cycle:
        pretrained_item = pretrained_cycle[(idx - 1) % len(pretrained_cycle)]

    fe_tag = pretrained_item["tag"] if pretrained_item else "none"
    blocks_per_stage = int(base_cfg["blocks_per_stage"])
    encoder_channels = tuple(base_cfg["encoder_channels"])
    dropout = float(base_cfg["dropout"])
    deepen = random.random() < float(threshold_preset["deepen_prob"]) and blocks_per_stage < 3
    if deepen:
        blocks_per_stage = int(min(3, blocks_per_stage + 1))
        deepen_candidates = [
            (32, 64, 160, 320),
            (32, 64, 128, 320),
        ]
        encoder_channels = random.choice(deepen_candidates)
        dropout = float(clip(dropout + 0.05, 0.25, 0.5))
    deepen_tag = f"_d{blocks_per_stage}" if deepen else ""

    cfg = {
        "name": (
            f"iter_auto{idx:03d}_bs{bs}_lr{int(round(lr*1e6))}"
            f"_a{int(round(lambda_adv*100))}"
            f"_p{int(round(lambda_proto*100))}"
            f"_fe{fe_tag}{deepen_tag}"
        ),
        "search_directive": directive,
        "epochs": int(epochs),
        "batch_size": int(bs),
        "lr": lr,
        "lambda_adv": lambda_adv,
        "lambda_proto": lambda_proto,
        "lambda_recon": float(base_cfg["lambda_recon"]),
        "supcon_temperature": supcon_temperature,
        "accept_percentile": accept_percentile,
        "reject_threshold_factor": reject_threshold_factor,
        "raw_open_score_blend": raw_open_score_blend,
        "raw_distance_percentile": raw_distance_percentile,
        "eval_interval": eval_interval,
        "early_stop_patience": early_stop_patience,
        "min_epochs_before_early_stop": min_epochs_before_early_stop,
        "min_epoch_ratio_before_early_stop": min_epoch_ratio_before_early_stop,
        "early_stop_min_lr_ratio": early_stop_min_lr_ratio,
        "early_stop_min_delta": early_stop_min_delta,
        "encoder_channels": encoder_channels,
        "blocks_per_stage": blocks_per_stage,
        "num_axial_heads": int(base_cfg["num_axial_heads"]),
        "dropout": dropout,
        "pretrained_feature_model": pretrained_item["path"] if pretrained_item else "",
        "pretrained_feature_arch": pretrained_item["arch"] if pretrained_item else "auto",
        "main_backbone": str(base_cfg.get("main_backbone", "gcms")),
        "main_backbone_model": str(base_cfg.get("main_backbone_model", "")),
        "main_feature_layers": str(base_cfg.get("main_feature_layers", "layer4")),
        "main_feature_fuse": str(base_cfg.get("main_feature_fuse", "concat")),
        "transformer_patch_size": int(base_cfg.get("transformer_patch_size", 16)),
        "transformer_embed_dim": int(base_cfg.get("transformer_embed_dim", 256)),
        "transformer_depth": int(base_cfg.get("transformer_depth", 6)),
        "transformer_num_heads": int(base_cfg.get("transformer_num_heads", 8)),
        "transformer_mlp_ratio": float(base_cfg.get("transformer_mlp_ratio", 4.0)),
    }
    return cfg


def launch_one(cfg: dict, warmup_guard_ref: float | None,
               warmup_ref_run: str | None, warmup_guard_epoch: int,
               launch_slot: int):
    name = cfg["name"]
    out_dir = OUTPUTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON,
        "-u",
        "run_experiment.py",
        "--name",
        name,
        "--epochs",
        str(cfg["epochs"]),
        "--batch_size",
        str(cfg["batch_size"]),
        "--lr",
        str(cfg["lr"]),
        "--lambda_adv",
        str(cfg["lambda_adv"]),
        "--lambda_proto",
        str(cfg["lambda_proto"]),
        "--lambda_recon",
        str(cfg["lambda_recon"]),
        "--supcon_temperature",
        str(cfg["supcon_temperature"]),
        "--accept_percentile",
        str(cfg["accept_percentile"]),
        "--reject_threshold_factor",
        str(cfg["reject_threshold_factor"]),
        "--raw_open_score_blend",
        str(cfg["raw_open_score_blend"]),
        "--raw_distance_percentile",
        str(cfg["raw_distance_percentile"]),
        "--encoder_channels",
        ",".join(str(int(c)) for c in cfg.get("encoder_channels", (32, 64, 128, 256))),
        "--blocks_per_stage",
        str(cfg.get("blocks_per_stage", 2)),
        "--num_axial_heads",
        str(cfg.get("num_axial_heads", 4)),
        "--dropout",
        str(cfg.get("dropout", 0.3)),
        "--eval_interval",
        str(cfg["eval_interval"]),
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
        "--pretrained_feature_model",
        str(cfg.get("pretrained_feature_model", "")),
        "--pretrained_feature_arch",
        str(cfg.get("pretrained_feature_arch", "auto")),
        "--main_backbone",
        str(cfg.get("main_backbone", "gcms")),
        "--main_backbone_model",
        str(cfg.get("main_backbone_model", "")),
        "--main_feature_layers",
        str(cfg.get("main_feature_layers", "layer4")),
        "--main_feature_fuse",
        str(cfg.get("main_feature_fuse", "concat")),
        "--transformer_patch_size",
        str(cfg.get("transformer_patch_size", 16)),
        "--transformer_embed_dim",
        str(cfg.get("transformer_embed_dim", 256)),
        "--transformer_depth",
        str(cfg.get("transformer_depth", 6)),
        "--transformer_num_heads",
        str(cfg.get("transformer_num_heads", 8)),
        "--transformer_mlp_ratio",
        str(cfg.get("transformer_mlp_ratio", 4.0)),
    ]
    if cfg.get("prepared_dir"):
        cmd.extend(["--prepared_dir", str(cfg["prepared_dir"])])
    if cfg.get("rt_bins") is not None:
        cmd.extend(["--rt_bins", str(int(cfg["rt_bins"]))])
    if cfg.get("mz_bins") is not None:
        cmd.extend(["--mz_bins", str(int(cfg["mz_bins"]))])
    if cfg.get("input_raw_pca_enabled") is True:
        cmd.append("--enable_input_raw_pca")
    elif cfg.get("input_raw_pca_enabled") is False:
        cmd.append("--disable_input_raw_pca")
    if cfg.get("input_raw_pca_components") is not None:
        cmd.extend(["--input_raw_pca_components", str(int(cfg["input_raw_pca_components"]))])
    if warmup_guard_ref is not None and warmup_guard_ref > 0:
        cmd.extend([
            "--warmup_guard_enabled",
            "--warmup_guard_epoch",
            str(warmup_guard_epoch),
            "--warmup_guard_best_at_epoch",
            str(warmup_guard_ref),
            "--warmup_guard_min_ratio",
            str(WARMUP_GUARD_MIN_RATIO),
        ])
        if WARMUP_GUARD_COMPARE_BEST:
            cmd.append("--warmup_guard_compare_best")
        else:
            cmd.append("--warmup_guard_no_compare_best")

    append_progress([
        f"- [{ts()}] {SCRIPT_TAG} START {name}",
        "  - target: 先全面超过对比算法，再达到工业化阈值",
        (
            "  - overrides: "
            f"epochs={cfg.get('epochs')}, batch_size={cfg.get('batch_size')}, lr={cfg.get('lr')}, "
            f"lambda_adv={cfg.get('lambda_adv')}, lambda_proto={cfg.get('lambda_proto')}, "
            f"lambda_recon={cfg.get('lambda_recon')}, supcon_temperature={cfg.get('supcon_temperature')}, "
            f"accept_percentile={cfg.get('accept_percentile')}, reject_threshold_factor={cfg.get('reject_threshold_factor')}, "
            f"raw_open_score_blend={cfg.get('raw_open_score_blend')}, raw_distance_percentile={cfg.get('raw_distance_percentile')}, "
            f"encoder_channels={cfg.get('encoder_channels')}, blocks_per_stage={cfg.get('blocks_per_stage')}, "
            f"num_axial_heads={cfg.get('num_axial_heads')}, dropout={cfg.get('dropout')}, eval_interval={cfg.get('eval_interval')}, "
            f"early_stop_patience={cfg.get('early_stop_patience')}, "
            f"min_epochs_before_early_stop={cfg.get('min_epochs_before_early_stop')}, "
            f"min_epoch_ratio_before_early_stop={cfg.get('min_epoch_ratio_before_early_stop')}, "
            f"early_stop_min_lr_ratio={cfg.get('early_stop_min_lr_ratio')}, "
            f"early_stop_min_delta={cfg.get('early_stop_min_delta')}, "
            f"main_backbone={cfg.get('main_backbone')}, main_backbone_model={cfg.get('main_backbone_model')}, "
            f"main_feature_layers={cfg.get('main_feature_layers')}, main_feature_fuse={cfg.get('main_feature_fuse')}, "
            f"transformer_patch_size={cfg.get('transformer_patch_size')}, "
            f"transformer_embed_dim={cfg.get('transformer_embed_dim')}, "
            f"transformer_depth={cfg.get('transformer_depth')}, "
            f"transformer_num_heads={cfg.get('transformer_num_heads')}, "
            f"transformer_mlp_ratio={cfg.get('transformer_mlp_ratio')}, "
            f"pretrained_feature_arch={cfg.get('pretrained_feature_arch')}, "
            f"pretrained_feature_model={cfg.get('pretrained_feature_model')}"
        ),
        (
            "  - input_format: "
            f"prepared_dir={cfg.get('prepared_dir')}, "
            f"input_raw_pca_enabled={cfg.get('input_raw_pca_enabled')}, "
            f"input_raw_pca_components={cfg.get('input_raw_pca_components')}, "
            f"rt_bins={cfg.get('rt_bins')}, mz_bins={cfg.get('mz_bins')}"
        ),
        (
            f"  - warmup_guard: epoch={warmup_guard_epoch}, "
            f"ref_run={warmup_ref_run}, best_at_epoch={warmup_guard_ref}, "
            f"compare_best={WARMUP_GUARD_COMPARE_BEST}, min_ratio={WARMUP_GUARD_MIN_RATIO}"
        ),
    ])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["GCMS_SHOW_PROGRESS"] = "0"
    if GPU_IDS:
        env["CUDA_VISIBLE_DEVICES"] = GPU_IDS[launch_slot % len(GPU_IDS)]

    log_file = open(out_dir / "run.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc, log_file


def main() -> int:
    constraint_anchor_metrics = _load_constraint_anchor_metrics() if CONSTRAINT_MODE else None
    constraint_anchor_status = "ready" if (not CONSTRAINT_MODE or constraint_anchor_metrics is not None) else "missing"

    append_progress([
        f"- [{ts()}] {SCRIPT_TAG} LOOP START",
        "  - script: auto_iterate_until_sci.py",
        f"  - keep_top_n={KEEP_TOP_N}, max_trials={MAX_TRIALS}",
        f"  - keep_all_run_dirs={KEEP_ALL_RUN_DIRS}",
        (
            "  - phase1: 相对对比算法全面碾压 "
            f"(SA_acc>best+{BASELINE_DOMINANCE_MARGIN['setting_a_accuracy']}, "
            f"SA_bal>best+{BASELINE_DOMINANCE_MARGIN['setting_a_balanced_acc']}, "
            f"AUROC>best+{BASELINE_DOMINANCE_MARGIN['open_set_AUROC']}, "
            f"FPR<best-{BASELINE_DOMINANCE_MARGIN['FPR_at_95TPR']}, "
            f"1-shot>best+{BASELINE_DOMINANCE_MARGIN['shot1_acc']}, "
            f"3-shot>best+{BASELINE_DOMINANCE_MARGIN['shot3_acc']})"
        ),
        (
            "  - phase2: 达到工业应用阈值 "
            f"(SA_acc>={INDUSTRIAL_TARGETS['setting_a_accuracy_min']}, "
            f"SA_bal>={INDUSTRIAL_TARGETS['setting_a_balanced_acc_min']}, "
            f"AUROC>={INDUSTRIAL_TARGETS['setting_b_open_set_AUROC_min']}, "
            f"FPR<={INDUSTRIAL_TARGETS['setting_b_fpr95_max']}, "
            f"1-shot>={INDUSTRIAL_TARGETS['setting_c_1shot_acc_min']}, "
            f"3-shot>={INDUSTRIAL_TARGETS['setting_c_3shot_acc_min']}, "
            f"gap>={INDUSTRIAL_TARGETS['known_unknown_gap_min']})"
        ),
        (
            "  - warmup_guard_policy: "
            f"epoch={WARMUP_GUARD_EPOCH}, fallback_epoch={WARMUP_GUARD_FALLBACK_EPOCH}, "
            f"compare_best={WARMUP_GUARD_COMPARE_BEST}, min_ratio={WARMUP_GUARD_MIN_RATIO}"
        ),
        (
            "  - pathA_constraints: "
            f"enabled={CONSTRAINT_MODE}, anchor_run={CONSTRAINT_ANCHOR_RUN}, anchor_status={constraint_anchor_status}, "
            f"max_A_drop={CONSTRAINT_MAX_A_DROP}, max_FPR_rise={CONSTRAINT_MAX_FPR_RISE}, max_C1_drop={CONSTRAINT_MAX_C1_DROP}"
        ),
        (
            "  - practical_record: "
            f"enabled={PRACTICAL_RECORD_ENABLED}, main_metric={PRACTICAL_MAIN_METRIC}, "
            f"main_gain_min={PRACTICAL_MAIN_GAIN_MIN}, fpr_headroom={PRACTICAL_FPR_HEADROOM}, "
            f"main_weight={PRACTICAL_MAIN_WEIGHT}, soft_weight={PRACTICAL_SOFT_WEIGHT}, weighted_min={PRACTICAL_WEIGHTED_MIN}"
        ),
        f"  - max_concurrent_trainings={MAX_CONCURRENT_TRAININGS}, effective_max_concurrent={EFFECTIVE_MAX_CONCURRENT}, poll_seconds={POLL_SECONDS}",
        f"  - gpu_ids={GPU_IDS}",
    ])

    if has_active_training():
        append_progress([
            f"- [{ts()}] {SCRIPT_TAG} WAIT EXISTING TRAININGS",
            "  - detected existing run_experiment.py processes; wait for them to finish before launching new trials",
        ])
        while has_active_training():
            time.sleep(POLL_SECONDS)
        append_progress([
            f"- [{ts()}] {SCRIPT_TAG} RESUME AFTER EXISTING TRAININGS",
            "  - no external training processes remain; continue with refreshed best-run search",
        ])

    base_cfg, base_from = derive_base_config()
    pretrained_cycle = discover_pretrained_cycle()
    best_run_init, best_metrics_init = derive_current_best_run()
    backbone_queue, idx_after_backbone, backbone_info_lines = build_main_backbone_ablation_queue(
        start_idx=next_auto_index(),
        base_cfg=base_cfg,
        pretrained_cycle=pretrained_cycle,
    )
    structured_queue, idx_after_structured, structured_info_lines = build_structured_local_queue(
        start_idx=idx_after_backbone,
        base_cfg=base_cfg,
        best_metrics=best_metrics_init,
        pretrained_cycle=pretrained_cycle,
    )
    forced_queue, trial_idx_after_forced, forced_info_lines = build_input_ablation_queue(
        start_idx=idx_after_structured,
        base_cfg=base_cfg,
        best_metrics=best_metrics_init,
    )

    staged_queue = backbone_queue + structured_queue + forced_queue

    if backbone_queue:
        append_progress([
            f"- [{ts()}] {SCRIPT_TAG} MAIN BACKBONE ABLATION QUEUED",
            f"  - based_on_best={best_run_init}",
            *[f"  - {line}" for line in backbone_info_lines],
        ])
    elif backbone_info_lines:
        append_progress([
            f"- [{ts()}] {SCRIPT_TAG} MAIN BACKBONE ABLATION SKIPPED",
            *[f"  - {line}" for line in backbone_info_lines],
        ])

    if structured_queue:
        append_progress([
            f"- [{ts()}] {SCRIPT_TAG} STRUCTURED LOCAL QUEUED",
            f"  - based_on_best={best_run_init}",
            *[f"  - {line}" for line in structured_info_lines],
        ])
    elif structured_info_lines:
        append_progress([
            f"- [{ts()}] {SCRIPT_TAG} STRUCTURED LOCAL SKIPPED",
            *[f"  - {line}" for line in structured_info_lines],
        ])

    if forced_queue:
        append_progress([
            f"- [{ts()}] {SCRIPT_TAG} INPUT FORMAT ABLATION QUEUED",
            f"  - based_on_best={best_run_init}",
            *[f"  - {line}" for line in forced_info_lines],
        ])
    elif forced_info_lines:
        append_progress([
            f"- [{ts()}] {SCRIPT_TAG} INPUT FORMAT ABLATION SKIPPED",
            *[f"  - {line}" for line in forced_info_lines],
        ])

    append_progress([
        f"- [{ts()}] {SCRIPT_TAG} BASE CONFIG",
        f"  - base_from={base_from}",
        (
            f"  - base: epochs={base_cfg['epochs']}, batch_size={base_cfg['batch_size']}, "
            f"lr={base_cfg['lr']}, lambda_adv={base_cfg['lambda_adv']}, "
            f"lambda_proto={base_cfg['lambda_proto']}, supcon_temperature={base_cfg['supcon_temperature']}, "
            f"accept_percentile={base_cfg['accept_percentile']}, reject_threshold_factor={base_cfg['reject_threshold_factor']}, "
            f"raw_open_score_blend={base_cfg['raw_open_score_blend']}, raw_distance_percentile={base_cfg['raw_distance_percentile']}, "
            f"encoder_channels={base_cfg['encoder_channels']}, blocks_per_stage={base_cfg['blocks_per_stage']}, "
            f"num_axial_heads={base_cfg['num_axial_heads']}, dropout={base_cfg['dropout']}, "
            f"main_backbone={base_cfg['main_backbone']}, main_backbone_model={base_cfg['main_backbone_model']}, "
            f"main_feature_layers={base_cfg['main_feature_layers']}, main_feature_fuse={base_cfg['main_feature_fuse']}, "
            f"transformer_patch_size={base_cfg['transformer_patch_size']}, "
            f"transformer_embed_dim={base_cfg['transformer_embed_dim']}, "
            f"transformer_depth={base_cfg['transformer_depth']}, "
            f"transformer_num_heads={base_cfg['transformer_num_heads']}, "
            f"transformer_mlp_ratio={base_cfg['transformer_mlp_ratio']}, "
            f"early_stop_patience={base_cfg['early_stop_patience']}, "
            f"min_epochs_before_early_stop={base_cfg['min_epochs_before_early_stop']}, "
            f"min_epoch_ratio_before_early_stop={base_cfg['min_epoch_ratio_before_early_stop']}, "
            f"early_stop_min_lr_ratio={base_cfg['early_stop_min_lr_ratio']}, "
            f"early_stop_min_delta={base_cfg['early_stop_min_delta']}"
        ),
        f"  - pretrained_cycle={pretrained_cycle if pretrained_cycle else 'none'}",
        f"  - staged_queue_size={len(staged_queue)}",
    ])

    trial_idx = trial_idx_after_forced
    launched_trials = 0
    running: dict[str, dict] = {}

    while launched_trials < MAX_TRIALS or running:
        best_run, best_metrics = derive_current_best_run()
        if best_run and best_metrics and meets_targets(best_metrics):
            append_progress([
                f"- [{ts()}] {SCRIPT_TAG} TARGET ALREADY ACHIEVED by {best_run}",
                (
                    f"  - metrics: SA_acc={best_metrics.get('setting_a_accuracy')}, "
                    f"SA_bal={best_metrics.get('setting_a_balanced_acc')}, "
                    f"AUROC={best_metrics['open_set_AUROC']}, FPR95={best_metrics['FPR_at_95TPR']}, "
                    f"1-shot={best_metrics.get('shot1_acc')}, 3-shot={best_metrics['shot3_acc']}"
                ),
            ])
            for _, info in running.items():
                try:
                    info["proc"].terminate()
                except Exception:
                    pass
                try:
                    info["log_file"].close()
                except Exception:
                    pass
            return 0

        while launched_trials < MAX_TRIALS and len(running) < EFFECTIVE_MAX_CONCURRENT:
            directive = "balanced_polish"
            directive_reason = "暂无历史结果，使用保守均衡搜索"
            if best_metrics:
                directive, directive_reason = diagnose_search_direction(best_metrics)

            append_progress([
                f"- [{ts()}] {SCRIPT_TAG} SEARCH DIRECTIVE",
                f"  - based_on={best_run}",
                f"  - directive={directive}",
                f"  - reason={directive_reason}",
            ])

            if staged_queue:
                cfg = staged_queue.pop(0)
            else:
                cfg = generate_candidate(trial_idx, base_cfg, pretrained_cycle, directive=directive)
                trial_idx += 1
            if read_summary(cfg["name"]) is not None:
                continue

            warmup_guard_epoch = WARMUP_GUARD_EPOCH
            warmup_guard_ref, warmup_ref_run = derive_warmup_guard_reference(epoch=warmup_guard_epoch)
            if warmup_guard_ref is None:
                warmup_guard_epoch = WARMUP_GUARD_FALLBACK_EPOCH
                warmup_guard_ref, warmup_ref_run = derive_warmup_guard_reference(epoch=warmup_guard_epoch)

            append_progress([
                f"- [{ts()}] {SCRIPT_TAG} WARMUP REF",
                (
                    f"  - reference_run={warmup_ref_run}, epoch={warmup_guard_epoch}, "
                    f"val_acc={warmup_guard_ref}, compare_best={WARMUP_GUARD_COMPARE_BEST}, min_ratio={WARMUP_GUARD_MIN_RATIO}"
                ),
            ])

            proc, log_file = launch_one(
                cfg,
                warmup_guard_ref=warmup_guard_ref,
                warmup_ref_run=warmup_ref_run,
                warmup_guard_epoch=warmup_guard_epoch,
                launch_slot=launched_trials,
            )
            running[cfg["name"]] = {
                "proc": proc,
                "log_file": log_file,
                "cfg": cfg,
            }
            launched_trials += 1

            append_progress([
                f"- [{ts()}] {SCRIPT_TAG} LAUNCHED {cfg['name']}",
                f"  - pid={proc.pid}, running={len(running)}/{EFFECTIVE_MAX_CONCURRENT}, launched_trials={launched_trials}/{MAX_TRIALS}",
            ])

        finished = []
        for run_name, info in running.items():
            code = info["proc"].poll()
            if code is not None:
                finished.append((run_name, code))

        if not finished:
            time.sleep(POLL_SECONDS)
            continue

        for run_name, code in finished:
            info = running.pop(run_name)
            try:
                info["log_file"].close()
            except Exception:
                pass

            cfg = info["cfg"]
            summary = read_summary(run_name)
            if summary is None:
                append_progress([
                    f"- [{ts()}] {SCRIPT_TAG} FAIL {run_name}",
                    f"  - exit_code: {code}",
                    f"  - summary_found: {summary is not None}",
                ])
                append_result_jsonl({
                    "time": ts(),
                    "phase": SCRIPT_TAG,
                    "name": run_name,
                    "status": "failed",
                    "exit_code": code,
                })
                continue

            m = extract_metrics(summary)
            ok = meets_targets(m)
            best_run, best_metrics = derive_current_best_run()
            top_runs = derive_top_runs(KEEP_TOP_N)
            pruned = cleanup_non_best_artifacts(top_runs, active_runs=set(running.keys()))
            directive, directive_reason = diagnose_search_direction(m)
            baseline_ok = dominates_readme_baselines(m)
            industrial_ok = meets_industrial_targets(m)
            practical_status = _practical_record_status(m)

            extra_status = ""
            if code != 0:
                extra_status = f", exit_code={code} (summary已生成，按完成处理)"

            append_progress([
                f"- [{ts()}] {SCRIPT_TAG} DONE {run_name}",
                (
                    f"  - metrics: SA_acc={m.get('setting_a_accuracy')}, SA_bal={m.get('setting_a_balanced_acc')}, "
                    f"AUROC={m['open_set_AUROC']}, FPR95={m['FPR_at_95TPR']}, gap={m.get('known_unknown_gap')}, "
                    f"1-shot={m.get('shot1_acc')}, 3-shot={m['shot3_acc']}, "
                    f"baseline_ok={baseline_ok}, industrial_ok={industrial_ok}, meet_target={ok}, "
                    f"practical_record={practical_status['record']}, practical_reason={practical_status['reason']}, "
                    f"main_gain={practical_status['main_gain']}, weighted_score={practical_status['weighted_score']}{extra_status}"
                ),
                f"  - summary: outputs/{run_name}/evaluation_summary.json",
                f"  - next_direction={directive}, reason={directive_reason}",
                f"  - current_best={best_run}, best_metrics={best_metrics}",
                f"  - keep_top_n={KEEP_TOP_N}, kept_runs={top_runs}",
                f"  - pruned_non_best={pruned}",
            ])
            append_result_jsonl({
                "time": ts(),
                "phase": SCRIPT_TAG,
                "name": run_name,
                "status": "done",
                "metrics": m,
                "meet_target": ok,
                "baseline_ok": baseline_ok,
                "industrial_ok": industrial_ok,
                "practical_record": practical_status,
                "next_direction": directive,
                "next_direction_reason": directive_reason,
                "current_best": best_run,
                "keep_top_n": KEEP_TOP_N,
                "kept_runs": top_runs,
                "pruned_non_best": pruned,
            })

            if practical_status["record"]:
                append_progress([
                    f"- [{ts()}] {SCRIPT_TAG} PRACTICAL CANDIDATE {run_name}",
                    (
                        f"  - reason=main_metric_improved_and_practical, main_metric={practical_status['main_metric']}, "
                        f"main_gain={practical_status['main_gain']}, weighted_score={practical_status['weighted_score']}, "
                        f"required_keys={practical_status['required_keys']}, soft_pass_ratio={practical_status['soft_pass_ratio']}, "
                        f"baseline_checks={practical_status['baseline_checks']}, guard_ok={practical_status['guard_ok']}"
                    ),
                    f"  - summary: outputs/{run_name}/evaluation_summary.json",
                ])
                append_practical_jsonl({
                    "time": ts(),
                    "phase": SCRIPT_TAG,
                    "name": run_name,
                    "status": "practical_candidate",
                    "main_metric": practical_status["main_metric"],
                    "main_gain": practical_status["main_gain"],
                    "required_keys": practical_status["required_keys"],
                    "soft_keys": practical_status["soft_keys"],
                    "soft_pass_ratio": practical_status["soft_pass_ratio"],
                    "weighted_score": practical_status["weighted_score"],
                    "baseline_checks": practical_status["baseline_checks"],
                    "guard_ok": practical_status["guard_ok"],
                    "metrics": m,
                    "current_best": best_run,
                    "meet_target": ok,
                    "baseline_ok": baseline_ok,
                    "industrial_ok": industrial_ok,
                })

            if ok:
                append_progress([f"- [{ts()}] {SCRIPT_TAG} TARGET ACHIEVED by {run_name}"])
                for _, remain in running.items():
                    try:
                        remain["proc"].terminate()
                    except Exception:
                        pass
                    try:
                        remain["log_file"].close()
                    except Exception:
                        pass
                return 0

            # 动态更新下一轮基准: 跟随当前最优
            base_cfg, base_from = derive_base_config()

    append_progress([
        f"- [{ts()}] {SCRIPT_TAG} LOOP END (max_trials={MAX_TRIALS}, target not achieved)",
    ])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
