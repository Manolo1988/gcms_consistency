"""Phase-2 auto iteration for GCMS target search.

This script is designed to continue after a currently running experiment finishes.
It appends progress to outputs/PROJECT_PROGRESS.md and summaries to
outputs/AUTO_SEARCH_RESULTS.jsonl.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROGRESS_LOG = OUTPUTS_DIR / "PROJECT_PROGRESS.md"
RESULTS_JSONL = OUTPUTS_DIR / "AUTO_SEARCH_RESULTS.jsonl"
PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")

TARGETS = {
    "setting_b_open_set_AUROC_min": 0.61,
    "setting_b_fpr95_max": 0.70,
    "setting_c_3shot_acc_min": 0.85,
}

WAIT_SECONDS = 60
WARMUP_GUARD_EPOCH = 10
KEEP_TOP_N = 5
KEEP_ALL_RUN_DIRS = os.environ.get("AUTO_KEEP_ALL_RUN_DIRS", "1") != "0"

# New candidates around the best-known region (iter1/iter2),
# with both bs=16 and bs=8 variants.
CANDIDATES = [
    {
        "name": "iter8_bs8_recover_e50",
        "epochs": 50,
        "batch_size": 8,
        "lr": 2.0e-4,
        "lambda_adv": 0.10,
        "lambda_proto": 0.60,
        "lambda_recon": 0.2,
        "supcon_temperature": 0.07,
        "accept_percentile": 95.0,
        "eval_interval": 5,
        "early_stop_patience": 4,
    },
    {
        "name": "iter9_midtrade_e45",
        "epochs": 45,
        "batch_size": 16,
        "lr": 2.1e-4,
        "lambda_adv": 0.12,
        "lambda_proto": 0.70,
        "lambda_recon": 0.2,
        "supcon_temperature": 0.065,
        "accept_percentile": 95.0,
        "eval_interval": 5,
        "early_stop_patience": 4,
    },
    {
        "name": "iter10_bs8_lowadv_e45",
        "epochs": 45,
        "batch_size": 8,
        "lr": 1.8e-4,
        "lambda_adv": 0.06,
        "lambda_proto": 0.55,
        "lambda_recon": 0.2,
        "supcon_temperature": 0.06,
        "accept_percentile": 96.0,
        "eval_interval": 5,
        "early_stop_patience": 4,
    },
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
    sb = summary.get("setting_b", {})
    sc = summary.get("setting_c", {}).get("3", {})
    baseline = summary.get("baseline_tic_pca_mlp", {})
    baseline_sb = baseline.get("setting_b", {})
    baseline_sc3 = baseline.get("setting_c", {}).get("3", {})

    m = {
        "open_set_AUROC": sb.get("open_set_AUROC"),
        "FPR_at_95TPR": sb.get("FPR_at_95TPR"),
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

    return m


def meets_targets(m: dict) -> bool:
    return (
        (m.get("open_set_AUROC") is not None)
        and (m.get("FPR_at_95TPR") is not None)
        and (m.get("shot3_acc") is not None)
        and (m["open_set_AUROC"] >= TARGETS["setting_b_open_set_AUROC_min"])
        and (m["FPR_at_95TPR"] <= TARGETS["setting_b_fpr95_max"])
        and (m["shot3_acc"] >= TARGETS["setting_c_3shot_acc_min"])
    )


def _rank_metrics(m: dict) -> tuple:
    auroc = m.get("open_set_AUROC")
    fpr95 = m.get("FPR_at_95TPR")
    shot3 = m.get("shot3_acc")
    auroc_v = -1.0 if auroc is None else float(auroc)
    fpr_v = 1e9 if fpr95 is None else float(fpr95)
    shot3_v = -1.0 if shot3 is None else float(shot3)
    return (auroc_v, -fpr_v, shot3_v)


def derive_current_best_run() -> tuple[str | None, dict | None]:
    candidates = []
    for summary_path in OUTPUTS_DIR.glob("*/evaluation_summary.json"):
        run_name = summary_path.parent.name
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        m = extract_metrics(summary)
        candidates.append((run_name, m))

    if not candidates:
        return None, None
    best_run, best_metrics = max(candidates, key=lambda x: _rank_metrics(x[1]))
    return best_run, best_metrics


def derive_top_runs(top_n: int = KEEP_TOP_N) -> list[str]:
    ranked = []
    for summary_path in OUTPUTS_DIR.glob("*/evaluation_summary.json"):
        run_name = summary_path.parent.name
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        m = extract_metrics(summary)
        ranked.append((run_name, m))

    ranked.sort(key=lambda x: _rank_metrics(x[1]), reverse=True)
    return [name for name, _ in ranked[:max(int(top_n), 1)]]


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
            if m_inline:
                if int(m_inline.group(1)) == epoch:
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
    for summary_path in OUTPUTS_DIR.glob("*/evaluation_summary.json"):
        run_name = summary_path.parent.name
        v = read_val_acc_at_epoch(run_name, epoch=epoch)
        if v is not None:
            refs.append((run_name, v))
    if not refs:
        return None, None
    run_name, v = max(refs, key=lambda x: x[1])
    return v, run_name


def cleanup_non_best_artifacts(keep_runs: list[str]) -> list[str]:
    if KEEP_ALL_RUN_DIRS:
        return []

    keep_set = set(keep_runs)
    if not keep_set:
        return []

    keep_files = {"run_config.json", "evaluation_summary.json"}
    pruned_runs = []

    for summary_path in OUTPUTS_DIR.glob("*/evaluation_summary.json"):
        run_dir = summary_path.parent
        run_name = run_dir.name
        if run_name in keep_set:
            continue

        changed = False
        for child in run_dir.iterdir():
            if child.name in keep_files:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                changed = True
            elif child.is_file():
                try:
                    child.unlink()
                    changed = True
                except FileNotFoundError:
                    pass
        if changed:
            pruned_runs.append(run_name)

    return pruned_runs


def run_one(
    cfg: dict,
    warmup_guard_ref: float | None,
    warmup_ref_run: str | None,
) -> int:
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
        "--eval_interval",
        str(cfg["eval_interval"]),
        "--early_stop_patience",
        str(cfg["early_stop_patience"]),
    ]
    if warmup_guard_ref is not None and warmup_guard_ref > 0:
        cmd.extend([
            "--warmup_guard_enabled",
            "--warmup_guard_compare_best",
            "--warmup_guard_epoch",
            str(WARMUP_GUARD_EPOCH),
            "--warmup_guard_best_at_epoch",
            str(warmup_guard_ref),
        ])

    append_progress([
        f"- [{ts()}] AUTO2 START {name}",
        "  - target: AUROC>=0.61, FPR@95<=0.70, 3-shot>=0.85",
        (
            "  - overrides: epochs={epochs}, batch_size={batch_size}, lr={lr}, "
            "lambda_adv={lambda_adv}, lambda_proto={lambda_proto}, "
            "lambda_recon={lambda_recon}, supcon_temperature={supcon_temperature}, "
            "accept_percentile={accept_percentile}, eval_interval={eval_interval}, "
            "early_stop_patience={early_stop_patience}"
        ).format(**cfg),
        (
            f"  - warmup_guard: epoch={WARMUP_GUARD_EPOCH}, "
            f"ref_run={warmup_ref_run}, best_at_epoch={warmup_guard_ref}, "
            f"mode=compare_best"
        ),
    ])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["GCMS_SHOW_PROGRESS"] = "0"

    with open(out_dir / "run.log", "a", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode


def main() -> int:
    append_progress([
        f"- [{ts()}] AUTO2 LOOP START",
        "  - script: auto_iterate_phase2.py",
        f"  - keep_all_run_dirs={KEEP_ALL_RUN_DIRS}",
    ])

    for cfg in CANDIDATES:
        while has_active_training():
            print(f"[{ts()}] waiting current auto/training to finish...", flush=True)
            time.sleep(WAIT_SECONDS)

        name = cfg["name"]
        existing = read_summary(name)
        if existing is not None:
            m = extract_metrics(existing)
            ok = meets_targets(m)
            best_run, best_metrics = derive_current_best_run()
            top_runs = derive_top_runs(KEEP_TOP_N)
            pruned = cleanup_non_best_artifacts(top_runs)
            append_progress([
                f"- [{ts()}] AUTO2 SKIP {name} (summary exists)",
                f"  - metrics: AUROC={m['open_set_AUROC']}, FPR95={m['FPR_at_95TPR']}, 3-shot={m['shot3_acc']}, meet_target={ok}",
                (
                    f"  - current_best={best_run}, "
                    f"best_metrics={best_metrics}"
                ),
                f"  - keep_top_n={KEEP_TOP_N}, kept_runs={top_runs}",
                f"  - pruned_non_best={pruned}",
            ])
            append_result_jsonl({
                "time": ts(),
                "phase": "AUTO2",
                "name": name,
                "status": "skip_exists",
                "metrics": m,
                "meet_target": ok,
                "current_best": best_run,
                "keep_top_n": KEEP_TOP_N,
                "kept_runs": top_runs,
                "pruned_non_best": pruned,
            })
            if ok:
                append_progress([f"- [{ts()}] AUTO2 TARGET ACHIEVED by {name}"])
                return 0
            continue

        warmup_guard_ref, warmup_ref_run = derive_warmup_guard_reference(
            epoch=WARMUP_GUARD_EPOCH
        )
        append_progress([
            f"- [{ts()}] AUTO2 WARMUP REF",
            (
                f"  - reference_run={warmup_ref_run}, epoch={WARMUP_GUARD_EPOCH}, "
                f"val_acc={warmup_guard_ref}, mode=compare_best"
            ),
        ])

        code = run_one(
            cfg,
            warmup_guard_ref=warmup_guard_ref,
            warmup_ref_run=warmup_ref_run,
        )
        summary = read_summary(name)
        if code != 0 or summary is None:
            append_progress([
                f"- [{ts()}] AUTO2 FAIL {name}",
                f"  - exit_code: {code}",
                f"  - summary_found: {summary is not None}",
            ])
            append_result_jsonl({
                "time": ts(),
                "phase": "AUTO2",
                "name": name,
                "status": "failed",
                "exit_code": code,
            })
            continue

        m = extract_metrics(summary)
        ok = meets_targets(m)
        best_run, best_metrics = derive_current_best_run()
        top_runs = derive_top_runs(KEEP_TOP_N)
        pruned = cleanup_non_best_artifacts(top_runs)
        append_progress([
            f"- [{ts()}] AUTO2 DONE {name}",
            f"  - metrics: AUROC={m['open_set_AUROC']}, FPR95={m['FPR_at_95TPR']}, 3-shot={m['shot3_acc']}, meet_target={ok}",
            f"  - summary: outputs/{name}/evaluation_summary.json",
            (
                f"  - current_best={best_run}, "
                f"best_metrics={best_metrics}"
            ),
            f"  - keep_top_n={KEEP_TOP_N}, kept_runs={top_runs}",
            f"  - pruned_non_best={pruned}",
        ])
        append_result_jsonl({
            "time": ts(),
            "phase": "AUTO2",
            "name": name,
            "status": "done",
            "metrics": m,
            "meet_target": ok,
            "current_best": best_run,
            "keep_top_n": KEEP_TOP_N,
            "kept_runs": top_runs,
            "pruned_non_best": pruned,
        })

        if ok:
            append_progress([f"- [{ts()}] AUTO2 TARGET ACHIEVED by {name}"])
            return 0

    append_progress([f"- [{ts()}] AUTO2 LOOP END (target not achieved)"])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
