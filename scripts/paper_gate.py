#!/usr/bin/env python3
"""Audit, prepare, and orchestrate the paper gate experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper_protocol import (
    PAPER_SHOTS,
    audit_split,
    build_cross_batch_episodes,
    load_json,
    save_json,
    validate_episode_manifest,
)


def _split_path(prepared_dir, model_dir=None):
    if model_dir:
        candidate = Path(model_dir) / "final_model" / "split.json"
        if candidate.exists():
            return candidate
    return Path(prepared_dir) / "split.json"


def command_audit(args):
    prepared_dir = Path(args.prepared_dir)
    report = audit_split(
        prepared_dir / "metadata.csv",
        _split_path(prepared_dir, args.model_dir),
        expected_holdout_products=tuple(args.holdout_products.split(",")),
    )
    save_json(report, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(2)


def command_episodes(args):
    prepared_dir = Path(args.prepared_dir)
    split = load_json(_split_path(prepared_dir, args.model_dir))
    manifest = build_cross_batch_episodes(
        prepared_dir / "metadata.csv",
        split["test_unknown_idx"],
        shots=tuple(args.shots),
        episode_count=args.episodes,
        seed_start=args.episode_seed_start,
    )
    manifest["validation"] = validate_episode_manifest(manifest)
    save_json(manifest, args.output)
    print(
        f"saved {manifest['episode_count']} episodes to {Path(args.output).resolve()} "
        f"({manifest['validation']['status']})"
    )
    if manifest["validation"]["status"] != "pass":
        raise SystemExit(2)


def command_commands(args):
    output_prefix = args.output_prefix.strip("/")
    shots = " ".join(str(value) for value in args.shots)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"PREPARED_DIR=\"{args.prepared_dir}\"",
        f"OUTPUT_PREFIX=\"{output_prefix}\"",
        "PYTHON_BIN=\"${PYTHON_BIN:-.venv/bin/python}\"",
        "OUTPUT_BASE=\"new_outputs/${OUTPUT_PREFIX}\"",
        "export CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES:-0}\"",
        "export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}\"",
        "",
        "if [[ ! -x \"${PYTHON_BIN}\" ]]; then",
        "  echo \"Python executable not found: ${PYTHON_BIN}\" >&2",
        "  exit 1",
        "fi",
        "mkdir -p \"${OUTPUT_BASE}\"",
        "\"${PYTHON_BIN}\" -u -c \"import torch; print('Python:', __import__('sys').executable); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())\"",
        "",
        "\"${PYTHON_BIN}\" -u scripts/paper_gate.py audit --prepared-dir \"${PREPARED_DIR}\"",
        "\"${PYTHON_BIN}\" -u scripts/paper_gate.py episodes --prepared-dir \"${PREPARED_DIR}\" "
        f"--shots {shots} --episodes {args.episodes} "
        f"--episode-seed-start {args.episode_seed_start}",
        "",
    ]
    for seed in args.seeds:
        main_name = f"main_s{seed}"
        lines.extend([
            f"MAIN_ROOT=\"${{OUTPUT_BASE}}/{main_name}\"",
            "\"${PYTHON_BIN}\" -u main.py train "
            "--output_dir \"${MAIN_ROOT}\" --prepared_dir \"${PREPARED_DIR}\" "
            f"--seed {seed} --epochs {args.epochs} --batch_size {args.batch_size} "
            f"--lr {args.lr} --main_backbone gcms --lambda_adv {args.lambda_adv} "
            f"--lambda_supcon {args.lambda_supcon} --lambda_proto {args.lambda_proto} "
            f"--lambda_recon {args.lambda_recon} --lambda_cls {args.lambda_cls} "
            f"--lambda_hard_pair {args.lambda_hard_pair} "
            f"--supcon_temperature {args.supcon_temperature} --accept_percentile 97.0 "
            "--rt_min 3.17 --rt_max 36.91 --mz_min 30 --mz_max 200 "
            "--no_auto_create_split_on_train --model_select_metric acc "
            "--model_select_min_epoch 100 --deterministic "
            "--dataloader_workers 4 --dataloader_prefetch_factor 2",
            "RUN_DIR=$(find \"${MAIN_ROOT}\" -maxdepth 1 -type d -name 'run_*' | sort | tail -n 1)",
            "if [[ -z \"${RUN_DIR}\" ]]; then echo \"No run directory found under ${MAIN_ROOT}\" >&2; exit 1; fi",
            "\"${PYTHON_BIN}\" -u main.py evaluate "
            "--output_dir \"${RUN_DIR}\" --prepared_dir \"${PREPARED_DIR}\" "
            f"--seed {seed} --main_backbone gcms --batch_size {args.batch_size} "
            "--rt_min 3.17 --rt_max 36.91 --mz_min 30 --mz_max 200 "
            "--skip_open_set --fewshot_repeats 1 --no_save_visualizations",
            "\"${PYTHON_BIN}\" -u scripts/evaluate_paper_checkpoint.py "
            "--run-dir \"${RUN_DIR}\" --prepared-dir \"${PREPARED_DIR}\" "
            f"--method-name main --shots {shots} --episodes {args.episodes} "
            f"--episode-seed-start {args.episode_seed_start} "
            "--episode-manifest result/paper_gate/fewshot_episodes.json",
            "",
            "\"${PYTHON_BIN}\" -u scripts/run_paper_dl_baseline.py "
            f"--method plain_cnn_ce --seed {seed} --prepared-dir \"${{PREPARED_DIR}}\" "
            f"--output-dir \"${{OUTPUT_BASE}}/plain_cnn_ce_s{seed}\" "
            f"--epochs {args.epochs} --batch-size {args.batch_size} --lr {args.lr} "
            f"--episodes {args.episodes} --episode-seed-start {args.episode_seed_start} "
            "--episode-manifest result/paper_gate/fewshot_episodes.json",
            "\"${PYTHON_BIN}\" -u scripts/run_paper_dl_baseline.py "
            f"--method plain_cnn_supcon --seed {seed} --prepared-dir \"${{PREPARED_DIR}}\" "
            f"--output-dir \"${{OUTPUT_BASE}}/plain_cnn_supcon_s{seed}\" "
            f"--epochs {args.epochs} --batch-size {args.batch_size} --lr {args.lr} "
            f"--episodes {args.episodes} --episode-seed-start {args.episode_seed_start} "
            "--episode-manifest result/paper_gate/fewshot_episodes.json",
            "",
        ])
    lines.append(
        "\"${PYTHON_BIN}\" -u scripts/summarize_paper_gate.py --root \"${OUTPUT_BASE}\" "
        "--main-method main"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"saved server commands to {output.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--prepared-dir", default="new_prepared_data_relabel_v1")
    audit_parser.add_argument("--model-dir")
    audit_parser.add_argument("--holdout-products", default="HMD,XCJ")
    audit_parser.add_argument("--output", default="result/paper_gate/protocol_audit.json")
    audit_parser.set_defaults(func=command_audit)

    episode_parser = subparsers.add_parser("episodes")
    episode_parser.add_argument("--prepared-dir", default="new_prepared_data_relabel_v1")
    episode_parser.add_argument("--model-dir")
    episode_parser.add_argument("--shots", nargs="+", type=int, default=list(PAPER_SHOTS))
    episode_parser.add_argument("--episodes", type=int, default=100)
    episode_parser.add_argument("--episode-seed-start", type=int, default=42000)
    episode_parser.add_argument("--output", default="result/paper_gate/fewshot_episodes.json")
    episode_parser.set_defaults(func=command_episodes)

    commands_parser = subparsers.add_parser("commands")
    commands_parser.add_argument("--prepared-dir", default="new_prepared_data_relabel_v1")
    commands_parser.add_argument("--output-prefix", default="paper_gate")
    commands_parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    commands_parser.add_argument("--shots", nargs="+", type=int, default=list(PAPER_SHOTS))
    commands_parser.add_argument("--episodes", type=int, default=100)
    commands_parser.add_argument("--episode-seed-start", type=int, default=42000)
    commands_parser.add_argument("--epochs", type=int, default=200)
    commands_parser.add_argument("--batch-size", type=int, default=64)
    commands_parser.add_argument("--lr", type=float, default=0.00026)
    commands_parser.add_argument("--lambda-adv", type=float, default=0.06)
    commands_parser.add_argument("--lambda-supcon", type=float, default=1.0)
    commands_parser.add_argument("--lambda-proto", type=float, default=0.75)
    commands_parser.add_argument("--lambda-recon", type=float, default=0.3)
    commands_parser.add_argument("--lambda-cls", type=float, default=0.25)
    commands_parser.add_argument("--lambda-hard-pair", type=float, default=0.05)
    commands_parser.add_argument("--supcon-temperature", type=float, default=0.07)
    commands_parser.add_argument("--output", default="scripts/run_paper_gate_server.sh")
    commands_parser.set_defaults(func=command_commands)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
