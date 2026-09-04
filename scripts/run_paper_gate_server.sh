#!/usr/bin/env bash
set -euo pipefail

PREPARED_DIR="new_prepared_data_relabel_v1"
OUTPUT_PREFIX="paper_gate"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" -u -c "import torch; print('Python:', __import__('sys').executable); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

"${PYTHON_BIN}" -u scripts/paper_gate.py audit --prepared-dir "${PREPARED_DIR}"
"${PYTHON_BIN}" -u scripts/paper_gate.py episodes --prepared-dir "${PREPARED_DIR}" --shots 1 3 5 --episodes 100 --episode-seed-start 42000

"${PYTHON_BIN}" -u main.py train --output_dir "outputs/paper_gate/main_s41" --prepared_dir "${PREPARED_DIR}" --seed 41 --epochs 200 --batch_size 64 --lr 0.00026 --lambda_adv 0.06 --lambda_supcon 1.0 --lambda_proto 0.75 --lambda_recon 0.3 --lambda_cls 0.25 --lambda_hard_pair 0.05 --supcon_temperature 0.07 --no_auto_create_split_on_train --deterministic
"${PYTHON_BIN}" -u main.py evaluate --output_dir "outputs/paper_gate/main_s41" --prepared_dir "${PREPARED_DIR}" --seed 41 --skip_open_set --fewshot_repeats 1 --no_save_visualizations
"${PYTHON_BIN}" -u scripts/evaluate_paper_checkpoint.py --run-dir "outputs/paper_gate/main_s41" --prepared-dir "${PREPARED_DIR}" --method-name main --shots 1 3 5 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json

"${PYTHON_BIN}" -u scripts/run_paper_dl_baseline.py --method plain_cnn_ce --seed 41 --prepared-dir "${PREPARED_DIR}" --output-dir "outputs/paper_gate/plain_cnn_ce_s41" --epochs 200 --batch-size 64 --lr 0.00026 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json
"${PYTHON_BIN}" -u scripts/run_paper_dl_baseline.py --method plain_cnn_supcon --seed 41 --prepared-dir "${PREPARED_DIR}" --output-dir "outputs/paper_gate/plain_cnn_supcon_s41" --epochs 200 --batch-size 64 --lr 0.00026 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json

"${PYTHON_BIN}" -u main.py train --output_dir "outputs/paper_gate/main_s42" --prepared_dir "${PREPARED_DIR}" --seed 42 --epochs 200 --batch_size 64 --lr 0.00026 --lambda_adv 0.06 --lambda_supcon 1.0 --lambda_proto 0.75 --lambda_recon 0.3 --lambda_cls 0.25 --lambda_hard_pair 0.05 --supcon_temperature 0.07 --no_auto_create_split_on_train --deterministic
"${PYTHON_BIN}" -u main.py evaluate --output_dir "outputs/paper_gate/main_s42" --prepared_dir "${PREPARED_DIR}" --seed 42 --skip_open_set --fewshot_repeats 1 --no_save_visualizations
"${PYTHON_BIN}" -u scripts/evaluate_paper_checkpoint.py --run-dir "outputs/paper_gate/main_s42" --prepared-dir "${PREPARED_DIR}" --method-name main --shots 1 3 5 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json

"${PYTHON_BIN}" -u scripts/run_paper_dl_baseline.py --method plain_cnn_ce --seed 42 --prepared-dir "${PREPARED_DIR}" --output-dir "outputs/paper_gate/plain_cnn_ce_s42" --epochs 200 --batch-size 64 --lr 0.00026 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json
"${PYTHON_BIN}" -u scripts/run_paper_dl_baseline.py --method plain_cnn_supcon --seed 42 --prepared-dir "${PREPARED_DIR}" --output-dir "outputs/paper_gate/plain_cnn_supcon_s42" --epochs 200 --batch-size 64 --lr 0.00026 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json

"${PYTHON_BIN}" -u main.py train --output_dir "outputs/paper_gate/main_s43" --prepared_dir "${PREPARED_DIR}" --seed 43 --epochs 200 --batch_size 64 --lr 0.00026 --lambda_adv 0.06 --lambda_supcon 1.0 --lambda_proto 0.75 --lambda_recon 0.3 --lambda_cls 0.25 --lambda_hard_pair 0.05 --supcon_temperature 0.07 --no_auto_create_split_on_train --deterministic
"${PYTHON_BIN}" -u main.py evaluate --output_dir "outputs/paper_gate/main_s43" --prepared_dir "${PREPARED_DIR}" --seed 43 --skip_open_set --fewshot_repeats 1 --no_save_visualizations
"${PYTHON_BIN}" -u scripts/evaluate_paper_checkpoint.py --run-dir "outputs/paper_gate/main_s43" --prepared-dir "${PREPARED_DIR}" --method-name main --shots 1 3 5 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json

"${PYTHON_BIN}" -u scripts/run_paper_dl_baseline.py --method plain_cnn_ce --seed 43 --prepared-dir "${PREPARED_DIR}" --output-dir "outputs/paper_gate/plain_cnn_ce_s43" --epochs 200 --batch-size 64 --lr 0.00026 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json
"${PYTHON_BIN}" -u scripts/run_paper_dl_baseline.py --method plain_cnn_supcon --seed 43 --prepared-dir "${PREPARED_DIR}" --output-dir "outputs/paper_gate/plain_cnn_supcon_s43" --epochs 200 --batch-size 64 --lr 0.00026 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json

"${PYTHON_BIN}" -u main.py train --output_dir "outputs/paper_gate/main_s44" --prepared_dir "${PREPARED_DIR}" --seed 44 --epochs 200 --batch_size 64 --lr 0.00026 --lambda_adv 0.06 --lambda_supcon 1.0 --lambda_proto 0.75 --lambda_recon 0.3 --lambda_cls 0.25 --lambda_hard_pair 0.05 --supcon_temperature 0.07 --no_auto_create_split_on_train --deterministic
"${PYTHON_BIN}" -u main.py evaluate --output_dir "outputs/paper_gate/main_s44" --prepared_dir "${PREPARED_DIR}" --seed 44 --skip_open_set --fewshot_repeats 1 --no_save_visualizations
"${PYTHON_BIN}" -u scripts/evaluate_paper_checkpoint.py --run-dir "outputs/paper_gate/main_s44" --prepared-dir "${PREPARED_DIR}" --method-name main --shots 1 3 5 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json

"${PYTHON_BIN}" -u scripts/run_paper_dl_baseline.py --method plain_cnn_ce --seed 44 --prepared-dir "${PREPARED_DIR}" --output-dir "outputs/paper_gate/plain_cnn_ce_s44" --epochs 200 --batch-size 64 --lr 0.00026 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json
"${PYTHON_BIN}" -u scripts/run_paper_dl_baseline.py --method plain_cnn_supcon --seed 44 --prepared-dir "${PREPARED_DIR}" --output-dir "outputs/paper_gate/plain_cnn_supcon_s44" --epochs 200 --batch-size 64 --lr 0.00026 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json

"${PYTHON_BIN}" -u main.py train --output_dir "outputs/paper_gate/main_s45" --prepared_dir "${PREPARED_DIR}" --seed 45 --epochs 200 --batch_size 64 --lr 0.00026 --lambda_adv 0.06 --lambda_supcon 1.0 --lambda_proto 0.75 --lambda_recon 0.3 --lambda_cls 0.25 --lambda_hard_pair 0.05 --supcon_temperature 0.07 --no_auto_create_split_on_train --deterministic
"${PYTHON_BIN}" -u main.py evaluate --output_dir "outputs/paper_gate/main_s45" --prepared_dir "${PREPARED_DIR}" --seed 45 --skip_open_set --fewshot_repeats 1 --no_save_visualizations
"${PYTHON_BIN}" -u scripts/evaluate_paper_checkpoint.py --run-dir "outputs/paper_gate/main_s45" --prepared-dir "${PREPARED_DIR}" --method-name main --shots 1 3 5 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json

"${PYTHON_BIN}" -u scripts/run_paper_dl_baseline.py --method plain_cnn_ce --seed 45 --prepared-dir "${PREPARED_DIR}" --output-dir "outputs/paper_gate/plain_cnn_ce_s45" --epochs 200 --batch-size 64 --lr 0.00026 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json
"${PYTHON_BIN}" -u scripts/run_paper_dl_baseline.py --method plain_cnn_supcon --seed 45 --prepared-dir "${PREPARED_DIR}" --output-dir "outputs/paper_gate/plain_cnn_supcon_s45" --epochs 200 --batch-size 64 --lr 0.00026 --episodes 100 --episode-seed-start 42000 --episode-manifest result/paper_gate/fewshot_episodes.json

"${PYTHON_BIN}" -u scripts/summarize_paper_gate.py --root outputs/paper_gate --main-method main
