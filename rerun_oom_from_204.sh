#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

LIST_FILE="outputs/oom_runs_from_204.txt"
RERUN_LOG="outputs/oom_rerun.log"
CONCURRENCY="${RERUN_CONCURRENCY:-5}"
PYTHON_BIN="/home/ubuntu/sunlong/gcms_consistency/.venv/bin/python"
FORCE_EPOCHS="${RERUN_FORCE_EPOCHS:-200}"
FORCE_EARLY_STOP_PATIENCE="${RERUN_EARLY_STOP_PATIENCE:-12}"
FORCE_MIN_EPOCHS_BEFORE_STOP="${RERUN_MIN_EPOCHS_BEFORE_EARLY_STOP:-120}"
FORCE_MIN_EPOCH_RATIO="${RERUN_MIN_EPOCH_RATIO_BEFORE_EARLY_STOP:-0.6}"
FORCE_MIN_LR_RATIO="${RERUN_EARLY_STOP_MIN_LR_RATIO:-0.1}"
FORCE_MIN_DELTA="${RERUN_EARLY_STOP_MIN_DELTA:-0.0003}"

if [[ ! -f "$LIST_FILE" ]]; then
  echo "[fatal] missing list file: $LIST_FILE" >&2
  exit 1
fi

touch "$RERUN_LOG"
echo "[$(date +"%F %T")] start rerun, concurrency=$CONCURRENCY, force_epochs=$FORCE_EPOCHS, early_stop_patience=$FORCE_EARLY_STOP_PATIENCE" >> "$RERUN_LOG"

launch_one() {
  local run="$1"
  local cfg="outputs/$run/run_config.json"
  local run_dir="outputs/$run"
  local run_log="$run_dir/run.log"
  local eval_summary="$run_dir/evaluation_summary.json"

  if [[ -f "$eval_summary" ]]; then
    echo "[$(date +"%F %T")] [skip] $run already has evaluation_summary" >> "$RERUN_LOG"
    return 0
  fi

  if [[ ! -f "$cfg" ]]; then
    echo "[$(date +"%F %T")] [skip] $run missing run_config" >> "$RERUN_LOG"
    return 0
  fi

  local idx
  idx="$(echo "$run" | sed -E 's/^iter_auto([0-9]+).*/\1/')"
  if [[ -z "$idx" ]]; then
    echo "[$(date +"%F %T")] [skip] $run parse idx failed" >> "$RERUN_LOG"
    return 0
  fi

  local gpu=$((10#$idx % 2))

  local epochs batch_size lr lambda_adv lambda_proto lambda_recon
  local supcon_temperature accept_percentile eval_interval early_stop_patience
  local min_epochs_before_early_stop min_epoch_ratio_before_early_stop
  local early_stop_min_lr_ratio early_stop_min_delta
  local pretrained_feature_model pretrained_feature_arch

  epochs="$FORCE_EPOCHS"
  batch_size="$(jq -r '.config.batch_size // 8' "$cfg")"
  lr="$(jq -r '.config.lr // 0.0002' "$cfg")"
  lambda_adv="$(jq -r '.config.lambda_adv // 0.1' "$cfg")"
  lambda_proto="$(jq -r '.config.lambda_proto // 0.8' "$cfg")"
  lambda_recon="$(jq -r '.config.lambda_recon // 0.2' "$cfg")"
  supcon_temperature="$(jq -r '.config.supcon_temperature // 0.07' "$cfg")"
  accept_percentile="$(jq -r '.config.accept_percentile // 95.0' "$cfg")"
  eval_interval="$(jq -r '.config.eval_interval // 5' "$cfg")"
  early_stop_patience="$FORCE_EARLY_STOP_PATIENCE"
  min_epochs_before_early_stop="$FORCE_MIN_EPOCHS_BEFORE_STOP"
  min_epoch_ratio_before_early_stop="$FORCE_MIN_EPOCH_RATIO"
  early_stop_min_lr_ratio="$FORCE_MIN_LR_RATIO"
  early_stop_min_delta="$FORCE_MIN_DELTA"
  pretrained_feature_model="$(jq -r '.config.pretrained_feature_model // ""' "$cfg")"
  pretrained_feature_arch="$(jq -r '.config.pretrained_feature_arch // "auto"' "$cfg")"

  rm -f "$run_dir/evaluation_summary.json"
  {
    echo
    echo "[$(date +"%F %T")] [resume] start rerun with forced epochs/early-stop"
  } >> "$run_log"

  echo "[$(date +"%F %T")] [start] $run gpu=$gpu bs=$batch_size epochs=$epochs" >> "$RERUN_LOG"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONUNBUFFERED=1 \
  GCMS_SHOW_PROGRESS=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" -u run_experiment.py \
    --name "$run" \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --lr "$lr" \
    --lambda_adv "$lambda_adv" \
    --lambda_proto "$lambda_proto" \
    --lambda_recon "$lambda_recon" \
    --supcon_temperature "$supcon_temperature" \
    --accept_percentile "$accept_percentile" \
    --eval_interval "$eval_interval" \
    --early_stop_patience "$early_stop_patience" \
    --min_epochs_before_early_stop "$min_epochs_before_early_stop" \
    --min_epoch_ratio_before_early_stop "$min_epoch_ratio_before_early_stop" \
    --early_stop_min_lr_ratio "$early_stop_min_lr_ratio" \
    --early_stop_min_delta "$early_stop_min_delta" \
    --pretrained_feature_model "$pretrained_feature_model" \
    --pretrained_feature_arch "$pretrained_feature_arch" \
    >> "$run_log" 2>&1

  local status=$?
  if [[ $status -eq 0 ]]; then
    echo "[$(date +"%F %T")] [done] $run exit=$status" >> "$RERUN_LOG"
  else
    echo "[$(date +"%F %T")] [fail] $run exit=$status" >> "$RERUN_LOG"
  fi
  return $status
}

running_jobs=0
while IFS= read -r run || [[ -n "$run" ]]; do
  [[ -z "$run" ]] && continue

  while (( running_jobs >= CONCURRENCY )); do
    wait -n || true
    running_jobs=$((running_jobs - 1))
  done

  launch_one "$run" &
  running_jobs=$((running_jobs + 1))
done < "$LIST_FILE"

wait || true
echo "[$(date +"%F %T")] rerun batch finished" >> "$RERUN_LOG"
