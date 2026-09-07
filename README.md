# GC-MS Closed-set and Few-shot Recognition

This repository contains the focused experiment for closed-set product
recognition and few-shot registration.

## Data layout

Run commands from the repository root:

```text
dataset/
new_prepared_data/
  tensors/
new_prepared_data_relabel_v1/
  metadata.csv
```

The metadata files use relative paths (`dataset/...` and
`new_prepared_data/tensors/...`), so the same checkout works locally and on a
server with the same directory layout.

## Environment

```bash
conda activate yolov10
```

## Audit and build manifests

```bash
python scripts/run_closeandfew.py audit \
  --metadata new_prepared_data_relabel_v1/metadata.csv

python scripts/run_closeandfew.py build \
  --metadata new_prepared_data_relabel_v1/metadata.csv \
  --output closeandfew_outputs/manifests \
  --folds 4 \
  --shots 1,3,5 \
  --episodes 100 \
  --seed 42
```

## Train one method

```bash
python scripts/run_closeandfew.py train \
  --metadata new_prepared_data_relabel_v1/metadata.csv \
  --manifest closeandfew_outputs/manifests/products_00_*.json \
  --method cnn_ce \
  --output closeandfew_outputs/cnn_ce \
  --epochs 100 \
  --device auto
```

Available methods are `cnn_ce`, `cnn_supcon`, and
`cnn_supcon_batchadv`.

## Evaluate exported features

```bash
python scripts/run_closeandfew.py evaluate \
  --metadata new_prepared_data_relabel_v1/metadata.csv \
  --manifest closeandfew_outputs/manifests/products_00_*.json \
  --feature cnn_ce=closeandfew_outputs/cnn_ce/features.npz \
  --output closeandfew_outputs/evaluation \
  --shots 1,3,5 \
  --episodes 100
```

## Full matrix

```bash
python scripts/run_closeandfew.py matrix \
  --metadata new_prepared_data_relabel_v1/metadata.csv \
  --manifests closeandfew_outputs/manifests \
  --output closeandfew_outputs/main \
  --seeds 41,42,43,44,45
```

## Tests

```bash
python -m unittest discover -s tests -p 'test_closeandfew_protocol.py' -v
```
