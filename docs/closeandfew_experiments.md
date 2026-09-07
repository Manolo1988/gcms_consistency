# Closed-set recognition and few-shot registration protocol

This directory implements the minimum experiment needed to support one focused
paper claim: a representation learned from known GC-MS products should remain
reliable on later batches and should admit new products from 1, 3, or 5
independent registration samples without retraining the backbone.

## Experimental contract

The split unit is `product_fine + lot_id`. When `lot_id` is missing, the sample
ID is used as a conservative fallback. Duplicate sample IDs are disambiguated
by a deterministic occurrence suffix and reported in the data audit. Repeated
measurements from one product
lot cannot cross closed-set partitions or the support/query boundary. Closed
validation and test batches occur after all training batches. The final novel
products and final closed-test batches are never used for preprocessing,
checkpoint selection, early stopping, or hyperparameter selection.

Few-shot evaluation is incremental, not a novel-only classification task. A
query is classified jointly against all base prototypes and newly registered
prototypes. Each episode uses exactly 1, 3, or 5 independent groups per novel
class from pre-cutoff batches; all novel queries are from batches after one
global temporal cutoff. Report base, novel, and all-class Macro-F1 and balanced
accuracy, their harmonic Macro-F1, old-to-new and new-to-old error rates, and
registration time.

## Minimum method matrix

Use one fixed plain CNN architecture for every deep row. Only the training
objective changes:

| Method | CE | SupCon | Batch adversarial |
|---|---:|---:|---:|
| `cnn_ce` | yes | no | no |
| `cnn_supcon` | yes | yes | no |
| `cnn_supcon_batchadv` | yes | yes | yes |

Traditional references are `tic_pca_proto`, `tic_pca_mahalanobis`, native
TIC-PCA-SVM, and native PLS-DA. PCA, scaling, covariance, SVM, and PLS-DA are
fit only on `train_idx`. The three exported traditional latent spaces use the
same frozen mean-prototype registration rule as the deep methods; native SVM
and PLS-DA results are additional closed-set references. Attention,
reconstruction, prototype redistribution, rejection, and open-set modules are
outside this minimum paper and should not be reintroduced before this matrix is
complete. A spherical prototype variant may be added later as one isolated
ablation, disabled by default.

Automatically selected outer folds use disjoint novel-product sets, so one
novel product is not counted repeatedly in the across-fold summary. With the
current metadata, four disjoint two-product folds satisfy the temporal 5-shot
constraint; requesting more folds returns the maximum feasible disjoint set.

## Reproducible run

From the repository root, first create immutable manifests:

```bash
python scripts/run_closeandfew.py build \
  --metadata new_prepared_data_relabel_v1/metadata.csv \
  --output closeandfew_outputs/manifests \
  --folds 4 --shots 1,3,5 --episodes 100 --seed 42
```

Run the complete matrix (the metadata contains obsolete absolute tensor paths,
so `--tensor-root` is required on a relocated machine):

```bash
python scripts/run_closeandfew.py matrix \
  --metadata new_prepared_data_relabel_v1/metadata.csv \
  --manifests closeandfew_outputs/manifests \
  --tensor-root new_prepared_data/tensors \
  --output closeandfew_outputs/main \
  --seeds 41,42,43,44,45 \
  --methods tic_pca_proto,tic_pca_mahalanobis,tic_plsda_latent,cnn_ce,cnn_supcon,cnn_supcon_batchadv
```

The default is five training seeds, 100 registration episodes per shot and
2,000 lot-level bootstrap replicates. Increase to ten training seeds and
1,000 episodes if compute permits. Methods within a fold use identical episode
seeds, enabling paired deltas in `paired_differences.csv`. Keep one additional
predeclared novel-product fold untouched if method or hyperparameter changes
are made after inspecting the initial final-fold results.

For a quick implementation check, use one fold, one seed, two episodes, ten
bootstrap replicates, and one epoch. Such a run verifies plumbing only and is
not publishable evidence.

## Required reporting

Report all folds and seeds rather than the best run. For closed-set recognition,
show Accuracy, Macro-F1, Balanced Accuracy, per-class recall, confusion matrix,
and lot-level 95% bootstrap intervals. For each shot, report mean, standard
deviation, and empirical 95% interval across episodes. Use paired episode
deltas for method comparisons. Document all exclusions and the manifest hash;
do not describe a result as independent if multiple injections share a lot.
