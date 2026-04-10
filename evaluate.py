"""
全面评估:
  1. 闭集产品识别 (Accuracy, F1, Confusion Matrix)
  2. 开集识别 (Open-Set AUROC, F1@FPR)
  3. 少样本评估 (N-shot Accuracy)
  4. 一致性评分 (AUROC, 分布)
  5. 批次鲁棒性 (Silhouette, 批次可预测性, t-SNE)
  6. 可解释性 (Grad-CAM 重叠率)
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    silhouette_score,
)
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from register import PrototypeStore, register_from_loader


# ═══════════════════════════════════════════════════════════
#  核心收集函数
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def collect_embeddings(model, loader, device):
    """收集所有样本的嵌入和元数据。"""
    model.eval()
    records = []

    for batch in loader:
        x = batch["input"].to(device)
        z = model.encode(x)

        for i in range(x.size(0)):
            records.append({
                "sample_id": batch["sample_id"][i] if isinstance(batch["sample_id"], list) else batch["sample_id"],
                "true_product": batch["product"][i].item(),
                "batch_label": batch["batch"][i].item(),
                "z": z[i].cpu().numpy(),
            })
    return records


@torch.no_grad()
def collect_predictions(model, loader, proto_store, device, reject_factor=2.0):
    """收集所有样本的预测结果 (基于原型匹配)。"""
    model.eval()
    records = []

    for batch in loader:
        x = batch["input"].to(device)
        z = model.encode(x)
        result = proto_store.predict(z)

        for i in range(x.size(0)):
            pred_idx = result["pred_idx"][i].item()
            score = result["scores"][i].item()
            min_dist = result["min_dists"][i].item()
            true_label = batch["product"][i].item()

            # 拒识判定
            is_known = proto_store.is_known(
                result["min_dists"][i:i+1], factor=reject_factor
            )[0]

            records.append({
                "sample_id": batch["sample_id"][i] if isinstance(batch["sample_id"], list) else batch["sample_id"],
                "true_product": true_label,
                "pred_product": pred_idx,
                "pred_class": result["pred_class"][i],
                "consistency_score": score,
                "min_dist": min_dist,
                "is_known": bool(is_known),
                "correct": pred_idx == true_label,
                "z": z[i].cpu().numpy(),
                "batch_label": batch["batch"][i].item(),
            })
    return records


# ═══════════════════════════════════════════════════════════
#  1. 闭集产品识别指标
# ═══════════════════════════════════════════════════════════

def classification_metrics(records):
    """产品分类指标。"""
    y_true = [r["true_product"] for r in records]
    y_pred = [r["pred_product"] for r in records]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


# ═══════════════════════════════════════════════════════════
#  2. 一致性评分质量
# ═══════════════════════════════════════════════════════════

def consistency_metrics(records):
    """一致性评分质量指标。"""
    correct = np.array([r["correct"] for r in records])
    scores = np.array([r["consistency_score"] for r in records])
    is_known = np.array([r["is_known"] for r in records])

    n = len(records)
    accept_rate = is_known.sum() / max(n, 1)

    # 正确识别且被接受 / 错误识别但被接受 等
    correct_accepted = (correct & is_known).sum()
    correct_rejected = (correct & ~is_known).sum()
    wrong_accepted = (~correct & is_known).sum()
    wrong_rejected = (~correct & ~is_known).sum()

    far = wrong_accepted / max(n, 1)
    frr = correct_rejected / max(n, 1)

    # 用一致性分数做 AUROC (正标签 = 预测正确)
    auroc, auprc = np.nan, np.nan
    if len(np.unique(correct)) > 1:
        auroc = roc_auc_score(correct.astype(int), scores)
        auprc = average_precision_score(correct.astype(int), scores)

    # EER (等错误率)
    eer = _compute_eer(correct, scores) if len(np.unique(correct)) > 1 else np.nan

    return {
        "accept_rate": float(accept_rate),
        "FAR": float(far),
        "FRR": float(frr),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "EER": float(eer),
        "correct_accepted": int(correct_accepted),
        "correct_rejected": int(correct_rejected),
        "wrong_accepted": int(wrong_accepted),
        "wrong_rejected": int(wrong_rejected),
    }


def _compute_eer(labels, scores, n_thresholds=1000):
    """计算等错误率 (EER)。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels.astype(int), scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


# ═══════════════════════════════════════════════════════════
#  3. 批次鲁棒性
# ═══════════════════════════════════════════════════════════

def batch_robustness_metrics(records):
    """批次鲁棒性指标。"""
    zs = np.stack([r["z"] for r in records])
    product_labels = np.array([r["true_product"] for r in records])
    batch_labels = np.array([r["batch_label"] for r in records])

    result = {}

    # Silhouette Score (按产品 — 应该高)
    if len(np.unique(product_labels)) > 1 and len(zs) > len(np.unique(product_labels)):
        result["silhouette_product"] = float(
            silhouette_score(zs, product_labels, sample_size=min(len(zs), 2000))
        )
    else:
        result["silhouette_product"] = np.nan

    # Silhouette Score (按批次 — 应该低, 代表批次信息被消除)
    if len(np.unique(batch_labels)) > 1 and len(zs) > len(np.unique(batch_labels)):
        result["silhouette_batch"] = float(
            silhouette_score(zs, batch_labels, sample_size=min(len(zs), 2000))
        )
    else:
        result["silhouette_batch"] = np.nan

    # 批次可预测性 (用嵌入预测批次标签的准确率 — 越低越好)
    if len(np.unique(batch_labels)) > 1 and len(zs) > 10:
        clf = LogisticRegression(max_iter=500, solver="lbfgs",
                                 multi_class="auto", random_state=42)
        clf.fit(zs, batch_labels)
        batch_pred_acc = clf.score(zs, batch_labels)
        result["batch_predictability"] = float(batch_pred_acc)
    else:
        result["batch_predictability"] = np.nan

    return result


# ═══════════════════════════════════════════════════════════
#  4. 开集识别指标
# ═══════════════════════════════════════════════════════════

def open_set_metrics(known_records, unknown_records):
    """
    开集评估: 已知类 vs 未知类的分离度。
    """
    known_scores = np.array([r["consistency_score"] for r in known_records])
    unknown_scores = np.array([r["consistency_score"] for r in unknown_records])

    # 标签: 1=已知, 0=未知
    labels = np.concatenate([np.ones(len(known_scores)),
                             np.zeros(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])

    auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else np.nan

    # F1 @ FPR=5%
    from sklearn.metrics import roc_curve, f1_score as f1_fn
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx_5pct = np.searchsorted(fpr, 0.05)
    if idx_5pct < len(thresholds):
        thresh_5pct = thresholds[idx_5pct]
        preds_5pct = (scores >= thresh_5pct).astype(int)
        f1_at_5pct = f1_fn(labels, preds_5pct)
    else:
        f1_at_5pct = np.nan

    return {
        "open_set_AUROC": float(auroc),
        "F1_at_FPR5pct": float(f1_at_5pct),
        "known_score_mean": float(known_scores.mean()),
        "unknown_score_mean": float(unknown_scores.mean()),
    }


# ═══════════════════════════════════════════════════════════
#  5. 少样本评估
# ═══════════════════════════════════════════════════════════

def few_shot_evaluate(model, dataset, ref_idx, test_idx, label_names,
                      device, cfg):
    """
    N-shot 评估: 用 ref_idx 注册原型，在 test_idx 上评估。
    """
    from torch.utils.data import DataLoader, Subset

    ref_loader = DataLoader(Subset(dataset, ref_idx),
                            batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx),
                             batch_size=cfg.batch_size, shuffle=False)

    # 用参考样本注册
    proto_store, _, _ = register_from_loader(
        model, ref_loader, label_names, device,
        percentile=cfg.accept_percentile
    )

    # 在测试集上评估
    records = collect_predictions(
        model, test_loader, proto_store, device,
        reject_factor=cfg.reject_threshold_factor
    )

    if not records:
        return {"accuracy": np.nan, "macro_f1": np.nan}

    cls_m = classification_metrics(records)
    return {
        "accuracy": cls_m["accuracy"],
        "macro_f1": cls_m["macro_f1"],
        "n_ref": len(ref_idx),
        "n_test": len(records),
    }


# ═══════════════════════════════════════════════════════════
#  6. 可视化
# ═══════════════════════════════════════════════════════════

def plot_embedding_tsne(records, save_path, color_by="product"):
    """t-SNE 嵌入可视化。"""
    from sklearn.manifold import TSNE

    zs = np.stack([r["z"] for r in records])
    labels = np.array([r["true_product"] if color_by == "product"
                       else r["batch_label"] for r in records])

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(zs)//4)))
    z2d = tsne.fit_transform(zs)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(z2d[mask, 0], z2d[mask, 1], label=str(lbl),
                   alpha=0.6, s=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title(f"t-SNE (colored by {color_by})")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(records, save_path):
    """一致性分数分布图 (正确 vs 错误)。"""
    correct_scores = [r["consistency_score"] for r in records if r["correct"]]
    wrong_scores = [r["consistency_score"] for r in records if not r["correct"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    if correct_scores:
        ax.hist(correct_scores, bins=30, alpha=0.6, label="Correct", color="green")
    if wrong_scores:
        ax.hist(wrong_scores, bins=30, alpha=0.6, label="Wrong", color="red")
    ax.legend()
    ax.set_xlabel("Consistency Score")
    ax.set_ylabel("Count")
    ax.set_title("Consistency Score Distribution")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  主评估入口
# ═══════════════════════════════════════════════════════════

def evaluate_fold(model, loader, proto_store, device, fold_name="",
                  save_dir=None, reject_factor=2.0):
    """评估一个 fold 的全部指标。"""
    records = collect_predictions(model, loader, proto_store, device,
                                 reject_factor=reject_factor)
    cls_m = classification_metrics(records)
    con_m = consistency_metrics(records)
    rob_m = batch_robustness_metrics(records)

    print(f"\n── 评估结果 [{fold_name}] ──")
    print(f"  分类准确率:       {cls_m['accuracy']:.4f}")
    print(f"  Macro-F1:         {cls_m['macro_f1']:.4f}")
    print(f"  一致性 AUROC:     {con_m['AUROC']:.4f}")
    print(f"  EER:              {con_m['EER']:.4f}")
    print(f"  接受率:           {con_m['accept_rate']:.4f}")
    print(f"  FAR / FRR:        {con_m['FAR']:.4f} / {con_m['FRR']:.4f}")
    print(f"  Silhouette(产品): {rob_m['silhouette_product']:.4f}")
    print(f"  Silhouette(批次): {rob_m['silhouette_batch']:.4f}")
    print(f"  批次可预测性:     {rob_m['batch_predictability']:.4f}")

    # 保存可视化
    if save_dir:
        save_dir = Path(save_dir)
        plot_embedding_tsne(records, save_dir / f"tsne_product_{fold_name}.png",
                            color_by="product")
        plot_embedding_tsne(records, save_dir / f"tsne_batch_{fold_name}.png",
                            color_by="batch")
        plot_score_distribution(records, save_dir / f"score_dist_{fold_name}.png")

    return {"classification": cls_m, "consistency": con_m,
            "robustness": rob_m, "records": records}


def evaluate_all_folds(fold_results, cfg):
    """汇总所有 fold 的评估结果。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    viz_dir = out_dir / "visualizations"

    all_metrics = []
    for fr in fold_results:
        m = evaluate_fold(
            fr["model"], fr["loader_val"], fr["proto_store"],
            device, fold_name=fr["test_batch"],
            save_dir=viz_dir,
            reject_factor=cfg.reject_threshold_factor,
        )
        m["fold"] = fr["fold"]
        m["test_batch"] = fr["test_batch"]
        all_metrics.append(m)

    # 汇总
    accs = [m["classification"]["accuracy"] for m in all_metrics]
    f1s = [m["classification"]["macro_f1"] for m in all_metrics]
    aurocs = [m["consistency"]["AUROC"] for m in all_metrics]
    eers = [m["consistency"]["EER"] for m in all_metrics]
    sil_prod = [m["robustness"]["silhouette_product"] for m in all_metrics]
    sil_batch = [m["robustness"]["silhouette_batch"] for m in all_metrics]
    batch_pred = [m["robustness"]["batch_predictability"] for m in all_metrics]

    print(f"\n{'='*60}")
    print(f"跨批次汇总 (Leave-One-Batch-Out)")
    print(f"{'='*60}")
    print(f"  Accuracy:           {np.nanmean(accs):.4f} ± {np.nanstd(accs):.4f}")
    print(f"  Macro-F1:           {np.nanmean(f1s):.4f} ± {np.nanstd(f1s):.4f}")
    _print_optional("AUROC", aurocs)
    _print_optional("EER", eers)
    _print_optional("Silhouette(产品)", sil_prod)
    _print_optional("Silhouette(批次)", sil_batch)
    _print_optional("批次可预测性", batch_pred)

    # 保存
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "accuracy_mean": float(np.nanmean(accs)),
        "accuracy_std": float(np.nanstd(accs)),
        "f1_mean": float(np.nanmean(f1s)),
        "f1_std": float(np.nanstd(f1s)),
        "folds": [
            {"fold": m["fold"], "test_batch": m["test_batch"],
             "accuracy": m["classification"]["accuracy"],
             "macro_f1": m["classification"]["macro_f1"],
             "AUROC": m["consistency"]["AUROC"],
             "EER": m["consistency"]["EER"],
             "accept_rate": m["consistency"]["accept_rate"],
             "FAR": m["consistency"]["FAR"],
             "FRR": m["consistency"]["FRR"],
             "silhouette_product": m["robustness"]["silhouette_product"],
             "silhouette_batch": m["robustness"]["silhouette_batch"],
             "batch_predictability": m["robustness"]["batch_predictability"]}
            for m in all_metrics
        ],
    }
    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return all_metrics


def _print_optional(name, values):
    clean = [v for v in values if not np.isnan(v)]
    if clean:
        print(f"  {name:20s} {np.mean(clean):.4f} ± {np.std(clean):.4f}")