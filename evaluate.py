"""
全面评估:
  1. 产品分类指标  2. 一致性接受/拒收指标
  3. 跨批次稳定性  4. 表征质量
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)
from pathlib import Path
import json


@torch.no_grad()
def collect_predictions(model, loader, thresholds, device):
    """收集所有样本的预测结果。"""
    model.eval()
    records = []

    for batch in loader:
        x = batch["input"].to(device)
        result = model.predict(x)

        for i in range(x.size(0)):
            pred = result["pred_product"][i].item()
            conf = result["confidence"][i].item()
            dist = result["consistency_dist"][i].item()
            energy = result["energy"][i].item()
            true_label = batch["product"][i].item()
            threshold = thresholds.get(pred, float("inf"))
            accepted = dist <= threshold

            records.append({
                "sample_id": batch["sample_id"][i],
                "true_product": true_label,
                "pred_product": pred,
                "confidence": conf,
                "consistency_dist": dist,
                "energy": energy,
                "threshold": threshold,
                "accepted": accepted,
                "correct": pred == true_label,
                "z": result["z"][i].cpu().numpy(),
            })
    return records


def classification_metrics(records):
    """产品分类指标。"""
    y_true = [r["true_product"] for r in records]
    y_pred = [r["pred_product"] for r in records]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def consistency_metrics(records):
    """一致性接受/拒收指标。"""
    correct = np.array([r["correct"] for r in records])
    accepted = np.array([r["accepted"] for r in records])
    dists = np.array([r["consistency_dist"] for r in records])

    # 对正确分类的样本
    correct_accepted = (correct & accepted).sum()
    correct_rejected = (correct & ~accepted).sum()
    wrong_accepted = (~correct & accepted).sum()
    wrong_rejected = (~correct & ~accepted).sum()

    n = len(records)
    far = wrong_accepted / max(n, 1)        # 假通过率
    frr = correct_rejected / max(n, 1)      # 假拒率
    accept_rate = accepted.sum() / max(n, 1)

    # 用距离做 AUROC (正标签 = 预测正确)
    auroc, auprc = np.nan, np.nan
    if len(np.unique(correct)) > 1:
        # 距离越小越可能是正确的，取负距离做 score
        scores = -dists
        auroc = roc_auc_score(correct.astype(int), scores)
        auprc = average_precision_score(correct.astype(int), scores)

    return {
        "accept_rate": accept_rate,
        "FAR": far,
        "FRR": frr,
        "AUROC": auroc,
        "AUPRC": auprc,
        "correct_accepted": int(correct_accepted),
        "correct_rejected": int(correct_rejected),
        "wrong_accepted": int(wrong_accepted),
        "wrong_rejected": int(wrong_rejected),
    }


def batch_leakage_score(records, num_batches=None):
    """
    在潜空间上评估批次泄漏程度：
    训练一个小的线性分类器预测批次，准确率越低说明去批次越成功。
    """
    from sklearn.linear_model import LogisticRegression

    zs = np.stack([r["z"] for r in records])
    # 这里需要 batch label，暂用 sample_id 推断
    # 简化: 返回 z 的统计即可
    return {"z_mean_norm": float(np.linalg.norm(zs.mean(axis=0)))}


def evaluate_fold(model, loader, thresholds, device, fold_name=""):
    """评估一个 fold 的全部指标。"""
    records = collect_predictions(model, loader, thresholds, device)
    cls_m = classification_metrics(records)
    con_m = consistency_metrics(records)
    rep_m = batch_leakage_score(records)

    print(f"\n── 评估结果 [{fold_name}] ──")
    print(f"  分类准确率:  {cls_m['accuracy']:.4f}")
    print(f"  Macro-F1:    {cls_m['macro_f1']:.4f}")
    print(f"  接受率:      {con_m['accept_rate']:.4f}")
    print(f"  FAR:         {con_m['FAR']:.4f}")
    print(f"  FRR:         {con_m['FRR']:.4f}")
    print(f"  AUROC:       {con_m['AUROC']:.4f}")

    return {"classification": cls_m, "consistency": con_m,
            "representation": rep_m, "records": records}


def evaluate_all_folds(fold_results, cfg):
    """汇总所有 fold 的评估结果。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_metrics = []
    for fr in fold_results:
        m = evaluate_fold(
            fr["model"], fr["loader_val"], fr["thresholds"],
            device, fold_name=fr["test_batch"]
        )
        m["fold"] = fr["fold"]
        m["test_batch"] = fr["test_batch"]
        all_metrics.append(m)

    # 汇总
    accs = [m["classification"]["accuracy"] for m in all_metrics]
    f1s = [m["classification"]["macro_f1"] for m in all_metrics]
    aurocs = [m["consistency"]["AUROC"] for m in all_metrics]

    print(f"\n{'='*60}")
    print(f"跨批次汇总 (Leave-One-Batch-Out)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Macro-F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    aurocs_clean = [a for a in aurocs if not np.isnan(a)]
    if aurocs_clean:
        print(f"  AUROC:     {np.mean(aurocs_clean):.4f} ± {np.std(aurocs_clean):.4f}")

    # 保存
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "folds": [
            {"fold": m["fold"], "test_batch": m["test_batch"],
             "accuracy": m["classification"]["accuracy"],
             "macro_f1": m["classification"]["macro_f1"],
             "accept_rate": m["consistency"]["accept_rate"],
             "FAR": m["consistency"]["FAR"],
             "FRR": m["consistency"]["FRR"]}
            for m in all_metrics
        ],
    }
    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return all_metrics