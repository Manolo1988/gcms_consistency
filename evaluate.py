"""统一评估 (五维度指标体系):
  1. 产品识别 (Setting A): Macro-F1, Accuracy, 跨批次 Δ
  2. 开放集 + 新品 (Setting B & C): AUROC, FPR@95TPR, N-shot Accuracy
  3. 一致性评分 (Setting A): AUROC(correct/wrong), Cohen's d
  4. 批次鲁棒性 (Setting A): 批次可预测性, Silhouette
  5. 可解释性: Grad-CAM (定性)
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    silhouette_score, roc_curve,
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
#  指标组 1: 产品识别 (Setting A)
# ═══════════════════════════════════════════════════════════

def product_identification_metrics(records, train_records=None):
    """
    Macro-F1, Accuracy, 跨批次准确率差 Δ。
    train_records: 训练集预测 (用于计算同批次准确率)。
    """
    y_true = [r["true_product"] for r in records]
    y_pred = [r["pred_product"] for r in records]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }

    if train_records is not None:
        acc_same = accuracy_score(
            [r["true_product"] for r in train_records],
            [r["pred_product"] for r in train_records])
        metrics["cross_batch_gap"] = acc_same - metrics["accuracy"]

    return metrics


# ═══════════════════════════════════════════════════════════
#  指标组 3: 一致性评分 (Setting A)
# ═══════════════════════════════════════════════════════════

def consistency_scoring_metrics(records):
    """AUROC (correct vs incorrect), Cohen's d。"""
    correct = np.array([r["correct"] for r in records])
    scores = np.array([r["consistency_score"] for r in records])
    is_known = np.array([r["is_known"] for r in records])

    result = {
        "accept_rate": float(is_known.sum() / max(len(records), 1)),
    }

    if len(np.unique(correct)) > 1:
        result["AUROC_correct"] = float(roc_auc_score(correct.astype(int), scores))

        # Cohen's d: 正确预测 vs 错误预测的分数分离度
        s_correct = scores[correct]
        s_wrong = scores[~correct]
        pooled_std = np.sqrt(
            (s_correct.var() * max(len(s_correct)-1, 0) +
             s_wrong.var() * max(len(s_wrong)-1, 0))
            / max(len(s_correct) + len(s_wrong) - 2, 1)
        )
        result["cohens_d"] = float(
            (s_correct.mean() - s_wrong.mean()) / max(pooled_std, 1e-8)
        )
    else:
        result["AUROC_correct"] = np.nan
        result["cohens_d"] = np.nan

    return result


# ═══════════════════════════════════════════════════════════
#  指标组 4: 批次鲁棒性 (Setting A)
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
#  指标组 2: 开放集 + 新品 (Setting B & C)
# ═══════════════════════════════════════════════════════════

def open_set_metrics(known_records, unknown_records):
    """
    开集评估: 已知类 vs 未知类的分离度。
    返回 AUROC 和 FPR@95TPR。
    """
    known_scores = np.array([r["consistency_score"] for r in known_records])
    unknown_scores = np.array([r["consistency_score"] for r in unknown_records])

    # 标签: 1=已知, 0=未知
    labels = np.concatenate([np.ones(len(known_scores)),
                             np.zeros(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])

    result = {}
    if len(np.unique(labels)) > 1:
        result["open_set_AUROC"] = float(roc_auc_score(labels, scores))

        # FPR@95TPR
        fpr, tpr, thresholds = roc_curve(labels, scores)
        idx_95 = np.searchsorted(tpr, 0.95)
        result["FPR_at_95TPR"] = float(fpr[min(idx_95, len(fpr) - 1)])
    else:
        result["open_set_AUROC"] = float(np.nan)
        result["FPR_at_95TPR"] = float(np.nan)

    result["known_score_mean"] = float(known_scores.mean()) if len(known_scores) else float(np.nan)
    result["unknown_score_mean"] = float(unknown_scores.mean()) if len(unknown_scores) else float(np.nan)
    return result


# ═══════════════════════════════════════════════════════════
#  少样本评估 (Setting C)
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
#  Setting A/B/C 统一评估入口
# ═══════════════════════════════════════════════════════════

def evaluate_setting_a(model, loader_train_noaug, loader_val, proto_store,
                       device, cfg, fold_name=""):
    """Setting A: 闭集跨批次 — 在已知类上评估。"""
    train_records = collect_predictions(
        model, loader_train_noaug, proto_store, device,
        reject_factor=cfg.reject_threshold_factor)
    val_records = collect_predictions(
        model, loader_val, proto_store, device,
        reject_factor=cfg.reject_threshold_factor)

    pid = product_identification_metrics(val_records, train_records)
    con = consistency_scoring_metrics(val_records)
    rob = batch_robustness_metrics(val_records)

    print(f"\n── Setting A [{fold_name}] ──")
    print(f"  Accuracy:       {pid['accuracy']:.4f}")
    print(f"  Macro-F1:       {pid['macro_f1']:.4f}")
    if "cross_batch_gap" in pid:
        print(f"  Cross-batch Δ:  {pid['cross_batch_gap']:.4f}")
    print(f"  Score AUROC:    {con['AUROC_correct']:.4f}")
    print(f"  Cohen's d:      {con['cohens_d']:.4f}")
    print(f"  Sil(product):   {rob['silhouette_product']:.4f}")
    print(f"  Sil(batch):     {rob['silhouette_batch']:.4f}")
    print(f"  Batch pred:     {rob['batch_predictability']:.4f}")

    return {
        "product_identification": pid,
        "consistency_scoring": con,
        "batch_robustness": rob,
        "records": val_records,
    }


def evaluate_setting_b(model, proto_store, loader_known, loader_unknown,
                       device, cfg, fold_name=""):
    """Setting B: 开放集 — 已知类 vs 未知类判别。"""
    known_records = collect_predictions(
        model, loader_known, proto_store, device,
        reject_factor=cfg.reject_threshold_factor)
    unknown_records = collect_predictions(
        model, loader_unknown, proto_store, device,
        reject_factor=cfg.reject_threshold_factor)

    osm = open_set_metrics(known_records, unknown_records)

    print(f"\n── Setting B [{fold_name}] ──")
    print(f"  Open-set AUROC:  {osm['open_set_AUROC']:.4f}")
    print(f"  FPR@95TPR:       {osm['FPR_at_95TPR']:.4f}")
    print(f"  Known mean:      {osm['known_score_mean']:.4f}")
    print(f"  Unknown mean:    {osm['unknown_score_mean']:.4f}")

    return {"open_set": osm}


def evaluate_setting_c(model, unknown_dataset, unknown_idx_splits,
                       label_names, device, cfg, fold_name=""):
    """Setting C: 少样本 — N-shot 注册未知类。"""
    results = {}
    print(f"\n── Setting C [{fold_name}] ──")
    for n_shot, split in unknown_idx_splits.items():
        if not split["test_idx"]:
            results[n_shot] = {"accuracy": np.nan, "macro_f1": np.nan}
            continue
        m = few_shot_evaluate(
            model, unknown_dataset, split["ref_idx"], split["test_idx"],
            label_names, device, cfg)
        results[n_shot] = m
        print(f"  {n_shot}-shot: acc={m['accuracy']:.4f}, "
              f"f1={m['macro_f1']:.4f}")
    return {"few_shot": results}


def evaluate_all_settings(fold_results, split_info, cfg):
    """汇总所有 fold 的 Setting A/B/C 评估。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    viz_dir = out_dir / "visualizations"

    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")

    from dataset import GCMSDataset, few_shot_from_unknown

    unknown_idx = split_info["unknown_idx"]
    if unknown_idx:
        unknown_ds = GCMSDataset(metadata_csv, product_col=product_col,
                                 augmentation=None, indices=unknown_idx)
        unknown_loader = torch.utils.data.DataLoader(
            unknown_ds, batch_size=cfg.batch_size, shuffle=False)
        fs_splits = few_shot_from_unknown(
            unknown_idx, metadata_csv, product_col=product_col,
            n_shot_values=cfg.n_shot_values, seed=cfg.seed)
        unknown_label_names = unknown_ds.get_label_name_map()
    else:
        unknown_loader = None
        fs_splits = {}
        unknown_label_names = {}

    all_a, all_b, all_c = [], [], []

    for fr in fold_results:
        model = fr["model"].to(device)
        proto_store = fr["proto_store"]
        fold_name = fr["test_batch"]
        ds_train = fr["ds_train"]

        # 无增强训练 loader (Setting A cross-batch gap)
        train_noaug = GCMSDataset(
            metadata_csv, product_col=product_col,
            augmentation=None, indices=ds_train.df.index.tolist())
        train_noaug.product_enc = ds_train.product_enc
        train_noaug.batch_enc = ds_train.batch_enc
        train_noaug.df["product_label"] = ds_train.product_enc.transform(
            train_noaug.df[product_col])
        train_noaug.df["batch_label"] = ds_train.batch_enc.transform(
            train_noaug.df["batch_idx"])
        loader_train_noaug = torch.utils.data.DataLoader(
            train_noaug, batch_size=cfg.batch_size, shuffle=False)

        # Setting A
        a = evaluate_setting_a(
            model, loader_train_noaug, fr["loader_val"], proto_store,
            device, cfg, fold_name)
        all_a.append(a)

        # Setting B
        if unknown_loader is not None:
            b = evaluate_setting_b(
                model, proto_store, fr["loader_val"], unknown_loader,
                device, cfg, fold_name)
        else:
            b = {"open_set": {"open_set_AUROC": np.nan,
                              "FPR_at_95TPR": np.nan,
                              "known_score_mean": np.nan,
                              "unknown_score_mean": np.nan}}
        all_b.append(b)

        # Setting C
        if unknown_idx and fs_splits:
            c = evaluate_setting_c(
                model, unknown_ds, fs_splits, unknown_label_names,
                device, cfg, fold_name)
        else:
            c = {"few_shot": {}}
        all_c.append(c)

        # 可视化
        if a.get("records"):
            plot_embedding_tsne(
                a["records"],
                viz_dir / f"tsne_product_{fold_name}.png",
                color_by="product")
            plot_embedding_tsne(
                a["records"],
                viz_dir / f"tsne_batch_{fold_name}.png",
                color_by="batch")
            plot_score_distribution(
                a["records"],
                viz_dir / f"score_dist_{fold_name}.png")

    _print_summary(all_a, all_b, all_c, cfg)
    _save_summary(all_a, all_b, all_c, fold_results, out_dir)

    return all_a, all_b, all_c


def _print_summary(all_a, all_b, all_c, cfg):
    """打印跨 fold 汇总。"""
    print(f"\n{'='*60}")
    print("跨批次汇总 (Leave-One-Batch-Out)")
    print(f"{'='*60}")

    accs = [a["product_identification"]["accuracy"] for a in all_a]
    f1s = [a["product_identification"]["macro_f1"] for a in all_a]
    gaps = [a["product_identification"].get("cross_batch_gap", np.nan)
            for a in all_a]
    aurocs_c = [a["consistency_scoring"]["AUROC_correct"] for a in all_a]
    cohens = [a["consistency_scoring"]["cohens_d"] for a in all_a]
    sil_p = [a["batch_robustness"]["silhouette_product"] for a in all_a]
    sil_b = [a["batch_robustness"]["silhouette_batch"] for a in all_a]
    bp = [a["batch_robustness"]["batch_predictability"] for a in all_a]

    print("\nSetting A (闭集跨批次):")
    _p("Accuracy", accs)
    _p("Macro-F1", f1s)
    _p("Cross-batch Δ", gaps)
    _p("Score AUROC", aurocs_c)
    _p("Cohen's d", cohens)
    _p("Sil(product)", sil_p)
    _p("Sil(batch)", sil_b)
    _p("Batch pred", bp)

    aurocs_os = [b["open_set"]["open_set_AUROC"] for b in all_b]
    fprs = [b["open_set"]["FPR_at_95TPR"] for b in all_b]
    print("\nSetting B (开放集):")
    _p("Open-set AUROC", aurocs_os)
    _p("FPR@95TPR", fprs)

    print("\nSetting C (少样本):")
    for n_shot in cfg.n_shot_values:
        n_accs = [c["few_shot"].get(n_shot, {}).get("accuracy", np.nan)
                  for c in all_c]
        _p(f"{n_shot}-shot Acc", n_accs)


def _p(name, values):
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if clean:
        print(f"  {name:20s} {np.mean(clean):.4f} ± {np.std(clean):.4f}")


def _save_summary(all_a, all_b, all_c, fold_results, out_dir):
    """保存评估结果到 JSON。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _s(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary = {"folds": []}
    for i, (a, b, c) in enumerate(zip(all_a, all_b, all_c)):
        fold_data = {
            "fold": fold_results[i]["fold"],
            "test_batch": fold_results[i]["test_batch"],
            "setting_a": {
                k: _s(v) for k, v in a["product_identification"].items()
                if k not in ("confusion", "report")
            },
            "setting_a_consistency": {
                k: _s(v) for k, v in a["consistency_scoring"].items()
            },
            "setting_a_robustness": {
                k: _s(v) for k, v in a["batch_robustness"].items()
            },
            "setting_b": {k: _s(v) for k, v in b["open_set"].items()},
            "setting_c": {
                str(k): {kk: _s(vv) for kk, vv in v.items()}
                for k, v in c["few_shot"].items()
            },
        }
        summary["folds"].append(fold_data)

    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)