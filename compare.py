"""
对比实验运行器:
  统一评估所有方法 (传统 / DL 基线 / 消融 / 本文方法)
  → 生成对比表格 (文本 + LaTeX) + 可视化图表
"""
import copy
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, silhouette_score,
)
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import pi

from config import Config
from dataset import GCMSDataset, leave_one_batch_out_splits
from train import set_seed, build_loaders, run_fold, train_one_epoch, validate
from models import GCMSConsistencyNet
from losses import MetricLearningLoss
from register import register_from_loader

from baselines import (
    extract_features,
    PCAMahalanobis, PLSDABaseline, SVMBaseline, RandomForestBaseline,
    PlainEncoder, BaselineCNN, BaselineLoss,
    train_dl_baseline_fold,
)


# ── 方法注册表 ────────────────────────────────────────────

TRADITIONAL_METHODS = {
    "PCA+Mahalanobis": PCAMahalanobis,
    "PLS-DA": PLSDABaseline,
    "SVM-RBF": SVMBaseline,
    "RandomForest": RandomForestBaseline,
}

DL_BASELINES = ["ResNet-CE", "ResNet-Triplet", "ResNet-Center"]

ABLATIONS = ["Ours-noDualAxis", "Ours-noBatchAdv", "Ours-Softmax"]

PROPOSED = "Ours(Full)"

ALL_METHODS = (
    list(TRADITIONAL_METHODS.keys())
    + DL_BASELINES
    + ABLATIONS
    + [PROPOSED]
)


# ═══════════════════════════════════════════════════════════
#  统一评估
# ═══════════════════════════════════════════════════════════

def evaluate_unified(preds, true_labels, scores, embeddings, batch_labels):
    """
    统一指标计算, 返回 dict 包含:
      accuracy, macro_f1, balanced_acc,
      consistency_auroc, silhouette_product, silhouette_batch,
      batch_predictability
    """
    preds = np.asarray(preds)
    true_labels = np.asarray(true_labels)
    scores = np.asarray(scores)
    embeddings = np.asarray(embeddings)
    batch_labels = np.asarray(batch_labels)

    correct = (preds == true_labels).astype(int)

    m = {
        "accuracy": float(accuracy_score(true_labels, preds)),
        "macro_f1": float(f1_score(true_labels, preds,
                                    average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(true_labels, preds)),
    }

    # 一致性 AUROC (分数能否区分正确/错误预测)
    if len(np.unique(correct)) > 1:
        m["consistency_auroc"] = float(roc_auc_score(correct, scores))
    else:
        m["consistency_auroc"] = float("nan")

    # 产品 Silhouette (嵌入按产品聚类质量, 越高越好)
    uniq_prod = np.unique(true_labels)
    if (len(uniq_prod) > 1
            and len(embeddings) > len(uniq_prod)):
        m["silhouette_product"] = float(silhouette_score(
            embeddings, true_labels,
            sample_size=min(len(embeddings), 2000)))
    else:
        m["silhouette_product"] = float("nan")

    # 批次 Silhouette (嵌入按批次聚类, 越低越好 = 批次不变)
    uniq_batch = np.unique(batch_labels)
    if (len(uniq_batch) > 1
            and len(embeddings) > len(uniq_batch)):
        m["silhouette_batch"] = float(silhouette_score(
            embeddings, batch_labels,
            sample_size=min(len(embeddings), 2000)))
    else:
        m["silhouette_batch"] = float("nan")

    # 批次可预测性 (训练线性分类器预测批次, 越低越好)
    if len(uniq_batch) > 1 and len(embeddings) > 10:
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(embeddings, batch_labels)
        m["batch_predictability"] = float(clf.score(embeddings, batch_labels))
    else:
        m["batch_predictability"] = float("nan")

    return m


# ═══════════════════════════════════════════════════════════
#  收集 DL 预测结果
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def collect_softmax_results(model, loader, device):
    """Softmax 推理 → (preds, scores, embeddings, true_labels, batch_labels)"""
    model.eval()
    all_pred, all_score, all_z, all_y, all_b = [], [], [], [], []

    for batch in loader:
        x = batch["input"].to(device)
        out = model(x)

        probs = torch.softmax(out["logits"], dim=1)
        preds = probs.argmax(dim=1)
        max_probs = probs.max(dim=1).values
        z = out["z"]

        all_pred.append(preds.cpu().numpy())
        all_score.append(max_probs.cpu().numpy())
        all_z.append(z.cpu().numpy())
        all_y.append(batch["product"].numpy())
        all_b.append(batch["batch"].numpy())

    return (np.concatenate(all_pred), np.concatenate(all_score),
            np.concatenate(all_z), np.concatenate(all_y),
            np.concatenate(all_b))


@torch.no_grad()
def collect_proto_results(model, loader, proto_store, device):
    """原型推理 → (preds, scores, embeddings, true_labels, batch_labels)"""
    model.eval()
    all_pred, all_score, all_z, all_y, all_b = [], [], [], [], []

    for batch in loader:
        x = batch["input"].to(device)
        z = model.encode(x)
        result = proto_store.predict(z)

        all_pred.append(result["pred_idx"].cpu().numpy())
        all_score.append(result["scores"].cpu().numpy())
        all_z.append(z.cpu().numpy())
        all_y.append(batch["product"].numpy())
        all_b.append(batch["batch"].numpy())

    return (np.concatenate(all_pred), np.concatenate(all_score),
            np.concatenate(all_z), np.concatenate(all_y),
            np.concatenate(all_b))


# ═══════════════════════════════════════════════════════════
#  方法运行器: 传统方法
# ═══════════════════════════════════════════════════════════

def _run_traditional_fold(method_name, train_idx, val_idx,
                          metadata_csv, cfg):
    """传统方法单 fold: 特征提取 → 拟合 → 预测 → 评估。"""
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")

    ds_train = GCMSDataset(metadata_csv, product_col=product_col,
                           augmentation=None, indices=train_idx)
    ds_val = GCMSDataset(metadata_csv, product_col=product_col,
                         augmentation=None, indices=val_idx)
    ds_val.product_enc = ds_train.product_enc
    ds_val.batch_enc = ds_train.batch_enc
    ds_val.df["product_label"] = ds_train.product_enc.transform(
        ds_val.df[product_col])
    ds_val.df["batch_label"] = ds_train.batch_enc.transform(
        ds_val.df["batch_idx"])

    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size,
                              shuffle=False, num_workers=0)
    loader_val = DataLoader(ds_val, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=0)

    X_train, y_train, _ = extract_features(loader_train)
    X_val, y_val, b_val = extract_features(loader_val)

    model_cls = TRADITIONAL_METHODS[method_name]
    model = model_cls()
    model.fit(X_train, y_train)
    preds, scores = model.predict(X_val)
    embeddings = model.get_embeddings(X_val)

    return evaluate_unified(preds, y_val, scores, embeddings, b_val)


# ═══════════════════════════════════════════════════════════
#  方法运行器: DL 基线
# ═══════════════════════════════════════════════════════════

def _build_proto_store(model, ds_train, train_idx, metadata_csv,
                       cfg, device):
    """从训练集构建原型库 (供 Triplet/Center 基线推理用)。"""
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")
    ds_noaug = GCMSDataset(metadata_csv, product_col=product_col,
                           augmentation=None, indices=train_idx)
    ds_noaug.product_enc = ds_train.product_enc
    ds_noaug.batch_enc = ds_train.batch_enc
    ds_noaug.df["product_label"] = ds_train.product_enc.transform(
        ds_noaug.df[product_col])
    ds_noaug.df["batch_label"] = ds_train.batch_enc.transform(
        ds_noaug.df["batch_idx"])
    loader_reg = DataLoader(ds_noaug, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=0)
    label_names = ds_train.get_label_name_map()
    proto_store, _, _ = register_from_loader(
        model, loader_reg, label_names, device,
        percentile=cfg.accept_percentile)
    return proto_store


def _run_dl_baseline_fold(method_name, train_idx, val_idx, batch_name,
                          metadata_csv, cfg):
    """DL 基线单 fold: 训练 → 推理 → 评估。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ds_train, ds_val, loader_val = train_dl_baseline_fold(
        method_name, train_idx, val_idx, batch_name, metadata_csv, cfg
    )

    if method_name == "ResNet-CE":
        preds, scores, zs, ys, bs = collect_softmax_results(
            model, loader_val, device)
    else:
        proto_store = _build_proto_store(
            model, ds_train, train_idx, metadata_csv, cfg, device)
        preds, scores, zs, ys, bs = collect_proto_results(
            model, loader_val, proto_store, device)

    return evaluate_unified(preds, ys, scores, zs, bs)


# ═══════════════════════════════════════════════════════════
#  方法运行器: 消融变体
# ═══════════════════════════════════════════════════════════

def _train_no_dualaxis_fold(fold_idx, train_idx, val_idx, batch_name,
                            metadata_csv, cfg):
    """消融: 完整度量学习管道 + PlainEncoder (无双轴注意力)。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")

    ds_train, ds_val, loader_train, loader_val = build_loaders(
        metadata_csv, train_idx, val_idx, cfg, product_col)

    num_products = ds_train.num_products
    num_batches = ds_train.num_batches
    print(f"    [noDualAxis] 产品={num_products}, 批次={num_batches}")

    # 创建标准模型后替换编码器
    model = GCMSConsistencyNet(num_products, num_batches, cfg).to(device)
    model.encoder = PlainEncoder(
        in_channels=cfg.in_channels,
        channels=cfg.encoder_channels,
        dropout=cfg.dropout,
        blocks_per_stage=cfg.blocks_per_stage,
    ).to(device)

    criterion = MetricLearningLoss(cfg).to(device)

    # Phase 1: 重建预训练
    opt1 = torch.optim.AdamW(model.parameters(), lr=cfg.lr_pretrain,
                             weight_decay=cfg.weight_decay)
    for epoch in range(cfg.epochs_pretrain):
        train_one_epoch(model, loader_train, criterion, opt1, device,
                        "pretrain", epoch, cfg.epochs_pretrain)
        if (epoch + 1) % 40 == 0:
            print(f"      Pretrain {epoch+1}/{cfg.epochs_pretrain}")

    # Phase 2: 度量学习
    opt2 = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": cfg.lr_finetune * 0.1},
         {"params": model.proj_head.parameters()},
         {"params": model.product_head.parameters()},
         {"params": model.domain_head.parameters()},
         {"params": model.decoder.parameters(), "lr": cfg.lr_finetune * 0.5}],
        lr=cfg.lr_finetune, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=cfg.epochs_finetune)

    best_acc = 0
    best_state = None
    for epoch in range(cfg.epochs_finetune):
        train_one_epoch(model, loader_train, criterion, opt2, device,
                        "finetune", epoch, cfg.epochs_finetune)
        m_val = validate(model, loader_val, criterion, device)
        scheduler.step()
        if m_val["acc"] > best_acc:
            best_acc = m_val["acc"]
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        if (epoch + 1) % 40 == 0:
            print(f"      Finetune {epoch+1}/{cfg.epochs_finetune} "
                  f"val_acc={m_val['acc']:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    # Phase 3: 原型注册
    proto_store = _build_proto_store(
        model, ds_train, train_idx, metadata_csv, cfg, device)

    return model, proto_store, ds_train, ds_val, loader_val


def _run_ablation_fold(ablation_name, fold_idx, train_idx, val_idx,
                       batch_name, metadata_csv, cfg):
    """消融变体单 fold。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ablation_name == "Ours-noBatchAdv":
        ab_cfg = copy.deepcopy(cfg)
        ab_cfg.lambda_adv = 0.0
        model, proto_store, ds_train, ds_val, loader_val = run_fold(
            fold_idx, train_idx, val_idx, batch_name, metadata_csv, ab_cfg)
        preds, scores, zs, ys, bs = collect_proto_results(
            model, loader_val, proto_store, device)

    elif ablation_name == "Ours-noDualAxis":
        model, proto_store, ds_train, ds_val, loader_val = \
            _train_no_dualaxis_fold(
                fold_idx, train_idx, val_idx, batch_name,
                metadata_csv, cfg)
        preds, scores, zs, ys, bs = collect_proto_results(
            model, loader_val, proto_store, device)

    elif ablation_name == "Ours-Softmax":
        model, proto_store, ds_train, ds_val, loader_val = run_fold(
            fold_idx, train_idx, val_idx, batch_name, metadata_csv, cfg)
        preds, scores, zs, ys, bs = collect_softmax_results(
            model, loader_val, device)

    else:
        raise ValueError(f"未知消融: {ablation_name}")

    return evaluate_unified(preds, ys, scores, zs, bs)


# ═══════════════════════════════════════════════════════════
#  方法运行器: 本文方法
# ═══════════════════════════════════════════════════════════

def _run_proposed_fold(fold_idx, train_idx, val_idx, batch_name,
                       metadata_csv, cfg):
    """本文完整方法单 fold。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, proto_store, ds_train, ds_val, loader_val = run_fold(
        fold_idx, train_idx, val_idx, batch_name, metadata_csv, cfg)
    preds, scores, zs, ys, bs = collect_proto_results(
        model, loader_val, proto_store, device)
    return evaluate_unified(preds, ys, scores, zs, bs)


# ═══════════════════════════════════════════════════════════
#  聚合
# ═══════════════════════════════════════════════════════════

def aggregate_fold_metrics(fold_metrics):
    """聚合多 fold 指标 → mean ± std。"""
    keys = fold_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics
                if not np.isnan(m.get(k, float("nan")))]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        else:
            agg[f"{k}_mean"] = float("nan")
            agg[f"{k}_std"] = float("nan")
    return agg


# ═══════════════════════════════════════════════════════════
#  结果格式化
# ═══════════════════════════════════════════════════════════

METRICS_DISPLAY = [
    # (key,           显示名,           越大越好?)
    ("accuracy",           "Accuracy↑",      True),
    ("macro_f1",           "Macro-F1↑",      True),
    ("consistency_auroc",  "Con.AUROC↑",     True),
    ("silhouette_product", "Sil(Prod)↑",     True),
    ("silhouette_batch",   "Sil(Batch)↓",    False),
    ("batch_predictability", "BatchPred↓",   False),
]


def _find_best(results):
    """找到每个指标的最优方法。"""
    best = {}
    for key, _, higher_better in METRICS_DISPLAY:
        vals = {}
        for m, r in results.items():
            v = r.get(f"{key}_mean", float("nan"))
            if not np.isnan(v):
                vals[m] = v
        if vals:
            best[key] = (max(vals, key=vals.get) if higher_better
                         else min(vals, key=vals.get))
    return best


def generate_comparison_table(results, save_path=None):
    """打印并保存文本对比表格。"""
    best = _find_best(results)

    col_w = 16
    header = f"{'Method':<24}"
    for _, disp, _ in METRICS_DISPLAY:
        header += f" | {disp:>{col_w}}"
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for method in ALL_METHODS:
        if method not in results:
            continue
        r = results[method]
        row = f"{method:<24}"
        for key, _, _ in METRICS_DISPLAY:
            mean = r.get(f"{key}_mean", float("nan"))
            std = r.get(f"{key}_std", float("nan"))
            if np.isnan(mean):
                cell = "N/A"
            else:
                cell = f"{mean:.4f}±{std:.4f}"
                if best.get(key) == method:
                    cell = f"*{cell}*"
            row += f" | {cell:>{col_w}}"
        lines.append(row)
    lines.append(sep)

    text = "\n".join(lines)
    print("\n" + text)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(text)
    return text


def generate_latex_table(results, save_path):
    """生成 LaTeX 对比表格。"""
    best = _find_best(results)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{跨批次一致性检测方法对比}",
        r"\label{tab:comparison}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l" + "c" * len(METRICS_DISPLAY) + "}",
        r"\toprule",
    ]

    header = "Method"
    for _, disp, _ in METRICS_DISPLAY:
        header += f" & {disp}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for method in ALL_METHODS:
        if method not in results:
            continue
        r = results[method]
        row = method.replace("_", r"\_")
        for key, _, _ in METRICS_DISPLAY:
            mean = r.get(f"{key}_mean", float("nan"))
            std = r.get(f"{key}_std", float("nan"))
            if np.isnan(mean):
                row += " & N/A"
            else:
                cell = f"{mean:.3f}$\\pm${std:.3f}"
                if best.get(key) == method:
                    cell = r"\textbf{" + cell + "}"
                row += f" & {cell}"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])

    text = "\n".join(lines)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(text)
    return text


# ═══════════════════════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════════════════════

# 方法分类颜色
_METHOD_COLORS = {}
for m in TRADITIONAL_METHODS:
    _METHOD_COLORS[m] = "#3498db"       # 蓝: 传统
for m in DL_BASELINES:
    _METHOD_COLORS[m] = "#2ecc71"       # 绿: DL 基线
for m in ABLATIONS:
    _METHOD_COLORS[m] = "#f39c12"       # 橙: 消融
_METHOD_COLORS[PROPOSED] = "#e74c3c"    # 红: 本文方法


def plot_comparison_bar(results, save_dir):
    """分组柱状图: 每个指标一张图。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in ALL_METHODS if m in results]
    if len(methods) < 2:
        return

    chart_items = [
        ("accuracy", "Accuracy", "产品识别准确率对比"),
        ("macro_f1", "Macro-F1", "Macro-F1 对比"),
        ("consistency_auroc", "AUROC", "一致性评分 AUROC 对比"),
        ("silhouette_product", "Silhouette", "产品聚类质量 Silhouette 对比"),
    ]

    for key, ylabel, title in chart_items:
        means = [results[m].get(f"{key}_mean", float("nan"))
                 for m in methods]
        stds = [results[m].get(f"{key}_std", 0) for m in methods]
        colors = [_METHOD_COLORS.get(m, "#95a5a6") for m in methods]

        fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.1), 6))
        x_pos = np.arange(len(methods))
        ax.bar(x_pos, means, yerr=stds, capsize=4,
               color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

        # 图例
        handles = [
            plt.Rectangle((0, 0), 1, 1, fc="#3498db", label="传统方法"),
            plt.Rectangle((0, 0), 1, 1, fc="#2ecc71", label="DL 基线"),
            plt.Rectangle((0, 0), 1, 1, fc="#f39c12", label="消融变体"),
            plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", label="本文方法"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8)
        plt.tight_layout()
        fig.savefig(save_dir / f"compare_{key}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_radar_chart(results, save_path):
    """雷达图: 多维度方法对比。"""
    metrics = ["accuracy", "macro_f1", "consistency_auroc",
               "silhouette_product"]
    labels = ["Accuracy", "Macro-F1", "Con. AUROC", "Sil(Product)"]

    methods = [m for m in ALL_METHODS if m in results]
    if len(methods) < 2:
        return

    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for method in methods:
        r = results[method]
        values = [r.get(f"{m}_mean", 0) for m in metrics]
        values += values[:1]
        color = _METHOD_COLORS.get(method, "#95a5a6")
        lw = 2.5 if method == PROPOSED else 1.0
        alpha = 1.0 if method == PROPOSED else 0.6
        ax.plot(angles, values, linewidth=lw, label=method,
                color=color, alpha=alpha)
        if method == PROPOSED:
            ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    ax.set_title("方法多维度对比", pad=20)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_batch_robustness(results, save_path):
    """批次鲁棒性对比: Silhouette(batch) + BatchPredictability 越低越好。"""
    methods = [m for m in ALL_METHODS if m in results]
    if len(methods) < 2:
        return

    sil_batch = [results[m].get("silhouette_batch_mean", float("nan"))
                 for m in methods]
    batch_pred = [results[m].get("batch_predictability_mean", float("nan"))
                  for m in methods]
    colors = [_METHOD_COLORS.get(m, "#95a5a6") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Silhouette(batch) - lower is better
    ax = axes[0]
    x = np.arange(len(methods))
    ax.bar(x, sil_batch, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Silhouette (Batch)")
    ax.set_title("批次聚类度 (越低越好)")
    ax.grid(axis="y", alpha=0.3)

    # Batch predictability - lower is better
    ax = axes[1]
    ax.bar(x, batch_pred, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Batch Predictability")
    ax.set_title("批次可预测性 (越低越好)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════

def run_comparison(cfg, methods=None):
    """
    运行对比实验。

    Parameters
    ----------
    cfg : Config
    methods : list[str] or None
        指定运行的方法, 默认全部。
    """
    set_seed(cfg.seed)

    if methods is None:
        methods = list(ALL_METHODS)

    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    splits = leave_one_batch_out_splits(metadata_csv)

    results = {}
    timing = {}

    for method in methods:
        if method not in ALL_METHODS:
            print(f"\n  [跳过] 未知方法: {method}")
            continue

        print(f"\n{'='*60}")
        print(f"  方法: {method}")
        print(f"{'='*60}")

        t0 = time.time()
        fold_metrics = []

        for fold_idx, (train_idx, val_idx, bname) in enumerate(splits):
            print(f"\n  Fold {fold_idx}: test_batch={bname}")

            try:
                if method in TRADITIONAL_METHODS:
                    m = _run_traditional_fold(
                        method, train_idx, val_idx, metadata_csv, cfg)
                elif method in DL_BASELINES:
                    m = _run_dl_baseline_fold(
                        method, train_idx, val_idx, bname,
                        metadata_csv, cfg)
                elif method in ABLATIONS:
                    m = _run_ablation_fold(
                        method, fold_idx, train_idx, val_idx,
                        bname, metadata_csv, cfg)
                elif method == PROPOSED:
                    m = _run_proposed_fold(
                        fold_idx, train_idx, val_idx, bname,
                        metadata_csv, cfg)
                else:
                    raise ValueError(f"方法未注册: {method}")

                fold_metrics.append(m)
                print(f"    Acc={m['accuracy']:.4f}, "
                      f"F1={m['macro_f1']:.4f}, "
                      f"AUROC={m.get('consistency_auroc', float('nan')):.4f}")

            except Exception as e:
                print(f"    错误: {e}")
                traceback.print_exc()

        elapsed = time.time() - t0
        timing[method] = elapsed

        if fold_metrics:
            results[method] = aggregate_fold_metrics(fold_metrics)
            print(f"\n  {method} 综合: "
                  f"Acc={results[method]['accuracy_mean']:.4f}±"
                  f"{results[method]['accuracy_std']:.4f}  "
                  f"耗时={elapsed:.1f}s")

    # ── 保存结果 ──
    out_dir = Path(cfg.output_dir) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    # ── 生成报告 ──
    if len(results) >= 2:
        generate_comparison_table(results,
                                  out_dir / "comparison_table.txt")
        generate_latex_table(results,
                             out_dir / "comparison_table.tex")
        plot_comparison_bar(results, out_dir)
        plot_batch_robustness(results,
                              out_dir / "batch_robustness.png")
    if len(results) >= 3:
        plot_radar_chart(results, out_dir / "radar_chart.png")

    print(f"\n{'='*60}")
    print(f"对比结果已保存到 {out_dir}")
    print(f"{'='*60}")

    return results
