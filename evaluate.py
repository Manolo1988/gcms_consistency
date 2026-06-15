"""统一评估 (五维度指标体系):
  1. 产品识别 (Setting A): Macro-F1, Accuracy, 跨批次 Δ
  2. 开放集 + 新品 (Setting B & C): AUROC, FPR@95TPR, N-shot Accuracy
  3. 一致性评分 (Setting A): AUROC(correct/wrong), Cohen's d
  4. 批次鲁棒性 (Setting A): 批次可预测性, Silhouette
  5. 可解释性: Grad-CAM (定性)
"""
import os
import numpy as np
import torch
from tqdm import tqdm
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


def _progress_enabled():
    """控制是否显示 tqdm 进度条。默认关闭，避免污染日志。"""
    return os.environ.get("GCMS_SHOW_PROGRESS", "0") == "1"


def _resolve_product_label_name_map(loader):
    """从 DataLoader/Subset 链路解析 product_label -> class_name 映射。"""
    ds = getattr(loader, "dataset", None)
    while ds is not None:
        enc = getattr(ds, "product_enc", None)
        if enc is not None and hasattr(enc, "classes_"):
            classes = list(enc.classes_)
            return {int(i): str(name) for i, name in enumerate(classes)}
        ds = getattr(ds, "dataset", None)
    return {}


def _batch_tic(batch, device):
    tic = batch.get("tic") if isinstance(batch, dict) else None
    return tic.to(device) if torch.is_tensor(tic) else None


# ═══════════════════════════════════════════════════════════
#  核心收集函数
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def collect_embeddings(model, loader, device):
    """收集所有样本的嵌入和元数据。"""
    model.eval()
    records = []

    for batch in tqdm(
        loader,
        desc="收集嵌入",
        leave=False,
        ncols=80,
        disable=not _progress_enabled(),
    ):
        x = batch["input"].to(device)
        z = model.encode(x, tic=_batch_tic(batch, device))

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
    label_name_map = _resolve_product_label_name_map(loader)

    for batch in tqdm(
        loader,
        desc="收集预测",
        leave=False,
        ncols=80,
        disable=not _progress_enabled(),
    ):
        x = batch["input"].to(device)
        z = model.encode(x, tic=_batch_tic(batch, device))
        result = proto_store.predict(z)

        for i in range(x.size(0)):
            pred_idx = result["pred_idx"][i].item()
            score = result["scores"][i].item()
            min_dist = result["min_dists"][i].item()
            true_label = batch["product"][i].item()
            true_class_name = label_name_map.get(int(true_label), str(int(true_label)))
            pred_class_name = str(result["pred_class"][i])

            # 拒识判定
            is_known = proto_store.is_known(
                result["min_dists"][i:i+1],
                factor=reject_factor,
                pred_idx=result["pred_idx"][i:i+1],
                use_spherical=bool(result.get("use_spherical", True)),
            )[0]

            records.append({
                "sample_id": batch["sample_id"][i] if isinstance(batch["sample_id"], list) else batch["sample_id"],
                "true_product": true_label,
                "true_class": true_class_name,
                "pred_product": pred_idx,
                "pred_class": pred_class_name,
                "consistency_score": score,
                "open_set_score": result["scores"][i].item(),
                "margin_score": result["margin_scores"][i].item(),
                "min_dist": min_dist,
                "is_known": bool(is_known),
                "correct": pred_class_name == true_class_name,
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
    use_class_names = all(
        ("true_class" in r and "pred_class" in r)
        for r in records
    )
    if use_class_names:
        y_true = [str(r["true_class"]) for r in records]
        y_pred = [str(r["pred_class"]) for r in records]
    else:
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
        train_use_class_names = all(
            ("true_class" in r and "pred_class" in r)
            for r in train_records
        )
        if train_use_class_names:
            y_true_train = [str(r["true_class"]) for r in train_records]
            y_pred_train = [str(r["pred_class"]) for r in train_records]
        else:
            y_true_train = [r["true_product"] for r in train_records]
            y_pred_train = [r["pred_product"] for r in train_records]
        acc_same = accuracy_score(y_true_train, y_pred_train)
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
                                 random_state=42)
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
    返回 AUROC, FPR@95TPR, EER, TPR@FPR5, TPR@FPR10。
    """
    known_scores = np.array([
        r.get("open_set_score", r.get("consistency_score", 0.0))
        for r in known_records
    ])
    unknown_scores = np.array([
        r.get("open_set_score", r.get("consistency_score", 0.0))
        for r in unknown_records
    ])

    from open_score_calibration import evaluate_calibration
    cal_metrics = evaluate_calibration(known_scores, unknown_scores)

    result = {
        "open_set_AUROC": cal_metrics["AUROC"],
        "FPR_at_95TPR": cal_metrics["FPR_at_95TPR"],
        "EER": cal_metrics["EER"],
        "TPR_at_FPR5": cal_metrics["TPR_at_FPR5"],
        "TPR_at_FPR10": cal_metrics["TPR_at_FPR10"],
        "known_score_mean": cal_metrics["known_score_mean"],
        "unknown_score_mean": cal_metrics["unknown_score_mean"],
        "AUPR": cal_metrics.get("AUPR", float("nan")),
    }
    return result


def _records_from_arrays(preds, scores, embeddings, true_labels, batch_labels,
                         is_known=True):
    """将数组结果组装成统一 records 结构。"""
    records = []
    n = len(true_labels)
    for i in range(n):
        pred_i = int(preds[i]) if len(preds) > i else -1
        true_i = int(true_labels[i])
        score_i = float(scores[i]) if len(scores) > i else 0.0
        z_i = embeddings[i] if len(embeddings) > i else np.zeros(1, dtype=np.float32)
        batch_i = int(batch_labels[i]) if len(batch_labels) > i else -1

        records.append({
            "sample_id": f"baseline_{i}",
            "true_product": true_i,
            "pred_product": pred_i,
            "pred_class": str(pred_i),
            "consistency_score": score_i,
            "open_set_score": score_i,
            "margin_score": score_i,
            "min_dist": float(1.0 - score_i),
            "is_known": bool(is_known),
            "correct": pred_i == true_i,
            "z": np.asarray(z_i),
            "batch_label": batch_i,
        })
    return records


README_BASELINE_ORDER = [
    "pca_mahalanobis",
    "pls_da",
    "svm_rbf",
    "tic_pca_mlp",
]

README_BASELINE_SPECS = [
    {
        "key": "pca_mahalanobis",
        "name": "PCA+Mahalanobis",
        "feature_mode": "raw",
    },
    {
        "key": "pls_da",
        "name": "PLS-DA",
        "feature_mode": "raw",
    },
    {
        "key": "svm_rbf",
        "name": "SVM-RBF",
        "feature_mode": "raw",
    },
    {
        "key": "tic_pca_mlp",
        "name": "TIC+PCA+MLP",
        "feature_mode": "tic",
    },
]


def _resolve_feature_mode_label(feature_mode, cfg):
    if feature_mode == "raw":
        return "raw"
    if feature_mode in ("pretrained", "pretrained_raw"):
        arch = str(getattr(cfg, "pretrained_feature_arch", "auto") or "auto")
        layers = str(getattr(cfg, "pretrained_feature_layers", "layer4") or "layer4")
        fuse = str(getattr(cfg, "pretrained_feature_fuse", "concat") or "concat")
        return f"pretrained:{arch}:{layers}:{fuse}"
    return feature_mode


def _extract_with_mode(loader, feature_mode, cfg):
    from baselines import (
        extract_features,
        extract_tic_features,
        extract_pretrained_resnet_features,
    )

    if feature_mode == "tic":
        return extract_tic_features(loader)

    if feature_mode == "raw":
        return extract_features(loader)

    if feature_mode not in ("pretrained", "pretrained_raw"):
        raise ValueError(f"未知特征模式: {feature_mode}")

    model_path = str(getattr(cfg, "pretrained_feature_model", "") or "").strip()
    if not model_path:
        raise ValueError(
            f"feature_mode={feature_mode} 需要配置 pretrained_feature_model"
        )
    arch = str(getattr(cfg, "pretrained_feature_arch", "auto") or "auto")
    layers = str(getattr(cfg, "pretrained_feature_layers", "layer4") or "layer4")
    fuse = str(getattr(cfg, "pretrained_feature_fuse", "concat") or "concat")
    return extract_pretrained_resnet_features(
        loader,
        weight_path=model_path,
        arch=arch,
        layers=layers,
        fuse=fuse,
    )


def _build_baseline_feature_cache(split, cfg, make_loader,
                                  metadata_csv, product_col, feature_modes):
    from torch.utils.data import DataLoader, Subset
    from dataset import GCMSDataset, few_shot_from_unknown

    cache = {
        "train": {},
        "setting_a": {},
        "setting_b": {},
        "setting_c": {},
    }

    _, loader_train = make_loader(split["train_idx"])
    for mode in feature_modes:
        cache["train"][mode] = _extract_with_mode(loader_train, mode, cfg)

    if split.get("test_batch_idx"):
        _, loader_test = make_loader(split["test_batch_idx"])
        for mode in feature_modes:
            cache["setting_a"][mode] = {
                "train": cache["train"][mode],
                "test": _extract_with_mode(loader_test, mode, cfg),
            }

    if split.get("test_unknown_idx"):
        known_idx = sorted(set(split.get("val_idx", []))
                           | set(split.get("test_batch_idx", [])))
        _, loader_known = make_loader(known_idx)
        _, loader_unknown = make_loader(split["test_unknown_idx"])
        for mode in feature_modes:
            cache["setting_b"][mode] = {
                "known": _extract_with_mode(loader_known, mode, cfg),
                "unknown": _extract_with_mode(loader_unknown, mode, cfg),
            }

        unknown_idx = split["test_unknown_idx"]
        ds_unknown = GCMSDataset(
            metadata_csv,
            product_col=product_col,
            augmentation=None,
            indices=unknown_idx,
            cfg=cfg,
        )
        fs_splits = few_shot_from_unknown(
            unknown_idx,
            metadata_csv,
            product_col=product_col,
            n_shot_values=cfg.n_shot_values,
            seed=cfg.seed,
        )
        for n_shot, fs in fs_splits.items():
            ref_idx = fs.get("ref_idx", [])
            test_idx = fs.get("test_idx", [])
            cache["setting_c"][str(n_shot)] = {
                "n_ref": int(len(ref_idx)),
                "n_test": int(len(test_idx)),
                "modes": {},
            }
            if not ref_idx or not test_idx:
                continue

            ref_loader = DataLoader(
                Subset(ds_unknown, ref_idx),
                batch_size=cfg.batch_size,
                shuffle=False,
            )
            test_loader = DataLoader(
                Subset(ds_unknown, test_idx),
                batch_size=cfg.batch_size,
                shuffle=False,
            )
            for mode in feature_modes:
                cache["setting_c"][str(n_shot)]["modes"][mode] = {
                    "ref": _extract_with_mode(ref_loader, mode, cfg),
                    "test": _extract_with_mode(test_loader, mode, cfg),
                }

    return cache


def _build_main_vs_baseline(result_a, result_b, result_c, baseline_result):
    """构建主模型相对 baseline 的差值(main - baseline)。"""

    def _delta(main_v, base_v):
        if main_v is None or base_v is None:
            return None
        try:
            return float(main_v) - float(base_v)
        except Exception:
            return None

    out = {}

    if result_a and baseline_result and baseline_result.get("setting_a"):
        pid = result_a.get("product_identification", {})
        ba = baseline_result.get("setting_a", {})
        out["setting_a"] = {
            "accuracy": _delta(pid.get("accuracy"), ba.get("accuracy")),
            "macro_f1": _delta(pid.get("macro_f1"), ba.get("macro_f1")),
        }

    if result_b and baseline_result and baseline_result.get("setting_b"):
        sb = result_b.get("open_set", {})
        bb = baseline_result.get("setting_b", {})
        out["setting_b"] = {
            "open_set_AUROC": _delta(sb.get("open_set_AUROC"),
                                       bb.get("open_set_AUROC")),
            "FPR_at_95TPR": _delta(sb.get("FPR_at_95TPR"),
                                    bb.get("FPR_at_95TPR")),
        }

    if result_c and baseline_result and baseline_result.get("setting_c"):
        out_c = {}
        main_c = result_c.get("few_shot", {})
        base_c = baseline_result.get("setting_c", {})
        n_values = sorted(
            set([str(k) for k in main_c.keys()]) | set([str(k) for k in base_c.keys()]),
            key=lambda x: int(x),
        )
        for n_str in n_values:
            main_m = main_c.get(int(n_str), {}) if n_str.isdigit() else {}
            base_m = base_c.get(n_str, {})
            out_c[n_str] = {
                "accuracy": _delta(main_m.get("accuracy"), base_m.get("accuracy")),
                "macro_f1": _delta(main_m.get("macro_f1"), base_m.get("macro_f1")),
            }
        out["setting_c"] = out_c

    return out


def _build_main_vs_readme_baselines(result_a, result_b, result_c,
                                    baselines_readme):
    out = {}
    for key in README_BASELINE_ORDER:
        baseline_result = baselines_readme.get(key)
        if not baseline_result:
            continue
        out[key] = _build_main_vs_baseline(
            result_a,
            result_b,
            result_c,
            baseline_result,
        )
    return out


def _evaluate_baseline_with_cache(model_cls, feature_mode, cache):
    result = {
        "setting_a": None,
        "setting_b": None,
        "setting_c": {},
    }

    train_pack = cache.get("train", {}).get(feature_mode)
    if train_pack is None:
        return result

    x_train, y_train, b_train = train_pack
    model_ab = None

    if cache.get("setting_a", {}).get(feature_mode):
        x_test, y_test, b_test = cache["setting_a"][feature_mode]["test"]

        model_ab = model_cls()
        model_ab.fit(x_train, y_train)

        pred_train, score_train = model_ab.predict(x_train)
        z_train = model_ab.get_embeddings(x_train)
        train_records = _records_from_arrays(
            pred_train, score_train, z_train, y_train, b_train,
            is_known=True,
        )

        pred_test, score_test = model_ab.predict(x_test)
        z_test = model_ab.get_embeddings(x_test)
        test_records = _records_from_arrays(
            pred_test, score_test, z_test, y_test, b_test,
            is_known=True,
        )

        pid = product_identification_metrics(test_records, train_records)
        con = consistency_scoring_metrics(test_records)
        rob = batch_robustness_metrics(test_records)

        result["setting_a"] = {
            "accuracy": float(pid.get("accuracy", np.nan)),
            "macro_f1": float(pid.get("macro_f1", np.nan)),
            "balanced_acc": float(pid.get("balanced_acc", np.nan)),
            "cross_batch_gap": float(pid.get("cross_batch_gap", np.nan)),
            "AUROC_correct": float(con.get("AUROC_correct", np.nan)),
            "cohens_d": float(con.get("cohens_d", np.nan)),
            "silhouette_product": float(rob.get("silhouette_product", np.nan)),
            "silhouette_batch": float(rob.get("silhouette_batch", np.nan)),
            "batch_predictability": float(rob.get("batch_predictability", np.nan)),
        }

    if cache.get("setting_b", {}).get(feature_mode):
        if model_ab is None:
            model_ab = model_cls()
            model_ab.fit(x_train, y_train)

        x_known, y_known, b_known = cache["setting_b"][feature_mode]["known"]
        x_unknown, y_unknown, b_unknown = cache["setting_b"][feature_mode]["unknown"]

        pred_k, score_k = model_ab.predict(x_known)
        z_k = model_ab.get_embeddings(x_known)
        known_records = _records_from_arrays(
            pred_k, score_k, z_k, y_known, b_known, is_known=True
        )

        pred_u, score_u = model_ab.predict(x_unknown)
        z_u = model_ab.get_embeddings(x_unknown)
        unknown_records = _records_from_arrays(
            pred_u, score_u, z_u, y_unknown, b_unknown, is_known=False
        )

        result["setting_b"] = open_set_metrics(known_records, unknown_records)

    for n_str, block in cache.get("setting_c", {}).items():
        n_ref = int(block.get("n_ref", 0))
        n_test = int(block.get("n_test", 0))
        mode_block = block.get("modes", {}).get(feature_mode)
        if not mode_block:
            result["setting_c"][n_str] = {
                "accuracy": float(np.nan),
                "macro_f1": float(np.nan),
                "n_ref": n_ref,
                "n_test": n_test,
            }
            continue

        x_ref, y_ref, _ = mode_block["ref"]
        x_test, y_test, _ = mode_block["test"]
        fs_model = model_cls()
        fs_model.fit(x_ref, y_ref)
        pred_t, _ = fs_model.predict(x_test)
        result["setting_c"][n_str] = {
            "accuracy": float(accuracy_score(y_test, pred_t)),
            "macro_f1": float(f1_score(y_test, pred_t,
                                        average="macro", zero_division=0)),
            "n_ref": n_ref,
            "n_test": int(len(y_test)),
        }

    return result


def evaluate_readme_baselines(split, cfg, make_loader,
                              metadata_csv, product_col):
    from baselines import (
        PCAMahalanobis,
        PLSDABaseline,
        SVMBaseline,
        TICPcaMLPBaseline,
    )

    cls_map = {
        "pca_mahalanobis": PCAMahalanobis,
        "pls_da": PLSDABaseline,
        "svm_rbf": SVMBaseline,
        "tic_pca_mlp": TICPcaMLPBaseline,
    }
    modes = sorted({spec["feature_mode"] for spec in README_BASELINE_SPECS})
    cache = _build_baseline_feature_cache(
        split,
        cfg,
        make_loader,
        metadata_csv,
        product_col,
        modes,
    )

    out = {}
    for spec in README_BASELINE_SPECS:
        key = spec["key"]
        feature_mode = spec["feature_mode"]
        baseline_result = _evaluate_baseline_with_cache(
            cls_map[key],
            feature_mode,
            cache,
        )
        baseline_result["name"] = spec["name"]
        baseline_result["feature_mode"] = _resolve_feature_mode_label(
            feature_mode,
            cfg,
        )
        out[key] = baseline_result

    return out


def evaluate_tic_pca_mlp_baseline(split, cfg, make_loader,
                                  metadata_csv, product_col):
    """兼容旧接口: 返回 TIC+PCA+MLP 在 A/B/C 的结果。"""
    all_results = evaluate_readme_baselines(
        split,
        cfg,
        make_loader,
        metadata_csv,
        product_col,
    )
    return all_results.get(
        "tic_pca_mlp",
        {
            "setting_a": None,
            "setting_b": None,
            "setting_c": {},
            "name": "TIC+PCA+MLP",
            "feature_mode": "tic",
        },
    )


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

    cls_m = product_identification_metrics(records)
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


def fit_open_score_calibrator_pseudo_unknown(model, cfg, split, make_loader,
                                             metadata_csv, product_col,
                                             save_dir=None):
    """Fit calibration on known-class pseudo unknowns only; never uses Setting B unknowns."""
    from open_score_calibration import OpenScoreCalibrator
    from register import register_from_loader
    import pandas as pd

    rng = np.random.RandomState(int(getattr(cfg, "open_score_calibration_seed", 42)))
    train_df = pd.read_csv(metadata_csv)
    train_df = train_df[(train_df["product_fine"] != "BLANK")
                        & (~train_df["is_special"])].reset_index(drop=True)
    train_df = train_df.iloc[split["train_idx"]]
    products = sorted(train_df[product_col].unique().tolist())
    n_holdout = int(getattr(cfg, "open_score_calibration_holdout_products", 1) or 1)
    n_holdout = min(max(n_holdout, 1), max(len(products) - 1, 1))
    if len(products) <= 1:
        raise ValueError("校准失败: 已知训练产品数不足，无法构造伪未知")

    eligible = []
    for p in products:
        n = int((train_df[product_col] == p).sum())
        if n >= 5:
            eligible.append(p)
    if len(eligible) < n_holdout:
        eligible = products
    pseudo_products = sorted(rng.choice(eligible, size=n_holdout, replace=False).tolist())

    known_idx = train_df.index[~train_df[product_col].isin(pseudo_products)].tolist()
    pseudo_idx = train_df.index[train_df[product_col].isin(pseudo_products)].tolist()
    if not known_idx or not pseudo_idx:
        raise ValueError("校准失败: 伪未知划分为空")

    ds_known, loader_cal_known = make_loader(known_idx)
    _, loader_cal_unknown = make_loader(pseudo_idx)
    label_names = ds_known.get_label_name_map()
    proto_cal, _, _ = register_from_loader(
        model, loader_cal_known, label_names, next(model.parameters()).device,
        percentile=cfg.accept_percentile,
    )

    features = [
        f.strip() for f in str(getattr(cfg, "open_score_features", "")).split(",")
        if f.strip()
    ]
    features = [{"base": "base_score", "margin": "margin_score"}.get(f, f) for f in features]
    calibrator = OpenScoreCalibrator(
        mode=getattr(cfg, "open_score_calibration_mode", "logistic"),
        features=features,
        seed=int(getattr(cfg, "open_score_calibration_seed", 42)),
    )

    known_features, pseudo_features = [], []
    known_scores, pseudo_scores = [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for loader, feature_list, score_list in [
            (loader_cal_known, known_features, known_scores),
            (loader_cal_unknown, pseudo_features, pseudo_scores),
        ]:
            for batch in loader:
                x = batch["input"].to(device)
                z = model.encode(x, tic=_batch_tic(batch, device))
                pred = proto_cal.predict(z)
                feature_list.append(calibrator.collect_features_from_predict(pred, proto_cal))
                score_list.extend(pred["scores"].detach().cpu().numpy().tolist())

    calibrator.fit(known_features, pseudo_features)
    metadata = {
        "fit_scope": "known_train_pseudo_unknown",
        "pseudo_unknown_products": pseudo_products,
        "n_fit_known": int(len(known_scores)),
        "n_fit_pseudo_unknown": int(len(pseudo_scores)),
        "features": features,
        "mode": getattr(cfg, "open_score_calibration_mode", "logistic"),
    }
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "calibration_fit_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    return calibrator, metadata


def apply_calibrator_to_loader(model, proto_store, calibrator, loader, device):
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            z = model.encode(x, tic=_batch_tic(batch, device))
            pred = proto_store.predict(z)
            fd = calibrator.collect_features_from_predict(pred, proto_store)
            n = fd["base_score"].shape[0]
            for i in range(n):
                scores.append(calibrator.apply({k: v[i] for k, v in fd.items()}))
    return scores


def evaluate_setting_c(model, unknown_dataset, unknown_idx_splits,
                       label_names, device, cfg, fold_name=""):
    """Setting C: 少样本 — N-shot 注册未知类。

    支持 Few-shot repeats:
      - 如果 unknown_idx_splits[n_shot] 是 dict (单次): 旧兼容模式
      - 如果 unknown_idx_splits[n_shot] 是 list (多次): 重复抽样模式,
        输出 mean/std/CI
    """
    results = {}
    repeats_per_n = {}
    print(f"\n── Setting C [{fold_name}] ──")
    for n_shot, split in unknown_idx_splits.items():
        if isinstance(split, list):
            # Multiple repeats
            accs, f1s = [], []
            for rep in split:
                if not rep.get("test_idx"):
                    continue
                m = few_shot_evaluate(
                    model, unknown_dataset, rep["ref_idx"], rep["test_idx"],
                    label_names, device, cfg)
                accs.append(m["accuracy"])
                f1s.append(m["macro_f1"])

            if not accs:
                results[n_shot] = {"accuracy": np.nan, "macro_f1": np.nan}
                continue

            acc_arr = np.array(accs, dtype=np.float64)
            f1_arr = np.array(f1s, dtype=np.float64)
            n_repeats = len(accs)

            # Compute statistics
            mean_acc = float(acc_arr.mean())
            std_acc = float(acc_arr.std(ddof=1)) if n_repeats > 1 else 0.0
            ci95_acc = 1.96 * std_acc / np.sqrt(n_repeats) if n_repeats > 1 else 0.0

            mean_f1 = float(f1_arr.mean())
            std_f1 = float(f1_arr.std(ddof=1)) if n_repeats > 1 else 0.0
            ci95_f1 = 1.96 * std_f1 / np.sqrt(n_repeats) if n_repeats > 1 else 0.0

            results[n_shot] = {
                "accuracy": mean_acc,
                "accuracy_mean": mean_acc,
                "accuracy_std": std_acc,
                "accuracy_ci95": ci95_acc,
                "accuracy_ci95_low": mean_acc - ci95_acc,
                "accuracy_ci95_high": mean_acc + ci95_acc,
                "macro_f1": mean_f1,
                "macro_f1_mean": mean_f1,
                "macro_f1_std": std_f1,
                "macro_f1_ci95": ci95_f1,
                "repeats": n_repeats,
                "accuracy_values": [float(v) for v in acc_arr],
                "macro_f1_values": [float(v) for v in f1_arr],
                "n_ref": split[0].get("n_ref", len(split[0].get("ref_idx", []))) if split else 0,
                "n_test": len(split[0].get("test_idx", [])) if split else 0,
            }
            repeats_per_n[n_shot] = n_repeats

            print(f"  {n_shot}-shot ({n_repeats} repeats): "
                  f"acc={mean_acc:.4f}±{std_acc:.4f} [{mean_acc-ci95_acc:.4f}, {mean_acc+ci95_acc:.4f}], "
                  f"f1={mean_f1:.4f}±{std_f1:.4f}")
        else:
            # Single repeat (compat)
            if not split.get("test_idx"):
                results[n_shot] = {"accuracy": np.nan, "macro_f1": np.nan}
                continue
            m = few_shot_evaluate(
                model, unknown_dataset, split["ref_idx"], split["test_idx"],
                label_names, device, cfg)
            results[n_shot] = m
            print(f"  {n_shot}-shot: acc={m['accuracy']:.4f}, "
                  f"f1={m['macro_f1']:.4f}")
    return {"few_shot": results}


# ═══════════════════════════════════════════════════════════
#  单模型评估 (使用 prepared_data/split.json 划分)
# ═══════════════════════════════════════════════════════════

def evaluate_single_model(cfg):
    """
    评估单一模型在三个 Setting 上的表现。

    Setting A: 已知产品 × 留出批次 → 批次一致性
    Setting B: 已知类 vs 留出产品类 → 开集检测
    Setting C: 留出产品类 N-shot 注册 → 少样本识别
    """
    if str(getattr(cfg, "primary_model", "")).strip().lower() == "raw_pca_mlp":
        from raw_pca_pipeline import evaluate_single_model_raw_pca

        return evaluate_single_model_raw_pca(
            cfg,
            baseline_eval_fn=evaluate_readme_baselines,
            print_summary_fn=_print_single_summary,
            save_summary_fn=_save_single_summary,
        )

    from config import get_device
    from dataset import GCMSDataset, load_data_split, few_shot_from_unknown
    from models import GCMSConsistencyNet
    from register import PrototypeStore

    device = get_device()
    split = load_data_split(cfg)
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")
    out_dir = Path(cfg.output_dir)
    viz_dir = out_dir / "visualizations"

    # ── 加载模型 ──
    model_dir = out_dir / "final_model"
    if not (model_dir / "model.pt").exists():
        print("未找到 final_model/model.pt, 请先运行 python main.py train")
        return

    # ── 输入变换: raw -> PCA(沿 m/z 轴) ──
    input_transform = None
    input_pca_path = model_dir / "input_rt_pca.pkl"
    if input_pca_path.exists():
        from input_pca import load_rt_axis_pca, RtAxisPcaTransform

        input_pca_model = load_rt_axis_pca(input_pca_path)
        input_transform = RtAxisPcaTransform(input_pca_model)
        cfg.mz_bins = int(getattr(input_pca_model, "n_components_", cfg.mz_bins))
        print(f"  [Input PCA] 已加载: {input_pca_path.name}, n_components={cfg.mz_bins}")

    # ── 全局编码器: 覆盖所有产品和批次 ──
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    full_df = pd.read_csv(metadata_csv)
    full_df = full_df[(full_df["product_fine"] != "BLANK")
                      & (~full_df["is_special"])].reset_index(drop=True)
    global_product_enc = LabelEncoder().fit(
        sorted(full_df[product_col].unique()))
    global_batch_enc = LabelEncoder().fit(
        sorted(full_df["batch_idx"].unique()))

    # 构建模型: domain_head 大小须与训练时一致
    meta_path = model_dir / "train_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            train_meta = json.load(f)
        num_batches_model = train_meta["num_batches"]
        if bool(train_meta.get("input_raw_pca_enabled", False)):
            cfg.mz_bins = int(train_meta.get("input_raw_pca_components", cfg.mz_bins))
        for key in [
            "tic_branch_enabled", "tic_encoder", "tic_embed_dim",
            "tic_fusion_mode", "tic_fusion_output_dim",
        ]:
            if key in train_meta:
                setattr(cfg, key, train_meta[key])
    else:
        # 回退: 从 state_dict 推断
        sd = torch.load(model_dir / "model.pt", map_location="cpu",
                        weights_only=True)
        num_batches_model = sd["domain_head.fc.2.weight"].shape[0]

    model = GCMSConsistencyNet(num_batches_model, cfg).to(device)
    model.load_state_dict(torch.load(
        model_dir / "model.pt", map_location=device, weights_only=True))

    proto_store = PrototypeStore()
    proto_dir = model_dir / "prototypes"
    if proto_dir.exists():
        proto_store.load(proto_dir)

    def _make_loader_main(indices):
        ds = GCMSDataset(metadata_csv, product_col=product_col,
                         augmentation=None, indices=indices,
                         input_transform=input_transform,
                         cfg=cfg)
        ds.product_enc = global_product_enc
        ds.batch_enc = global_batch_enc
        ds.df["product_label"] = global_product_enc.transform(
            ds.df[product_col])
        ds.df["batch_label"] = global_batch_enc.transform(
            ds.df["batch_idx"])
        ds.num_products = len(global_product_enc.classes_)
        ds.num_batches = len(global_batch_enc.classes_)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False)
        return ds, loader

    def _make_loader_baseline(indices):
        ds = GCMSDataset(metadata_csv, product_col=product_col,
                         augmentation=None, indices=indices,
                         cfg=cfg)
        ds.product_enc = global_product_enc
        ds.batch_enc = global_batch_enc
        ds.df["product_label"] = global_product_enc.transform(
            ds.df[product_col])
        ds.df["batch_label"] = global_batch_enc.transform(
            ds.df["batch_idx"])
        ds.num_products = len(global_product_enc.classes_)
        ds.num_batches = len(global_batch_enc.classes_)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False)
        return ds, loader

    # 训练集 loader (Setting A 计算 cross-batch gap)
    _, loader_train = _make_loader_main(split["train_idx"])

    print(f"\n{'='*60}")
    print("单模型评估")
    print(f"{'='*60}")

    # ═══ Setting A: 批次一致性 ═══
    if split["test_batch_idx"]:
        ds_test_batch, loader_test_batch = _make_loader_main(
            split["test_batch_idx"])
        result_a = evaluate_setting_a(
            model, loader_train, loader_test_batch, proto_store,
            device, cfg, f"留出批次 {split['holdout_batches']}")

        if result_a.get("records"):
            plot_embedding_tsne(
                result_a["records"],
                viz_dir / "tsne_product_settingA.png", color_by="product")
            plot_embedding_tsne(
                result_a["records"],
                viz_dir / "tsne_batch_settingA.png", color_by="batch")
            plot_score_distribution(
                result_a["records"],
                viz_dir / "score_dist_settingA.png")
    else:
        result_a = None
        print("\n── Setting A: 无留出批次测试数据 ──")

    # ═══ Setting B: 开集检测 ═══
    if split["test_unknown_idx"]:
        # 已知类采用 val + 留出批次并集，降低单一批次分布偏差
        known_idx = sorted(
            set(split["val_idx"]) | set(split["test_batch_idx"])
        )
        _, loader_known = _make_loader_main(known_idx)
        ds_unknown, loader_unknown = _make_loader_main(split["test_unknown_idx"])
        result_b = evaluate_setting_b(
            model, proto_store, loader_known, loader_unknown,
            device, cfg, f"留出产品 {split['holdout_products']}")

        # ═══ Open-set Score Calibration (optional) ═══
        if getattr(cfg, "open_score_calibration_enabled", False):
            try:
                from open_score_calibration import (
                    evaluate_calibration,
                )
                cal_dir = out_dir / "calibration"
                cal_dir.mkdir(parents=True, exist_ok=True)

                calibrator, cal_fit_meta = fit_open_score_calibrator_pseudo_unknown(
                    model, cfg, split, _make_loader_main, metadata_csv,
                    product_col, save_dir=cal_dir,
                )
                calibrator.save(cal_dir)

                known_records = collect_predictions(
                    model, loader_known, proto_store, device,
                    reject_factor=cfg.reject_threshold_factor,
                )
                unknown_records = collect_predictions(
                    model, loader_unknown, proto_store, device,
                    reject_factor=cfg.reject_threshold_factor,
                )
                metrics_uncal = evaluate_calibration(
                    [r["open_set_score"] for r in known_records],
                    [r["open_set_score"] for r in unknown_records],
                )

                known_cal = apply_calibrator_to_loader(
                    model, proto_store, calibrator, loader_known, device)
                unknown_cal = apply_calibrator_to_loader(
                    model, proto_store, calibrator, loader_unknown, device)
                metrics_cal = evaluate_calibration(known_cal, unknown_cal)

                result_b["open_set"]["calibration"] = {
                    "enabled": True,
                    "mode": cfg.open_score_calibration_mode,
                    "fit_metadata": cal_fit_meta,
                    "pre_calibration": metrics_uncal,
                    "post_calibration": metrics_cal,
                    "delta_AUROC": metrics_cal.get("AUROC", float("nan")) - metrics_uncal.get("AUROC", float("nan")),
                    "delta_FPR95": metrics_cal.get("FPR_at_95TPR", float("nan")) - metrics_uncal.get("FPR_at_95TPR", float("nan")),
                }
                print(f"\n  [Calibration] delta_AUROC={result_b['open_set']['calibration']['delta_AUROC']:.4f}, "
                      f"delta_FPR95={result_b['open_set']['calibration']['delta_FPR95']:.4f}")
            except Exception as e:
                print(f"\n  [Calibration] Failed: {e}")
                result_b["open_set"]["calibration"] = {"enabled": True, "error": str(e)}
    else:
        result_b = None
        print("\n── Setting B: 无留出产品测试数据 ──")

    # ═══ Setting C: 少样本 ═══
    if split["test_unknown_idx"]:
        unknown_idx = split["test_unknown_idx"]
        ds_unknown_full = GCMSDataset(
            metadata_csv, product_col=product_col,
            augmentation=None, indices=unknown_idx,
            input_transform=input_transform,
            cfg=cfg)
        unknown_label_names = ds_unknown_full.get_label_name_map()

        fs_repeats = int(getattr(cfg, "fewshot_repeats", 1))
        fs_seed_start = int(getattr(cfg, "fewshot_seed_start", cfg.seed))

        fs_splits = few_shot_from_unknown(
            unknown_idx, metadata_csv, product_col=product_col,
            n_shot_values=cfg.n_shot_values, seed=cfg.seed,
            repeats=fs_repeats, seed_start=fs_seed_start)
        result_c = evaluate_setting_c(
            model, ds_unknown_full, fs_splits, unknown_label_names,
            device, cfg, f"留出产品 {split['holdout_products']}")

        # Save per-repeat CSV if repeats > 1
        if fs_repeats > 1:
            import csv
            repeats_dir = out_dir / "fewshot"
            repeats_dir.mkdir(parents=True, exist_ok=True)
            for n_shot, data in result_c.get("few_shot", {}).items():
                if "accuracy_values" in data:
                    csv_path = repeats_dir / f"repeats_{n_shot}shot.csv"
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["repeat", "accuracy", "macro_f1"])
                        for i, (acc, f1) in enumerate(zip(
                            data["accuracy_values"], data["macro_f1_values"])):
                            writer.writerow([i, acc, f1])
    else:
        result_c = None
        print("\n── Setting C: 无留出产品测试数据 ──")

    # ═══ Baselines: README 对比方法 ═══
    try:
        baselines_readme = evaluate_readme_baselines(
            split, cfg, _make_loader_baseline, metadata_csv, product_col)
    except Exception as e:
        baselines_readme = {}
        print(f"\n[README Baselines] 评估失败: {e}")

    # ═══ 汇总打印 ═══
    _print_single_summary(result_a, result_b, result_c, cfg,
                          baselines_readme=baselines_readme)
    _save_single_summary(result_a, result_b, result_c, split, out_dir,
                         baselines_readme=baselines_readme,
                         cfg=cfg)

    return result_a, result_b, result_c


def _print_single_summary(result_a, result_b, result_c, cfg,
                          baselines_readme=None):
    """打印单模型评估汇总。"""
    print(f"\n{'='*60}")
    print("单模型评估汇总")
    print(f"{'='*60}")

    if result_a:
        pid = result_a["product_identification"]
        con = result_a["consistency_scoring"]
        rob = result_a["batch_robustness"]
        print("\nSetting A (批次一致性 — 已知产品 × 留出批次):")
        print(f"  Accuracy:          {pid['accuracy']:.4f}")
        print(f"  Macro-F1:          {pid['macro_f1']:.4f}")
        if "cross_batch_gap" in pid:
            print(f"  Cross-batch Δ:     {pid['cross_batch_gap']:.4f}")
        print(f"  Score AUROC:       {con['AUROC_correct']:.4f}")
        print(f"  Cohen's d:         {con['cohens_d']:.4f}")
        print(f"  Sil(product):      {rob['silhouette_product']:.4f}")
        print(f"  Sil(batch):        {rob['silhouette_batch']:.4f}")
        print(f"  Batch pred:        {rob['batch_predictability']:.4f}")

    if result_b:
        osm = result_b["open_set"]
        print("\nSetting B (开集检测 — 已知类 vs 留出产品类):")
        print(f"  Open-set AUROC:    {osm['open_set_AUROC']:.4f}")
        print(f"  FPR@95TPR:         {osm['FPR_at_95TPR']:.4f}")
        print(f"  Known score mean:  {osm['known_score_mean']:.4f}")
        print(f"  Unknown score mean:{osm['unknown_score_mean']:.4f}")

    if result_c:
        print("\nSetting C (少样本 — 留出产品类 N-shot 注册):")
        for n_shot in cfg.n_shot_values:
            m = result_c["few_shot"].get(n_shot, {})
            acc = m.get("accuracy", float("nan"))
            f1 = m.get("macro_f1", float("nan"))
            print(f"  {n_shot:2d}-shot: acc={acc:.4f}, f1={f1:.4f}")

    if baselines_readme:
        print("\nREADME Baselines 对比:")
        cmp_all = _build_main_vs_readme_baselines(
            result_a,
            result_b,
            result_c,
            baselines_readme,
        )
        for key in README_BASELINE_ORDER:
            baseline_result = baselines_readme.get(key)
            if not baseline_result:
                continue

            name = baseline_result.get("name", key)
            mode = baseline_result.get("feature_mode", "raw")
            bb = baseline_result.get("setting_b") or {}
            bc3 = (baseline_result.get("setting_c") or {}).get("3", {})
            cmp_sb = (cmp_all.get(key) or {}).get("setting_b", {})
            cmp_sc3 = ((cmp_all.get(key) or {}).get("setting_c", {}) or {}).get("3", {})

            print(f"  [{name}] feature={mode}")
            print(
                "    Setting B: "
                f"AUROC={bb.get('open_set_AUROC', float('nan')):.4f}, "
                f"FPR95={bb.get('FPR_at_95TPR', float('nan')):.4f}"
            )
            print(
                "    Setting C 3-shot: "
                f"acc={bc3.get('accuracy', float('nan')):.4f}, "
                f"f1={bc3.get('macro_f1', float('nan')):.4f}"
            )
            print(
                "    Main-Baseline: "
                f"d_AUROC={cmp_sb.get('open_set_AUROC', float('nan')):.4f}, "
                f"d_FPR95={cmp_sb.get('FPR_at_95TPR', float('nan')):.4f}, "
                f"d_3shot={cmp_sc3.get('accuracy', float('nan')):.4f}"
            )


def _save_single_summary(result_a, result_b, result_c, split, out_dir,
                         baselines_readme=None, cfg=None):
    """保存单模型评估结果到 JSON。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _s(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _s_recursive(obj):
        if isinstance(obj, dict):
            return {str(k): _s_recursive(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_s_recursive(v) for v in obj]
        return _s(obj)

    summary = {
        "split": {
            "known_products": split["known_products"],
            "holdout_products": split["holdout_products"],
            "holdout_batches": split["holdout_batches"],
            "stats": split["stats"],
        },
    }

    if result_a:
        pid = result_a["product_identification"]
        summary["setting_a"] = {
            k: _s(v) for k, v in pid.items()
            if k not in ("confusion", "report")
        }
        summary["setting_a_consistency"] = {
            k: _s(v) for k, v in result_a["consistency_scoring"].items()
        }
        summary["setting_a_robustness"] = {
            k: _s(v) for k, v in result_a["batch_robustness"].items()
        }

    if result_b:
        summary["setting_b"] = {
            k: _s(v) for k, v in result_b["open_set"].items()
        }

    if result_c:
        summary["setting_c"] = {
            str(k): {kk: _s(vv) for kk, vv in v.items()}
            for k, v in result_c["few_shot"].items()
        }

    if baselines_readme:
        summary["baselines_readme"] = {}
        for key in README_BASELINE_ORDER:
            baseline_result = baselines_readme.get(key)
            if not baseline_result:
                continue
            summary["baselines_readme"][key] = {
                "name": baseline_result.get("name", key),
                "feature_mode": baseline_result.get("feature_mode", "raw"),
                "setting_a": _s_recursive(baseline_result.get("setting_a") or {}),
                "setting_b": _s_recursive(baseline_result.get("setting_b") or {}),
                "setting_c": _s_recursive(baseline_result.get("setting_c") or {}),
            }

        # 兼容旧字段: baseline_tic_pca_mlp + main_vs_baseline
        tic = summary["baselines_readme"].get("tic_pca_mlp")
        if tic:
            summary["baseline_tic_pca_mlp"] = {
                "setting_a": tic.get("setting_a", {}),
                "setting_b": tic.get("setting_b", {}),
                "setting_c": tic.get("setting_c", {}),
            }
            summary["main_vs_baseline"] = _s_recursive(
                _build_main_vs_baseline(
                    result_a,
                    result_b,
                    result_c,
                    tic,
                )
            )

        summary["main_vs_readme_baselines"] = _s_recursive(
            _build_main_vs_readme_baselines(
                result_a,
                result_b,
                result_c,
                summary["baselines_readme"],
            )
        )

    model_path = str(getattr(cfg, "pretrained_feature_model", "") or "").strip() if cfg else ""
    summary["pretrained_feature_extractor"] = {
        "model_path": model_path,
        "arch": (str(getattr(cfg, "pretrained_feature_arch", "auto") or "auto") if cfg else "auto"),
        "layers": (str(getattr(cfg, "pretrained_feature_layers", "layer4") or "layer4") if cfg else "layer4"),
        "fuse": (str(getattr(cfg, "pretrained_feature_fuse", "concat") or "concat") if cfg else "concat"),
        "enabled": bool(model_path),
    }

    summary["main_model_backbone"] = {
        "backbone": (str(getattr(cfg, "main_backbone", "gcms") or "gcms") if cfg else "gcms"),
        "model_path": (str(getattr(cfg, "main_backbone_model", "") or "") if cfg else ""),
        "layers": (str(getattr(cfg, "main_feature_layers", "layer4") or "layer4") if cfg else "layer4"),
        "fuse": (str(getattr(cfg, "main_feature_fuse", "concat") or "concat") if cfg else "concat"),
        "transformer_patch_size": (int(getattr(cfg, "transformer_patch_size", 16)) if cfg else 16),
        "transformer_embed_dim": (int(getattr(cfg, "transformer_embed_dim", 256)) if cfg else 256),
        "transformer_depth": (int(getattr(cfg, "transformer_depth", 6)) if cfg else 6),
        "transformer_num_heads": (int(getattr(cfg, "transformer_num_heads", 8)) if cfg else 8),
        "transformer_mlp_ratio": (float(getattr(cfg, "transformer_mlp_ratio", 4.0)) if cfg else 4.0),
    }

    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_s)
    print(f"\n评估结果已保存到 {out_dir / 'evaluation_summary.json'}")


# ═══════════════════════════════════════════════════════════
#  旧版: LOBO 多 fold 评估 (保留用于交叉验证)
# ═══════════════════════════════════════════════════════════

def evaluate_all_settings(fold_results, split_info, cfg):
    """汇总所有 fold 的 Setting A/B/C 评估 (LOBO 模式)。"""
    from config import get_device
    device = get_device()
    out_dir = Path(cfg.output_dir)
    viz_dir = out_dir / "visualizations"

    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")

    from dataset import GCMSDataset, few_shot_from_unknown

    unknown_idx = split_info["unknown_idx"]
    if unknown_idx:
        unknown_ds = GCMSDataset(metadata_csv, product_col=product_col,
                                 augmentation=None, indices=unknown_idx,
                                 cfg=cfg)
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

    for fi, fr in enumerate(
        tqdm(
            fold_results,
            desc="评估 folds",
            ncols=80,
            disable=not _progress_enabled(),
        )
    ):
        model = fr["model"].to(device)
        proto_store = fr["proto_store"]
        fold_name = fr["test_batch"]
        ds_train = fr["ds_train"]
        train_idx = fr.get("train_idx")

        # 无增强训练 loader (Setting A cross-batch gap)
        if train_idx is not None:
            train_noaug = GCMSDataset(
                metadata_csv, product_col=product_col,
                augmentation=None, indices=train_idx,
                cfg=cfg)
        else:
            train_noaug = GCMSDataset(
                metadata_csv, product_col=product_col,
                augmentation=None, indices=ds_train.df.index.tolist(),
                cfg=cfg)
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
            "fold": _s(fold_results[i]["fold"]),
            "test_batch": str(fold_results[i]["test_batch"]),
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
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_s)
