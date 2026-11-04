# -*- coding: utf-8 -*-
"""时间校正的可达验证: 训练批次拟合 -> 测试批次应用。

只读两份落盘嵌入 (需 evaluate --dump-embeddings):
  embeddings_train.npz       训练集 (用于拟合趋势)
  setting_a_embeddings_01.npz 留出批次 (用于评估)
不训练、不改模型权重, 只改"原型放在哪"或"样本挪到哪"。

四个版本:
  V0 固定原型        : proto = 训练样本均值 (现状)
  V1 漂移原型        : proto_p(t) = a_p + b_p*(t - t_ref)      [只挪原型]
  V2 特征去趋势      : z' = z - b_p*(t - t_ref)                [挪样本]
  V3 共享方向漂移原型: proto_p(t) = a_p + b*(t - t_ref), b = 各产品斜率的均值

判读:
  V1/V3 只改判定, 样本本身没动, 所以批次可预测性/轮廓与 V0 完全相同,
  它们的收益只能看准确率和"离自己原型的归一化距离"。
  V2 改了样本, 三个批次指标都会变, 但也最有副作用风险。

用法 (项目根目录):
  .venv/bin/python scripts/time_drift_apply.py --train <embeddings_train.npz> \
      --test <setting_a_embeddings_01.npz> [--out 目录]
"""
import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score


def to_days(name):
    s = str(name)[:8]
    return float(dt.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal())


def load(path):
    d = np.load(path, allow_pickle=True)
    return (np.asarray(d["z"], dtype=np.float64),
            np.asarray(d["product"]).astype(str),
            np.asarray(d["batch"]).astype(str))


def accuracy(Z, P, y, classes):
    d = ((Z[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    pred = np.array(classes)[d.argmin(1)]
    return accuracy_score(y, pred), f1_score(y, pred, average="macro"), pred


def norm_dist_to_own(Z, P, y, classes):
    idx = {c: i for i, c in enumerate(classes)}
    rows = np.array([idx[v] for v in y])
    d = np.sqrt(((Z - P[rows]) ** 2).sum(-1))
    return float(np.median(d))


def batch_predictability(Z, batch):
    if len(np.unique(batch)) < 2 or len(Z) < 11:
        return float("nan")
    clf = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
    clf.fit(Z, batch)
    return float(clf.score(Z, batch))


def safe_silhouette(Z, labels):
    if len(np.unique(labels)) < 2 or len(Z) <= len(np.unique(labels)):
        return float("nan")
    return float(silhouette_score(Z, labels, sample_size=min(len(Z), 2000),
                                  random_state=42))


def main():
    ap = argparse.ArgumentParser(description="时间校正可达验证")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    Ztr, ytr, btr = load(args.train)
    Zte, yte, bte = load(args.test)
    print(f"训练 {len(Ztr)} 条 / 测试 {len(Zte)} 条 / 维度 {Ztr.shape[1]}")
    print(f"测试批次 {sorted(set(bte))}")

    ttr_raw = np.array([to_days(b) for b in btr])
    tte_raw = np.array([to_days(b) for b in bte])
    lo, hi = ttr_raw.min(), ttr_raw.max()
    ttr = (ttr_raw - lo) / max(hi - lo, 1.0)
    tte = (tte_raw - lo) / max(hi - lo, 1.0)   # 可 >1, 表示时间外推
    t_ref = float(ttr.mean())

    classes = sorted(set(ytr))
    cls_idx = {c: i for i, c in enumerate(classes)}
    A = np.zeros((len(classes), Ztr.shape[1]))      # 参考时刻位置
    S = np.zeros((len(classes), Ztr.shape[1]))      # 每天的漂移方向
    fitted = []
    for c in classes:
        m = ytr == c
        if m.sum() < 4 or len(set(ttr[m])) < 2:
            A[cls_idx[c]] = Ztr[m].mean(0)
            S[cls_idx[c]] = 0.0
            continue
        X = np.column_stack([np.ones(int(m.sum())), ttr[m]])
        coef, *_ = np.linalg.lstsq(X, Ztr[m], rcond=None)
        A[cls_idx[c]] = coef[0] + coef[1] * t_ref
        S[cls_idx[c]] = coef[1]
        fitted.append(c)

    if fitted:
        S_shared = S[[cls_idx[c] for c in fitted]].mean(0, keepdims=True)
        S_shared = np.repeat(S_shared, len(classes), axis=0)
    else:
        S_shared = np.zeros_like(S)

    Ztr_c = Ztr.copy()
    Zte_c = Zte.copy()
    for c in classes:
        i = cls_idx[c]
        mtr, mte = ytr == c, yte == c
        Ztr_c[mtr] = Ztr[mtr] - np.outer(ttr[mtr] - t_ref, S[i])
        Zte_c[mte] = Zte[mte] - np.outer(tte[mte] - t_ref, S[i])

    results = {}

    def per_sample_protos(Z, t):
        """按样本自己的时间取原型 (同一 t 只算一次)."""
        out = np.zeros((len(Z), len(classes), Z.shape[1]))
        for tv in np.unique(t):
            m = t == tv
            out[m] = A[None, :, :] + S[None, :, :] * (tv - t_ref)
        return out

    def per_sample_protos_shared(Z, t):
        out = np.zeros((len(Z), len(classes), Z.shape[1]))
        for tv in np.unique(t):
            m = t == tv
            out[m] = A[None, :, :] + S_shared[None, :, :] * (tv - t_ref)
        return out

    def score_variant(name, Z, P_fixed=None, P_per=None):
        if P_per is not None:
            # 每个样本一组原型: 逐个算最近类
            preds = []
            for i in range(len(Z)):
                d = ((Z[i][None, :] - P_per[i]) ** 2).sum(-1)
                preds.append(classes[int(d.argmin())])
            preds = np.array(preds)
            acc = accuracy_score(yte, preds)
            f1 = f1_score(yte, preds, average="macro")
        else:
            acc, f1, preds = accuracy(Z, P_fixed, yte, classes)
        results[name] = {"accuracy": float(acc), "macro_f1": float(f1),
                         "pred": preds}
        return preds

    # V0 固定原型
    P0 = np.stack([Ztr[ytr == c].mean(0) for c in classes])
    p0 = score_variant("V0_固定原型", Zte, P_fixed=P0)

    # V1 漂移原型
    p1 = score_variant("V1_漂移原型", Zte, P_per=per_sample_protos(Zte, tte))

    # V2 特征去趋势
    P2 = np.stack([Ztr_c[ytr == c].mean(0) for c in classes])
    p2 = score_variant("V2_特征去趋势", Zte_c, P_fixed=P2)

    # V3 共享方向漂移原型
    p3 = score_variant("V3_共享方向原型", Zte, P_per=per_sample_protos_shared(Zte, tte))

    print("\n版本                A准确率   A_macroF1   改判  救回  弄坏")
    for name, r in results.items():
        diff = r["pred"] != p0
        fixed = int(((p0 != yte) & (r["pred"] == yte)).sum())
        broke = int(((p0 == yte) & (r["pred"] != yte)).sum())
        print(f"  {name:<18} {r['accuracy']:.4f}   {r['macro_f1']:.4f}   "
              f"{int(diff.sum()):>4}  {fixed:>4}  {broke:>4}")

    print("\n批次侧指标 (只有改了样本的版本才会变)")
    rows = [("V0_固定原型", Zte, P0, p0), ("V2_特征去趋势", Zte_c, P2, p2)]
    print("  版本               批次可预测性  产品轮廓  批次轮廓  原生距离中位数")
    for name, Z, P, _ in rows:
        print(f"  {name:<18} {batch_predictability(Z, bte):>10.3f} "
              f"{safe_silhouette(Z, yte):>9.3f} {safe_silhouette(Z, bte):>9.3f} "
              f"{norm_dist_to_own(Z, P, yte, classes):>12.3f}")

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "new_result" / "checks"
        / f"time_drift_apply_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "variants.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "accuracy", "macro_f1", "changed",
                    "fixed", "broken"])
        for name, r in results.items():
            w.writerow([name, f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}",
                        int((r["pred"] != p0).sum()),
                        int(((p0 != yte) & (r["pred"] == yte)).sum()),
                        int(((p0 == yte) & (r["pred"] != yte)).sum())])
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
                   for k, v in results.items()},
                  f, ensure_ascii=False, indent=2)
    print(f"\n已写入: {out_dir}")


if __name__ == "__main__":
    main()
