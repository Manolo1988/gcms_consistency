# -*- coding: utf-8 -*-
"""时间漂移探针: 批次效应里有多少是"随采集时间的趋势"?

只读 evaluate 落盘的 setting_a_embeddings.npz (需 --dump-embeddings), 不训练、不改模型。

核心做法 (全部在同一产品内部做, 避免把"产品构成不同"算成"批次漂移"):
  1. 用批次 one-hot 回归 z, 得到"批次能解释的方差"        -> R2_batch
  2. 用采集日期 t 的线性项回归 z, 得到"时间趋势能解释的方差" -> R2_time
  3. 比值 R2_time / R2_batch = 批次效应里"是趋势"的比例
  4. 把趋势项扣掉 (z' = z - b*(t - t_ref), 保留参考时刻的平均位置),
     重算批次可预测性 / 产品轮廓 / 批次轮廓

判读:
  比值中位数 >= 0.6  -> 时间趋势是批次效应的主体, 时间方案有据可依
  0.3 ~ 0.6          -> 部分是趋势, 还有明显的"每批一口气"式偏移
  < 0.3              -> 不是趋势, 时间方案可以放弃

用法 (项目根目录):
  .venv/bin/python scripts/time_drift_probe.py <setting_a_embeddings.npz> [--out 目录]
"""
import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

REPO = Path(__file__).resolve().parents[1]


def batch_to_days(name):
    s = str(name)[:8]
    return float(dt.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal())


def r2_total(Z, Zhat):
    ss_res = float(((Z - Zhat) ** 2).sum())
    ss_tot = float(((Z - Z.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_lstsq(Z, X):
    coef, *_ = np.linalg.lstsq(X, Z, rcond=None)
    return coef, r2_total(Z, X @ coef)


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
    ap = argparse.ArgumentParser(description="时间漂移探针")
    ap.add_argument("npz", help="setting_a_embeddings.npz")
    ap.add_argument("--out", default=None, help="输出目录")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    Z = np.asarray(d["z"], dtype=np.float64)
    prod = np.asarray(d["product"]).astype(str)
    batch = np.asarray(d["batch"]).astype(str)
    t = np.array([batch_to_days(b) for b in batch], dtype=np.float64)
    span = max(t.max() - t.min(), 1.0)
    t = (t - t.min()) / span
    t_ref = float(t.mean())

    print(f"样本 {len(Z)}  z 维度 {Z.shape[1]}  产品 {len(set(prod))}  批次 {len(set(batch))}")
    print(f"时间跨度 {span:.0f} 天 ({t.min():.2f}~{t.max():.2f} 归一化)\n")

    rows = []
    Z_corr = Z.copy()
    for p in sorted(set(prod)):
        m = prod == p
        Zp, tp, bp = Z[m], t[m], batch[m]
        n_batch = len(set(bp))
        if n_batch < 2 or int(m.sum()) < 6:
            rows.append({"product": p, "n": int(m.sum()), "n_batch": n_batch,
                         "r2_batch": float("nan"), "r2_time": float("nan"),
                         "trend_share": float("nan")})
            continue
        X_batch = np.zeros((m.sum(), n_batch))
        for j, b in enumerate(sorted(set(bp))):
            X_batch[bp == b, j] = 1.0
        X_time = np.column_stack([np.ones_like(tp), tp])

        _, r2_b = fit_lstsq(Zp, X_batch)
        coef_t, r2_t = fit_lstsq(Zp, X_time)
        share = r2_t / r2_b if r2_b and r2_b > 1e-9 else float("nan")

        slope = coef_t[1]
        Z_corr[m] = Zp - np.outer(tp - t_ref, slope)

        rows.append({"product": p, "n": int(m.sum()), "n_batch": n_batch,
                     "r2_batch": r2_b, "r2_time": r2_t, "trend_share": share})

    print("产品       n  批次数   R2(批次)  R2(时间趋势)  趋势占比")
    for r in rows:
        print(f"  {r['product']:<6} {r['n']:>4} {r['n_batch']:>5}   "
              f"{r['r2_batch']:>8.3f} {r['r2_time']:>12.3f} "
              f"{r['trend_share']:>9.3f}")

    shares = [r["trend_share"] for r in rows if r["trend_share"] == r["trend_share"]]
    median_share = float(np.median(shares)) if shares else float("nan")

    before = {
        "batch_predictability": batch_predictability(Z, batch),
        "silhouette_product": safe_silhouette(Z, prod),
        "silhouette_batch": safe_silhouette(Z, batch),
    }
    after = {
        "batch_predictability": batch_predictability(Z_corr, batch),
        "silhouette_product": safe_silhouette(Z_corr, prod),
        "silhouette_batch": safe_silhouette(Z_corr, batch),
    }

    print("\n去掉线性时间趋势前后 (全体, 同一套指标口径)")
    print("  指标                    原值     去趋势后")
    for k in ("batch_predictability", "silhouette_product", "silhouette_batch"):
        print(f"  {k:<22} {before[k]:>7.3f}   {after[k]:>7.3f}")

    if median_share == median_share:
        if median_share >= 0.6:
            verdict = "时间趋势是批次效应的主体 (>=0.6), 时间方案有据可依"
        elif median_share >= 0.3:
            verdict = "部分是趋势 (0.3~0.6), 还有明显的每批独立偏移"
        else:
            verdict = "不是趋势 (<0.3), 时间方案不建议继续"
    else:
        verdict = "数据不足, 无法判断"
    print(f"\n结论: 趋势占比中位数 {median_share:.3f} -> {verdict}")

    out_dir = Path(args.out) if args.out else (
        REPO / "new_result" / "checks" / f"time_drift_probe_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_product.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["product", "n", "n_batch",
                                          "r2_batch", "r2_time", "trend_share"])
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"source_npz": str(args.npz), "median_trend_share": median_share,
                   "before": before, "after": after, "verdict": verdict,
                   "per_product": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n已写入: {out_dir}")


if __name__ == "__main__":
    main()
