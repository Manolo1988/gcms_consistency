# -*- coding: utf-8 -*-
"""训练集上的批次可预测性: 方法有没有在自己作用的地盘上把批次信息压下去?

读多个 embeddings_train.npz (需 evaluate --dump-embeddings 生成), 对每个算:
  batch_predictability : 逻辑回归从 z 猜 26 个训练批次 (与 evaluate 同口径, 越低越好)
  silhouette_batch    : 批次轮廓 (越低越好)
  silhouette_product  : 产品轮廓 (越高越好)

注意: 训练集上的批次可预测性天生偏高, 因为部分产品只出现在少数批次
(产品构成与批次绑定)。但这个偏置对每个模型相同, 横向比"相对基线的变化"依然有效。

用法 (项目根目录):
  .venv/bin/python scripts/train_batch_probe.py <a.npz> [<b.npz> ...] --label x --label y
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score


def load(path):
    d = np.load(path, allow_pickle=True)
    return (np.asarray(d["z"], dtype=np.float64),
            np.asarray(d["product"]).astype(str),
            np.asarray(d["batch"]).astype(str))


def batch_predictability(Z, batch):
    if len(np.unique(batch)) < 2 or len(Z) < 11:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(Z, batch)
    return float(clf.score(Z, batch))


def safe_silhouette(Z, labels):
    if len(np.unique(labels)) < 2 or len(Z) <= len(np.unique(labels)):
        return float("nan")
    return float(silhouette_score(Z, labels, sample_size=min(len(Z), 3000),
                                  random_state=42))


def main():
    ap = argparse.ArgumentParser(description="训练集上的批次可预测性对比")
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--label", action="append", default=None)
    args = ap.parse_args()

    print("%-16s %-10s %-10s %-10s %-7s" % ("run", "可预测↓", "批轮廓↓", "产轮廓↑", "n"))
    for i, path in enumerate(args.npz):
        Z, prod, batch = load(path)
        label = args.label[i] if args.label and i < len(args.label) else path
        print("%-16s %-10.3f %-10.3f %-10.3f %-7d" % (
            label, batch_predictability(Z, batch),
            safe_silhouette(Z, batch), safe_silhouette(Z, prod), len(Z)))


if __name__ == "__main__":
    main()
