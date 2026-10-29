# -*- coding: utf-8 -*-
"""自检: 直接抽取 evaluate.py 里的 _rbf_mmd2 / mmd_batch_metrics 源码执行。

不修改、不复刻, 保证测的就是将来要跑的那段代码。
造数据用真实 Setting A 的「批次 x 产品」构成 (从 seed41 的
setting_a_predictions.csv 统计得到), z 为 256 维单位向量。

几何校准: 实测 silhouette_product=0.68 (6 产品 / 413 样本)。本脚本会打印
合成数据的 silhouette, 必须落在 0.6~0.75 才算「仿真可信」。
"""
import re
import numpy as np

from pathlib import Path
SRC = str(Path(__file__).resolve().parents[1] / "evaluate.py")
src = open(SRC, encoding="utf-8").read()


def extract(name):
    m = re.search(r"^def %s\(" % re.escape(name), src, re.M)
    if not m:
        raise SystemExit("找不到函数 %s" % name)
    nxt = re.search(r"^(?:def |# ══)", src[m.end():], re.M)
    end = m.end() + nxt.start() if nxt else len(src)
    return src[m.start():end]


ns = {"np": np}
for fn in ("_rbf_mmd2", "mmd_batch_metrics"):
    exec(extract(fn), ns)
mmd_fn = ns["mmd_batch_metrics"]
import inspect
print("[代码] 真实抽取, 签名 = mmd_batch_metrics%s" % (inspect.signature(mmd_fn),))

D = 256
# 真实 test 集「批次 x 产品」样本数 (来自 setting_a_predictions.csv, 已排除 RJD)
REAL = {
    "20240308": {"H88": 10, "H99": 4, "HRJ": 33, "HRY": 29, "HYZ": 20, "YJD": 0},
    "20250905": {"H88": 28, "H99": 9, "HRJ": 0,  "HRY": 0,  "HYZ": 73, "YJD": 11},
    "20250912": {"H88": 13, "H99": 9, "HRJ": 0,  "HRY": 2,  "HYZ": 31, "YJD": 6},
    "20250920": {"H88": 22, "H99": 9, "HRJ": 0,  "HRY": 0,  "HYZ": 73, "YJD": 14},
    "20260313": {"H88": 0,  "H99": 0, "HRJ": 0,  "HRY": 17, "HYZ": 0,  "YJD": 0},
}
PRODUCTS = ["H88", "H99", "HRJ", "HRY", "HYZ", "YJD"]
WITHIN = 0.45          # 单位球上「同产品」内部弦距 (校准到 silhouette≈0.68)


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def build(per_batch_shift=None, within=WITHIN, seed=0, composition=None):
    """per_batch_shift: {批次: 平移强度(以「同产品内部距离」为单位)}。
    未列出的批次 = 0。平移方向随机。"""
    rng = np.random.default_rng(seed)
    comp = composition or REAL
    centers = unit(rng.normal(size=(len(PRODUCTS), D)))
    eps_std = within / np.sqrt(D)
    shifts = {}
    for b in comp:
        s = (per_batch_shift or {}).get(b, 0.0)
        shifts[b] = unit(rng.normal(size=D)) * s * within
    records = []
    for b, row in comp.items():
        for pi, p in enumerate(PRODUCTS):
            n = row.get(p, 0)
            if n == 0:
                continue
            z = centers[pi] + rng.normal(scale=eps_std, size=(n, D)) + shifts[b]
            for r in unit(z):
                records.append({"z": r.astype(np.float32),
                                "batch_name": b, "true_class": p})
    return records


def geometry(records):
    """合成数据的几何体检: 产品内/产品间平均距离 + silhouette。"""
    zs = np.stack([r["z"] for r in records]).astype(np.float64)
    ys = np.array([r["true_class"] for r in records])
    dm = np.sqrt(np.maximum(
        (zs * zs).sum(1)[:, None] + (zs * zs).sum(1)[None, :] - 2 * zs @ zs.T, 0))
    same = ys[:, None] == ys[None, :]
    iu = np.triu_indices(len(zs), 1)
    w = dm[iu][same[iu]].mean()
    b = dm[iu][~same[iu]].mean()
    sil = []
    for i in range(0, len(zs), 3):
        a = dm[i][same[i] & (np.arange(len(zs)) != i)]
        bs = [dm[i][(ys == c) & (np.arange(len(zs)) != i)].mean()
              for c in np.unique(ys) if c != ys[i]]
        a, bb = a.mean(), min(bs)
        sil.append((bb - a) / max(a, bb))
    return w, b, float(np.mean(sil))


def show(r):
    for k in ("mmd_batch_mean", "mmd_batch_max", "mmd_batch_argmax",
              "mmd_batch_min", "mmd_n_products", "mmd_n_pairs",
              "mmd_z_strat", "mmd_p_strat", "mmd_p_max", "mmd_n_perm",
              "mmd_unevaluable_products"):
        v = r.get(k)
        print("    %-24s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))
    print("    per_product             %s" % (
        {k: round(v, 4) for k, v in (r.get("mmd_per_product") or {}).items()},))
    print("    pairs_per_product       %s" % (r.get("mmd_pairs_per_product"),))


base = build(seed=11)
w, b, sil = geometry(base)
print("[几何] 产品内均距=%.3f  产品间均距=%.3f  比值=%.2f  silhouette=%.3f"
      % (w, b, b / w, sil))
print("       实测模型 silhouette_product=0.68 (6 产品/413 样本), 合成须接近它")
print("       批次数=%d, 样本数=%d" % (len(set(r["batch_name"] for r in base)), len(base)))

print("\n" + "=" * 74)
print("检查 1  组合不变性: 批次内产品构成完全不同, 但产品分布本身没变")
print("        (旧版「全体跨批次 MMD」在这种情况会报出巨大差异 -> 已废弃)")
print("=" * 74)
comp_a = {"B1": {"H88": 30, "H99": 20, "HRY": 10},
          "B2": {"HYZ": 30, "YJD": 20, "H88": 10}}
comp_b = {"B1": {"HYZ": 30, "YJD": 20, "HRY": 10},
          "B2": {"H88": 30, "H99": 20, "HYZ": 10}}
for tag, c in (("A", comp_a), ("B", comp_b)):
    r = mmd_fn(build(seed=7, composition=c), n_perm=200, seed=42)
    print("  构成 %s: mean=%.4f max=%.4f p_mean=%.4f p_max=%.4f"
          % (tag, r["mmd_batch_mean"], r["mmd_batch_max"],
             r["mmd_p_strat"], r["mmd_p_max"]))
print("  -> 两者都应不显著 (p>0.05)")

print("\n" + "=" * 74)
print("检查 2  无批次效应 (真实构成, 平移=0)  -> 期望 p 不显著")
print("=" * 74)
show(mmd_fn(build(seed=11), n_perm=200, seed=42))

print("\n" + "=" * 74)
print("检查 3  所有产品整体平移 0/0.5/1/1.5/2.5 个「产品内距离」-> 期望单调上升")
print("=" * 74)
print("  平移   mean     max      argmax   z       p_mean   p_max")
for s in (0.0, 0.5, 1.0, 1.5, 2.5):
    r = mmd_fn(build({x: s for x in REAL}, seed=11), n_perm=200, seed=42)
    print("  %-6.1f %-8.4f %-8.4f %-8s %-7.2f %-8.4f %-8.4f" % (
        s, r["mmd_batch_mean"], r["mmd_batch_max"], r["mmd_batch_argmax"],
        r["mmd_z_strat"], r["mmd_p_strat"], r["mmd_p_max"]))

print("\n" + "=" * 74)
print("检查 4  只有 HRY 漂 (真实构成: HRY 只有 20240308 vs 20260313 一对)")
print("        -> 期望 argmax 指向 HRY, p_max 变小")
print("=" * 74)
print("  HRY平移 mean     max      argmax   z       p_mean   p_max    HRY值")
for s in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    r = mmd_fn(build({"20260313": s}, seed=11), n_perm=200, seed=42)
    pp = r.get("mmd_per_product") or {}
    print("  %-7.1f %-8.4f %-8.4f %-8s %-7.2f %-8.4f %-8.4f %.4f" % (
        s, r["mmd_batch_mean"], r["mmd_batch_max"], r["mmd_batch_argmax"],
        r["mmd_z_strat"], r["mmd_p_strat"], r["mmd_p_max"],
        pp.get("HRY", float("nan"))))

print("\n" + "=" * 74)
print("检查 5  零校准: 10 组独立「无批次效应」数据的 p 值 (应大致均匀, 少显著)")
print("=" * 74)
ps, pms = [], []
for k in range(10):
    r = mmd_fn(build(seed=100 + k), n_perm=100, seed=42 + k)
    ps.append(r["mmd_p_strat"])
    pms.append(r["mmd_p_max"])
print("  p_mean:", " ".join("%.3f" % x for x in ps))
print("  p_max :", " ".join("%.3f" % x for x in pms))
print("  显著个数 (期望 ≈0.5/10): p_mean=%d, p_max=%d"
      % (sum(1 for x in ps if x < 0.05), sum(1 for x in pms if x < 0.05)))

print("\n" + "=" * 74)
print("检查 6  边界情况")
print("=" * 74)
r = mmd_fn([], n_perm=50)
print("  空 records        -> mean=%s max=%s" % (r["mmd_batch_mean"], r["mmd_batch_max"]))
r = mmd_fn(build(seed=11)[:3], n_perm=50)
print("  只 3 条           -> mean=%s" % (r["mmd_batch_mean"],))
one = [dict(x, batch_name="B1") for x in build(seed=11)[:40]]
r = mmd_fn(one, n_perm=50)
print("  只 1 个批次       -> mean=%s" % (r["mmd_batch_mean"],))
r = mmd_fn(build(seed=11), n_perm=0)
print("  n_perm=0          -> mean=%.4f z=%s p=%s" % (
    r["mmd_batch_mean"], r["mmd_z_strat"], r["mmd_p_strat"]))
r = mmd_fn(build(seed=11), n_perm=30, seed=42, min_group=30)
print("  min_group=30      -> mean=%.4f products=%d pairs=%d 不可评估=%s" % (
    r["mmd_batch_mean"], r["mmd_n_products"], r["mmd_n_pairs"],
    r["mmd_unevaluable_products"]))
recs = build(seed=11)
for x in recs:
    x["batch_label"] = x.pop("batch_name")
r = mmd_fn(recs, n_perm=20)
print("  退回 batch_label  -> mean=%.4f products=%d" % (
    r["mmd_batch_mean"], r["mmd_n_products"]))
recs = build(seed=11)
for x in recs:
    x["true_product"] = x.pop("true_class")
r = mmd_fn(recs, n_perm=20)
print("  退回 true_product -> mean=%.4f products=%d" % (
    r["mmd_batch_mean"], r["mmd_n_products"]))
print("\n自检完成, 无异常。")
