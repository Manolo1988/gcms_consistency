"""
扫描 dataset 目录 → 建元数据表 → 逐样本转换为 .npz 张量。
运行方式: python data_prepare.py

.D 文件夹命名规则（标准批次）：
    NNNN#<规格代号><生产线>CS<批次编号>.D
    例：0001#HYZBCS125030302.D
        ├─ 0001   检测顺序号（可不连续）
        ├─ HYZ    规格代号（见 SPEC_NAME_MAP）
        ├─ B      生产线代码（A/B/C/...）
        ├─ CS     固定分隔符
        └─ 125030302  批次编号（1=工厂代码，25=年，03=月，03=日，02=序号）
"""
import re, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from data_reader import d_folder_to_tensor

cfg = Config()

# ── 规格代号 → 规格名称（来自实验室对照表）────────────
SPEC_NAME_MAP = {
    "H88":  "红河（88）",
    "H99":  "红河（99）",
    "HRJ":  "红河（软甲）",
    "HDC":  "红河（道）彩膜版",
    "HV8":  "红河（V8）",
    "HYZ":  "云烟（紫）",
    "HRY":  "云烟（软如意）",
    "HYX":  "云烟（印象）",
    "HRL":  "云烟（软礼印象）",
    "XCJ":  "云烟（中支小重九）",
    "RJD":  "红塔山（软经典）",
    "YJD":  "红塔山（硬经典）",
    "HMD":  "牡丹（软）",
}

# ── fine 编码 → coarse 规格代号 ─────────────────────
#   fine = 规格代号 + 生产线，e.g. H88A / H88B / H88C → H88
FINE_TO_COARSE = {
    "H88A": "H88", "H88B": "H88", "H88C": "H88",
    "H99C": "H99",
    "HRJA": "HRJ", "HRJB": "HRJ",
    "HDCA": "HDC",
    "HV8A": "HV8",
    "HRYA": "HRY", "HRYB": "HRY",
    "HYXA": "HYX",
    "HRLA": "HRL",
    "XCJA": "XCJ",
    "RJDA": "RJD",
    "YJDA": "YJD",
    "HMDA": "HMD",
    "HYZB": "HYZ", "HYZC": "HYZ",
}

# ── 正则：标准英文样品编号 ───────────────────────────
#   规格代号：字母开头，后跟 2 个字母或数字（共 3 字符，如 H88 / HYZ / HV8）
#   生产线：1 个字母（A/B/C/...）
#   分隔符：固定 CS
_RE_STD = re.compile(r"^([A-Za-z][A-Za-z0-9]{2})([A-Za-z])CS", re.IGNORECASE)


def parse_d_name(folder_name: str) -> dict:
    """
    从 .D 文件夹名解析样品元数据。

    返回字典：
        seq_no      检测顺序号（int，无则为 None）
        fine_code   规格+生产线编码，e.g. "HYZB"
        coarse_code 规格代号，e.g. "HYZ"
        spec_name   规格名称，e.g. "云烟（紫）"
        line_code   生产线代码，e.g. "B"
        lot_id      批次编号字符串，e.g. "125030302"
        is_special  True 表示非产品样本（空白/清洗剂/环境样等）
    """
    stem = folder_name.removesuffix(".D").removesuffix(".d").strip()
    # 去除名称中可能存在的空格（如 "HYZBC S125..."）
    stem_nsp = stem.replace(" ", "")
    stem_up = stem_nsp.upper()

    # ── 1. 空白样 ────────────────────────────────────
    if any(k in stem_up for k in ("空白", "BLANK", "BALNK")):
        return _special("BLANK", "BLANK", "空白样", stem)

    # ── 2. 特殊物料 ──────────────────────────────────
    for keywords, code, name in (
        (("管道清洗剂",), "CLEANER", "管道清洗剂"),
        (("热熔胶",),    "HOTMELT", "热熔胶"),
        (("糖料",),      "SUGAR",   "糖料"),
        (("内标",),      "INSTD",   "内标"),
    ):
        if any(k in stem for k in keywords):
            return _special(code, code, name, stem)

    # ── 3. 疑似污染 ──────────────────────────────────
    if any(k in stem for k in ("凝似被污染", "污染")):
        return _special("CONTAM", "CONTAM", "疑似污染", stem)

    # ── 4. 乌兰参照样（WYZX 前缀，参照样品不归入产品） ──
    no_seq = re.sub(r"^\d+#", "", stem_nsp)
    if no_seq.upper().startswith("WYZX"):
        return _special("WYZX", "WYZ", "乌兰参照样", stem)

    # ── 5. 气调杀虫取样点（序号#N号取样点描述）────────
    # no_seq = 去掉序号后的 stem，此时在第 4 步之后已定义
    if re.match(r"^\d+号取样点", no_seq):
        return _special("ENV", "ENV", "环境取样", stem)

    # ── 6. 标准英文编码：NNNN#<规格><生产线>CS<批次> ──
    # 先提取顺序号
    seq_match = re.match(r"^(\d+)#", stem_nsp)
    seq_no = int(seq_match.group(1)) if seq_match else None

    # 修正遗漏 H 前缀的个别文件（如 13443#88CCS...）
    body = no_seq
    if re.match(r"^[89][89][A-C]CS", body, re.IGNORECASE):
        body = "H" + body

    m = _RE_STD.match(body)
    if m:
        spec_code = m.group(1).upper()   # 规格代号，e.g. "HYZ"
        line_code  = m.group(2).upper()  # 生产线，e.g. "B"
        fine_code  = spec_code + line_code  # e.g. "HYZB"
        # 跳过 "<spec><line>CS"，len(spec_code)+1+2 = len(spec_code)+3
        lot_id     = body[len(spec_code) + 1 + 2:]  # 跳过生产线+"CS"，e.g. "125030302"
        # 去除括号注释（部分批次编号后有括号说明）
        lot_id = re.sub(r"\(.*", "", lot_id).strip()

        coarse_code = FINE_TO_COARSE.get(fine_code)
        if coarse_code is None:
            # fine 编码不在映射表中，降级为规格代号本身
            coarse_code = spec_code
        spec_name = SPEC_NAME_MAP.get(coarse_code, coarse_code)
        is_special = coarse_code not in SPEC_NAME_MAP

        return {
            "seq_no":      seq_no,
            "fine_code":   fine_code,
            "coarse_code": coarse_code,
            "spec_name":   spec_name,
            "line_code":   line_code,
            "lot_id":      lot_id,
            "is_special":  is_special,
        }

    # ── 7. 第一批次中文命名兜底 ──────────────────────
    for ch_kw, coarse_code in (
        ("云烟（紫）", "HYZ"), ("云烟(紫)", "HYZ"), ("中支小重九", "XCJ"),
    ):
        if ch_kw in stem:
            spec_name = SPEC_NAME_MAP.get(coarse_code, coarse_code)
            is_ref = "对照" in stem or "参照" in stem
            return {
                "seq_no":      seq_no if seq_match else None,
                "fine_code":   coarse_code,
                "coarse_code": coarse_code,
                "spec_name":   spec_name,
                "line_code":   "",
                "lot_id":      "",
                "is_special":  is_ref,
            }

    # ── 8. 无法识别 ──────────────────────────────────
    return _special("UNKNOWN", "UNKNOWN", "未知", stem)


def _special(fine_code: str, coarse_code: str, spec_name: str, stem: str) -> dict:
    seq_match = re.match(r"^(\d+)#", stem)
    return {
        "seq_no":      int(seq_match.group(1)) if seq_match else None,
        "fine_code":   fine_code,
        "coarse_code": coarse_code,
        "spec_name":   spec_name,
        "line_code":   "",
        "lot_id":      "",
        "is_special":  True,
    }


def scan_dataset(root: str) -> pd.DataFrame:
    """
    扫描 dataset/ 目录，收集所有 .D 文件夹的元数据。

    返回列：
        sample_id       全局唯一样品 ID，格式 B{批次序号:02d}_{顺序号:04d}_{批次编号}
        d_path          .D 文件夹绝对路径
        d_name          .D 文件夹名称
        batch_idx       批次序号（0-based，按文件夹名排序）
        batch_name      批次文件夹名，e.g. "20250310"
        seq_no          检测顺序号
        fine_code       规格+生产线编码，e.g. "HYZB"
        coarse_code     规格代号，e.g. "HYZ"
        spec_name       规格名称，e.g. "云烟（紫）"
        line_code       生产线代码，e.g. "B"
        lot_id          批次编号，e.g. "125030302"
        product_fine    同 fine_code（dataset.py 兼容列）
        product_coarse  同 coarse_code（dataset.py 兼容列）
        is_special      True 表示非产品样本
    """
    root = Path(root)
    rows = []

    batch_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.name
    )

    for batch_idx, batch_dir in enumerate(batch_dirs):
        batch_name = batch_dir.name
        d_folders = sorted(
            [p for p in batch_dir.iterdir() if p.is_dir() and p.suffix.lower() == ".d"]
        )
        for d_path in d_folders:
            info = parse_d_name(d_path.name)
            seq_str = f"{info['seq_no']:04d}" if info["seq_no"] is not None else "0000"
            sample_id = f"B{batch_idx:02d}_{seq_str}_{info['lot_id'] or d_path.stem}"
            rows.append({
                "sample_id":      sample_id,
                "d_path":         str(d_path),
                "d_name":         d_path.name,
                "batch_idx":      batch_idx,
                "batch_name":     batch_name,
                "seq_no":         info["seq_no"],
                "fine_code":      info["fine_code"],
                "coarse_code":    info["coarse_code"],
                "spec_name":      info["spec_name"],
                "line_code":      info["line_code"],
                "lot_id":         info["lot_id"],
                # 保持与 dataset.py 兼容的列名
                "product_fine":   info["fine_code"],
                "product_coarse": info["coarse_code"],
                "is_special":     info["is_special"],
            })

    df = pd.DataFrame(rows)
    print(f"扫描完成: {len(df)} 个样本, "
          f"{df['batch_idx'].nunique()} 个批次, "
          f"{df['fine_code'].nunique()} 种产品(fine), "
          f"{df['coarse_code'].nunique()} 种产品(coarse)")
    return df


def convert_all(metadata: pd.DataFrame, out_dir: str, cfg: Config):
    """逐样本读取 .D 并生成 .npz 张量。"""
    out_dir = Path(out_dir)
    tensor_dir = out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    rt_ranges = []
    success, fail = 0, 0
    tensor_paths = []

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="转换中"):
        npz_path = tensor_dir / f"{idx:04d}.npz"
        tensor_paths.append(str(npz_path))

        if npz_path.exists():
            success += 1
            continue

        try:
            tensor, grid, actual_rt = d_folder_to_tensor(
                row["d_path"],
                rt_bins=cfg.rt_bins, mz_bins=cfg.mz_bins,
                rt_range=cfg.rt_range, mz_range=cfg.mz_range,
            )
            np.savez_compressed(npz_path, tensor=tensor, grid=grid)
            rt_ranges.append(actual_rt)
            success += 1
        except Exception as e:
            print(f"\n  ✗ {row['d_name']}: {e}")
            # 保存空张量做占位
            empty = np.zeros((cfg.in_channels, cfg.rt_bins, cfg.mz_bins), dtype=np.float32)
            np.savez_compressed(npz_path, tensor=empty, grid=np.zeros(1))
            fail += 1

    metadata = metadata.copy()
    metadata["tensor_path"] = tensor_paths

    meta_path = out_dir / "metadata.csv"
    metadata.to_csv(meta_path, index=False, encoding="utf-8-sig")

    info = {
        "rt_bins": cfg.rt_bins, "mz_bins": cfg.mz_bins,
        "mz_range": list(cfg.mz_range),
        "success": success, "fail": fail,
    }
    (out_dir / "grid_info.json").write_text(json.dumps(info, indent=2))
    print(f"\n转换完成: 成功 {success}, 失败 {fail}")
    print(f"元数据: {meta_path}")
    return metadata


# ── CLI ──────────────────────────────────────────────────
if __name__ == "__main__":
    metadata = scan_dataset(cfg.dataset_root)
    print("\n产品分布（fine）:")
    print(metadata[["fine_code", "spec_name"]].drop_duplicates()
          .sort_values("fine_code").to_string(index=False))
    print("\n各规格样本数:")
    print(metadata["fine_code"].value_counts().to_string())
    print("\n批次分布:")
    print(metadata.groupby("batch_name")["sample_id"].count().to_string())
    print(f"\nUNKNOWN 样本数: {(metadata['fine_code']=='UNKNOWN').sum()}")
    unknown_df = metadata[metadata["fine_code"] == "UNKNOWN"][["batch_name", "d_name"]]
    if len(unknown_df):
        print(unknown_df.to_string(index=False))

    ans = input("\n开始转换张量? (y/n): ").strip().lower()
    if ans == "y":
        convert_all(metadata, cfg.prepared_dir, cfg)