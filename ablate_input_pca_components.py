"""Input-PCA component ablation with prepared-data reuse.

Usage:
  .venv/bin/python ablate_input_pca_components.py \
    --base_run iter_auto252_bs16_lr172_a7_p90_fewr50 \
    --components 128,171,192 \
    --lambda_adv 0.07 --epochs 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

from config import Config, get_device
from data_prepare import convert_all, scan_dataset
from dataset import GCMSDataset, create_data_split, load_data_split
from evaluate import evaluate_setting_a
from models import GCMSConsistencyNet
from register import PrototypeStore
from train import train_single_model


def _parse_components(raw: str) -> List[int]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    comps = sorted({int(v) for v in vals})
    if not comps:
        raise ValueError("components 不能为空")
    return comps


def _load_base_config(project_root: Path, base_run: str) -> Dict:
    cfg_path = project_root / "outputs" / base_run / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到 base run config: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return dict(payload.get("config", {}))


def _make_cfg(base_cfg: Dict) -> Config:
    cfg = Config()
    for k, v in base_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _prepared_dir_has_component(prepared_dir: Path, comp: int) -> bool:
    metadata_ok = (prepared_dir / "metadata.csv").exists()
    grid_info_path = prepared_dir / "grid_info.json"
    if not metadata_ok or not grid_info_path.exists():
        return False
    try:
        with open(grid_info_path, "r", encoding="utf-8") as f:
            grid = json.load(f)
        comp_req = int(comp)
        pca_requested = int(grid.get("input_pca_requested_components", -1))
        pca_actual = int(grid.get("input_pca_components", -1))

        # 优先按“请求维度”判断，适配新版本 grid_info。
        if pca_requested == comp_req:
            return True

        # 兼容旧版本: 若请求维度超过原始宽度，实际维度会被截断到 width-1。
        ref_width = int(grid.get("input_pca_ref_mz_axis_len", 0) or 0)
        if ref_width > 1:
            expected_actual = min(comp_req, ref_width - 1)
            return pca_actual == expected_actual

        return pca_actual == comp_req
    except Exception:
        return False


def _ensure_prepared_dir(
    project_root: Path,
    base_cfg: Dict,
    comp: int,
    force_prepare: bool,
) -> Tuple[Path, bool]:
    if comp == 128:
        prepared_dir = project_root / "prepared_data"
    else:
        prepared_dir = project_root / f"prepared_data_pca{comp}"

    has_ready = _prepared_dir_has_component(prepared_dir, comp)
    prepared_now = False

    if force_prepare or not has_ready:
        cfg_prep = _make_cfg(base_cfg)
        cfg_prep.prepared_dir = str(prepared_dir)
        cfg_prep.input_raw_pca_enabled = True
        cfg_prep.input_raw_pca_components = int(comp)
        cfg_prep.prepare_direct_raw_pca = True
        cfg_prep.save_prepare_plots = False
        cfg_prep.save_prepare_tables = False

        print("\n" + "=" * 88)
        print(f"[PCA-ABLATE] PREPARE comp={comp} dir={prepared_dir}")
        print("=" * 88)
        metadata = scan_dataset(cfg_prep.dataset_root)
        convert_all(metadata, str(prepared_dir), cfg_prep)
        prepared_now = True

    cfg_split = _make_cfg(base_cfg)
    cfg_split.prepared_dir = str(prepared_dir)
    cfg_split.input_raw_pca_enabled = True
    cfg_split.input_raw_pca_components = int(comp)
    metadata_csv = str(prepared_dir / "metadata.csv")
    product_col = "product_fine" if cfg_split.product_granularity == "fine" else "product_coarse"
    create_data_split(metadata_csv, cfg_split, product_col=product_col)

    return prepared_dir, prepared_now


def _loader_kwargs(cfg: Config, device: torch.device):
    workers = max(int(getattr(cfg, "dataloader_workers", 0) or 0), 0)
    pin = bool(getattr(cfg, "dataloader_pin_memory", True)) and device.type == "cuda"
    kwargs = {"num_workers": workers, "pin_memory": pin}
    if workers > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg, "dataloader_persistent_workers", True))
        kwargs["prefetch_factor"] = max(int(getattr(cfg, "dataloader_prefetch_factor", 2) or 2), 1)
    return kwargs


def _evaluate_setting_a_only(cfg: Config) -> Dict:
    split = load_data_split(cfg)
    device = get_device()
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = "product_fine" if cfg.product_granularity == "fine" else "product_coarse"
    out_dir = Path(cfg.output_dir)
    model_dir = out_dir / "final_model"

    input_transform = None
    input_pca_path = model_dir / "input_rt_pca.pkl"
    if input_pca_path.exists():
        from input_pca import RtAxisPcaTransform, load_rt_axis_pca

        input_pca_model = load_rt_axis_pca(input_pca_path)
        input_transform = RtAxisPcaTransform(input_pca_model)
        cfg.mz_bins = int(getattr(input_pca_model, "n_components_", cfg.mz_bins))

    from sklearn.preprocessing import LabelEncoder

    full_df = pd.read_csv(metadata_csv)
    full_df = full_df[(full_df["product_fine"] != "BLANK") & (~full_df["is_special"])].reset_index(drop=True)
    global_product_enc = LabelEncoder().fit(sorted(full_df[product_col].unique()))
    global_batch_enc = LabelEncoder().fit(sorted(full_df["batch_idx"].unique()))

    def make_loader(indices):
        ds = GCMSDataset(
            metadata_csv,
            product_col=product_col,
            augmentation=None,
            indices=indices,
            input_transform=input_transform,
        )
        ds.product_enc = global_product_enc
        ds.batch_enc = global_batch_enc
        ds.df["product_label"] = global_product_enc.transform(ds.df[product_col])
        ds.df["batch_label"] = global_batch_enc.transform(ds.df["batch_idx"])
        ds.num_products = len(global_product_enc.classes_)
        ds.num_batches = len(global_batch_enc.classes_)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            **_loader_kwargs(cfg, device),
        )
        return ds, loader

    with open(model_dir / "train_meta.json", "r", encoding="utf-8") as f:
        train_meta = json.load(f)
    num_batches_model = int(train_meta["num_batches"])
    if bool(train_meta.get("input_raw_pca_enabled", False)):
        cfg.mz_bins = int(train_meta.get("input_raw_pca_components", cfg.mz_bins))

    model = GCMSConsistencyNet(num_batches_model, cfg).to(device)
    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
    )

    proto_store = PrototypeStore()
    proto_store.load(model_dir / "prototypes")

    _, loader_train = make_loader(split["train_idx"])
    _, loader_test_batch = make_loader(split["test_batch_idx"])
    result_a = evaluate_setting_a(
        model,
        loader_train,
        loader_test_batch,
        proto_store,
        device,
        cfg,
        fold_name=f"pca_comp={cfg.input_raw_pca_components}",
    )

    pid = result_a["product_identification"]
    return {
        "setting_a_accuracy": float(pid.get("accuracy", float("nan"))),
        "setting_a_macro_f1": float(pid.get("macro_f1", float("nan"))),
        "setting_a_balanced_acc": float(pid.get("balanced_acc", float("nan"))),
        "cross_batch_gap": float(pid.get("cross_batch_gap", float("nan"))),
        "model_select_holdout_batches": split.get("model_select_holdout_batches", []),
        "holdout_batches": split.get("holdout_batches", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Ablate input_raw_pca_components with prepared-data reuse")
    parser.add_argument("--base_run", required=True, help="Base run name under outputs/")
    parser.add_argument("--components", default="128,171,192", help="Comma-separated PCA components")
    parser.add_argument("--lambda_adv", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--name_prefix", default="ablate_p1_pca")
    parser.add_argument(
        "--report_path",
        default="outputs/ablation_pca_components_settingA_p1.json",
        help="Where to store summary JSON",
    )
    parser.add_argument("--force_prepare", action="store_true", help="Force re-prepare even if cache exists")
    parser.add_argument("--skip_existing", action="store_true", help="Skip training if final model already exists")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    base_cfg = _load_base_config(project_root, args.base_run)
    components = _parse_components(args.components)

    results = []
    for comp in components:
        prepared_dir, prepared_now = _ensure_prepared_dir(
            project_root=project_root,
            base_cfg=base_cfg,
            comp=int(comp),
            force_prepare=bool(args.force_prepare),
        )

        run_name = f"{args.name_prefix}{comp}_adv{int(round(args.lambda_adv * 100)):02d}_e{args.epochs}"
        output_dir = project_root / "outputs" / run_name

        cfg = _make_cfg(base_cfg)
        cfg.output_dir = str(output_dir)
        cfg.prepared_dir = str(prepared_dir)
        cfg.input_raw_pca_enabled = True
        cfg.input_raw_pca_components = int(comp)
        cfg.lambda_adv = float(args.lambda_adv)
        cfg.epochs = int(args.epochs)

        # 低成本网格默认开启: 搜索阶段稀疏验证 + 收敛阶段密验证
        cfg.eval_interval_search = max(int(getattr(cfg, "eval_interval_search", 10) or 10), 1)
        cfg.eval_interval_final = max(int(getattr(cfg, "eval_interval_final", 5) or 5), 1)
        cfg.eval_final_start_ratio = float(getattr(cfg, "eval_final_start_ratio", 0.7) or 0.7)

        model_path = output_dir / "final_model" / "model.pt"
        if args.skip_existing and model_path.exists():
            print(f"[PCA-ABLATE] skip train (exists): {run_name}")
        else:
            print("\n" + "=" * 88)
            print(f"[PCA-ABLATE] TRAIN comp={comp} run={run_name}")
            print("=" * 88)
            train_single_model(cfg)

        metric = _evaluate_setting_a_only(cfg)
        row = {
            "run_name": run_name,
            "prepared_dir": str(prepared_dir),
            "prepared_reused": bool(not prepared_now),
            "input_raw_pca_components": int(comp),
            "lambda_adv": float(cfg.lambda_adv),
            "epochs": int(cfg.epochs),
            **metric,
        }
        results.append(row)
        print(
            f"[PCA-ABLATE] DONE comp={comp}: "
            f"A_acc={row['setting_a_accuracy']:.4f}, A_bal={row['setting_a_balanced_acc']:.4f}, "
            f"reuse={row['prepared_reused']}"
        )

    results = sorted(results, key=lambda x: x["setting_a_accuracy"], reverse=True)
    report_path = project_root / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88)
    print("[PCA-ABLATE] SUMMARY (sorted by Setting A accuracy)")
    print("=" * 88)
    for r in results:
        print(
            f"pca={r['input_raw_pca_components']} | "
            f"A_acc={r['setting_a_accuracy']:.4f} | "
            f"A_bal={r['setting_a_balanced_acc']:.4f} | "
            f"A_f1={r['setting_a_macro_f1']:.4f} | "
            f"gap={r['cross_batch_gap']:.4f} | "
            f"reuse={r['prepared_reused']}"
        )
    print(f"[PCA-ABLATE] report: {report_path}")


if __name__ == "__main__":
    main()
