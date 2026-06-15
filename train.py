"""
单阶段统一训练引擎:
  L = L_supcon + λ₁·L_adv + λ₂·L_proto + λ_recon·L_recon
  验证使用原型匹配准确率；训练结束后注册最终原型。

训练单一模型 (非 LOBO), 数据划分由 prepared_data/split.json 决定。
"""
import json, time
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

from config import Config
from dataset import GCMSDataset, GCMSAugmentation, load_data_split
from models import GCMSConsistencyNet
from losses import UnifiedLoss
from register import register_from_loader


def _progress_enabled():
    """控制是否显示 tqdm 进度条。默认关闭，避免污染日志。"""
    return os.environ.get("GCMS_SHOW_PROGRESS", "0") == "1"


def _batch_tic(batch, device):
    tic = batch.get("tic") if isinstance(batch, dict) else None
    return tic.to(device, non_blocking=True) if torch.is_tensor(tic) else None


def _cuda_amp_enabled(cfg, device):
    return bool(
        getattr(cfg, "amp_enabled", True)
        and getattr(device, "type", "cpu") == "cuda"
        and torch.cuda.is_available()
    )


def _amp_dtype(cfg):
    dtype_name = str(getattr(cfg, "amp_dtype", "float16") or "float16").lower()
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


def _channels_last_enabled(cfg, device):
    return bool(
        getattr(cfg, "channels_last", True)
        and getattr(device, "type", "cpu") == "cuda"
    )


def _move_batch_to_device(batch, device, channels_last=False):
    batch_dev = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True)
            if k == "input" and channels_last and v.ndim == 4:
                v = v.contiguous(memory_format=torch.channels_last)
        batch_dev[k] = v
    return batch_dev


def _maybe_optimize_model(model, cfg, device):
    if _channels_last_enabled(cfg, device):
        model = model.to(memory_format=torch.channels_last)

    if bool(getattr(cfg, "torch_compile", False)) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("  torch.compile: enabled")
        except Exception as exc:
            print(f"  torch.compile: disabled ({exc})")
    return model


def _model_state_dict_for_save(model):
    return getattr(model, "_orig_mod", model).state_dict()


def _load_model_state(model, state_dict):
    getattr(model, "_orig_mod", model).load_state_dict(state_dict)


def _make_grad_scaler(cfg, device):
    enabled = _cuda_amp_enabled(cfg, device)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_early_stop_controls(cfg, total_epochs, initial_lr):
    """解析早停保护参数，避免在未进入收敛区时提前停止。"""
    patience = int(getattr(cfg, "early_stop_patience", 0) or 0)
    if patience <= 0:
        return {
            "enabled": False,
            "patience": 0,
            "min_epochs": 0,
            "min_lr": 0.0,
            "min_delta": 0.0,
        }

    ratio = float(getattr(cfg, "min_epoch_ratio_before_early_stop", 0.6) or 0.6)
    ratio = float(min(max(ratio, 0.0), 1.0))
    min_epochs_cfg = int(getattr(cfg, "min_epochs_before_early_stop", 0) or 0)
    min_epochs_ratio = int(np.ceil(total_epochs * ratio))
    min_epochs = max(min_epochs_cfg, min_epochs_ratio)
    min_epochs = min(max(min_epochs, 1), int(total_epochs))

    lr_ratio = float(getattr(cfg, "early_stop_min_lr_ratio", 0.2) or 0.2)
    lr_ratio = float(min(max(lr_ratio, 0.0), 1.0))
    min_lr = float(initial_lr) * lr_ratio

    min_delta = float(getattr(cfg, "early_stop_min_delta", 0.0) or 0.0)
    min_delta = max(min_delta, 0.0)

    return {
        "enabled": True,
        "patience": patience,
        "min_epochs": min_epochs,
        "min_lr": min_lr,
        "min_delta": min_delta,
    }


def _resolve_eval_interval(cfg, epoch_num, total_epochs):
    """分阶段验证频率: 搜索阶段稀疏, 收敛阶段加密。"""
    fallback = max(int(getattr(cfg, "eval_interval", 10) or 10), 1)
    search_interval = max(
        int(getattr(cfg, "eval_interval_search", fallback) or fallback), 1
    )
    final_interval = max(
        int(getattr(cfg, "eval_interval_final", fallback) or fallback), 1
    )
    final_ratio = float(getattr(cfg, "eval_final_start_ratio", 0.7) or 0.7)
    final_ratio = float(min(max(final_ratio, 0.0), 1.0))
    final_start_epoch = max(1, int(np.ceil(total_epochs * final_ratio)))
    in_final_stage = epoch_num >= final_start_epoch
    interval = final_interval if in_final_stage else search_interval
    return interval, in_final_stage


def _loader_runtime_kwargs(cfg, device):
    workers = max(int(getattr(cfg, "dataloader_workers", 0) or 0), 0)
    pin_memory_cfg = bool(getattr(cfg, "dataloader_pin_memory", True))
    pin_memory = bool(pin_memory_cfg and getattr(device, "type", "cpu") == "cuda")

    kwargs = {
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(
            getattr(cfg, "dataloader_persistent_workers", True)
        )
        kwargs["prefetch_factor"] = max(
            int(getattr(cfg, "dataloader_prefetch_factor", 2) or 2), 1
        )
    return kwargs


def _build_proto_subset_loader(dataset, cfg, device):
    """训练中期验证: 仅用训练子集构原型, 降低验证成本。"""
    n_total = len(dataset)
    if n_total <= 0:
        return None

    ratio = float(getattr(cfg, "proto_val_subset_ratio", 0.35) or 0.35)
    ratio = float(min(max(ratio, 0.0), 1.0))
    min_samples = max(int(getattr(cfg, "proto_val_subset_min_samples", 256) or 256), 1)
    max_samples = max(int(getattr(cfg, "proto_val_subset_max_samples", 1024) or 1024), min_samples)

    target = int(round(n_total * ratio))
    target = max(target, min_samples)
    target = min(target, max_samples, n_total)
    if target >= n_total:
        return None

    labels = dataset.df["product_label"].to_numpy(dtype=np.int64)
    rng = np.random.RandomState(int(getattr(cfg, "seed", 42) or 42) + 2026)

    chosen = []
    all_idx = np.arange(n_total)
    for cls in np.unique(labels):
        cls_idx = all_idx[labels == cls]
        if len(cls_idx) == 0:
            continue
        chosen.append(int(rng.choice(cls_idx)))

    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < target:
        pool = np.setdiff1d(all_idx, np.array(chosen, dtype=np.int64), assume_unique=False)
        need = target - len(chosen)
        if need > 0 and len(pool) > 0:
            extra = rng.choice(pool, size=min(need, len(pool)), replace=False)
            chosen.extend(extra.tolist())

    chosen = sorted(set(int(i) for i in chosen))
    if len(chosen) >= n_total:
        return None

    subset = Subset(dataset, chosen)
    return DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        **_loader_runtime_kwargs(cfg, device),
    )


def build_loaders(metadata_csv, train_idx, val_idx, cfg, product_col,
                  input_transform=None, device=None):
    aug = GCMSAugmentation(cfg)
    ds_train = GCMSDataset(metadata_csv, product_col=product_col,
                                                     augmentation=aug, indices=train_idx,
                                                     input_transform=input_transform,
                                                     cfg=cfg)
    ds_val = GCMSDataset(metadata_csv, product_col=product_col,
                                                 augmentation=None, indices=val_idx,
                                                 input_transform=input_transform,
                                                 cfg=cfg)

    # 统一编码器: 合并 train/val 所有可能的标签值，避免 LOBO 中
    # 验证批次不在训练编码器中的问题
    all_batch_vals = sorted(
        set(ds_train.df["batch_idx"].unique())
        | set(ds_val.df["batch_idx"].unique())
    )
    all_product_vals = sorted(
        set(ds_train.df[product_col].unique())
        | set(ds_val.df[product_col].unique())
    )

    shared_batch_enc = LabelEncoder().fit(all_batch_vals)
    shared_product_enc = LabelEncoder().fit(all_product_vals)

    for ds in (ds_train, ds_val):
        ds.product_enc = shared_product_enc
        ds.batch_enc = shared_batch_enc
        ds.df["product_label"] = shared_product_enc.transform(
            ds.df[product_col]
        )
        ds.df["batch_label"] = shared_batch_enc.transform(
            ds.df["batch_idx"]
        )
        ds.num_products = len(shared_product_enc.classes_)
        ds.num_batches = len(shared_batch_enc.classes_)

    device = torch.device("cpu") if device is None else device
    loader_kwargs = _loader_runtime_kwargs(cfg, device)
    loader_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        sampler=_build_balanced_sampler(ds_train),
        drop_last=True,
        **loader_kwargs,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return ds_train, ds_val, loader_train, loader_val


def _build_input_transform_for_training(cfg, metadata_csv, train_idx):
    """可选输入变换: raw tensor 先做 RT 轴 PCA 再送入 README 主模型。"""
    enabled = bool(getattr(cfg, "input_raw_pca_enabled", False))
    if not enabled:
        return None, None

    n_comp = int(getattr(cfg, "input_raw_pca_components", 128) or 128)

    grid_info_path = Path(cfg.prepared_dir) / "grid_info.json"
    if grid_info_path.exists():
        try:
            grid_info = json.loads(grid_info_path.read_text(encoding="utf-8"))
            if bool(grid_info.get("input_pca_precomputed", False)):
                precomputed_comp = int(grid_info.get("input_pca_components", n_comp))
                precomputed_rt = int(grid_info.get("input_pca_rt_bins", cfg.rt_bins))
                cfg.rt_bins = precomputed_rt
                cfg.mz_bins = precomputed_comp
                setattr(cfg, "_input_pca_precomputed_active", True)
                print(
                    "  [Input PCA] 检测到 prepared_data 已预先分解, "
                    f"直接使用落盘 tensor (rt_bins={precomputed_rt}, n_components={precomputed_comp})"
                )
                return None, None
        except Exception as e:
            print(f"  [Input PCA] 读取 grid_info 失败, 回退运行时 PCA: {e}")

    from input_pca import RtAxisPcaTransform, load_or_fit_rt_axis_pca

    cache_root = Path(cfg.prepared_dir) / "cache" / "input_pca"
    print(f"  [Input PCA] 检查缓存: n_components={n_comp}, cache_dir={cache_root}")
    pca_model, pca_meta, cache_hit, cache_model_path = load_or_fit_rt_axis_pca(
        metadata_csv=metadata_csv,
        indices=train_idx,
        n_components=n_comp,
        cache_root=cache_root,
    )
    source = "cache_hit" if cache_hit else "cache_miss_fit"
    print(
        f"  [Input PCA] 准备完成({source}): "
        f"samples={pca_meta.get('n_tensors', 'na')}, rows={pca_meta.get('n_rows', 'na')}, "
        f"width={pca_meta.get('input_width', 'na')}->{pca_meta.get('n_components', n_comp)}, "
        f"model={cache_model_path}"
    )

    # 解码器重建尺寸需要和变换后的输入宽度保持一致。
    cfg.mz_bins = int(n_comp)
    return RtAxisPcaTransform(pca_model), pca_model


def _build_balanced_sampler(dataset):
    """
    SAIM 风格类别均衡采样器。

    参考 gyfseer/SAIM base_dataset.py 的 prob_based_sample:
    每个类采样等量样本, 类内按 softmax(score) 概率分布采样。
    此处初始 score 均为 1.0 (等概率), 保证少数类不被淹没。

    原理: 先计算每个样本的采样权重 = 1 / (该类的样本数),
    使得总体采样时各类期望出现次数相等。
    """
    labels = dataset.df["product_label"].values
    class_counts = np.bincount(labels)
    # 权重 = 1 / class_count (每个样本)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(dataset),
        replacement=True,
    )
    return sampler


def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    total_epochs, cfg=None, scaler=None):
    model.train()
    running = {}
    amp_enabled = _cuda_amp_enabled(cfg, device) if cfg is not None else False
    amp_dtype = _amp_dtype(cfg) if cfg is not None else torch.float16
    channels_last = _channels_last_enabled(cfg, device) if cfg is not None else False

    # DANN alpha 渐进增大
    p = epoch / total_epochs
    alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    model.domain_head.set_alpha(alpha)

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{total_epochs}",
        leave=False,
        ncols=100,
        disable=not _progress_enabled(),
    )
    for batch in pbar:
        batch_dev = _move_batch_to_device(batch, device, channels_last=channels_last)
        x = batch_dev["input"]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            out = model(x, tic=batch_dev.get("tic"))
            losses = criterion(out, batch_dev)
            loss = losses["total"]

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.item()

        pbar.set_postfix(loss=f"{loss.item():.3f}")

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def validate_with_prototypes(model, train_loader_noaug, val_loader,
                              label_names, device, cfg):
    """构建训练集原型，在验证集上做原型匹配评估。"""
    model.eval()
    proto_store, _, _ = register_from_loader(
        model, train_loader_noaug, label_names, device,
        percentile=cfg.accept_percentile)

    correct, total = 0, 0
    correct_flags = []
    score_vals = []
    for batch in tqdm(
        val_loader,
        desc="验证",
        leave=False,
        ncols=80,
        disable=not _progress_enabled(),
    ):
        x = batch["input"].to(device)
        z = model.encode(x, tic=_batch_tic(batch, device))
        result = proto_store.predict(z)
        corr = (result["pred_idx"].cpu() == batch["product"])
        correct += corr.sum().item()
        total += len(batch["product"])
        correct_flags.extend(corr.numpy().astype(int).tolist())
        score_vals.extend(result["scores"].detach().cpu().numpy().tolist())

    val_acc = correct / max(total, 1)
    auroc_correct = float("nan")
    if len(set(correct_flags)) > 1:
        auroc_correct = float(roc_auc_score(correct_flags, score_vals))

    # 联合指标: 精度为主, 分数可分性为辅
    val_metric = val_acc
    if not np.isnan(auroc_correct):
        val_metric = val_acc + 0.05 * auroc_correct

    return {
        "acc": float(val_acc),
        "auroc_correct": float(auroc_correct),
        "metric": float(val_metric),
    }, proto_store


def run_fold(fold_idx, train_idx, val_idx, batch_name, metadata_csv, cfg):
    """运行一个 fold 的单阶段统一训练。(保留用于 LOBO 交叉验证)"""
    from config import get_device
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}: 测试批次 = {batch_name}, device = {device}")
    print(f"{'='*60}")

    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")

    input_transform, _ = _build_input_transform_for_training(
        cfg, metadata_csv, train_idx
    )

    ds_train, ds_val, loader_train, loader_val = build_loaders(
        metadata_csv, train_idx, val_idx, cfg, product_col,
        input_transform=input_transform,
        device=device,
    )

    num_batches = ds_train.num_batches
    print(f"  训练: {len(ds_train)} 样本, 验证: {len(ds_val)} 样本")
    print(f"  产品数: {ds_train.num_products}, 批次数(训练): {num_batches}")
    print(
        "  DataLoader: "
        f"workers={max(int(getattr(cfg, 'dataloader_workers', 0) or 0), 0)}, "
        f"pin_memory={bool(getattr(cfg, 'dataloader_pin_memory', True))}, "
        f"prefetch_factor={int(getattr(cfg, 'dataloader_prefetch_factor', 2) or 2)}"
    )
    print(
        "  Performance: "
        f"amp={_cuda_amp_enabled(cfg, device)}, "
        f"amp_dtype={getattr(cfg, 'amp_dtype', 'float16')}, "
        f"channels_last={_channels_last_enabled(cfg, device)}, "
        f"dataset_cache={bool(getattr(cfg, 'dataset_cache_in_memory', False))}, "
        f"torch_compile={bool(getattr(cfg, 'torch_compile', False))}"
    )

    if getattr(device, "type", "cpu") == "cuda" and bool(getattr(cfg, "cuda_benchmark", True)):
        torch.backends.cudnn.benchmark = True

    model = GCMSConsistencyNet(num_batches, cfg).to(device)
    model = _maybe_optimize_model(model, cfg, device)
    criterion = UnifiedLoss(cfg).to(device)

    # 无增强训练集 (原型计算用)
    ds_train_noaug = GCMSDataset(metadata_csv, product_col=product_col,
                                 augmentation=None, indices=train_idx,
                                 input_transform=input_transform,
                                 cfg=cfg)
    ds_train_noaug.product_enc = ds_train.product_enc
    ds_train_noaug.batch_enc = ds_train.batch_enc
    ds_train_noaug.df["product_label"] = ds_train.product_enc.transform(
        ds_train_noaug.df[product_col]
    )
    ds_train_noaug.df["batch_label"] = ds_train.batch_enc.transform(
        ds_train_noaug.df["batch_idx"]
    )
    loader_train_noaug = DataLoader(
        ds_train_noaug,
        batch_size=cfg.batch_size,
        shuffle=False,
        **_loader_runtime_kwargs(cfg, device),
    )
    loader_train_noaug_subset = _build_proto_subset_loader(ds_train_noaug, cfg, device)
    label_names = ds_train.get_label_name_map()

    # ── 单阶段训练 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )
    scaler = _make_grad_scaler(cfg, device)
    early_stop_ctrl = _resolve_early_stop_controls(cfg, cfg.epochs, cfg.lr)

    best_acc = 0
    best_metric = -1e9
    best_state = None
    no_improve_checks = 0
    eval_checks = 0

    for epoch in range(cfg.epochs):
        m_train = train_one_epoch(model, loader_train, criterion, optimizer,
                                  device, epoch, cfg.epochs, cfg=cfg, scaler=scaler)
        scheduler.step()

        # 关闭 tqdm 时仍保留每轮可观测日志，避免看起来像“卡住”
        print(f"  Epoch {epoch+1}/{cfg.epochs}  "
              f"supcon={m_train.get('supcon',0):.3f} "
              f"adv={m_train.get('adv',0):.3f} "
              f"proto={m_train.get('proto',0):.3f} "
              f"total={m_train.get('total',0):.3f}")

        # 按配置间隔做原型验证；若启用前N轮淘汰，确保该轮次会触发验证
        epoch_num = epoch + 1
        eval_interval, in_final_stage = _resolve_eval_interval(cfg, epoch_num, cfg.epochs)
        guard_epoch = max(int(getattr(cfg, "warmup_guard_epoch", 10)), 1)
        should_eval = (
            (epoch_num % eval_interval == 0)
            or (epoch == cfg.epochs - 1)
            or (epoch_num == guard_epoch)
        )
        if should_eval:
            eval_checks += 1
            full_every = max(int(getattr(cfg, "proto_val_full_every", 3) or 3), 1)
            use_full_proto = (
                loader_train_noaug_subset is None
                or in_final_stage
                or (epoch == cfg.epochs - 1)
                or (eval_checks % full_every == 0)
            )
            proto_loader = loader_train_noaug if use_full_proto else loader_train_noaug_subset

            val_m, _ = validate_with_prototypes(
                model, proto_loader, loader_val,
                label_names, device, cfg)
            val_acc = float(val_m["acc"])
            val_metric = float(val_m["metric"])
            val_auroc = float(val_m["auroc_correct"])
            if val_metric > (best_metric + early_stop_ctrl["min_delta"]):
                best_metric = val_metric
                best_acc = val_acc
                best_state = {k: v.cpu().clone()
                              for k, v in _model_state_dict_for_save(model).items()}
                no_improve_checks = 0
            else:
                no_improve_checks += 1
            print(
                f"    -> val_acc={val_acc:.3f}, val_auroc={val_auroc:.3f}, "
                f"val_metric={val_metric:.3f} (best_metric={best_metric:.3f}, "
                f"proto_scope={'full' if use_full_proto else 'subset'})"
            )

            # 规则: 前N轮若显著落后于当前最佳方案, 直接淘汰该策略
            guard_enabled = bool(getattr(cfg, "warmup_guard_enabled", False))
            guard_ref = float(getattr(cfg, "warmup_guard_best_at_epoch", 0.0) or 0.0)
            guard_compare_best = bool(getattr(cfg, "warmup_guard_compare_best", True))
            guard_ratio = float(getattr(cfg, "warmup_guard_min_ratio", 1.0) or 1.0)
            if guard_enabled and guard_ref > 0 and epoch_num == guard_epoch:
                min_allowed = guard_ref if guard_compare_best else (guard_ref * guard_ratio)
                if val_acc < min_allowed:
                    rule_info = (
                        f"compare_to_best@{guard_epoch}"
                        if guard_compare_best
                        else f"ratio={guard_ratio:.2f}"
                    )
                    print(
                        f"    -> warmup guard stop: val_acc={val_acc:.3f} < "
                        f"{min_allowed:.3f} (best@{guard_epoch}={guard_ref:.3f}, "
                        f"{rule_info})"
                    )
                    break

            if early_stop_ctrl["enabled"] and no_improve_checks >= early_stop_ctrl["patience"]:
                current_lr = float(optimizer.param_groups[0].get("lr", cfg.lr))
                epoch_ready = epoch_num >= early_stop_ctrl["min_epochs"]
                lr_ready = current_lr <= early_stop_ctrl["min_lr"]
                if epoch_ready and lr_ready:
                    print(
                        "    -> early stop triggered "
                        f"(patience={early_stop_ctrl['patience']}, "
                        f"epoch={epoch_num}>={early_stop_ctrl['min_epochs']}, "
                        f"lr={current_lr:.6g}<={early_stop_ctrl['min_lr']:.6g})"
                    )
                    break
                print(
                    "    -> hold early stop "
                    f"(no_improve={no_improve_checks}/{early_stop_ctrl['patience']}, "
                    f"epoch_ready={epoch_ready}, lr_ready={lr_ready})"
                )

    if best_state is not None:
        _load_model_state(model, best_state)

    # 注册最终原型
    proto_store, all_z, all_labels = register_from_loader(
        model, loader_train_noaug, label_names, device,
        percentile=cfg.accept_percentile
    )
    proto_store.summary()

    # 保存
    fold_dir = Path(cfg.output_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_model_state_dict_for_save(model), fold_dir / "model.pt")
    proto_store.save(fold_dir / "prototypes")
    with open(fold_dir / "product_classes.json", "w") as f:
        json.dump(list(ds_train.product_enc.classes_), f)

    return model, proto_store, ds_train, ds_val, loader_val


# ═════════════════════════════════════════════════════════
#  单模型训练 (使用 prepared_data/split.json 划分)
# ═════════════════════════════════════════════════════════
def train_single_model(cfg: Config):
    """
    训练一个最终模型 (非交叉验证)。

        从 split.json 读取固定划分:
            train_idx → 训练
            val_idx   → 伪 holdout 批次验证 (模型选择)
    训练结束后注册原型并保存。
    """
    if str(getattr(cfg, "primary_model", "")).strip().lower() == "raw_pca_mlp":
        from raw_pca_pipeline import train_single_model_raw_pca

        return train_single_model_raw_pca(cfg)

    from config import get_device
    set_seed(cfg.seed)

    split = load_data_split(cfg)
    device = get_device()
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")

    print(f"\n{'='*60}")
    print(f"训练单一模型, device = {device}")
    print(f"{'='*60}")
    print(f"  已知产品: {split['known_products']}")
    print(f"  留出产品: {split['holdout_products']}  (Setting B/C)")
    print(f"  留出批次: {split['holdout_batches']}  (Setting A)")
    if split.get("model_select_holdout_batches"):
        print(
            f"  伪验证批次: {split.get('model_select_holdout_batches')} "
            "(训练早停选模)"
        )

    input_transform, input_pca_model = _build_input_transform_for_training(
        cfg, metadata_csv, split["train_idx"]
    )

    # ── 构建数据集 ──
    ds_train, ds_val, loader_train, loader_val = build_loaders(
        metadata_csv, split["train_idx"], split["val_idx"], cfg, product_col,
        input_transform=input_transform,
        device=device,
    )

    num_batches = ds_train.num_batches
    print(f"  训练: {len(ds_train)} 样本, 验证: {len(ds_val)} 样本")
    print(f"  产品数: {ds_train.num_products}, 批次数: {num_batches}")
    print(
        "  DataLoader: "
        f"workers={max(int(getattr(cfg, 'dataloader_workers', 0) or 0), 0)}, "
        f"pin_memory={bool(getattr(cfg, 'dataloader_pin_memory', True))}, "
        f"prefetch_factor={int(getattr(cfg, 'dataloader_prefetch_factor', 2) or 2)}"
    )
    print(
        "  Performance: "
        f"amp={_cuda_amp_enabled(cfg, device)}, "
        f"amp_dtype={getattr(cfg, 'amp_dtype', 'float16')}, "
        f"channels_last={_channels_last_enabled(cfg, device)}, "
        f"dataset_cache={bool(getattr(cfg, 'dataset_cache_in_memory', False))}, "
        f"torch_compile={bool(getattr(cfg, 'torch_compile', False))}"
    )

    if getattr(device, "type", "cpu") == "cuda" and bool(getattr(cfg, "cuda_benchmark", True)):
        torch.backends.cudnn.benchmark = True

    model = GCMSConsistencyNet(num_batches, cfg).to(device)
    model = _maybe_optimize_model(model, cfg, device)
    criterion = UnifiedLoss(cfg).to(device)

    # 无增强训练集 (原型计算用)
    ds_train_noaug = GCMSDataset(metadata_csv, product_col=product_col,
                                 augmentation=None,
                                 indices=split["train_idx"],
                                 input_transform=input_transform,
                                 cfg=cfg)
    ds_train_noaug.product_enc = ds_train.product_enc
    ds_train_noaug.batch_enc = ds_train.batch_enc
    ds_train_noaug.df["product_label"] = ds_train.product_enc.transform(
        ds_train_noaug.df[product_col])
    ds_train_noaug.df["batch_label"] = ds_train.batch_enc.transform(
        ds_train_noaug.df["batch_idx"])
    loader_train_noaug = DataLoader(
        ds_train_noaug,
        batch_size=cfg.batch_size,
        shuffle=False,
        **_loader_runtime_kwargs(cfg, device),
    )
    loader_train_noaug_subset = _build_proto_subset_loader(ds_train_noaug, cfg, device)
    label_names = ds_train.get_label_name_map()

    # ── 训练 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs)
    scaler = _make_grad_scaler(cfg, device)
    early_stop_ctrl = _resolve_early_stop_controls(cfg, cfg.epochs, cfg.lr)

    best_acc = 0
    best_metric = -1e9
    best_state = None
    no_improve_checks = 0
    eval_checks = 0

    for epoch in range(cfg.epochs):
        m_train = train_one_epoch(model, loader_train, criterion, optimizer,
                                  device, epoch, cfg.epochs, cfg=cfg, scaler=scaler)
        scheduler.step()

        # 关闭 tqdm 时仍保留每轮可观测日志，避免看起来像“卡住”
        print(f"  Epoch {epoch+1}/{cfg.epochs}  "
              f"supcon={m_train.get('supcon',0):.3f} "
              f"adv={m_train.get('adv',0):.3f} "
              f"proto={m_train.get('proto',0):.3f} "
              f"total={m_train.get('total',0):.3f}")

        epoch_num = epoch + 1
        eval_interval, in_final_stage = _resolve_eval_interval(cfg, epoch_num, cfg.epochs)
        guard_epoch = max(int(getattr(cfg, "warmup_guard_epoch", 10)), 1)
        should_eval = (
            (epoch_num % eval_interval == 0)
            or (epoch == cfg.epochs - 1)
            or (epoch_num == guard_epoch)
        )
        if should_eval:
            eval_checks += 1
            full_every = max(int(getattr(cfg, "proto_val_full_every", 3) or 3), 1)
            use_full_proto = (
                loader_train_noaug_subset is None
                or in_final_stage
                or (epoch == cfg.epochs - 1)
                or (eval_checks % full_every == 0)
            )
            proto_loader = loader_train_noaug if use_full_proto else loader_train_noaug_subset

            val_m, _ = validate_with_prototypes(
                model, proto_loader, loader_val,
                label_names, device, cfg)
            val_acc = float(val_m["acc"])
            val_metric = float(val_m["metric"])
            val_auroc = float(val_m["auroc_correct"])
            if val_metric > (best_metric + early_stop_ctrl["min_delta"]):
                best_metric = val_metric
                best_acc = val_acc
                best_state = {k: v.cpu().clone()
                              for k, v in _model_state_dict_for_save(model).items()}
                no_improve_checks = 0
            else:
                no_improve_checks += 1
            print(
                f"    -> val_acc={val_acc:.3f}, val_auroc={val_auroc:.3f}, "
                f"val_metric={val_metric:.3f} (best_metric={best_metric:.3f}, "
                f"proto_scope={'full' if use_full_proto else 'subset'})"
            )

            # 规则: 前N轮若显著落后于当前最佳方案, 直接淘汰该策略
            guard_enabled = bool(getattr(cfg, "warmup_guard_enabled", False))
            guard_ref = float(getattr(cfg, "warmup_guard_best_at_epoch", 0.0) or 0.0)
            guard_compare_best = bool(getattr(cfg, "warmup_guard_compare_best", True))
            guard_ratio = float(getattr(cfg, "warmup_guard_min_ratio", 1.0) or 1.0)
            if guard_enabled and guard_ref > 0 and epoch_num == guard_epoch:
                min_allowed = guard_ref if guard_compare_best else (guard_ref * guard_ratio)
                if val_acc < min_allowed:
                    rule_info = (
                        f"compare_to_best@{guard_epoch}"
                        if guard_compare_best
                        else f"ratio={guard_ratio:.2f}"
                    )
                    print(
                        f"    -> warmup guard stop: val_acc={val_acc:.3f} < "
                        f"{min_allowed:.3f} (best@{guard_epoch}={guard_ref:.3f}, "
                        f"{rule_info})"
                    )
                    break

            if early_stop_ctrl["enabled"] and no_improve_checks >= early_stop_ctrl["patience"]:
                current_lr = float(optimizer.param_groups[0].get("lr", cfg.lr))
                epoch_ready = epoch_num >= early_stop_ctrl["min_epochs"]
                lr_ready = current_lr <= early_stop_ctrl["min_lr"]
                if epoch_ready and lr_ready:
                    print(
                        "    -> early stop triggered "
                        f"(patience={early_stop_ctrl['patience']}, "
                        f"epoch={epoch_num}>={early_stop_ctrl['min_epochs']}, "
                        f"lr={current_lr:.6g}<={early_stop_ctrl['min_lr']:.6g})"
                    )
                    break
                print(
                    "    -> hold early stop "
                    f"(no_improve={no_improve_checks}/{early_stop_ctrl['patience']}, "
                    f"epoch_ready={epoch_ready}, lr_ready={lr_ready})"
                )

    if best_state is not None:
        _load_model_state(model, best_state)

    # ── 注册最终原型 ──
    proto_store, all_z, all_labels = register_from_loader(
        model, loader_train_noaug, label_names, device,
        percentile=cfg.accept_percentile)
    proto_store.summary()

    # ── 保存 ──
    model_dir = Path(cfg.output_dir) / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_model_state_dict_for_save(model), model_dir / "model.pt")
    proto_store.save(model_dir / "prototypes")
    if input_pca_model is not None:
        from input_pca import save_rt_axis_pca

        save_rt_axis_pca(input_pca_model, model_dir / "input_rt_pca.pkl")
    with open(model_dir / "product_classes.json", "w") as f:
        json.dump(list(ds_train.product_enc.classes_), f)
    with open(model_dir / "train_meta.json", "w") as f:
        pca_precomputed = bool(getattr(cfg, "_input_pca_precomputed_active", False))
        pca_enabled = bool(input_pca_model is not None) or pca_precomputed
        json.dump({
            "num_batches": int(num_batches),
            "num_products": int(ds_train.num_products),
            "input_raw_pca_enabled": pca_enabled,
            "input_raw_pca_precomputed": pca_precomputed,
            "input_raw_pca_components": int(getattr(cfg, "input_raw_pca_components", cfg.mz_bins)),
            "input_raw_pca_rt_bins": int(getattr(cfg, "rt_bins", 0)),
            "tic_branch_enabled": bool(getattr(cfg, "tic_branch_enabled", False)),
            "tic_encoder": str(getattr(cfg, "tic_encoder", "cnn1d")),
            "tic_embed_dim": int(getattr(cfg, "tic_embed_dim", 64)),
            "tic_fusion_mode": str(getattr(cfg, "tic_fusion_mode", "concat")),
            "tic_fusion_output_dim": int(getattr(cfg, "tic_fusion_output_dim", cfg.feature_dim)),
        }, f, indent=2)

    print(f"\n模型已保存到 {model_dir}")
    print(f"  最佳验证准确率: {best_acc:.4f}")

    return model, proto_store, ds_train, ds_val


def train_all_folds(cfg: Config):
    """Leave-one-batch-out 全部 fold 训练 (保留用于交叉验证)。"""
    from dataset import unified_splits
    set_seed(cfg.seed)
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")

    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")
    split_info = unified_splits(
        metadata_csv, product_col=product_col,
        num_open_classes=cfg.num_open_test_classes,
        seed=cfg.seed)

    print(f"已知类 ({len(split_info['known_classes'])}): "
          f"{split_info['known_classes']}")
    print(f"未知类 ({len(split_info['unknown_classes'])}): "
          f"{split_info['unknown_classes']}")

    # 保存切分信息
    split_file = Path(cfg.output_dir) / "split_info.json"
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump({
            "known_classes": split_info["known_classes"],
            "unknown_classes": split_info["unknown_classes"],
            "unknown_idx": split_info["unknown_idx"],
            "num_folds": len(split_info["folds"]),
        }, f, indent=2)

    fold_results = []
    num_folds = len(split_info["folds"])
    for fi, fold in enumerate(split_info["folds"]):
        print(f"\n━━ Fold {fi+1}/{num_folds} ━━")
        model, proto_store, ds_train, ds_val, loader_val = run_fold(
            fold["fold_idx"], fold["train_idx"], fold["val_idx"],
            fold["test_batch"], metadata_csv, cfg
        )
        fold_results.append({
            "fold": fold["fold_idx"],
            "test_batch": fold["test_batch"],
            "train_idx": fold["train_idx"],
            "model": model,
            "proto_store": proto_store,
            "ds_train": ds_train,
            "ds_val": ds_val,
            "loader_val": loader_val,
        })

    return fold_results, split_info
