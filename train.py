"""
单阶段统一训练引擎:
  L = L_supcon + λ₁·L_adv + λ₂·L_proto + λ_recon·L_recon
  验证使用原型匹配准确率；训练结束后注册最终原型。
"""
import json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import GCMSDataset, GCMSAugmentation, unified_splits
from models import GCMSConsistencyNet
from losses import UnifiedLoss
from register import register_from_loader


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(metadata_csv, train_idx, val_idx, cfg, product_col):
    aug = GCMSAugmentation(cfg)
    ds_train = GCMSDataset(metadata_csv, product_col=product_col,
                           augmentation=aug, indices=train_idx)
    ds_val = GCMSDataset(metadata_csv, product_col=product_col,
                         augmentation=None, indices=val_idx)

    # 确保编码器一致
    ds_val.product_enc = ds_train.product_enc
    ds_val.batch_enc = ds_train.batch_enc
    ds_val.df["product_label"] = ds_train.product_enc.transform(
        ds_val.df[product_col]
    )
    ds_val.df["batch_label"] = ds_train.batch_enc.transform(
        ds_val.df["batch_idx"]
    )

    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    loader_val = DataLoader(ds_val, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=0)
    return ds_train, ds_val, loader_train, loader_val


def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    total_epochs):
    model.train()
    running = {}

    # DANN alpha 渐进增大
    p = epoch / total_epochs
    alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    model.domain_head.set_alpha(alpha)

    for batch in loader:
        x = batch["input"].to(device)
        batch_dev = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
        batch_dev["input"] = x

        out = model(x)
        losses = criterion(out, batch_dev)
        loss = losses["total"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.item()

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
    for batch in val_loader:
        x = batch["input"].to(device)
        z = model.encode(x)
        result = proto_store.predict(z)
        correct += (result["pred_idx"].cpu() == batch["product"]).sum().item()
        total += len(batch["product"])

    return correct / max(total, 1), proto_store


def run_fold(fold_idx, train_idx, val_idx, batch_name, metadata_csv, cfg):
    """运行一个 fold 的单阶段统一训练。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}: 测试批次 = {batch_name}, device = {device}")
    print(f"{'='*60}")

    product_col = ("product_fine" if cfg.product_granularity == "fine"
                   else "product_coarse")

    ds_train, ds_val, loader_train, loader_val = build_loaders(
        metadata_csv, train_idx, val_idx, cfg, product_col
    )

    num_batches = ds_train.num_batches
    print(f"  训练: {len(ds_train)} 样本, 验证: {len(ds_val)} 样本")
    print(f"  产品数: {ds_train.num_products}, 批次数(训练): {num_batches}")

    model = GCMSConsistencyNet(num_batches, cfg).to(device)
    criterion = UnifiedLoss(cfg).to(device)

    # 无增强训练集 (原型计算用)
    ds_train_noaug = GCMSDataset(metadata_csv, product_col=product_col,
                                 augmentation=None, indices=train_idx)
    ds_train_noaug.product_enc = ds_train.product_enc
    ds_train_noaug.batch_enc = ds_train.batch_enc
    ds_train_noaug.df["product_label"] = ds_train.product_enc.transform(
        ds_train_noaug.df[product_col]
    )
    ds_train_noaug.df["batch_label"] = ds_train.batch_enc.transform(
        ds_train_noaug.df["batch_idx"]
    )
    loader_train_noaug = DataLoader(ds_train_noaug, batch_size=cfg.batch_size,
                                    shuffle=False, num_workers=0)
    label_names = ds_train.get_label_name_map()

    # ── 单阶段训练 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    best_acc = 0
    best_state = None

    for epoch in range(cfg.epochs):
        m_train = train_one_epoch(model, loader_train, criterion, optimizer,
                                  device, epoch, cfg.epochs)
        scheduler.step()

        # 每 10 轮做一次原型验证
        if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            val_acc, _ = validate_with_prototypes(
                model, loader_train_noaug, loader_val,
                label_names, device, cfg)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1}/{cfg.epochs}  "
                  f"supcon={m_train.get('supcon',0):.3f} "
                  f"adv={m_train.get('adv',0):.3f} "
                  f"proto={m_train.get('proto',0):.3f}  "
                  f"val_acc={val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # 注册最终原型
    proto_store, all_z, all_labels = register_from_loader(
        model, loader_train_noaug, label_names, device,
        percentile=cfg.accept_percentile
    )
    proto_store.summary()

    # 保存
    fold_dir = Path(cfg.output_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fold_dir / "model.pt")
    proto_store.save(fold_dir / "prototypes")
    with open(fold_dir / "product_classes.json", "w") as f:
        json.dump(list(ds_train.product_enc.classes_), f)

    return model, proto_store, ds_train, ds_val, loader_val


def train_all_folds(cfg: Config):
    """Leave-one-batch-out 全部 fold 训练 (仅在已知类上)。"""
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
    for fold in split_info["folds"]:
        model, proto_store, ds_train, ds_val, loader_val = run_fold(
            fold["fold_idx"], fold["train_idx"], fold["val_idx"],
            fold["test_batch"], metadata_csv, cfg
        )
        fold_results.append({
            "fold": fold["fold_idx"],
            "test_batch": fold["test_batch"],
            "model": model,
            "proto_store": proto_store,
            "ds_train": ds_train,
            "ds_val": ds_val,
            "loader_val": loader_val,
        })

    return fold_results, split_info