"""
三阶段训练引擎:
  Phase 1: MAE 预训练
  Phase 2: 多任务监督微调
  Phase 3: 一致性阈值校准
"""
import json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import GCMSDataset, GCMSAugmentation, leave_one_batch_out_splits
from models import GCMSConsistencyNet
from losses import MultiTaskLoss


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


def train_one_epoch(model, loader, criterion, optimizer, device, phase, epoch,
                    total_epochs):
    model.train()
    running = {}

    # 域对抗 alpha 渐进增大
    p = epoch / total_epochs
    alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    model.domain_head.set_alpha(alpha)

    for batch in loader:
        x = batch["input"].to(device)
        batch_dev = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
        batch_dev["input"] = x

        out = model(x)
        losses = criterion(out, batch_dev, phase=phase)
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
def validate(model, loader, criterion, device):
    model.eval()
    running = {}
    all_pred, all_true = [], []

    for batch in loader:
        x = batch["input"].to(device)
        batch_dev = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
        batch_dev["input"] = x

        out = model(x)
        losses = criterion(out, batch_dev, phase="finetune")

        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.item()

        pred = out["logits"].argmax(dim=1)
        all_pred.append(pred.cpu())
        all_true.append(batch["product"])

    n = max(len(loader), 1)
    metrics = {k: v / n for k, v in running.items()}

    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    metrics["acc"] = (all_pred == all_true).float().mean().item()
    return metrics


def calibrate_thresholds(model, loader, device, percentile=95.0):
    """在验证集上为每个产品校准一致性距离阈值。"""
    model.eval()
    dists_per_product = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            result = model.predict(x)
            labels = batch["product"]
            for i in range(len(labels)):
                y = labels[i].item()
                d = result["consistency_dist"][i].item()
                dists_per_product.setdefault(y, []).append(d)

    thresholds = {}
    for y, ds in dists_per_product.items():
        thresholds[y] = float(np.percentile(ds, percentile))
    return thresholds


def run_fold(fold_idx, train_idx, val_idx, batch_name, metadata_csv, cfg):
    """运行一个 leave-one-batch-out fold 的完整三阶段训练。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}: 测试批次 = {batch_name}, device = {device}")
    print(f"{'='*60}")

    product_col = cfg.product_granularity
    if product_col == "fine":
        product_col = "product_fine"
    else:
        product_col = "product_coarse"

    ds_train, ds_val, loader_train, loader_val = build_loaders(
        metadata_csv, train_idx, val_idx, cfg, product_col
    )

    num_products = ds_train.num_products
    num_batches = ds_train.num_batches
    print(f"  训练: {len(ds_train)} 样本, 验证: {len(ds_val)} 样本")
    print(f"  产品数: {num_products}, 批次数(训练): {num_batches}")

    model = GCMSConsistencyNet(num_products, num_batches, cfg).to(device)
    criterion = MultiTaskLoss(num_products, cfg.feature_dim, cfg).to(device)

    # ── Phase 1: 预训练 ──
    print("\n[Phase 1] MAE 预训练")
    opt1 = torch.optim.AdamW(model.parameters(), lr=cfg.lr_pretrain,
                             weight_decay=cfg.weight_decay)
    for epoch in range(cfg.epochs_pretrain):
        m = train_one_epoch(model, loader_train, criterion, opt1, device,
                            "pretrain", epoch, cfg.epochs_pretrain)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{cfg.epochs_pretrain}  "
                  f"recon={m.get('recon', 0):.4f}")

    # ── Phase 2: 监督微调 ──
    print("\n[Phase 2] 多任务监督微调")
    opt2 = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": cfg.lr_finetune * 0.1},
         {"params": model.product_head.parameters()},
         {"params": model.domain_head.parameters()},
         {"params": model.proto_head.parameters()},
         {"params": model.energy_head.parameters()},
         {"params": model.decoder.parameters(), "lr": cfg.lr_finetune * 0.5},
         {"params": criterion.center_loss.parameters()}],
        lr=cfg.lr_finetune, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=cfg.epochs_finetune
    )

    best_val_acc = 0
    best_state = None

    for epoch in range(cfg.epochs_finetune):
        m_train = train_one_epoch(model, loader_train, criterion, opt2, device,
                                  "finetune", epoch, cfg.epochs_finetune)
        m_val = validate(model, loader_val, criterion, device)
        scheduler.step()

        if m_val["acc"] > best_val_acc:
            best_val_acc = m_val["acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{cfg.epochs_finetune}  "
                  f"train_cls={m_train.get('cls',0):.3f} "
                  f"val_acc={m_val['acc']:.3f}  "
                  f"val_proto={m_val.get('proto',0):.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Phase 3: 阈值校准 ──
    print("\n[Phase 3] 一致性阈值校准")
    thresholds = calibrate_thresholds(
        model, loader_train, device, percentile=cfg.accept_percentile
    )
    print(f"  阈值: {thresholds}")

    # 保存
    fold_dir = Path(cfg.output_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fold_dir / "model.pt")
    with open(fold_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(fold_dir / "product_classes.json", "w") as f:
        json.dump(list(ds_train.product_enc.classes_), f)

    return model, thresholds, ds_train, ds_val, loader_val


def train_all_folds(cfg: Config):
    """Leave-one-batch-out 全部 fold 训练。"""
    set_seed(cfg.seed)
    metadata_csv = str(Path(cfg.prepared_dir) / "metadata.csv")
    splits = leave_one_batch_out_splits(metadata_csv)

    fold_results = []
    for i, (train_idx, val_idx, bname) in enumerate(splits):
        model, thresholds, ds_train, ds_val, loader_val = run_fold(
            i, train_idx, val_idx, bname, metadata_csv, cfg
        )
        fold_results.append({
            "fold": i, "test_batch": bname,
            "model": model, "thresholds": thresholds,
            "ds_train": ds_train, "ds_val": ds_val,
            "loader_val": loader_val,
        })

    return fold_results