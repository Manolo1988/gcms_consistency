"""
统一度量学习损失 (单阶段, 无 softmax 分类):
  L = L_supcon + λ₁·L_adv + λ_focal·L_focal + λ₂·L_proto + λ_hard·L_hardpair + λ_recon·L_recon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance (Lin et al., 2017).

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Reduces the loss contribution from easy examples and focuses on hard ones.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha  # class weights tensor (C,) or None
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B, C) — raw logits
        targets: (B,)  — class indices
        """
        alpha = self.alpha
        if alpha is not None:
            # 保证 class weight 与 logits 在同一设备
            alpha = alpha.to(logits.device)
        ce_loss = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
        pt = torch.exp(-ce_loss)  # p_t for the correct class
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class SupConLoss(nn.Module):
    """
    监督对比损失 (Supervised Contrastive Learning, Khosla et al. 2020)。
    同产品样本在投影空间中拉近，不同产品推远。
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (B, D) — L2 归一化的投影向量
        labels:   (B,)   — 产品标签
        """
        device = features.device
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 余弦相似度矩阵
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # 正样本掩码: 同一类别 (排除自身)
        labels_col = labels.unsqueeze(1)
        mask_pos = (labels_col == labels_col.T).float()
        diag_mask = torch.eye(B, dtype=torch.float32, device=device)
        mask_pos = mask_pos * (1.0 - diag_mask)

        # 数值稳定性: 对角设为极小值后取 max
        sim_for_max = sim - diag_mask * 1e9
        logits_max = sim_for_max.max(dim=1, keepdim=True).values
        logits = sim - logits_max.detach()

        # 排除自身: 对角设为极小值 (不用 in-place)
        logits = logits - diag_mask * 1e9

        # Log-softmax
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # 正样本对的平均 log-prob
        num_pos = mask_pos.sum(dim=1)
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (num_pos + 1e-8)

        # 仅对至少有一个正样本的样本计算损失
        valid = num_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -mean_log_prob_pos[valid].mean()
        return loss


class BatchPrototypeLoss(nn.Module):
    """
    批内原型距离损失: 拉近样本到同类原型，推远异类原型。
    原型由批内样本均值动态计算。
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z, labels):
        """
        z:      (B, D) — 嵌入向量
        labels: (B,)   — 产品标签
        """
        device = z.device
        unique_labels = labels.unique()

        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 计算每类原型
        prototypes = []
        proto_labels = []
        for lbl in unique_labels:
            mask = labels == lbl
            prototypes.append(z[mask].mean(dim=0))
            proto_labels.append(lbl)
        prototypes = torch.stack(prototypes)  # (K, D)

        # 每个样本到所有原型的距离 (避免 cdist, 兼容 MPS)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        z_sq = (z * z).sum(dim=1, keepdim=True)       # (B, 1)
        p_sq = (prototypes * prototypes).sum(dim=1)    # (K,)
        cross = z @ prototypes.T                        # (B, K)
        dists = (z_sq + p_sq.unsqueeze(0) - 2 * cross).clamp(min=1e-12).sqrt()  # (B, K)

        # 拉近同类原型 + 推远异类原型
        loss_pull = torch.tensor(0.0, device=device)
        loss_push = torch.tensor(0.0, device=device)
        for i, lbl in enumerate(proto_labels):
            mask = labels == lbl
            pos_dists = dists[mask, i]
            loss_pull = loss_pull + pos_dists.mean()

            neg_mask = ~mask
            if neg_mask.any():
                neg_dists = dists[neg_mask, i]
                loss_push = loss_push + F.relu(self.margin - neg_dists).mean()

        K = len(proto_labels)
        return (loss_pull + 0.5 * loss_push) / K


class HardPairMarginLoss(nn.Module):
    """
    针对已知易混产品对的批内原型边界损失。

    对每个易混对 (A, B)，当一个 batch 同时含有 A/B 时，要求：
      dist(sample_A, proto_B) - dist(sample_A, proto_A) >= margin
      dist(sample_B, proto_A) - dist(sample_B, proto_B) >= margin

    同时维护 EMA 原型，确保即使 A/B 不在同一 batch 也能施加约束。
    """

    def __init__(self, margin=0.35, ema_decay=0.99):
        super().__init__()
        self.margin = float(margin)
        self.ema_decay = float(ema_decay)
        self.pair_label_ids = []
        # EMA prototypes: label_id -> running average embedding
        self.ema_protos: dict[int, torch.Tensor] = {}
        # Track which labels exist in EMA
        self._ema_initialized = set()

    def set_label_names(self, label_names):
        name_to_id = {str(name): int(idx) for idx, name in label_names.items()}
        pair_names = getattr(self, "pair_names", ())
        self.pair_label_ids = [
            (name_to_id[a], name_to_id[b])
            for a, b in pair_names
            if a in name_to_id and b in name_to_id
        ]
        self.ema_protos.clear()
        self._ema_initialized.clear()

    def _update_ema(self, label_id, new_proto):
        if label_id not in self.ema_protos:
            self.ema_protos[label_id] = new_proto.detach().clone()
            self._ema_initialized.add(label_id)
        else:
            self.ema_protos[label_id] = (
                self.ema_decay * self.ema_protos[label_id]
                + (1.0 - self.ema_decay) * new_proto.detach()
            )

    def forward(self, z, labels):
        device = z.device
        if not self.pair_label_ids:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Update EMA prototypes for all classes present in this batch
        unique_labels = labels.unique().tolist()
        for lbl_id in unique_labels:
            if any(lbl_id in pair for pair in self.pair_label_ids):
                mask = labels == lbl_id
                proto = z[mask].mean(dim=0)
                self._update_ema(lbl_id, proto)

        losses = []
        for a_id, b_id in self.pair_label_ids:
            mask_a = labels == a_id
            mask_b = labels == b_id

            # Case 1: both classes present → use batch prototypes
            if mask_a.any() and mask_b.any():
                proto_a = z[mask_a].mean(dim=0)
                proto_b = z[mask_b].mean(dim=0)

                d_aa = (z[mask_a] - proto_a).norm(dim=1)
                d_ab = (z[mask_a] - proto_b).norm(dim=1)
                d_bb = (z[mask_b] - proto_b).norm(dim=1)
                d_ba = (z[mask_b] - proto_a).norm(dim=1)

                losses.append(F.relu(self.margin + d_aa - d_ab).mean())
                losses.append(F.relu(self.margin + d_bb - d_ba).mean())

            # Case 2: only class A present, use EMA for class B
            elif mask_a.any() and b_id in self._ema_initialized:
                proto_a = z[mask_a].mean(dim=0)
                ema_b = self.ema_protos[b_id].to(device)
                d_aa = (z[mask_a] - proto_a).norm(dim=1)
                d_ab = (z[mask_a] - ema_b).norm(dim=1)
                losses.append(F.relu(self.margin + d_aa - d_ab).mean())

            # Case 3: only class B present, use EMA for class A
            elif mask_b.any() and a_id in self._ema_initialized:
                proto_b = z[mask_b].mean(dim=0)
                ema_a = self.ema_protos[a_id].to(device)
                d_bb = (z[mask_b] - proto_b).norm(dim=1)
                d_ba = (z[mask_b] - ema_a).norm(dim=1)
                losses.append(F.relu(self.margin + d_bb - d_ba).mean())

            # Case 4: neither present but both have EMA → triplet loss on EMA protos
            elif a_id in self._ema_initialized and b_id in self._ema_initialized:
                # Ensure EMA prototypes stay separated
                ema_a = self.ema_protos[a_id].to(device)
                ema_b = self.ema_protos[b_id].to(device)
                dist_ab = (ema_a - ema_b).norm()
                losses.append(F.relu(self.margin * 3.0 - dist_ab))

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).mean()


class UnifiedLoss(nn.Module):
    """
    统一损失:
      L = L_supcon + λ₁·L_adv + λ_cls·L_focal + λ₂·L_proto + λ_hard·L_hardpair + λ_recon·L_recon
    """

    def __init__(self, cfg):
        super().__init__()
        self.supcon_loss = SupConLoss(temperature=cfg.supcon_temperature)
        self.proto_loss = BatchPrototypeLoss(margin=cfg.proto_margin)
        self.hard_pair_loss = HardPairMarginLoss(
            margin=getattr(cfg, "hard_pair_margin", 0.35)
        )
        self.hard_pair_loss.pair_names = tuple(
            (str(a), str(b))
            for a, b in getattr(cfg, "hard_pair_names", ())
        )
        self.domain_loss = nn.CrossEntropyLoss()
        # Use Focal Loss for classification to handle class imbalance
        focal_gamma = float(getattr(cfg, "focal_gamma", 2.0))
        focal_alpha = getattr(cfg, "focal_alpha", None)
        if focal_alpha is not None and not isinstance(focal_alpha, torch.Tensor):
            focal_alpha = torch.as_tensor(focal_alpha, dtype=torch.float32)
        self.cls_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="mean")
        self.recon_loss = nn.MSELoss()

        self.lam_supcon = cfg.lambda_supcon
        self.lam_adv = cfg.lambda_adv
        self.lam_cls = getattr(cfg, "lambda_cls", 0.0)
        self.lam_proto = cfg.lambda_proto
        self.lam_recon = cfg.lambda_recon
        self.lam_hard_pair = getattr(cfg, "lambda_hard_pair", 0.0)

    def set_label_names(self, label_names):
        self.hard_pair_loss.set_label_names(label_names)

    def forward(self, model_out, batch):
        labels = batch["product"]
        batch_labels = batch["batch"]

        # 监督对比损失 (在投影空间, 类间可分)
        l_supcon = self.supcon_loss(model_out["proj"], labels)

        # 批次对抗损失 (GRL, 去批次)
        l_adv = self.domain_loss(model_out["domain_logits"], batch_labels)

        # Focal 分类辅助损失 (处理类别不平衡)
        l_cls = torch.tensor(0.0, device=labels.device)
        if model_out.get("cls_logits") is not None:
            l_cls = self.cls_loss(model_out["cls_logits"], labels)

        # 原型距离损失 (在嵌入空间, 类内紧凑)
        l_proto = self.proto_loss(model_out["z"], labels)

        # 易混产品对边界损失 (带 EMA 原型)
        l_hard_pair = self.hard_pair_loss(model_out["z"], labels)

        # 重建损失 (防止表征退化)
        l_recon = self.recon_loss(model_out["recon"], batch["input"])

        total = (self.lam_supcon * l_supcon
                 + self.lam_adv * l_adv
                 + self.lam_cls * l_cls
                 + self.lam_proto * l_proto
                 + self.lam_hard_pair * l_hard_pair
                 + self.lam_recon * l_recon)

        return {
            "supcon": l_supcon, "adv": l_adv, "cls": l_cls,
            "proto": l_proto, "hardpair": l_hard_pair, "recon": l_recon,
            "total": total,
        }
