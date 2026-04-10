"""
度量学习多任务损失:
  L = λ_supcon·L_supcon + λ_adv·L_adv + λ_proto·L_proto
      + λ_recon·L_recon + λ_cls·L_cls
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        mask_pos.fill_diagonal_(0)

        # 数值稳定性
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # 排除自身
        mask_self = torch.eye(B, device=device)
        logits = logits - mask_self * 1e9

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

        # 每个样本到所有原型的距离
        dists = torch.cdist(z, prototypes)  # (B, K)

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


class MetricLearningLoss(nn.Module):
    """
    组合损失:
      L = λ_supcon·L_supcon + λ_adv·L_adv + λ_proto·L_proto
          + λ_recon·L_recon + λ_cls·L_cls
    """

    def __init__(self, cfg):
        super().__init__()
        self.supcon_loss = SupConLoss(temperature=cfg.supcon_temperature)
        self.proto_loss = BatchPrototypeLoss(margin=cfg.proto_margin)
        self.cls_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.CrossEntropyLoss()
        self.recon_loss = nn.MSELoss()

        self.lam_supcon = cfg.lambda_supcon
        self.lam_adv = cfg.lambda_adv
        self.lam_proto = cfg.lambda_proto
        self.lam_recon = cfg.lambda_recon
        self.lam_cls = cfg.lambda_cls

    def forward(self, model_out, batch, phase="finetune"):
        losses = {}

        if phase == "pretrain":
            losses["recon"] = self.recon_loss(model_out["recon"], batch["input"])
            losses["total"] = losses["recon"]
            return losses

        labels = batch["product"]
        batch_labels = batch["batch"]

        # 监督对比损失 (在投影空间)
        l_supcon = self.supcon_loss(model_out["proj"], labels)

        # 批次对抗损失
        l_adv = self.domain_loss(model_out["domain_logits"], batch_labels)

        # 原型距离损失 (在嵌入空间)
        l_proto = self.proto_loss(model_out["z"], labels)

        # 重建损失
        l_recon = self.recon_loss(model_out["recon"], batch["input"])

        # 辅助分类损失
        l_cls = self.cls_loss(model_out["logits"], labels)

        total = (self.lam_supcon * l_supcon
                 + self.lam_adv * l_adv
                 + self.lam_proto * l_proto
                 + self.lam_recon * l_recon
                 + self.lam_cls * l_cls)

        losses.update({
            "supcon": l_supcon, "adv": l_adv,
            "proto": l_proto, "recon": l_recon,
            "cls": l_cls, "total": total,
        })
        return losses