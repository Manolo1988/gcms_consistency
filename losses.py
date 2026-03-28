"""
多任务损失函数:
  L = L_cls + λ₁ L_domain + λ₂ L_proto + λ₃ L_center + λ₄ L_recon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLoss(nn.Module):
    """同类拉近原型，异类推开。"""
    def __init__(self, margin=10.0):
        super().__init__()
        self.margin = margin

    def forward(self, proto_dists, labels):
        # proto_dists: (B, K), labels: (B,)
        B = labels.size(0)
        pos_dist = proto_dists[torch.arange(B), labels]
        loss_pull = pos_dist.mean()

        mask_neg = torch.ones_like(proto_dists, dtype=torch.bool)
        mask_neg[torch.arange(B), labels] = False
        neg_dists = proto_dists[mask_neg].reshape(B, -1)
        loss_push = F.relu(self.margin - neg_dists).mean()

        return loss_pull + 0.5 * loss_push


class CenterLoss(nn.Module):
    """类内紧凑损失。"""
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim) * 0.1)

    def forward(self, z, labels):
        centers_batch = self.centers[labels]
        return ((z - centers_batch) ** 2).sum(dim=1).mean()


class MultiTaskLoss(nn.Module):
    """组合多任务损失。"""
    def __init__(self, num_products, feature_dim, cfg):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.CrossEntropyLoss()
        self.proto_loss = PrototypeLoss(margin=10.0)
        self.center_loss = CenterLoss(feature_dim, num_products)
        self.recon_loss = nn.MSELoss()

        self.lam_domain = cfg.lambda_domain
        self.lam_proto = cfg.lambda_proto
        self.lam_center = cfg.lambda_center
        self.lam_recon = cfg.lambda_recon

    def forward(self, model_out, batch, phase="finetune"):
        losses = {}

        if phase == "pretrain":
            losses["recon"] = self.recon_loss(model_out["recon"], batch["input"])
            losses["total"] = losses["recon"]
            return losses

        labels = batch["product"]
        batch_labels = batch["batch"]

        l_cls = self.cls_loss(model_out["logits"], labels)
        l_domain = self.domain_loss(model_out["domain_logits"], batch_labels)
        l_proto = self.proto_loss(model_out["proto_dists"], labels)
        l_center = self.center_loss(model_out["z"], labels)
        l_recon = self.recon_loss(model_out["recon"], batch["input"])

        total = (l_cls
                 + self.lam_domain * l_domain
                 + self.lam_proto * l_proto
                 + self.lam_center * l_center
                 + self.lam_recon * l_recon)

        losses.update({
            "cls": l_cls, "domain": l_domain,
            "proto": l_proto, "center": l_center,
            "recon": l_recon, "total": total,
        })
        return losses