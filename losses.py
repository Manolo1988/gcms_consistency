"""
统一度量学习损失 (单阶段, 无 softmax 分类):
  L = L_supcon + λ₁·L_adv + λ₂·L_proto + λ_recon·L_recon
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
    这样只强化指定相似对的局部边界，不会把所有类别一刀切推远。
    """

    def __init__(self, margin=0.35):
        super().__init__()
        self.margin = float(margin)
        self.pair_label_ids = []
        self.proto_momentum = 0.9
        self.register_buffer("_proto_labels", torch.empty(0, dtype=torch.long))
        self.register_buffer("_proto_memory", torch.empty(0, 0))

    def set_label_names(self, label_names):
        name_to_id = {str(name): int(idx) for idx, name in label_names.items()}
        pair_names = getattr(self, "pair_names", ())
        self.pair_label_ids = [
            (name_to_id[a], name_to_id[b])
            for a, b in pair_names
            if a in name_to_id and b in name_to_id
        ]

    def _memory_lookup(self, label_id, device, dtype):
        if self._proto_labels.numel() == 0 or self._proto_memory.numel() == 0:
            return None
        hits = (self._proto_labels.to(device) == int(label_id)).nonzero(as_tuple=False)
        if hits.numel() == 0:
            return None
        idx = int(hits[0].item())
        return self._proto_memory.to(device=device, dtype=dtype)[idx]

    @torch.no_grad()
    def _update_memory(self, labels, batch_protos):
        if not batch_protos:
            return
        device = labels.device
        proto_labels = sorted(batch_protos.keys())
        proto_tensor = torch.stack([batch_protos[k].detach() for k in proto_labels])
        proto_labels_t = torch.tensor(proto_labels, device=device, dtype=torch.long)

        if self._proto_labels.numel() == 0 or self._proto_memory.numel() == 0:
            self._proto_labels = proto_labels_t.cpu()
            self._proto_memory = proto_tensor.cpu()
            return

        old_labels = self._proto_labels.to(device)
        old_memory = self._proto_memory.to(device=device, dtype=proto_tensor.dtype)
        memory = {int(k.item()): v for k, v in zip(old_labels, old_memory)}
        for label, proto in zip(proto_labels_t, proto_tensor):
            key = int(label.item())
            if key in memory:
                memory[key] = F.normalize(
                    self.proto_momentum * memory[key]
                    + (1.0 - self.proto_momentum) * proto,
                    dim=0,
                )
            else:
                memory[key] = proto

        merged_labels = sorted(memory.keys())
        self._proto_labels = torch.tensor(merged_labels, dtype=torch.long)
        self._proto_memory = torch.stack([memory[k].detach().cpu() for k in merged_labels])

    def forward(self, z, labels):
        device = z.device
        if not self.pair_label_ids:
            return torch.tensor(0.0, device=device, requires_grad=True)

        losses = []
        z_unit = F.normalize(z, dim=1)
        pair_ids = sorted({int(x) for pair in self.pair_label_ids for x in pair})
        batch_protos = {}
        for label_id in pair_ids:
            mask = labels == label_id
            if mask.any():
                batch_protos[label_id] = F.normalize(
                    z_unit[mask].mean(dim=0), dim=0
                )
        for a_id, b_id in self.pair_label_ids:
            mask_a = labels == a_id
            mask_b = labels == b_id
            proto_a = batch_protos.get(int(a_id))
            proto_b = batch_protos.get(int(b_id))
            if proto_a is None:
                proto_a = self._memory_lookup(a_id, device, z_unit.dtype)
            if proto_b is None:
                proto_b = self._memory_lookup(b_id, device, z_unit.dtype)
            if proto_a is None or proto_b is None:
                continue

            if mask_a.any():
                sim_aa = z_unit[mask_a] @ proto_a
                sim_ab = z_unit[mask_a] @ proto_b
                losses.append(F.relu(self.margin + sim_ab - sim_aa).mean())
            if mask_b.any():
                sim_bb = z_unit[mask_b] @ proto_b
                sim_ba = z_unit[mask_b] @ proto_a
                losses.append(F.relu(self.margin + sim_ba - sim_bb).mean())

        self._update_memory(labels, batch_protos)

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).mean()


class UnifiedLoss(nn.Module):
    """
    统一损失:
      L = L_supcon + λ₁·L_adv + λ_cls·L_cls + λ₂·L_proto + λ_recon·L_recon
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
        self.cls_loss = nn.CrossEntropyLoss()
        self.recon_loss = nn.MSELoss()

        self.lam_supcon = cfg.lambda_supcon
        self.lam_adv = cfg.lambda_adv
        self.lam_cls = getattr(cfg, "lambda_cls", 0.0)
        self.lam_tic_cls = getattr(cfg, "lambda_tic_cls", 0.0)
        self.lam_proto = cfg.lambda_proto
        self.lam_recon = cfg.lambda_recon
        self.lam_hard_pair = getattr(cfg, "lambda_hard_pair", 0.0)
        self.lam_tic_residual = getattr(cfg, "lambda_tic_residual", 0.0)
        self.lam_tic_anchor = getattr(cfg, "lambda_tic_anchor", 0.0)
        self.lam_tic_gate = getattr(cfg, "lambda_tic_gate", 0.0)

    def set_label_names(self, label_names):
        self.hard_pair_loss.set_label_names(label_names)

    def forward(self, model_out, batch):
        labels = batch["product"]
        batch_labels = batch["batch"]

        # 监督对比损失 (在投影空间, 类间可分)
        l_supcon = self.supcon_loss(model_out["proj"], labels)

        # 批次对抗损失 (GRL, 去批次)
        l_adv = self.domain_loss(model_out["domain_logits"], batch_labels)

        # CE 分类辅助损失 (加速产品判别)
        l_cls = torch.tensor(0.0, device=labels.device)
        if model_out.get("cls_logits") is not None:
            l_cls = self.cls_loss(model_out["cls_logits"], labels)

        l_tic_cls = torch.tensor(0.0, device=labels.device)
        if model_out.get("tic_cls_logits") is not None:
            l_tic_cls = self.cls_loss(model_out["tic_cls_logits"], labels)

        # 原型距离损失 (在嵌入空间, 类内紧凑)
        l_proto = self.proto_loss(model_out["z"], labels)

        # 易混产品对边界损失
        l_hard_pair = self.hard_pair_loss(model_out["z"], labels)

        # 重建损失 (防止表征退化)
        l_recon = self.recon_loss(model_out["recon"], batch["input"])

        # TIC 分支正则: 让 TIC 做小幅、稳定修正，而不是改写主嵌入空间。
        l_tic_residual = torch.tensor(0.0, device=labels.device)
        tic_residual = model_out.get("tic_residual")
        if tic_residual is not None:
            l_tic_residual = tic_residual.pow(2).mean()

        l_tic_anchor = torch.tensor(0.0, device=labels.device)
        tic_anchor = model_out.get("tic_main_anchor")
        if tic_anchor is not None:
            fused = model_out["z_raw"]
            if fused.shape == tic_anchor.shape:
                l_tic_anchor = 1.0 - F.cosine_similarity(fused, tic_anchor, dim=1).mean()

        l_tic_gate = torch.tensor(0.0, device=labels.device)
        tic_gate_raw = model_out.get("tic_gate_raw_mean")
        tic_gate = model_out.get("tic_gate_mean")
        if tic_gate_raw is not None:
            l_tic_gate = tic_gate_raw
        elif tic_gate is not None:
            l_tic_gate = tic_gate

        total = (self.lam_supcon * l_supcon
                 + self.lam_adv * l_adv
                 + self.lam_cls * l_cls
                 + self.lam_tic_cls * l_tic_cls
                 + self.lam_proto * l_proto
                 + self.lam_hard_pair * l_hard_pair
                 + self.lam_recon * l_recon
                 + self.lam_tic_residual * l_tic_residual
                 + self.lam_tic_anchor * l_tic_anchor
                 + self.lam_tic_gate * l_tic_gate)

        return {
            "supcon": l_supcon, "adv": l_adv, "cls": l_cls,
            "ticcls": l_tic_cls,
            "proto": l_proto, "hardpair": l_hard_pair, "recon": l_recon,
            "ticres": l_tic_residual, "ticanchor": l_tic_anchor,
            "ticgate_loss": l_tic_gate,
            "total": total,
        }
