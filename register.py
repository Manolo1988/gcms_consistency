"""
原型注册与推理:
  - 冻结 Backbone，从参考样本计算类原型和半径
  - 基于原型距离的产品识别、一致性评分、拒识判定
  - SAIM 风格球面原型调整：增量注册时保持超球面均匀分布
  - 增量微调：新产品注册时微调编码器末层，保持旧类性能
"""
import json
import random
import numpy as np
import torch
from pathlib import Path


class SphericalPrototypeAdjuster:
    """
    SAIM 风格球面原型调整器。

    将类原型投影到单位超球面，通过迭代斥力使之均匀分布。
    新产品注册时调用，防止原型坍缩或过度聚集。

    参考: gyfseer/SAIM — Self-Adaptive Incremental Model (state.py)
    """

    # 预计算的收敛阈值 (列和的平方和)
    CONVERGENCE_THRESHOLDS = {
        2: 6.017e-3, 3: 7.695e-3, 4: 7.190e-3, 5: 7.133e-3,
        6: 7.367e-3, 7: 7.862e-3, 8: 7.292e-3, 9: 8.633e-3,
        10: 7.319e-3, 11: 8.644e-3, 12: 8.328e-3, 13: 1.595e-2,
        14: 1.435e-2, 15: 1.319e-2, 16: 2.130e-2, 17: 1.562e-2,
        18: 2.077e-2, 19: 2.246e-2, 20: 2.168e-2, 21: 2.719e-2,
        22: 3.808e-2, 23: 2.971e-2, 24: 2.574e-2,
    }

    @staticmethod
    def normalize(x):
        """投影到单位球面。"""
        return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def adjust(proto_matrix, max_iters=1000, lr=1e-3):
        """
        迭代调整原型使之在单位超球面上均匀分布。

        原理：
          - 计算原型间余弦相似度矩阵
          - 每个原型受到最近邻的斥力，沿远离方向移动
          - 重叠原型 (< 1°) 施加随机扰动
          - 重新归一化到单位球面
          - 当列和平方和 < 阈值时认为已收敛

        Args:
            proto_matrix: (K, D) 原型向量
            max_iters:    最大迭代次数
            lr:           斥力学习率

        Returns:
            (K, D) 调整后的单位球面原型
        """
        K = proto_matrix.shape[0]
        if K < 2:
            return SphericalPrototypeAdjuster.normalize(proto_matrix)

        state = SphericalPrototypeAdjuster.normalize(proto_matrix.clone())
        threshold = SphericalPrototypeAdjuster.CONVERGENCE_THRESHOLDS.get(
            K, 0.03 + 0.002 * K  # 超出预计算范围的默认值
        )

        for _ in range(max_iters):
            cov = state @ state.T
            n = cov.shape[0]

            # 最近异类邻居
            cov[range(n), range(n)] = -2.0
            _values, indices = torch.max(cov, dim=1)

            # 收敛判定: 列和 → 0 表示均匀分布
            col_sum = state.sum(dim=0)
            uniformity = (col_sum * col_sum).sum().item()
            if uniformity < threshold:
                break

            # 斥力迭代
            new_state = []
            for i in range(K):
                j = indices[i].item()
                gap = state[i:i+1] - state[j:j+1]
                gap_norm = (gap * gap).sum().item()

                if gap_norm < 0.0175:
                    # 近似重叠 (< ~1°), 随机扰动
                    rand_mov = torch.zeros_like(state[j:j+1])
                    dim_idx = random.randint(0, rand_mov.shape[1] - 1)
                    rand_mov[0, dim_idx] = 0.034
                    new_state.append(state[i:i+1] + rand_mov)
                else:
                    scale = lr * (random.random() + 1) / 2
                    new_state.append(state[i:i+1] + scale * gap)

            new_state = torch.cat(new_state, dim=0)

            # 若新状态更差且已低于阈值，停止
            new_col_sum = new_state.sum(dim=0)
            new_uniformity = (new_col_sum * new_col_sum).sum().item()
            if uniformity <= new_uniformity and uniformity < threshold:
                break

            state = SphericalPrototypeAdjuster.normalize(new_state)

        return state


class PrototypeStore:
    """
    管理产品原型向量和一致性半径。

    注册阶段 (不涉及梯度更新):
        p_k = mean(z_1, ..., z_N)          类原型
        r_k = percentile(‖z_i - p_k‖, 95)  一致性半径

    推理阶段:
        产品识别: argmin_k ‖z_q - p_k‖
        一致性分数: S = exp(-‖z_q - p_k*‖ / r_k*)
        拒识判定: min_k ‖z_q - p_k‖ > θ → 未知产品
    """

    def __init__(self):
        self.prototypes = {}   # class_name -> torch.Tensor (D,)
        self.radii = {}        # class_name -> float
        self.class_names = []  # ordered class name list
        self.spherical_prototypes = {}  # 球面调整后的原型

    def register(self, class_name, embeddings, percentile=95.0):
        """
        注册一个产品类别。

        embeddings: (N, D) — N 个参考样本的嵌入向量
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        embeddings = embeddings.float()

        proto = embeddings.mean(dim=0)
        dists = torch.norm(embeddings - proto, dim=1)
        radius = float(np.percentile(dists.cpu().numpy(), percentile))
        radius = max(radius, 1e-6)

        self.prototypes[class_name] = proto.cpu()
        self.radii[class_name] = radius
        if class_name not in self.class_names:
            self.class_names.append(class_name)

    def redistribute_on_sphere(self, max_iters=1000, lr=1e-3):
        """
        将所有已注册原型调整为超球面均匀分布 (SAIM 风格)。

        应在以下时机调用:
          - 所有已知类注册完毕后
          - 增量注册新产品后

        调整后的球面原型用于 predict() 中的余弦相似度匹配，
        原始原型和半径仍用于一致性评分。
        """
        if len(self.class_names) < 2:
            for c in self.class_names:
                self.spherical_prototypes[c] = (
                    self.prototypes[c] /
                    self.prototypes[c].norm().clamp(min=1e-8)
                )
            return

        proto_mat = torch.stack([self.prototypes[c] for c in self.class_names])
        adjusted = SphericalPrototypeAdjuster.adjust(
            proto_mat, max_iters=max_iters, lr=lr
        )
        for i, c in enumerate(self.class_names):
            self.spherical_prototypes[c] = adjusted[i]

    def predict(self, z, use_spherical=True):
        """
        基于原型匹配的推理。

        z: (B, D) — 待测样本嵌入
        use_spherical: 是否使用球面调整后的原型 (余弦距离)
        返回 dict:
            pred_class:  list[str]  — 预测产品名称
            pred_idx:    (B,)       — 预测类别索引
            scores:      (B,)       — 一致性分数 [0, 1]
            min_dists:   (B,)       — 到最近原型距离
            all_dists:   (B, K)     — 到所有原型距离
        """
        if not self.class_names:
            raise RuntimeError("原型库为空，请先注册产品")

        # 选择球面原型或原始原型
        if use_spherical and self.spherical_prototypes:
            proto_mat = torch.stack(
                [self.spherical_prototypes[c] for c in self.class_names]
            )
            proto_mat = proto_mat.to(z.device)
            # 余弦距离: 1 - cos_sim
            z_norm = z / z.norm(dim=1, keepdim=True).clamp(min=1e-8)
            cos_sim = z_norm @ proto_mat.T  # (B, K)
            all_dists = 1.0 - cos_sim
        else:
            proto_mat = torch.stack(
                [self.prototypes[c] for c in self.class_names]
            )
            proto_mat = proto_mat.to(z.device)
            # 欧式距离 (避免 cdist, 兼容 MPS)
            z_sq = (z * z).sum(dim=1, keepdim=True)
            p_sq = (proto_mat * proto_mat).sum(dim=1)
            cross = z @ proto_mat.T
            all_dists = (z_sq + p_sq.unsqueeze(0) - 2 * cross).clamp(min=1e-12).sqrt()

        # 最近原型
        min_dists, pred_idx = all_dists.min(dim=1)  # (B,), (B,)

        # 一致性分数: S = exp(-dist / radius)
        radii_tensor = torch.tensor(
            [self.radii[self.class_names[i]] for i in pred_idx.cpu().tolist()],
            device=z.device, dtype=z.dtype
        )
        scores = torch.exp(-min_dists / radii_tensor)

        pred_class = [self.class_names[i] for i in pred_idx.cpu().tolist()]

        return {
            "pred_class": pred_class,
            "pred_idx": pred_idx,
            "scores": scores,
            "min_dists": min_dists,
            "all_dists": all_dists,
        }

    def is_known(self, min_dists, factor=2.0):
        """
        拒识判定: 距离 > factor * 所有已注册类的平均半径 → 未知
        """
        mean_radius = np.mean(list(self.radii.values()))
        threshold = factor * mean_radius
        return min_dists.detach().cpu().numpy() <= threshold

    def save(self, path):
        """保存原型库到 JSON + PT 文件。"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存元数据
        meta = {
            "class_names": self.class_names,
            "radii": {k: float(v) for k, v in self.radii.items()},
        }
        with open(path / "proto_meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # 保存原型向量
        proto_dict = {k: v for k, v in self.prototypes.items()}
        torch.save(proto_dict, path / "prototypes.pt")

        # 保存球面原型
        if self.spherical_prototypes:
            sph_dict = {k: v for k, v in self.spherical_prototypes.items()}
            torch.save(sph_dict, path / "spherical_prototypes.pt")

    def load(self, path):
        """加载已保存的原型库。"""
        path = Path(path)

        with open(path / "proto_meta.json") as f:
            meta = json.load(f)

        self.class_names = meta["class_names"]
        self.radii = {k: float(v) for k, v in meta["radii"].items()}
        self.prototypes = torch.load(path / "prototypes.pt",
                                     map_location="cpu", weights_only=False)

        sph_path = path / "spherical_prototypes.pt"
        if sph_path.exists():
            self.spherical_prototypes = torch.load(
                sph_path, map_location="cpu", weights_only=False
            )
        else:
            self.spherical_prototypes = {}

    @property
    def num_classes(self):
        return len(self.class_names)

    def summary(self):
        """打印原型库概览。"""
        sph = "球面调整已启用" if self.spherical_prototypes else "未调整"
        print(f"原型库: {self.num_classes} 个产品 ({sph})")
        for c in self.class_names:
            p = self.prototypes[c]
            info = f"  {c}: radius={self.radii[c]:.4f}, proto_norm={p.norm().item():.4f}"
            if c in self.spherical_prototypes:
                sp = self.spherical_prototypes[c]
                info += f", sph_norm={sp.norm().item():.4f}"
            print(info)


@torch.no_grad()
def compute_prototypes(model, dataloader, device, percentile=95.0):
    """
    从数据集计算所有产品的原型和半径。

    返回:
        store:       PrototypeStore
        all_z:       (N, D) 所有样本嵌入
        all_labels:  (N,) 所有样本标签
    """
    model.eval()
    all_z = []
    all_labels = []
    all_sample_ids = []

    for batch in dataloader:
        x = batch["input"].to(device)
        z = model.encode(x)
        all_z.append(z.cpu())
        all_labels.append(batch["product"])
        all_sample_ids.extend(
            batch["sample_id"] if isinstance(batch["sample_id"], list)
            else [batch["sample_id"]]
        )

    all_z = torch.cat(all_z, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    store = PrototypeStore()
    unique_labels = all_labels.unique().tolist()

    return all_z, all_labels, unique_labels, store


def register_from_loader(model, dataloader, label_names, device,
                         percentile=95.0):
    """
    从 DataLoader 注册所有类原型。

    label_names: dict[int -> str] 标签索引到名称的映射
    """
    all_z, all_labels, unique_labels, store = compute_prototypes(
        model, dataloader, device, percentile
    )

    for lbl in unique_labels:
        mask = all_labels == lbl
        name = label_names.get(lbl, str(lbl))
        store.register(name, all_z[mask], percentile=percentile)

    # 球面原型调整: 均匀分布在单位超球面上
    store.redistribute_on_sphere()

    return store, all_z, all_labels


# ═══════════════════════════════════════════════════════════
#  增量微调: 新产品注册时微调编码器 + 重新注册所有原型
# ═══════════════════════════════════════════════════════════
def finetune_for_new_product(model, old_store, new_loader, old_loader,
                             cfg, device, new_label_names=None,
                             old_label_names=None):
    """
    检测到新产品后，微调编码器末层使新旧类特征均匀分布在球面上，
    同时保持对旧产品的一致性检测能力不下降。

    策略:
      1. 冻结编码器前几个 stage，只微调最后 stage + 投影头
      2. 使用旧类经验回放 (从 old_loader 采样) + 新类数据混合训练
      3. 加入原型蒸馏损失: 旧类嵌入向旧原型方向对齐，防止遗忘
      4. 微调后重新计算所有原型并做球面调整

    Args:
        model:           GCMSConsistencyNet (已加载旧权重)
        old_store:       PrototypeStore (旧类原型)
        new_loader:      DataLoader 新产品数据
        old_loader:      DataLoader 旧产品数据 (经验回放)
        cfg:             Config
        device:          torch.device
        new_label_names: dict[int -> str] 新类标签映射
        old_label_names: dict[int -> str] 旧类标签映射

    Returns:
        model:      微调后的模型
        new_store:  包含新旧所有类的 PrototypeStore
    """
    import torch.nn.functional as F
    from losses import SupConLoss, BatchPrototypeLoss

    # ── 1. 冻结编码器前几个 stage ──
    freeze_stages = cfg.finetune_freeze_encoder_stages
    frozen_modules = []
    encoder = model.encoder
    stage_map = [
        ("stem", encoder.stem),
        ("stage1", encoder.stage1), ("attn1", encoder.attn1),
        ("stage2", encoder.stage2), ("attn2", encoder.attn2),
        ("stage3", encoder.stage3), ("attn3", encoder.attn3),
    ]
    # stem 算 stage 0, stage1+attn1 = stage 1, ...
    n_frozen = 0
    n_frozen += 1  # stem
    frozen_modules.append(stage_map[0][1])
    for i in range(1, min(freeze_stages, 3) + 1):
        idx_stage = 2 * i - 1
        idx_attn = 2 * i
        if idx_stage < len(stage_map):
            frozen_modules.append(stage_map[idx_stage][1])
        if idx_attn < len(stage_map):
            frozen_modules.append(stage_map[idx_attn][1])

    for mod in frozen_modules:
        for param in mod.parameters():
            param.requires_grad_(False)

    # 不训练 domain_head 和 decoder
    for param in model.domain_head.parameters():
        param.requires_grad_(False)
    for param in model.decoder.parameters():
        param.requires_grad_(False)

    # ── 2. 保存旧类原型用于蒸馏 ──
    old_protos = {}
    for c in old_store.class_names:
        old_protos[c] = old_store.prototypes[c].to(device)

    # ── 3. 准备损失和优化器 ──
    supcon = SupConLoss(temperature=cfg.supcon_temperature).to(device)
    proto_loss_fn = BatchPrototypeLoss(margin=cfg.proto_margin).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.finetune_lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.finetune_epochs)

    # ── 4. 经验回放: 预提取旧类数据 ──
    old_data_cache = []
    for batch in old_loader:
        old_data_cache.append(batch)
    n_replay = max(1, int(len(old_data_cache) * cfg.finetune_replay_ratio))

    # ── 5. 微调循环 ──
    model.train()
    for epoch in range(cfg.finetune_epochs):
        total_loss = 0.0
        n_batches = 0

        # 随机选择经验回放批次
        replay_batches = random.sample(
            old_data_cache, min(n_replay, len(old_data_cache))
        )

        for new_batch in new_loader:
            x_new = new_batch["input"].to(device)
            y_new = new_batch["product"].to(device)
            z_new = model.encode(x_new)

            # 混合新旧数据
            if replay_batches:
                old_batch = replay_batches[n_batches % len(replay_batches)]
                x_old = old_batch["input"].to(device)
                y_old = old_batch["product"].to(device)
                z_old = model.encode(x_old)

                # 拼接
                z_all = torch.cat([z_new, z_old], dim=0)
                y_all = torch.cat([y_new, y_old], dim=0)
            else:
                z_all = z_new
                y_all = y_new

            # SupCon + Proto 损失
            proj_all = model.proj_head(z_all)
            proj_all = F.normalize(proj_all, dim=1)
            l_supcon = supcon(proj_all, y_all)
            l_proto = proto_loss_fn(z_all, y_all)

            # 原型蒸馏损失: 旧类嵌入保持接近旧原型
            l_distill = torch.tensor(0.0, device=device)
            if replay_batches and old_label_names:
                z_old_part = z_all[len(z_new):]
                y_old_part = y_all[len(z_new):]
                for c_name, old_proto in old_protos.items():
                    # 找旧类标签值
                    for lbl_idx, name in old_label_names.items():
                        if name == c_name:
                            mask = y_old_part == lbl_idx
                            if mask.any():
                                z_cls = z_old_part[mask]
                                # cosine距离: 保持方向对齐
                                cos_sim = F.cosine_similarity(
                                    z_cls, old_proto.unsqueeze(0), dim=1)
                                l_distill = l_distill + (1.0 - cos_sim).mean()
                            break
                n_old = max(len(old_protos), 1)
                l_distill = l_distill / n_old

            loss = (cfg.lambda_supcon * l_supcon
                    + cfg.lambda_proto * l_proto
                    + 0.5 * l_distill)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  微调 Epoch {epoch+1}/{cfg.finetune_epochs}  "
                  f"loss={avg_loss:.4f}")

    # ── 6. 解冻所有参数 (恢复正常状态) ──
    for param in model.parameters():
        param.requires_grad_(True)

    # ── 7. 重新注册所有原型 (新+旧) ──
    model.eval()
    new_store = PrototypeStore()

    # 注册旧类
    if old_loader is not None and old_label_names:
        old_z, old_labels, old_uniq, _ = compute_prototypes(
            model, old_loader, device, cfg.accept_percentile)
        for lbl in old_uniq:
            mask = old_labels == lbl
            name = old_label_names.get(lbl, str(lbl))
            new_store.register(name, old_z[mask],
                               percentile=cfg.accept_percentile)

    # 注册新类
    if new_label_names:
        new_z, new_labels, new_uniq, _ = compute_prototypes(
            model, new_loader, device, cfg.accept_percentile)
        for lbl in new_uniq:
            mask = new_labels == lbl
            name = new_label_names.get(lbl, str(lbl))
            new_store.register(name, new_z[mask],
                               percentile=cfg.accept_percentile)

    # 球面调整
    new_store.redistribute_on_sphere()
    new_store.summary()

    return model, new_store
