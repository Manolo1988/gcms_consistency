"""
原型注册与推理:
  - 冻结 Backbone，从参考样本计算类原型和半径
  - 基于原型距离的产品识别、一致性评分、拒识判定
  - SAIM 风格球面原型调整：增量注册时保持超球面均匀分布
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
            all_dists = torch.cdist(z, proto_mat)  # (B, K)

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
                                     map_location="cpu", weights_only=True)

        sph_path = path / "spherical_prototypes.pt"
        if sph_path.exists():
            self.spherical_prototypes = torch.load(
                sph_path, map_location="cpu", weights_only=True
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
