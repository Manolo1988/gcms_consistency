"""
原型注册与推理:
  - 冻结 Backbone，从参考样本计算类原型和半径
  - 基于原型距离的产品识别、一致性评分、拒识判定
"""
import json
import numpy as np
import torch
from pathlib import Path


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

    def predict(self, z):
        """
        基于原型匹配的推理。

        z: (B, D) — 待测样本嵌入
        返回 dict:
            pred_class:  list[str]  — 预测产品名称
            pred_idx:    (B,)       — 预测类别索引
            scores:      (B,)       — 一致性分数 [0, 1]
            min_dists:   (B,)       — 到最近原型距离
            all_dists:   (B, K)     — 到所有原型距离
        """
        if not self.class_names:
            raise RuntimeError("原型库为空，请先注册产品")

        proto_mat = torch.stack([self.prototypes[c] for c in self.class_names])
        proto_mat = proto_mat.to(z.device)  # (K, D)

        # 计算到所有原型的距离
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

    def load(self, path):
        """加载已保存的原型库。"""
        path = Path(path)

        with open(path / "proto_meta.json") as f:
            meta = json.load(f)

        self.class_names = meta["class_names"]
        self.radii = {k: float(v) for k, v in meta["radii"].items()}
        self.prototypes = torch.load(path / "prototypes.pt",
                                     map_location="cpu", weights_only=True)

    @property
    def num_classes(self):
        return len(self.class_names)

    def summary(self):
        """打印原型库概览。"""
        print(f"原型库: {self.num_classes} 个产品")
        for c in self.class_names:
            p = self.prototypes[c]
            print(f"  {c}: radius={self.radii[c]:.4f}, "
                  f"proto_norm={p.norm().item():.4f}")


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

    return store, all_z, all_labels
