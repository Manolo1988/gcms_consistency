"""
Grad-CAM 解释:
  在 RT × m/z 张量上定位关键区域，映射回 RT 和 m/z 坐标。
  支持基于分类 logits 和基于嵌入距离两种 CAM 模式。
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class GradCAM:
    """
    针对 GCMSConsistencyNet 的 Grad-CAM。
    支持两种目标:
      - "logits":    对辅助分类头的 logits 做 CAM (默认)
      - "embedding": 对嵌入到目标原型距离做 CAM (更贴合度量学习)
    """

    def __init__(self, model, target_layer=None, mode="logits"):
        self.model = model
        self.mode = mode
        self.gradients = None
        self.activations = None

        if target_layer is None:
            target_layer = model.encoder.stage3
        self._hook(target_layer)

    def _hook(self, layer):
        def fwd_hook(module, input, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer.register_forward_hook(fwd_hook)
        layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x, target_class=None, target_proto=None):
        """
        x: (1, C, H, W) 单个样本
        target_proto: (D,) 目标原型向量 (embedding 模式下使用)
        返回: (H, W) 的热力图
        """
        self.model.eval()
        x = x.requires_grad_(True)
        out = self.model(x, return_feat_map=True)

        self.model.zero_grad()

        if self.mode == "embedding" and target_proto is not None:
            # 基于嵌入距离: 最小化到目标原型的距离
            z = out["z"]
            target_proto = target_proto.to(z.device).unsqueeze(0)
            neg_dist = -torch.norm(z - target_proto, dim=1)
            neg_dist.backward()
        else:
            # 基于分类 logits
            logits = out["logits"]
            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            logits[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        return cam


def find_top_regions(cam, rt_range, mz_range, top_k=10, min_size=5):
    """
    在 Grad-CAM 热力图中找到最重要的区域，返回 RT 和 m/z 坐标。
    """
    H, W = cam.shape
    rt_min, rt_max = rt_range
    mz_min, mz_max = mz_range

    # 简单方法: 把 cam 按行/列积分，找峰
    rt_profile = cam.mean(axis=1)
    mz_profile = cam.mean(axis=0)

    rt_axis = np.linspace(rt_min, rt_max, H)
    mz_axis = np.linspace(mz_min, mz_max, W)

    # 取 top-k 的 RT 位置
    rt_top_idx = np.argsort(rt_profile)[-top_k:][::-1]
    mz_top_idx = np.argsort(mz_profile)[-top_k:][::-1]

    regions = []
    for ri in rt_top_idx:
        for mi in mz_top_idx:
            if cam[ri, mi] > 0.3:
                regions.append({
                    "rt": float(rt_axis[ri]),
                    "mz": float(mz_axis[mi]),
                    "importance": float(cam[ri, mi]),
                })

    regions.sort(key=lambda r: r["importance"], reverse=True)
    return regions[:top_k]


def plot_interpretation(x_np, cam, rt_range, mz_range,
                        sample_id="", consistency_score=None,
                        pred_class=None, save_dir=None):
    """
    绘制: 原始二维图 + Grad-CAM 叠加 + RT/m/z 投影。
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rt_min, rt_max = rt_range
    mz_min, mz_max = mz_range

    # (0,0) 原始输入 ch0
    ax = axes[0, 0]
    ax.imshow(x_np[0], aspect="auto", origin="lower",
              extent=[mz_min, mz_max, rt_min, rt_max], cmap="viridis")
    ax.set_title("Input Ch0 (Absolute)")
    ax.set_xlabel("m/z")
    ax.set_ylabel("RT (min)")

    # (0,1) Grad-CAM 叠加
    ax = axes[0, 1]
    ax.imshow(x_np[0], aspect="auto", origin="lower",
              extent=[mz_min, mz_max, rt_min, rt_max], cmap="gray", alpha=0.5)
    ax.imshow(cam, aspect="auto", origin="lower",
              extent=[mz_min, mz_max, rt_min, rt_max], cmap="jet", alpha=0.5)
    ax.set_title("Grad-CAM Overlay")
    ax.set_xlabel("m/z")
    ax.set_ylabel("RT (min)")

    # (1,0) RT 投影
    ax = axes[1, 0]
    rt_axis = np.linspace(rt_min, rt_max, cam.shape[0])
    ax.plot(rt_axis, cam.mean(axis=1), color="red")
    ax.fill_between(rt_axis, cam.mean(axis=1), alpha=0.3, color="red")
    ax.set_title("RT Contribution Profile")
    ax.set_xlabel("RT (min)")
    ax.set_ylabel("Importance")

    # (1,1) m/z 投影
    ax = axes[1, 1]
    mz_axis = np.linspace(mz_min, mz_max, cam.shape[1])
    ax.plot(mz_axis, cam.mean(axis=0), color="blue")
    ax.fill_between(mz_axis, cam.mean(axis=0), alpha=0.3, color="blue")
    ax.set_title("m/z Contribution Profile")
    ax.set_xlabel("m/z")
    ax.set_ylabel("Importance")

    title = f"Interpretation: {sample_id}"
    if pred_class:
        title += f" | Predicted: {pred_class}"
    if consistency_score is not None:
        title += f" | Score: {consistency_score:.3f}"
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"interpret_{sample_id}.png", dpi=150)
    plt.close(fig)
    return fig