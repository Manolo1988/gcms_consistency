"""
多任务批次不变深度网络：
  2D CNN Stem → ResBlocks → Axial Attention → 共享特征
  → 产品分类头 / 域对抗头 / 原型距离头 / 能量拒识头
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ═══════════════════════════════════════════════════════════
#  基础模块
# ═══════════════════════════════════════════════════════════
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力。"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResBlock2D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)

        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + self.shortcut(x), inplace=True)


# ═══════════════════════════════════════════════════════════
#  轴向注意力
# ═══════════════════════════════════════════════════════════
class AxialAttention(nn.Module):
    """沿指定轴做 multi-head self-attention。"""

    def __init__(self, dim, num_heads=4, axis="height"):
        super().__init__()
        self.axis = axis
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.axis == "height":         # 沿 RT 轴
            x_t = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        else:                             # 沿 m/z 轴
            x_t = x.permute(0, 2, 3, 1).reshape(B * H, W, C)

        out, _ = self.attn(x_t, x_t, x_t)
        out = self.norm(out + x_t)

        if self.axis == "height":
            out = out.reshape(B, W, H, C).permute(0, 3, 2, 1)
        else:
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


# ═══════════════════════════════════════════════════════════
#  编码器
# ═══════════════════════════════════════════════════════════
class GCMSEncoder(nn.Module):
    def __init__(self, in_channels=2, channels=(32, 64, 128, 256),
                 num_heads=4, dropout=0.3):
        super().__init__()
        # Stem: 快速下采样
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # 残差块
        self.layer1 = ResBlock2D(channels[0], channels[1], stride=2)
        self.layer2 = ResBlock2D(channels[1], channels[2], stride=2)
        self.layer3 = ResBlock2D(channels[2], channels[3], stride=2)

        # 轴向注意力（在最深层做，空间已经很小）
        self.axial_rt = AxialAttention(channels[3], num_heads, axis="height")
        self.axial_mz = AxialAttention(channels[3], num_heads, axis="width")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.out_dim = channels[3]

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.axial_rt(x)
        x = self.axial_mz(x)
        feat_map = x                              # 保留，用于 Grad-CAM
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return x, feat_map


# ═══════════════════════════════════════════════════════════
#  任务头
# ═══════════════════════════════════════════════════════════
class ProductHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        return self.fc(z)


class DomainHead(nn.Module):
    def __init__(self, in_dim, num_domains, alpha=1.0):
        super().__init__()
        self.grl = GradientReversal(alpha)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, num_domains),
        )

    def forward(self, z):
        return self.fc(self.grl(z))

    def set_alpha(self, alpha):
        self.grl.alpha = alpha


class PrototypeHead(nn.Module):
    """每个产品一个可学习原型中心。"""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_dim) * 0.1)

    def forward(self, z, labels=None):
        # z: (B, D),  prototypes: (K, D)
        dists = torch.cdist(z.unsqueeze(0), self.prototypes.unsqueeze(0)).squeeze(0)
        # dists: (B, K)
        return dists

    def consistency_score(self, z, pred_labels):
        """返回每个样本到其预测产品原型的距离。"""
        dists = torch.cdist(z.unsqueeze(0), self.prototypes.unsqueeze(0)).squeeze(0)
        scores = dists[torch.arange(len(pred_labels)), pred_labels]
        return scores


class EnergyHead(nn.Module):
    """基于能量的拒识。"""
    def __init__(self, in_dim, temperature=1.0):
        super().__init__()
        self.T = temperature
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z, logits=None):
        if logits is not None:
            energy = -self.T * torch.logsumexp(logits / self.T, dim=1)
        else:
            energy = self.fc(z).squeeze(-1)
        return energy


# ═══════════════════════════════════════════════════════════
#  MAE 预训练解码器
# ═══════════════════════════════════════════════════════════
class MAEDecoder(nn.Module):
    """轻量卷积解码器，用于重构预训练。"""
    def __init__(self, in_channels=256, out_channels=2,
                 target_h=1024, target_w=256):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        # 从 encoder 的 feature_map 上采样回原图大小
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_channels, 4, stride=2, padding=1),
        )

    def forward(self, feat_map):
        x = self.up(feat_map)
        x = F.interpolate(x, size=(self.target_h, self.target_w),
                          mode="bilinear", align_corners=False)
        return x


# ═══════════════════════════════════════════════════════════
#  完整模型
# ═══════════════════════════════════════════════════════════
class GCMSConsistencyNet(nn.Module):
    def __init__(self, num_products, num_batches, cfg):
        super().__init__()
        self.encoder = GCMSEncoder(
            in_channels=cfg.in_channels,
            channels=cfg.encoder_channels,
            num_heads=cfg.num_axial_heads,
            dropout=cfg.dropout,
        )
        dim = self.encoder.out_dim
        self.product_head = ProductHead(dim, num_products)
        self.domain_head = DomainHead(dim, num_batches)
        self.proto_head = PrototypeHead(dim, num_products)
        self.energy_head = EnergyHead(dim)
        self.decoder = MAEDecoder(
            dim, cfg.in_channels, cfg.rt_bins, cfg.mz_bins
        )

    def forward(self, x, return_feat_map=False):
        z, feat_map = self.encoder(x)
        logits = self.product_head(z)
        domain_logits = self.domain_head(z)
        proto_dists = self.proto_head(z)
        energy = self.energy_head(z, logits)
        recon = self.decoder(feat_map)

        out = {
            "z": z,
            "logits": logits,
            "domain_logits": domain_logits,
            "proto_dists": proto_dists,
            "energy": energy,
            "recon": recon,
        }
        if return_feat_map:
            out["feat_map"] = feat_map
        return out

    def predict(self, x):
        """推理：输出产品预测、一致性分数、拒识判定。"""
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            pred = out["logits"].argmax(dim=1)
            confidence = F.softmax(out["logits"], dim=1).max(dim=1).values
            consist_dist = self.proto_head.consistency_score(out["z"], pred)
            energy = out["energy"]
        return {
            "pred_product": pred,
            "confidence": confidence,
            "consistency_dist": consist_dist,
            "energy": energy,
            "z": out["z"],
        }