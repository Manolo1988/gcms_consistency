"""
基于度量学习的批次不变深度网络：
  Stem → [ResBlocks + 双轴注意力] × 3 → 嵌入空间
  训练头: 投影头(SupCon) / 辅助分类头 / 批次对抗头 / 重建头
  推理: 原型匹配 → 产品识别 + 一致性评分 + 拒识判定
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
#  RT × m/z 双轴结构化注意力
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


class DualAxisAttention(nn.Module):
    """RT 方向 + m/z 方向双轴注意力融合。"""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.rt_attn = AxialAttention(dim, num_heads, axis="height")
        self.mz_attn = AxialAttention(dim, num_heads, axis="width")

    def forward(self, x):
        x = self.rt_attn(x)
        x = self.mz_attn(x)
        return x


# ═══════════════════════════════════════════════════════════
#  编码器 (多级双轴注意力)
# ═══════════════════════════════════════════════════════════
def _make_stage(in_c, out_c, num_blocks=2, stride=2):
    """构建一个包含 num_blocks 个 ResBlock 的阶段。"""
    layers = [ResBlock2D(in_c, out_c, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(ResBlock2D(out_c, out_c, stride=1))
    return nn.Sequential(*layers)


class GCMSEncoder(nn.Module):
    """
    Stem → [ResBlocks×N + DualAxisAttention] × 3 → GlobalAvgPool → 嵌入
    每阶段后施加 RT + m/z 双轴注意力。
    """

    def __init__(self, in_channels=2, channels=(32, 64, 128, 256),
                 num_heads=4, dropout=0.3, blocks_per_stage=2):
        super().__init__()
        # Stem: 快速下采样
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Stage 1: 捕获小范围峰形
        self.stage1 = _make_stage(channels[0], channels[1], blocks_per_stage, stride=2)
        self.attn1 = DualAxisAttention(channels[1], num_heads)

        # Stage 2: 中尺度色谱模式
        self.stage2 = _make_stage(channels[1], channels[2], blocks_per_stage, stride=2)
        self.attn2 = DualAxisAttention(channels[2], num_heads)

        # Stage 3: 全局指纹
        self.stage3 = _make_stage(channels[2], channels[3], blocks_per_stage, stride=2)
        self.attn3 = DualAxisAttention(channels[3], num_heads)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.out_dim = channels[3]

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.attn1(x)

        x = self.stage2(x)
        x = self.attn2(x)

        x = self.stage3(x)
        x = self.attn3(x)

        feat_map = x                              # 保留，用于 Grad-CAM
        z = self.pool(x).flatten(1)
        z = self.drop(z)
        return z, feat_map


# ═══════════════════════════════════════════════════════════
#  任务头
# ═══════════════════════════════════════════════════════════
class ProjectionHead(nn.Module):
    """对比学习投影头: z → p (仅训练时使用)。"""

    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, z):
        p = self.net(z)
        return F.normalize(p, dim=1)


class ProductHead(nn.Module):
    """辅助分类头 (训练辅助，推理可不用)。"""

    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        return self.fc(z)


class DomainHead(nn.Module):
    """批次对抗头: 梯度反转消除批次信息。"""

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


# ═══════════════════════════════════════════════════════════
#  重建解码器 (可选正则)
# ═══════════════════════════════════════════════════════════
class ReconDecoder(nn.Module):
    """轻量卷积解码器，从 feature map 重建输入。"""

    def __init__(self, in_channels=256, out_channels=2,
                 target_h=1024, target_w=256):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
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
    """
    训练时输出: z(嵌入), proj(对比投影), logits(辅助分类),
               domain_logits(批次对抗), recon(重建)
    推理时: 仅需 z，配合 PrototypeStore 实现产品识别+一致性评分+拒识
    """

    def __init__(self, num_products, num_batches, cfg):
        super().__init__()
        self.embed_normalize = cfg.embed_normalize

        self.encoder = GCMSEncoder(
            in_channels=cfg.in_channels,
            channels=cfg.encoder_channels,
            num_heads=cfg.num_axial_heads,
            dropout=cfg.dropout,
            blocks_per_stage=cfg.blocks_per_stage,
        )
        dim = self.encoder.out_dim

        # 对比学习投影头
        self.proj_head = ProjectionHead(dim, cfg.proj_dim)
        # 辅助分类头 (训练辅助)
        self.product_head = ProductHead(dim, num_products)
        # 批次对抗头 (仅训练)
        self.domain_head = DomainHead(dim, num_batches)
        # 重建解码器 (可选正则)
        self.decoder = ReconDecoder(
            dim, cfg.in_channels, cfg.rt_bins, cfg.mz_bins
        )

    def forward(self, x, return_feat_map=False):
        z_raw, feat_map = self.encoder(x)

        # 归一化嵌入 (度量学习标准做法)
        if self.embed_normalize:
            z = F.normalize(z_raw, dim=1)
        else:
            z = z_raw

        proj = self.proj_head(z_raw)
        logits = self.product_head(z_raw)
        domain_logits = self.domain_head(z_raw)
        recon = self.decoder(feat_map)

        out = {
            "z": z,
            "z_raw": z_raw,
            "proj": proj,
            "logits": logits,
            "domain_logits": domain_logits,
            "recon": recon,
        }
        if return_feat_map:
            out["feat_map"] = feat_map
        return out

    def encode(self, x):
        """仅提取嵌入向量 (推理用)。"""
        z_raw, _ = self.encoder(x)
        if self.embed_normalize:
            return F.normalize(z_raw, dim=1)
        return z_raw