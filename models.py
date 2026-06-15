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

from tic_branch import compute_tic_from_tensor, TICEncoder1D, TICFusion


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
            x_t = x.permute(0, 3, 2, 1).contiguous().reshape(B * W, H, C)
        else:                             # 沿 m/z 轴
            x_t = x.permute(0, 2, 3, 1).contiguous().reshape(B * H, W, C)

        out, _ = self.attn(x_t, x_t, x_t)
        out = self.norm(out + x_t)

        if self.axis == "height":
            out = out.reshape(B, W, H, C).permute(0, 3, 2, 1).contiguous()
        else:
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
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
        self.feat_map_dim = channels[3]

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


def _parse_backbone_layers(layers):
    valid = {"layer1", "layer2", "layer3", "layer4"}
    if layers is None:
        return ["layer4"]

    if isinstance(layers, (list, tuple)):
        raw = [str(x).strip().lower() for x in layers if str(x).strip()]
    else:
        raw = [s.strip().lower() for s in str(layers).split(",") if s.strip()]

    if not raw:
        return ["layer4"]

    parsed = []
    for item in raw:
        if item not in valid:
            raise ValueError(f"不支持的主干层: {item}, 仅支持 {sorted(valid)}")
        if item not in parsed:
            parsed.append(item)
    return parsed


def _build_2d_sincos_pos_embed(h, w, dim, device, dtype):
    """生成二维 sin-cos 位置编码，避免固定输入尺寸带来的参数插值问题。"""
    if dim % 4 != 0:
        raise ValueError(f"transformer_embed_dim 必须能被 4 整除，当前={dim}")

    half = dim // 2
    quarter = half // 2
    omega = torch.arange(quarter, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max(float(quarter), 1.0)))

    grid_h = torch.arange(h, device=device, dtype=dtype)
    grid_w = torch.arange(w, device=device, dtype=dtype)

    out_h = torch.einsum("i,j->ij", grid_h, omega)
    out_w = torch.einsum("i,j->ij", grid_w, omega)

    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)

    pos = torch.zeros(h, w, dim, device=device, dtype=dtype)
    pos[:, :, :half] = emb_h[:, None, :]
    pos[:, :, half:] = emb_w[None, :, :]
    return pos.view(1, h * w, dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class GCMSTransformerEncoder(nn.Module):
    """GC-MS 主体 Transformer 编码器: patch embedding + token mixer blocks。"""

    def __init__(
        self,
        in_channels=2,
        patch_size=16,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        layers="layer4",
        fuse="concat",
        dropout=0.3,
    ):
        super().__init__()
        self.patch_size = int(max(4, patch_size))
        self.embed_dim = int(embed_dim)
        self.depth = int(max(2, depth))

        self.layer_list = _parse_backbone_layers(layers)
        self.fuse_mode = str(fuse or "concat").strip().lower()
        if self.fuse_mode not in {"concat", "last"}:
            raise ValueError("main_feature_fuse 仅支持 concat/last")

        self.patch_embed = nn.Conv2d(
            in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=False,
        )
        self.pre_norm = nn.LayerNorm(self.embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.embed_dim,
                num_heads=int(max(1, num_heads)),
                mlp_ratio=float(max(1.0, mlp_ratio)),
                dropout=float(max(0.0, min(dropout, 0.5))),
            )
            for _ in range(self.depth)
        ])
        self.post_norm = nn.LayerNorm(self.embed_dim)
        self.drop = nn.Dropout(float(max(0.0, min(dropout, 0.5))))

        self.feat_map_dim = self.embed_dim
        if self.fuse_mode == "last":
            self.out_dim = self.embed_dim
        else:
            self.out_dim = self.embed_dim * len(self.layer_list)

    def _stage_indices(self):
        depth = self.depth
        return {
            "layer1": max(1, depth // 4),
            "layer2": max(1, depth // 2),
            "layer3": max(1, (3 * depth) // 4),
            "layer4": depth,
        }

    def _tokens_to_map(self, tokens, h, w):
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.embed_dim, h, w)

    def forward(self, x):
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.pre_norm(tokens + _build_2d_sincos_pos_embed(h, w, c, x.device, x.dtype))

        stage_target = self._stage_indices()
        stage_maps = {}
        for i, blk in enumerate(self.blocks, start=1):
            tokens = blk(tokens)
            for layer_name, layer_idx in stage_target.items():
                if i == layer_idx and layer_name not in stage_maps:
                    stage_maps[layer_name] = self._tokens_to_map(tokens, h, w)

        tokens = self.post_norm(tokens)
        feat_map = self._tokens_to_map(tokens, h, w)
        stage_maps["layer4"] = feat_map

        vecs = [stage_maps[layer].mean(dim=(2, 3)) for layer in self.layer_list]
        if self.fuse_mode == "last":
            z = vecs[-1]
        else:
            z = torch.cat(vecs, dim=1)
        z = self.drop(z)
        return z, feat_map


def _resolve_tv_builder(arch: str):
    import torchvision.models as tv_models

    builders = {
        "resnet18": tv_models.resnet18,
        "resnet50": tv_models.resnet50,
        "wide_resnet50_2": tv_models.wide_resnet50_2,
    }
    if arch not in builders:
        raise ValueError(f"不支持的主模型骨干: {arch}")
    return builders[arch]


def _load_tv_resnet_local_weights(backbone, weight_path: str, arch: str):
    obj = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        state = obj
        for key in ("state_dict", "model", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
    else:
        raise ValueError(f"无效权重文件格式: {weight_path}")

    cleaned = {}
    prefixes = ("module.", "model.", "backbone.", "encoder.")
    for k, v in state.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        cleaned[nk] = v

    model_state = backbone.state_dict()
    matched = 0
    for k, v in cleaned.items():
        if k in model_state and getattr(v, "shape", None) == model_state[k].shape:
            matched += 1
    matched_ratio = matched / max(len(model_state), 1)

    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    missing_non_fc = [k for k in missing if not k.startswith("fc.")]
    unexpected_non_fc = [k for k in unexpected if not k.startswith("fc.")]

    if matched_ratio < 0.8:
        raise ValueError(
            "主模型骨干权重匹配率过低: "
            f"matched={matched}/{len(model_state)} ({matched_ratio:.1%}), "
            f"arch={arch}, weight={weight_path}"
        )

    if missing_non_fc or unexpected_non_fc:
        print(
            "[WARN] 主模型骨干检测到非fc参数键不匹配: "
            f"missing={len(missing_non_fc)}, unexpected={len(unexpected_non_fc)}, "
            f"arch={arch}, weight={weight_path}"
        )


class TorchvisionResNetEncoder(nn.Module):
    """可选主模型编码器: torchvision ResNet + 可配置多层融合。"""

    _layer_dims = {
        "resnet18": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
        "resnet50": {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
        "wide_resnet50_2": {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
    }

    def __init__(self, arch="resnet50", weight_path="", layers="layer4", fuse="concat", dropout=0.3):
        super().__init__()
        self.arch = str(arch)
        self.layer_list = _parse_backbone_layers(layers)
        self.fuse_mode = str(fuse or "concat").strip().lower()
        if self.fuse_mode not in {"concat", "last"}:
            raise ValueError("main_feature_fuse 仅支持 concat/last")

        builder = _resolve_tv_builder(self.arch)
        backbone = builder(weights=None)
        if weight_path:
            _load_tv_resnet_local_weights(backbone, weight_path, self.arch)

        # 与旧 encoder 接口兼容，便于增量微调/解释模块复用。
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4
        self.attn1 = nn.Identity()
        self.attn2 = nn.Identity()
        self.attn3 = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(float(dropout))

        layer_dims = self._layer_dims[self.arch]
        self.feat_map_dim = int(layer_dims["layer4"])
        if self.fuse_mode == "last":
            self.out_dim = int(layer_dims[self.layer_list[-1]])
        else:
            self.out_dim = int(sum(layer_dims[layer] for layer in self.layer_list))

        self.register_buffer("_img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _preprocess(self, x):
        x = x[:, :1, :, :].repeat(1, 3, 1, 1)
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-6)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self._img_mean) / self._img_std
        return x

    def forward(self, x):
        x = self._preprocess(x)
        x = self.stem(x)

        f1 = self.attn1(self.stage1(x))
        f2 = self.attn2(self.stage2(f1))
        f3 = self.attn3(self.stage3(f2))
        f4 = self.stage4(f3)

        feat_map = f4
        layer_map = {
            "layer1": f1,
            "layer2": f2,
            "layer3": f3,
            "layer4": f4,
        }
        vecs = [self.pool(layer_map[layer]).flatten(1) for layer in self.layer_list]
        if self.fuse_mode == "last":
            z = vecs[-1]
        else:
            z = torch.cat(vecs, dim=1)
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
    统一度量学习模型:
      训练: z(嵌入), proj(对比投影), domain_logits(批次对抗), recon(重建)
      推理: 仅需 z，配合 PrototypeStore 实现产品识别+一致性评分+拒识
      无 softmax 分类头: 类别数不写入网络权重, 支持注册即用
    """

    def __init__(self, num_batches, cfg):
        super().__init__()
        self.embed_normalize = cfg.embed_normalize
        self.tic_branch_enabled = bool(getattr(cfg, "tic_branch_enabled", False))

        backbone = str(getattr(cfg, "main_backbone", "gcms") or "gcms").lower()
        if backbone in {"gcms", "deep_consistency"}:
            self.encoder = GCMSEncoder(
                in_channels=cfg.in_channels,
                channels=cfg.encoder_channels,
                num_heads=cfg.num_axial_heads,
                dropout=cfg.dropout,
                blocks_per_stage=cfg.blocks_per_stage,
            )
        elif backbone in {"transformer", "vit", "gcms_transformer"}:
            self.encoder = GCMSTransformerEncoder(
                in_channels=cfg.in_channels,
                patch_size=int(getattr(cfg, "transformer_patch_size", 16)),
                embed_dim=int(getattr(cfg, "transformer_embed_dim", 256)),
                depth=int(getattr(cfg, "transformer_depth", 6)),
                num_heads=int(getattr(cfg, "transformer_num_heads", 8)),
                mlp_ratio=float(getattr(cfg, "transformer_mlp_ratio", 4.0)),
                layers=getattr(cfg, "main_feature_layers", "layer4"),
                fuse=getattr(cfg, "main_feature_fuse", "concat"),
                dropout=cfg.dropout,
            )
        else:
            weight_path = str(
                getattr(cfg, "main_backbone_model", "")
                or getattr(cfg, "pretrained_feature_model", "")
                or ""
            )
            self.encoder = TorchvisionResNetEncoder(
                arch=backbone,
                weight_path=weight_path,
                layers=getattr(cfg, "main_feature_layers", "layer4"),
                fuse=getattr(cfg, "main_feature_fuse", "concat"),
                dropout=cfg.dropout,
            )
        dim_2d = self.encoder.out_dim
        feat_map_dim = getattr(self.encoder, "feat_map_dim", dim_2d)

        dim = dim_2d
        if self.tic_branch_enabled:
            tic_dim = int(getattr(cfg, "tic_embed_dim", 64))
            fusion_mode = str(getattr(cfg, "tic_fusion_mode", "concat") or "concat").lower()
            fusion_out = int(getattr(cfg, "tic_fusion_output_dim", dim_2d) or dim_2d)
            if fusion_mode == "sum":
                fusion_out = dim_2d
            self.tic_encoder = TICEncoder1D(
                input_length=int(getattr(cfg, "rt_bins", 1024)),
                embed_dim=tic_dim,
                encoder_type=str(getattr(cfg, "tic_encoder", "cnn1d") or "cnn1d").lower(),
            )
            self.tic_fusion = TICFusion(
                z_dim=dim_2d,
                tic_dim=tic_dim,
                output_dim=fusion_out,
                mode=fusion_mode,
            )
            dim = fusion_out
        else:
            self.tic_encoder = None
            self.tic_fusion = None

        # 对比学习投影头 (仅训练)
        self.proj_head = ProjectionHead(dim, cfg.proj_dim)
        # 批次对抗头: 梯度反转消除批次信息 (训练后丢弃)
        self.domain_head = DomainHead(dim, num_batches)
        # 重建解码器 (正则项)
        self.decoder = ReconDecoder(
            feat_map_dim, cfg.in_channels, cfg.rt_bins, cfg.mz_bins
        )

    def _fuse_tic(self, x, z_raw):
        if not self.tic_branch_enabled:
            return z_raw
        tic = compute_tic_from_tensor(x)
        z_tic = self.tic_encoder(tic)
        return self.tic_fusion(z_raw, z_tic)

    def forward(self, x, tic=None, return_feat_map=False):
        z_2d_raw, feat_map = self.encoder(x)
        if self.tic_branch_enabled:
            tic_in = compute_tic_from_tensor(x) if tic is None else tic
            z_raw = self.tic_fusion(z_2d_raw, self.tic_encoder(tic_in))
        else:
            z_raw = z_2d_raw

        # 归一化嵌入 (度量学习标准做法)
        if self.embed_normalize:
            z = F.normalize(z_raw, dim=1)
        else:
            z = z_raw

        proj = self.proj_head(z_raw)
        domain_logits = self.domain_head(z_raw)
        recon = self.decoder(feat_map)

        out = {
            "z": z,
            "z_raw": z_raw,
            "proj": proj,
            "domain_logits": domain_logits,
            "recon": recon,
        }
        if return_feat_map:
            out["feat_map"] = feat_map
        return out

    def encode(self, x, tic=None):
        """仅提取嵌入向量 (推理用)。"""
        z_2d_raw, _ = self.encoder(x)
        if self.tic_branch_enabled:
            tic_in = compute_tic_from_tensor(x) if tic is None else tic
            z_raw = self.tic_fusion(z_2d_raw, self.tic_encoder(tic_in))
        else:
            z_raw = z_2d_raw
        if self.embed_normalize:
            return F.normalize(z_raw, dim=1)
        return z_raw
