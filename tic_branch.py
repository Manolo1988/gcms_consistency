"""
TIC (Total Ion Chromatogram) auxiliary branch for GC-MS product recognition.

Provides:
  - TIC computation from RT x m/z tensor (sum over m/z axis)
  - TICEncoder1D: 1D CNN / MLP / Transformer encoder for TIC profiles
  - Fusion modules: concat, gated, sum

All components are optional and controlled via Config flags:
  tic_branch_enabled=False, tic_source="from_tensor", etc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_tic_from_tensor(x, ch0_is_zscore=True, grid=None):
    """
    Compute TIC profile from RT x m/z tensor.

    x: (B, C, H, W) tensor
       - Channel 0: z-score normalized absolute intensity
       - Channel 1: log-relative composition
    ch0_is_zscore: if True, channel 0 is z-score normalized

    Returns tic: (B, H) - TIC profile along RT axis

    If grid is provided, use it for absolute intensity TIC.
    Otherwise, fall back to channel 0 sum over m/z.
    """
    if grid is not None:
        # grid: (H, W) raw absolute intensity (before z-score + log)
        if isinstance(grid, np.ndarray):
            grid = torch.from_numpy(grid).to(x.device, dtype=x.dtype)
        if grid.dim() == 2:
            grid = grid.unsqueeze(0)  # (1, H, W)
        # Sum over m/z axis
        tic = grid.sum(dim=2)  # (B, H)
    else:
        # Use channel 0: sum over m/z dimension
        # Channel 0 is z-score normalized, but the relative profile is preserved
        ch0 = x[:, 0, :, :]  # (B, H, W)
        tic = ch0.sum(dim=2)  # (B, H)

    # Normalize to [0, 1] per sample
    tic_min = tic.min(dim=1, keepdim=True).values
    tic_max = tic.max(dim=1, keepdim=True).values
    tic = (tic - tic_min) / (tic_max - tic_min + 1e-8)

    return tic


class TICEncoder1D(nn.Module):
    """
    1D CNN encoder for TIC profiles.

    Input: (B, 1, H) - TIC profile
    Output: (B, embed_dim) - TIC embedding
    """

    def __init__(self, input_length=1152, embed_dim=64, encoder_type="cnn1d"):
        super().__init__()
        self.encoder_type = encoder_type
        self.input_length = input_length
        self.embed_dim = embed_dim

        if encoder_type == "cnn1d":
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
        elif encoder_type == "mlp":
            flat_dim = input_length
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, embed_dim),
            )
        elif encoder_type == "transformer":
            self.input_proj = nn.Linear(1, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=4, dim_feedforward=256,
                dropout=0.1, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError(f"Unknown TIC encoder type: {encoder_type}")

    def forward(self, tic):
        """
        tic: (B, H) or (B, 1, H)
        Returns: (B, embed_dim)
        """
        if tic.dim() == 2:
            tic = tic.unsqueeze(1)  # (B, 1, H)

        if self.encoder_type == "transformer":
            x = tic.transpose(1, 2)  # (B, H, 1)
            x = self.input_proj(x)   # (B, H, embed_dim)
            x = self.transformer(x)
            x = x.transpose(1, 2)    # (B, embed_dim, H)
            x = self.pool(x).squeeze(-1)
        else:
            x = self.encoder(tic)
            x = x.squeeze(-1)

        return x


class TICFusion(nn.Module):
    """
    Fusion module for combining 2D encoder output with TIC embedding.
    """

    def __init__(self, z_dim, tic_dim, output_dim=256, mode="concat"):
        super().__init__()
        self.mode = mode
        self.z_dim = z_dim
        self.tic_dim = tic_dim

        if mode == "concat":
            self.fusion_proj = nn.Sequential(
                nn.Linear(z_dim + tic_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True),
            )
        elif mode == "gated":
            total_dim = z_dim + tic_dim
            self.gate_net = nn.Sequential(
                nn.Linear(total_dim, total_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(total_dim // 2, z_dim),
                nn.Sigmoid(),
            )
            self.tic_proj = nn.Linear(tic_dim, z_dim)
            self.fusion_proj = nn.Sequential(
                nn.Linear(z_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True),
            )
        elif mode == "sum":
            self.tic_proj = nn.Linear(tic_dim, z_dim)
            # No extra projection needed, sum is direct
            self.fusion_proj = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(self, z_2d, z_tic):
        """
        z_2d: (B, z_dim) - 2D encoder output (unnormalized)
        z_tic: (B, tic_dim) - TIC encoder output
        Returns: (B, output_dim) - fused embedding
        """
        if self.mode == "concat":
            fused = torch.cat([z_2d, z_tic], dim=1)
            return self.fusion_proj(fused)

        elif self.mode == "gated":
            concat = torch.cat([z_2d, z_tic], dim=1)
            gate = self.gate_net(concat)
            tic_proj = self.tic_proj(z_tic)
            gated = gate * z_2d + (1 - gate) * tic_proj
            return self.fusion_proj(gated)

        elif self.mode == "sum":
            tic_proj = self.tic_proj(z_tic)
            return z_2d + tic_proj

        return z_2d
