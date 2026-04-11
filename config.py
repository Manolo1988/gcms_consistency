"""全局配置：路径、网格参数、模型参数、训练超参数。"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# MPS fallback: 某些算子在 MPS 上未实现时回退到 CPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device():
    """自动选择最佳可用设备: CUDA > MPS > CPU。
    
    MPS 上的部分算子 (ConvTranspose2d backward 等) 存在兼容问题，
    通过试运行检测实际可用性。
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # 快速试运行: 模拟 AxialAttention 中 permute+reshape 的前向/反向
            x = torch.randn(1, 4, 8, 8, device="mps", requires_grad=True)
            xt = x.permute(0, 3, 2, 1).contiguous().reshape(8, 8, 4)
            out = xt.sum()
            out.backward()
            # 测试 ConvTranspose2d (ReconDecoder 使用)
            ct = torch.nn.ConvTranspose2d(4, 2, 4, stride=2, padding=1).to("mps")
            y = torch.randn(1, 4, 8, 8, device="mps")
            ct(y).sum().backward()
            # 测试 MultiheadAttention
            mha = torch.nn.MultiheadAttention(4, 1, batch_first=True).to("mps")
            q = torch.randn(2, 4, 4, device="mps")
            mha(q, q, q)[0].sum().backward()
            del x, xt, out, ct, y, mha, q
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


@dataclass
class Config:
    # ── 路径 ──────────────────────────────────────────────
    dataset_root: str = str(
        Path(__file__).resolve().parent / "dataset"
    )
    output_dir: str = str(
        Path(__file__).resolve().parent / "outputs"
    )
    prepared_dir: str = str(
        Path(__file__).resolve().parent / "prepared_data"
    )

    # ── RT × m/z 网格 ────────────────────────────────────
    rt_bins: int = 1024
    mz_bins: int = 256
    rt_range: Optional[tuple] = (0.0, 40.0)
    mz_range: tuple = (0.0, 200.0)
    log_transform: bool = True
    rt_range_percentiles: tuple = (1.0, 99.5)
    mz_range_percentiles: tuple = (1.0, 99.0)

    # ── 模型 ─────────────────────────────────────────────
    in_channels: int = 2                      # 绝对 + 相对通道
    feature_dim: int = 256                    # 嵌入空间维度
    proj_dim: int = 128                       # 对比学习投影头输出维度
    encoder_channels: tuple = (32, 64, 128, 256)
    blocks_per_stage: int = 2                 # 每阶段 ResBlock 数量
    num_axial_heads: int = 4
    dropout: float = 0.3
    embed_normalize: bool = True              # L2 归一化嵌入

    # ── 训练 (单阶段) ──────────────────────────────────────
    epochs: int = 200
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # ── 损失权重 ─────────────────────────────────────────
    # L = L_supcon + λ₁·L_adv + λ₂·L_proto + λ_recon·L_recon
    # 无 softmax 分类损失: 类别数不写入网络权重, 支持注册即用
    lambda_supcon: float = 1.0                # 监督对比损失 (类间可分)
    lambda_adv: float = 0.1                   # λ₁ 批次对抗 (去批次)
    lambda_proto: float = 0.5                 # λ₂ 原型紧凑 (类内紧凑)
    lambda_recon: float = 0.2                 # 重建正则
    supcon_temperature: float = 0.07          # SupCon 温度参数
    proto_margin: float = 1.0                 # 原型损失推斥间距

    # ── 一致性与原型 ─────────────────────────────────────
    accept_percentile: float = 95.0           # 一致性径阈值百分位
    reject_threshold_factor: float = 2.0      # 拒识: dist > factor * radius

    # ── 增量注册微调 ─────────────────────────────────────
    finetune_epochs: int = 20                 # 微调轮数
    finetune_lr: float = 1e-4                 # 微调学习率 (低于初始 lr)
    finetune_freeze_encoder_stages: int = 3   # 冻结编码器前 N 个 stage
    finetune_replay_ratio: float = 0.3        # 旧类经验回放样本比例

    # ── 实验设置 ─────────────────────────────────────────
    # 训练一个模型, 三个 Setting 共用同一模型:
    #   A: 闭集跨批次 (已知类性能 + 批次鲁棒性)
    #   B: 开放集 (已知 vs 未知类判别)
    #   C: 少样本注册 (N-shot 新品扩展)
    num_open_test_classes: int = 2            # 留出的未知类数量
    n_shot_values: tuple = (1, 3, 5, 10)     # 少样本注册样本数列表
    holdout_batch_ratio: float = 0.1          # 留出批次比例 (Setting A)
    min_samples_per_product: int = 10         # 产品最低样本数量, 低于此排除
    val_ratio: float = 0.1                    # 训练集中验证子集比例

    # ── 数据增强 ──────────────────────────────────────────
    aug_intensity_scale: tuple = (0.8, 1.2)
    aug_noise_std: float = 0.05
    aug_mask_ratio: float = 0.15
    aug_rt_shift_max: int = 8
    aug_mz_shift_max: int = 2
    # GC-MS 专用增强
    aug_baseline_wander_amp: float = 0.03     # 基线漂移幅度
    aug_baseline_wander_freq: int = 3         # 基线漂移正弦周期数
    aug_peak_broaden_sigma: float = 1.5       # 峰展宽高斯 sigma 上限
    aug_rt_warp_strength: float = 0.02        # RT 非线性扭曲幅度
    aug_mz_channel_drop: float = 0.05         # m/z 通道随机丢弃比例
    aug_tic_jitter: float = 0.1               # TIC 归一化抖动幅度

    # ── 产品标签粒度 ─────────────────────────────────────
    product_granularity: str = "fine"

    # ── 排除规则 ──────────────────────────────────────────
    exclude_blanks: bool = True
    exclude_special: bool = True

    # ── 数据准备可视化 ────────────────────────────────────
    save_prepare_plots: bool = True
    prepare_plot_max_samples: Optional[int] = None
    prepare_plot_dpi: int = 120
    tag_output_with_batch_and_product: bool = True

    # ── 数据准备表格导出 ─────────────────────────────────
    save_prepare_tables: bool = True

    seed: int = 42
