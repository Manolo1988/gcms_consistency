"""全局配置：路径、网格参数、模型参数、训练超参数。"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


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

    # ── 训练 ─────────────────────────────────────────────
    epochs_pretrain: int = 80
    epochs_finetune: int = 120
    batch_size: int = 8
    lr_pretrain: float = 1e-3
    lr_finetune: float = 3e-4
    weight_decay: float = 1e-4

    # ── 损失权重 ─────────────────────────────────────────
    # L_total = L_supcon + λ_adv * L_adv + λ_proto * L_proto + λ_recon * L_recon
    lambda_supcon: float = 1.0                # 监督对比损失权重
    lambda_adv: float = 0.3                   # 批次对抗损失权重
    lambda_proto: float = 1.0                 # 原型距离损失权重
    lambda_recon: float = 0.2                 # 重建正则损失权重
    lambda_cls: float = 0.5                   # 辅助分类损失权重 (训练辅助)
    supcon_temperature: float = 0.07          # SupCon 温度参数
    proto_margin: float = 1.0                 # 原型损失推斥间距

    # ── 一致性与原型 ─────────────────────────────────────
    accept_percentile: float = 95.0           # 一致性径阈值百分位
    reject_threshold_factor: float = 2.0      # 拒识: dist > factor * radius

    # ── 实验设置 ─────────────────────────────────────────
    # "closed": 闭集跨批次 (所有类参与训练, 按批次划分)
    # "open":   开集 (留出部分类不参与训练)
    # "fewshot": 少样本 (全类训练, 测试时模拟 N-shot 注册)
    split_mode: str = "closed"
    num_open_test_classes: int = 2            # 开集: 测试用类数
    num_open_val_classes: int = 1             # 开集: 验证用类数
    n_shot_values: tuple = (1, 3, 5, 10)     # 少样本: 注册样本数列表

    # ── 数据增强 ──────────────────────────────────────────
    aug_intensity_scale: tuple = (0.8, 1.2)
    aug_noise_std: float = 0.05
    aug_mask_ratio: float = 0.15
    aug_rt_shift_max: int = 8
    aug_mz_shift_max: int = 2

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
