"""全局配置：路径、网格参数、模型参数、训练超参数。"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── 路径 ──────────────────────────────────────────────
    # dataset/ 目录与本文件同级
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
    rt_range: Optional[tuple] = (0.0, 40.0)   # 固定保留时间范围 (min)
    mz_range: tuple = (0.0, 200.0)            # 固定 m/z 范围
    log_transform: bool = True
    # 用于兼容旧流程，固定范围下不再自动推断
    rt_range_percentiles: tuple = (1.0, 99.5)
    mz_range_percentiles: tuple = (1.0, 99.0)

    # ── 模型 ─────────────────────────────────────────────
    in_channels: int = 2                      # 绝对 + 相对通道
    feature_dim: int = 256
    encoder_channels: tuple = (32, 64, 128, 256)
    num_axial_heads: int = 4
    dropout: float = 0.3

    # ── 训练 ─────────────────────────────────────────────
    epochs_pretrain: int = 80
    epochs_finetune: int = 120
    batch_size: int = 8
    lr_pretrain: float = 1e-3
    lr_finetune: float = 3e-4
    weight_decay: float = 1e-4
    # 损失权重
    lambda_domain: float = 0.3
    lambda_proto: float = 1.0
    lambda_center: float = 0.5
    lambda_recon: float = 0.2
    # 一致性阈值百分位
    accept_percentile: float = 95.0

    # ── 数据增强 ──────────────────────────────────────────
    aug_intensity_scale: tuple = (0.8, 1.2)
    aug_noise_std: float = 0.05
    aug_mask_ratio: float = 0.15
    aug_rt_shift_max: int = 8                 # 像素
    aug_mz_shift_max: int = 2                 # 像素

    # ── 产品标签粒度 ─────────────────────────────────────
    # "fine":   H88A / H88B / H88C 各自独立
    # "coarse": H88A / H88B / H88C → H88
    product_granularity: str = "fine"

    # ── 排除规则 ──────────────────────────────────────────
    exclude_blanks: bool = True
    exclude_special: bool = True              # 排除空白/清洗剂/环境样等非产品样本

    # ── 数据准备可视化 ────────────────────────────────────
    save_prepare_plots: bool = True
    # None 表示全部保存；若设为整数则仅保存前 N 个样本图
    prepare_plot_max_samples: Optional[int] = None
    prepare_plot_dpi: int = 120
    # 数据与图片文件名是否包含批次/产品标签
    tag_output_with_batch_and_product: bool = True

    # ── 数据准备表格导出 ─────────────────────────────────
    save_prepare_tables: bool = True

    seed: int = 42
