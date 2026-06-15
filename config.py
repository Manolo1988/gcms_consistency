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
    pretrained_feature_model: str = ""        # 预训练特征提取权重路径
    pretrained_feature_arch: str = "auto"    # auto/resnet18/resnet50/wide_resnet50_2
    main_backbone: str = "gcms"              # gcms/transformer/resnet18/resnet50/wide_resnet50_2
    main_backbone_model: str = ""            # 主模型骨干本地权重路径(可空，空则回退 pretrained_feature_model)
    main_feature_layers: str = "layer4"      # 主模型特征层, 逗号分隔: layer2,layer3,layer4
    main_feature_fuse: str = "concat"        # 主模型层融合: concat/last
    transformer_patch_size: int = 16          # transformer patch 大小
    transformer_embed_dim: int = 256          # transformer token 维度
    transformer_depth: int = 6                # transformer block 层数
    transformer_num_heads: int = 8            # transformer 多头注意力头数
    transformer_mlp_ratio: float = 4.0        # transformer FFN 扩展比例

    # ── 主算法选择 (默认回到 README 主模型) ─────────────────
    primary_model: str = "deep_consistency"   # deep_consistency / raw_pca_mlp
    input_raw_pca_enabled: bool = True         # 仅 deep_consistency: raw -> PCA -> model
    input_raw_pca_components: int = 128
    raw_pca_components: int = 128
    raw_pca_hidden: str = "128,64"
    raw_pca_max_iter: int = 300
    raw_pca_alpha: float = 1e-4
    raw_pca_lr_init: float = 1e-3
    raw_open_score_blend: float = 1.0
    raw_distance_percentile: float = 95.0
    raw_fewshot_c_3shot: float = 2.0

    # ── 训练 (单阶段) ──────────────────────────────────────
    epochs: int = 200
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    eval_interval: int = 10                # 兼容开关: 未启用分阶段验证时使用
    eval_interval_search: int = 10         # 搜索阶段验证间隔
    eval_interval_final: int = 5           # 收敛阶段验证间隔(更密)
    eval_final_start_ratio: float = 0.7    # 从总训练进度该比例起切到收敛阶段
    early_stop_patience: int = 0           # 早停耐心(验证次数), 0=关闭
    min_epochs_before_early_stop: int = 0  # 早停前最少训练 epoch, 0=按比例自动计算
    min_epoch_ratio_before_early_stop: float = 0.6  # 早停前最少训练比例
    early_stop_min_lr_ratio: float = 0.2   # 仅当 lr <= 初始lr*ratio 才允许早停
    early_stop_min_delta: float = 5e-4     # 判定“有提升”的最小 metric 增量
    proto_val_subset_ratio: float = 0.35   # 训练中期验证: 原型构建使用训练子集比例
    proto_val_subset_min_samples: int = 256
    proto_val_subset_max_samples: int = 1024
    proto_val_full_every: int = 3          # 每隔 N 次验证做一次全量原型验证
    warmup_guard_enabled: bool = False     # 前N轮与最佳方案对比淘汰
    warmup_guard_epoch: int = 10           # 对比轮次 (默认第10轮)
    warmup_guard_best_at_epoch: float = 0.0  # 最佳方案在该轮的val_acc参考
    warmup_guard_compare_best: bool = True  # True: 直接对比当前最优迭代
    warmup_guard_min_ratio: float = 1.0    # 兼容: compare_best=False 时生效

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
    min_batches_per_product: int = 3          # 产品最少覆盖批次数, 低于此排除
    holdout_product_min_samples: int = 50     # 留出产品的最少样本数
    holdout_product_min_batches: int = 8      # 留出产品的最少批次覆盖
    holdout_batch_min_samples: int = 60       # 留出批次的最少样本数(已知类)
    holdout_batch_min_classes: int = 5        # 留出批次最少覆盖的已知产品数
    preferred_holdout_products: Tuple[str, ...] = ()
    preferred_holdout_batches: Tuple[str, ...] = ()
    val_ratio: float = 0.1                    # train_batches 内伪验证批次比例
    split_seed: int = 42                      # 数据划分随机种子

    # ── Few-shot 重复抽样 ────────────────────────────────
    fewshot_repeats: int = 1                  # 每个 N-shot 重复抽样次数, 1=不重复
    fewshot_seed_start: int = 42              # 重复抽样起始种子

    # ── Open-set Score Calibration ──────────────────────
    open_score_calibration_enabled: bool = False
    open_score_calibration_mode: str = "logistic"  # pseudo_unknown / grid_search / logistic
    open_score_features: str = "base,margin,min_dist,radius_norm,second_dist"
    open_score_weights: str = ""              # 保存校准后权重 (JSON string)
    open_score_calibration_holdout_products: int = 1  # 留出伪未知类数量
    open_score_calibration_seed: int = 42

    # ── TIC Auxiliary Branch ────────────────────────────
    tic_branch_enabled: bool = False          # 启用 TIC 辅助分支
    tic_source: str = "from_tensor"           # from_tensor / raw_file
    tic_encoder: str = "cnn1d"               # cnn1d / mlp / transformer
    tic_embed_dim: int = 64                   # TIC 编码器输出维度
    tic_fusion_mode: str = "concat"           # concat / gated / sum
    tic_fusion_output_dim: int = 256          # 融合后输出维度

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
    aug_peak_broaden_prob: float = 0.1        # 峰展宽触发概率(降负担)
    aug_rt_warp_strength: float = 0.02        # RT 非线性扭曲幅度
    aug_rt_warp_prob: float = 0.2             # RT 扭曲触发概率(降负担)
    aug_mz_channel_drop: float = 0.05         # m/z 通道随机丢弃比例
    aug_tic_jitter: float = 0.1               # TIC 归一化抖动幅度

    # ── DataLoader 性能 ──────────────────────────────────
    dataloader_workers: int = 4               # 建议 4~8
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2

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

    # ── 数据准备: 直接原始矩阵PCA(不走栅格化bins) ─────────────
    prepare_direct_raw_pca: bool = True

    # ── 数据准备表格导出 ─────────────────────────────────
    save_prepare_tables: bool = True

    seed: int = 42
