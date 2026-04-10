# GC-MS 跨批次一致性检测系统 — 代码总览

## 1. 项目定位

本项目针对烟草制品 GC-MS (气相色谱–质谱) 检测场景，构建**跨批次产品一致性**深度学习系统。核心目标是：给定一个待检 GC-MS 样本，**自动识别属于哪个产品**，并给出**量化的一致性评分**，同时能**拒识未知产品**。

与传统 softmax 分类不同，本系统基于**度量学习 + 原型匹配**框架，天然支持开集识别和少样本注册。

---

## 2. 数据流与处理管线

```
安捷伦 .D 文件夹   ──→   RT × m/z 二维张量 (.npz)   ──→   PyTorch DataLoader
(data_reader.py)       (data_prepare.py)                (dataset.py)
```

### 2.1 data_reader.py — 原始数据读取

| 功能 | 说明 |
|------|------|
| 读取后端 | 优先 `rainbow-api` → 备选 `pyteomics`(mzML) → TIC CSV fallback |
| 轴校正 | 自动判断二维矩阵是 (RT, m/z) 还是 (m/z, RT)，根据数值范围打分后标准化为 (RT, m/z) |
| 栅格化 | 将不规则扫描数据插值到固定 `rt_bins × mz_bins` 网格 (默认 1024 × 256) |
| 输出 | 双通道张量: 通道 0 = log 绝对强度，通道 1 = 行归一化相对强度 |

### 2.2 data_prepare.py — 数据集扫描与预处理

| 功能 | 说明 |
|------|------|
| `scan_dataset()` | 递归扫描 `dataset/` 目录，解析 .D 文件夹名 → 产品代码、批次、产线、批号等 |
| `parse_d_name()` | 从文件夹名提取元数据，支持标准样品和特殊样品 (空白、管道清洗、内标等) |
| `convert_all()` | 逐样本调用 `d_folder_to_tensor()` 生成 `.npz`，输出 `metadata.csv` + `grid_info.json` |
| 可视化 | 可选保存每个样本的二维热图 (png) 和长表 (csv) |

### 2.3 dataset.py — PyTorch Dataset + 切分策略

**数据增强** (`GCMSAugmentation`):
- 随机强度缩放 (±20%)、高斯噪声、随机矩形 mask、RT/m/z 平移

**三种切分模式**:

| 模式 | 函数 | 说明 |
|------|------|------|
| `closed` | `leave_one_batch_out_splits()` | 每次留出一个批次做测试，其余做训练 |
| `open` | `open_set_splits()` | 留出若干产品类不参与训练，评估开集识别 |
| `fewshot` | `few_shot_splits()` | 训练完成后，在测试批次模拟 N-shot (1,3,5,10) 注册 |

---

## 3. 模型架构 (models.py)

```
输入 (B, 2, 1024, 256)
        │
   ┌────▼────┐
   │  Stem   │ Conv7×7, s=2 → BN → ReLU → MaxPool s=2
   └────┬────┘
        │   (B, 32, 256, 64)
   ┌────▼────────────────┐
   │ Stage 1             │ ResBlock×2 (32→64, s=2)
   │ + DualAxisAttention │ RT轴注意力 + m/z轴注意力
   └────┬────────────────┘
        │   (B, 64, 128, 32)
   ┌────▼────────────────┐
   │ Stage 2             │ ResBlock×2 (64→128, s=2)
   │ + DualAxisAttention │
   └────┬────────────────┘
        │   (B, 128, 64, 16)
   ┌────▼────────────────┐
   │ Stage 3             │ ResBlock×2 (128→256, s=2)
   │ + DualAxisAttention │
   └────┬────────────────┘
        │   (B, 256, 32, 8)
   ┌────▼────┐
   │ GAP     │ 全局平均池化 → (B, 256) → Dropout
   └────┬────┘
        │   z_raw (B, 256)
        ├──→ L2 归一化 ──→ z (嵌入)
        ├──→ ProjectionHead ──→ proj (B, 128)  [SupCon 用]
        ├──→ ProductHead ──→ logits (B, K)      [辅助分类]
        ├──→ DomainHead (GRL) ──→ d_logits      [批次对抗]
        └──→ ReconDecoder ──→ recon (B, 2, H, W) [重建正则]
```

### 核心模块说明

| 模块 | 文件位置 | 说明 |
|------|----------|------|
| `ResBlock2D` | models.py | 带 SE 注意力的残差块，3×3 卷积 + shortcut |
| `DualAxisAttention` | models.py | RT 方向 + m/z 方向的多头自注意力，每阶段后独立施加 |
| `GradientReversal` | models.py | 梯度反转层 (DANN)，前传不变，反传取反，消除嵌入中的批次信息 |
| `ProjectionHead` | models.py | 两层 MLP，输出 L2 归一化向量，仅 SupCon 训练用 |
| `ReconDecoder` | models.py | 5 层反卷积 + 插值，从 feature map 重建输入，正则用 |

**参数量**: PlainEncoder (无注意力) 约 285 万，GCMSEncoder (含双轴注意力) 约 354 万。

---

## 4. 损失函数 (losses.py)

训练阶段总损失:

$$L_{\text{total}} = \lambda_{\text{supcon}} L_{\text{supcon}} + \lambda_{\text{adv}} L_{\text{adv}} + \lambda_{\text{proto}} L_{\text{proto}} + \lambda_{\text{recon}} L_{\text{recon}} + \lambda_{\text{cls}} L_{\text{cls}}$$

| 损失 | 类名 | 作用 | 默认权重 |
|------|------|------|----------|
| $L_{\text{supcon}}$ | `SupConLoss` | 同产品投影拉近、异产品推远 (Khosla 2020) | 1.0 |
| $L_{\text{adv}}$ | CE on `domain_logits` | 梯度反转，消除嵌入中批次信息 | 0.3 |
| $L_{\text{proto}}$ | `BatchPrototypeLoss` | 拉近样本到批内同类原型，推远异类原型 | 1.0 |
| $L_{\text{recon}}$ | MSE on `recon` | 防止表征退化的正则项 | 0.2 |
| $L_{\text{cls}}$ | CE on `logits` | 辅助分类，提供可靠梯度 | 0.5 |

预训练阶段 (Phase 1) 仅使用 $L_{\text{recon}}$。

---

## 5. 训练引擎 (train.py)

三阶段训练:

| 阶段 | 内容 | Epochs |
|------|------|--------|
| Phase 1 | 重建预训练: 仅 $L_{\text{recon}}$，预热编码器 | 80 |
| Phase 2 | 度量学习: 全部损失联合优化，编码器 LR×0.1，余弦退火 | 120 |
| Phase 3 | 原型注册: 冻结模型，从训练集计算原型和半径 | — |

**DANN alpha 调度**: $\alpha = \frac{2}{1+e^{-10p}} - 1$，$p = \text{epoch}/\text{total\_epochs}$，从 0 渐增至 1。

**验证策略**: 保存最优验证准确率对应的模型权重。

---

## 6. 原型注册与推理 (register.py)

### 注册

$$\mathbf{p}_k = \frac{1}{N}\sum_{i=1}^{N}\mathbf{z}_i, \quad r_k = \text{Percentile}(\{\|\mathbf{z}_i - \mathbf{p}_k\|\}, 95)$$

### 推理

| 任务 | 公式 |
|------|------|
| 产品识别 | $k^* = \arg\min_k \|\mathbf{z}_q - \mathbf{p}_k\|$ |
| 一致性评分 | $S = \exp\left(-\frac{\|\mathbf{z}_q - \mathbf{p}_{k^*}\|}{r_{k^*}}\right) \in [0, 1]$ |
| 拒识判定 | $\min_k \|\mathbf{z}_q - \mathbf{p}_k\| > \theta \Rightarrow$ 未知产品 |

`PrototypeStore` 支持 `save()`/`load()` 序列化，可独立部署。

---

## 7. 评估体系 (evaluate.py)

六维评估:

| 维度 | 指标 | 说明 |
|------|------|------|
| 1. 闭集分类 | Accuracy, Macro-F1, Balanced Accuracy | 标准分类指标 |
| 2. 一致性评分 | AUROC, AUPRC, EER, FAR/FRR | 分数能否区分正确/错误预测 |
| 3. 批次鲁棒性 | Silhouette(产品), Silhouette(批次), 批次可预测性 | Sil(产品)↑, Sil(批次)↓, 可预测性↓ |
| 4. 开集识别 | Open-Set AUROC, F1@FPR5% | 已知 vs 未知分离度 |
| 5. 少样本 | N-shot Accuracy (N=1,3,5,10) | 模拟新产品快速接入 |
| 6. 可解释性 | Grad-CAM 重叠率 | 关键区域可解释 |

可视化: t-SNE 嵌入图 (按产品/按批次着色)、一致性分数分布图。

---

## 8. 可解释性 (interpret.py)

`GradCAM` 支持两种模式:

| 模式 | 目标函数 | 适用场景 |
|------|----------|----------|
| `"logits"` | $\frac{\partial y_c}{\partial A}$ (分类 logits) | 常规分类解释 |
| `"embedding"` | $\frac{\partial (-\|\mathbf{z}-\mathbf{p}\|)}{\partial A}$ (距原型距离) | 度量学习解释，更贴合实际推理逻辑 |

输出: 原始输入图 + CAM 叠加图 + RT 贡献投影 + m/z 贡献投影。

---

## 9. 对比实验 (baselines.py + compare.py)

### 9.1 对比方法 (baselines.py)

**传统方法** (4 种):

| 方法 | 流程 |
|------|------|
| PCA + Mahalanobis | 降维 → 各类协方差逆 → Mahalanobis 距离分类 |
| PLS-DA | PCA 预处理 → PLS 回归 one-hot Y → argmax |
| SVM-RBF | PCA → RBF 核 SVM (Platt 概率校准) |
| Random Forest | PCA → 200 棵树集成 |

**DL 基线** (3 种):

| 方法 | 编码器 | 损失 | 推理 |
|------|--------|------|------|
| ResNet-CE | PlainEncoder (无双轴注意力) | CrossEntropy | Softmax |
| ResNet-Triplet | PlainEncoder | CE + 在线难样本三元组 | 原型匹配 |
| ResNet-Center | PlainEncoder | CE + CenterLoss | 原型匹配 |

**消融变体** (3 种):

| 变体 | 修改内容 |
|------|----------|
| Ours-noDualAxis | 完整损失管道 + PlainEncoder (移除双轴注意力) |
| Ours-noBatchAdv | 完整模型 + $\lambda_{\text{adv}}=0$ (移除批次对抗) |
| Ours-Softmax | 完整训练 + Softmax 推理 (替代原型匹配) |

### 9.2 对比运行器 (compare.py)

统一评估框架:
- 所有方法计算相同的 7 个指标 (Accuracy, Macro-F1, Con.AUROC, Sil(产品), Sil(批次), 批次可预测性)
- Leave-one-batch-out 多 fold 聚合 (mean ± std)
- 输出: 文本表格 + LaTeX 表格 + 柱状图 + 雷达图 + 批次鲁棒性图

---

## 10. CLI 命令 (main.py)

```bash
# 数据准备
python main.py prepare [--save_plot] [--save_table]

# 训练 (三种模式)
python main.py train --split_mode closed     # 闭集跨批次
python main.py train --split_mode open       # 开集
python main.py train --split_mode fewshot    # 少样本

# 原型注册
python main.py register --fold 0

# 评估
python main.py evaluate

# Grad-CAM 解释
python main.py interpret --fold 0 --sample_idx 0

# 对比实验
python main.py compare                                    # 全部 11 种方法
python main.py compare --methods "SVM-RBF,ResNet-CE,Ours(Full)"  # 指定方法
```

---

## 11. 配置中心 (config.py)

所有超参数集中管理，关键分组:

| 分组 | 示例参数 |
|------|----------|
| 网格 | `rt_bins=1024`, `mz_bins=256`, `rt_range`, `mz_range` |
| 模型 | `feature_dim=256`, `proj_dim=128`, `blocks_per_stage=2`, `num_axial_heads=4` |
| 训练 | `epochs_pretrain=80`, `epochs_finetune=120`, `batch_size=8`, `lr_finetune=3e-4` |
| 损失 | `lambda_supcon=1.0`, `lambda_adv=0.3`, `supcon_temperature=0.07`, `proto_margin=1.0` |
| 原型 | `accept_percentile=95`, `reject_threshold_factor=2.0` |
| 增强 | `aug_intensity_scale`, `aug_noise_std`, `aug_mask_ratio`, `aug_rt_shift_max` |

---

## 12. 文件依赖关系

```
config.py           ← 所有文件导入
data_reader.py      ← data_prepare.py
data_prepare.py     ← main.py (prepare)
dataset.py          ← train.py, evaluate.py, compare.py, main.py
models.py           ← train.py, evaluate.py, baselines.py, main.py
losses.py           ← train.py, compare.py
register.py         ← train.py, evaluate.py, compare.py, main.py
train.py            ← main.py, compare.py
evaluate.py         ← main.py, compare.py
interpret.py        ← main.py
baselines.py        ← compare.py
compare.py          ← main.py (compare)
main.py             → CLI 入口
```

---

## 13. 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据 (将 .D 文件夹转为张量)
python main.py prepare --rt_min 3.17 --rt_max 36.91

# 3. 训练 (闭集跨批次)
python main.py train --split_mode closed

# 4. 运行对比实验
python main.py compare

# 5. 查看结果
# 输出目录: outputs/comparison/
#   - comparison_table.txt (文本表格)
#   - comparison_table.tex (LaTeX)
#   - compare_accuracy.png / radar_chart.png / ...
```
