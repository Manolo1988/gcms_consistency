# 中间模型：每阶段 1 个残差块，保留双轴注意力

这次补一组中间模型，区分“层数减少”和“移除双轴注意力”分别产生了什么影响。

| 模型 | 每阶段残差块数 | 三个阶段后的双轴注意力 |
| --- | ---: | --- |
| 上轮 A：原模型，两项损失 | 2 | 保留 |
| 上轮 B：轻量 CNN，两项损失 | 1 | 移除 |
| 本次中间模型 | 1 | 保留 |

本次与 A 比较层数，与 B 比较双轴注意力。单个种子的结果先用于判断方向，不能单独证明哪个结构稳定更好。

## 结构具体怎么变

三个阶段都从 `残差块 → 残差块 → 双轴注意力` 变成 `残差块 → 双轴注意力`。残差块负责提取局部峰形；双轴注意力分别沿保留时间 RT 和质荷比 m/z 联系不同位置的信息。

代码已有这个能力，因此没有改 `models.py`，只设置 `main_backbone="gcms"`、`blocks_per_stage=1`。

上一轮 B 使用已有的 `PlainEncoder`：三个阶段直接依次连接，前向计算中没有 `attn1/attn2/attn3`。它与本次使用相同的残差块类型，块内仍有 SE 通道门控，即给不同特征通道分配权重。因此准确说法是“移除双轴注意力”，不是移除全部注意力机制。

## 数据和训练参数

直接读取 `new_prepared_data_preproc_v1/representations/dual_channel_no_pca`，不重新预处理、不拟合 PCA。

| 项目 | 本次设置，沿用上轮 |
| --- | --- |
| 输入 | 绝对、相对双通道；`2 × 2048 × 503`；RT 0–40，m/z 30–532 |
| 固定划分 | 训练 1999、验证 143、已知产品测试 468、未知产品测试 133 |
| 已知产品 | H88、H99、HRJ、HRY、HYZ、RJD、YJD |
| 未知产品 | HMD 68 个，XCJ 65 个 |
| 网络宽度 | 32/64/128/256；表征 256 维，对比学习投影 128 维 |
| 训练 | seed 42；100 轮；batch size 32；dropout 0.3 |
| 优化 | AdamW；学习率 0.0003；权重衰减 0.0001；余弦下降；梯度范数上限 5 |
| 损失 | `1.0 × SupCon + 0.40 × 加权分类损失`；温度 0.07；gamma 0、beta 0.999 |
| 抽样与增强 | 沿用类别均衡有放回抽样及原增强；完整值见保存的 `run_config.json` |
| 验证与选模 | 每 10 轮验证；先比 Macro-F1，相同再比各类召回率平均值；完全相同保留更早轮次 |
| 其他 | 不启用混合精度、TF32、SWA、提前停止；保留确定性设置；4 个数据加载进程 |

其他损失权重为 0，相关辅助头不执行前向计算，批次预测头和重建头不更新。训练启动前核对冻结样本清单、划分和全部张量头。

## 评估规则与能看到的结果

统一使用 `main.py train` 和 `main.py evaluate`。入口统一，当前实验选择沿用上轮的评估协议；主入口原来的默认协议仍可用于其他实验。

- **A，已知产品识别**：在 468 个留出批次样本上计算识别率、Macro-F1、逐产品结果和混淆矩阵，并保留一致性评分、批次鲁棒性、困难批次诊断及已有图表。
- **B，未知产品拒识**：只比较 468 个已知测试样本和 133 个未知样本。分数是 `-d/r`，d 为到最近产品原型的欧氏距离，r 为该产品训练距离的 95 分位数。分数越大越像已知产品；阈值只由 143 个验证样本的分数 5 分位数确定，不做校准或自动混分。除 AUROC、FPR@95TPR 外，保存已知接受率、未知误接收率及逐样本判定。
- **C，少样本识别**：沿用 100 组固定跨批次任务，种子 42000–42099；同组 1/3/5-shot 的参考样本逐步包含，参考与测试来自不同批次。对参考表征取原始均值作为新产品原型，按欧氏距离匹配，不微调。保存各组及各产品结果，并补充 Macro-F1 展示。

A/B 的训练原型也是单位表征的原始均值，不再次归一化，半径下限为 1e-6。A 新增的一致性展示使用 `exp(-d/r)`；B 仍使用 `-d/r`，两个分数的数值不能混用。B 的 CSV 分别记录原半径规则和验证阈值规则，正式接受率使用后者。

比较范围：上轮专用脚本用 NumPy float64 直接计算距离，统一原型流程用 float32 张量计算，可能有微小数值差异；不能保证逐位复现。A 的一致性、批次诊断及 C 的 F1 是补全展示，不能凭空给旧实验补上同项结论。

本地已核对 C 的 100 组参考/测试样本分配与上轮下载清单完全一致。含路径的 metadata 指纹会因本地与服务器路径不同而变化；样本分配没有变化。

## 新增和修改的文件

| 文件 | 作用 |
| --- | --- |
| `light_cnn_protocol.py` | 新增中间模型配置和 B 的冻结评估适配；原 A/B 分组及专用脚本保持原行为 |
| `main.py` | 增加 `--experiment_preset light_cnn_v1_attention_1block`；评估先恢复训练配置，再应用显式参数 |
| `config.py` | 保存实验/评估协议选项；恢复模型配置；避免评估时误用默认结构 |
| `train.py` | 接入 Macro-F1 选模及并列规则；显示验证指标；保存训练/验证记录、完整模型配置和最佳轮次 |
| `evaluate.py` | 将冻结 B/C 接入完整 A/B/C 流程；导出逐样本拒识、固定少样本清单及逐组结果 |
| `paper_protocol.py` | 在已有少样本识别结果上补算 F1，不改抽样和预测方式 |
| `tests/test_main_macro_f1_selection.py` | 无需 torch 的配置恢复、选模、阈值及少样本导出检查 |
| 本说明 | 解释结构、参数、评估、文件变更和运行方法 |

本地语法检查、13 项轻量测试和 `git diff --check` 已通过；未在本地训练或完整评估，服务器运行结果尚未验证。代码尚未自动提交或推送。

## 服务器正式实验

先将本次代码同步到服务器。在**项目根目录**运行下面命令。它会先训练，成功后自动完整评估；训练失败则停止。不要把它称作检查任务，它会实际训练和评估模型。

输出任务为 `new_result/experiments/attention_1block_<时间>/`：`output/` 放模型、指标和图表，`logs/` 放后台启动日志和进程号。输入不依赖本地整理后的旧结果目录。

```bash
prepared_dir="new_prepared_data_preproc_v1/representations/dual_channel_no_pca"
task_dir="new_result/experiments/attention_1block_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$task_dir/logs"
nohup setsid env PYTHONHASHSEED=42 PYTHONUNBUFFERED=1 bash -c '
  set -e
  echo "$$" > "$2/logs/runner.pid"
  .venv/bin/python main.py train \
    --experiment_preset light_cnn_v1_attention_1block \
    --prepared_dir "$1" --output_dir "$2/output"
  exec .venv/bin/python main.py evaluate \
    --prepared_dir "$1" --output_dir "$2/output" --save_visualizations
' _ "$prepared_dir" "$task_dir" > "$task_dir/logs/runner.log" 2>&1 &
echo "启动进程：$!"
echo "任务目录：$task_dir"
```

查看日志和进程；停止整组任务，包含训练的子进程：

```bash
tail -f "$task_dir/logs/runner.log"
ps -p "$(cat "$task_dir/logs/runner.pid")" -f
# 需要停止时执行：
kill -- "-$(cat "$task_dir/logs/runner.pid")"
```

若换了终端，先把 `task_dir` 设回启动时打印的具体目录。

主要产物：

- `output/final_model/model.pt`、`train_meta.json` 和 `prototypes/`：最佳模型、还原配置与产品原型。
- `output/run_config.json`、`training_history.json`、`validation_history.json`：参数和训练过程。
- `output/evaluation_summary.json`：A/B/C 及一致性、批次鲁棒性等汇总指标。
- `output/setting_a_predictions.csv`、`setting_a_errors.csv`、`setting_a_confusion_matrix.csv`：已知产品逐样本结果、错误与混淆矩阵。
- `output/setting_b_predictions.csv`：未知拒识分数及接受判定。
- `output/fewshot_episodes.json`、`setting_c_episodes.csv`：固定少样本任务及逐组结果。
- `output/visualizations/`：原流程图表；`output/train.log`、`eval.log`：原流程内部日志；`logs/runner.log`：后台总日志。
