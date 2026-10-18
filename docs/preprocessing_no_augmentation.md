# 三套预处理输入：不增强对照

这轮只关闭输入数据增强，沿用上一轮分类 CNN：通道 32/64/128/256、每阶段 2 个残差块、dropout 0.3、单一交叉熵损失。保留类别均衡抽样，固定 seed 42、100 轮、batch size 32、学习率 0.0003、AdamW、余弦学习率。Dropout 属于模型正则，不是输入数据增强，这轮仍开启。

三组共用冻结划分，不重新预处理、不拟合 PCA。每 10 轮验证一次，按验证准确率选择模型，与上一轮一致。测试集不参与训练或选模型；由于这份测试集结果已经被查看，后续反复据此调整方案得到的结果应视为探索性比较。

## 服务器运行

先将本次修改同步到服务器。在项目根目录执行：

```bash
.venv/bin/python scripts/run_preprocessing_comparison.py --check-only
mkdir -p result
task_tag="preproc_compare_noaug_$(date +%Y%m%d_%H%M%S)"
nohup .venv/bin/python -u scripts/run_preprocessing_comparison.py \
  --output-dir "result/${task_tag}" \
  --seed 42 --epochs 100 --batch-size 32 --lr 0.0003 \
  > "result/${task_tag}.log" 2>&1 &
task_pid=$!
echo "$task_pid" > "result/${task_tag}.pid"
echo "PID=$task_pid"
echo "结果：result/${task_tag}"
```

三组顺序运行。输入检查或任意一组失败时停止，不覆盖旧结果。

```bash
tail -f "result/${task_tag}.log"
ps -p "$task_pid" -f
```

`Ctrl+C` 只退出日志查看。停止训练：

```bash
kill "$task_pid"
```

重新登录后，先找到对应日志名；PID 保存在同名前缀的 `.pid` 文件中。没有断点续训功能，中途停止后请使用新输出目录重新开始。

## 文件与指标

每个输入方案子目录包含：

| 文件 | 内容 |
|---|---|
| `training_history.csv/json` | 每轮训练 loss、训练准确率、学习率；每 10 轮验证 loss/准确率 |
| `training_curves.png` | loss、准确率曲线 |
| `training_summary.json` | 最佳轮次、最佳验证准确率、最新完成轮次 |
| `model.pt` | 验证集选出的最佳权重，训练中每次提升就保存 |
| `train_metrics.json`、`val_metrics.json`、`test_metrics.json` | 最佳模型在三个集合上的完整分类指标 |
| `test_per_class_metrics.csv` | 各产品 Precision、Recall、F1、样本数 |
| `test_predictions.csv`、`test_errors.csv` | 样本 ID、原始路径、批次、质量标记、真实/预测产品、各类概率 |
| `test_confusion_matrix.csv/png` | 计数混淆矩阵和按真实产品归一化的图 |
| `test_confusion_matrix_normalized.csv` | 按行归一化的混淆矩阵 |
| `test_per_batch_metrics.csv` | 各留出批次的样本数与准确率 |
| `fewshot_summary.csv` | 1/3/5 条参考样本下的新品识别均值、抽样波动 |
| `fewshot_episodes.json` | 少样本参考/查询样本清单，可核对三组是否相同 |
| `effective_config.json`、`split.json` | 实际模型/增强设置、输入契约、环境版本、完整冻结划分 |
| `paper_gate_result.json` | 总结果，兼容原有字段 |
| `run_status.json` | `complete` 表示该组训练和评估均完成 |

训练集与验证集也有各自前缀的预测、逐类指标和混淆矩阵。总体指标包括 Accuracy、Balanced accuracy、Macro Precision/Recall/F1、Weighted F1、交叉熵 log loss（越低越好）。当所有产品都有样本时，Macro Recall 与 Balanced accuracy 数值相同。

三组运行完成后，父目录的 `comparison_metrics.csv/json/png` 汇总各组测试指标、验证指标、训练时间、参数量和少样本结果；`comparison_status.json` 的 `complete` 表示三组都完成。

曲线的训练指标来自均衡抽样、dropout 开启时的训练批次；最终 `train_metrics.json` 则在完整训练集上关闭 dropout 后计算，适合与验证/测试指标比较。不能把这两种训练分数直接当作相同口径。

这是单个训练种子的对照。100 次少样本抽样会重复使用相同样本，不等于训练了 100 个模型，也不能作为训练稳定性的证据。少样本识别是在提供新品参考样本后区分 HMD/XCJ，不是未知产品拒识；本轮不生成开放集 AUROC 或缺少真实一致性标签的一致性评分。

旧实验只有汇总分数，无法还原逐样本结果和训练曲线；如果服务器仍有 `model.pt`，可另做一次评估补回预测明细，历史训练曲线仍无法恢复。

## 本地轻量检查

```bash
python -m unittest discover -s tests -p test_baseline_reporting.py -v
python scripts/run_preprocessing_comparison.py --check-only
```

本地检查不依赖 PyTorch、不进行训练，只检验指标计算、样本标签对齐、图表导出及划分契约。
