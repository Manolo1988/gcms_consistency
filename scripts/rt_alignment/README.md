# RT 对齐偏移表（随代码分发）

批次级保留时间偏移，供 `scripts/prepare_preprocessing_variants.py --rt-offset-table` 读取。

放在 `scripts/` 下是因为 `new_prepared_data_preproc_v1/` 被 `.gitignore` 整体忽略，
而这张表是生成张量必需的标定参数，必须跟代码一起走。

- 生成脚本：`new_result/checks/rt_drift_20261023_1030/build_batch_offset_table.py`
- 生成日期：2026-10-23
- 参考批次：`20240304`（该批次 4 个产品的全部样本平均 TIC）
- 单位：秒。正值表示该批次比参考批次**更早**出峰，预处理时整段 RT 轴向后平移该值
- 覆盖：31/31 批次，偏移 0.00–15.60 秒
- 一致性：同批次内不同产品各自估计的偏移，极差中位 0.6 秒、最大 1.2 秒（网格精度即 0.6 秒）
- 可信度偏低（对齐相关系数 < 0.80）：20250417、20250619、20250808

脚本只用到 `batch` 和 `offset_seconds` 两列，其余列是溯源信息。
完整诊断（TIC 叠加图、逐样本明细、产品间一致性）见 `new_result/checks/rt_drift_20261023_1030/`。

重新生成表后，请同步覆盖本文件，保持与诊断产物一致。
