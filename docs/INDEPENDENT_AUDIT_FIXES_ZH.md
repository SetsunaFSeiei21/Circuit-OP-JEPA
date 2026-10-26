# 独立审查修复记录（0.4.0）

日期：2026-10-26。修复基于远端 `main` 提交 `6b8595b5491db6f2eb128d1ee8bb739008f24286`，对应被审代码 `85ca47392ce5a21ff765008fefe634da1c8378ee` 及随后补充的交接文档。逐项审查依据为配套 Word 中的研究协议、仓库实现和用户提供的 Opamp2 数据。

本轮修复不改变研究约定：JEPA 两分支共享同一个纯结构前缀、共享一次前向的输出并共同回传梯度；无 EMA、stop-gradient 或独立 predictor。第二阶段冻结完整输入 encoder，JEPA 推理不读取 OP。两个直接 OP 基线仍明确要求训练和推理均提供 OP。原始数据、私有逐样本预测和 checkpoint 不随此次提交发布。

## 修复清单

| 编号 | 审查问题 | 修复及回归证据 |
| --- | --- | --- |
| F01 | 第二阶段用当前配置覆盖真实预训练经历，比较可能把不同预训练协议当成同一方法 | 保留不可被阶段二配置替代的 `pretrain_provenance`，分别记录两阶段 seed；协议签名包含真实预训练预算、优化器、loss、架构、OP 配置与变换。比较核验来源内容及协议一致性；缺失旧来源不能标记严格完整。新增权重和来源一致性检查。 |
| F02 | 训练中无观测标签的指标仍影响验证选模 | 五模式统一以训练统计 `count>0` 的指标建立选模 mask，记录 `selection_metric_names`；验证集中完全没有可支持指标则明确失败。改变仅验证可见的未训练指标不会改变 history 或选中权重。 |
| F03 | 器件输入只知道“连到外部端口”，丢失 VINP/VOUT 等具体身份 | contract 3 引入训练集命名端口词表和端子×端口 multi-hot 特征；同一 net 上的多个别名均保留。五模式的端口交换回归均能观测输入/输出差异。 |
| F04 | model_name 未进入模型输入和设计指纹 | 加入大小写规范化的物理模型身份；训练集词表区分 PAD、UNK、NONE。不同模型不再被错误合并，未见模型不会在 held-out 数据中扩充词表。 |
| F05 | +0.0/-0.0 被指纹当成不同设计，可能跨 split 泄漏 | 仅在设计指纹中统一数字零；原始参数与 OP 的符号保留。manifest 强制声明新指纹版本，拒绝沿用旧分组。 |
| F06 | SQLite 两来源的状态/原因发生冲突时被静默覆盖 | 规范化后比较重复字段，保留 NULL 与显式空列表的语义差异；矛盾原因、状态或条件明确报错。真实原始数据仍逐字段一致。 |
| F07 | 大偏置小方差目标导致 R² 计算错误 | 采用二进制缩放和相对锚点的中心化计算；`[1e16, 1e16+2]` 的反例恢复正确 R²=-1，并覆盖极小、极大数值及退化情形。 |
| F08 | 真实电路名 `__all__` 与总体汇总哨兵冲突 | 所有指标和配对聚合显式区分 `scope=pooled` 与 `scope=circuit`；单种子不会被误计成两个种子或生成虚假 SD。 |
| F09 | 直接调用阶段入口会覆盖 checkpoint，并遗留无关报告 | 所有训练入口执行阶段目录检查，拒绝重复阶段产物、陈旧文件和无关 manifest；允许验证一致的 stage-one→stage-two 同目录衔接及仅补充性能标签。没有伪装的 resume。 |
| F10 | 每表一百万行硬限制阻止预期数据规模导入，并同时驻留完整 raw 与 canonical | 同一只读事务中预检查行数，再按样本流式 join。保留默认安全阈值，增加显式可配置上限和 `none`；1,020,000 行合成 reader 检查通过。最终 canonical/state/JSON 仍为全内存对象。 |
| F11 | test/OOD 报告重复累计同一训练成本，丢失 optimizer steps | 以真实 checkpoint run identity 去重训练成本，保留两阶段 steps；OP 覆盖按 split 单列，实测采集成本按明确 scope 单列。同一 run 的成本矛盾会报错。 |
| F12 | `.end` 被当作可跳过指令，随后 DUT 仍被解析 | `.end` 要求已完成 DUT，之后出现有效输入行即拒绝；合法最终 `.end` 和后续注释仍支持。 |

新增回归集中在 [adapter](../tests/test_adapter_audit_fixes.py)、[data identity](../tests/test_data_identity.py)、[input identity](../tests/test_input_identity.py)、[training](../tests/test_training_audit_fixes.py) 和 [comparison](../tests/test_comparison_audit_fixes.py)。原有测试保留，并将旧输入 fixture、输出目录和成本表预期调整到明确的新契约。

## 兼容性及运行方式

- 软件版本为 0.4.0；canonical 数据 schema 仍为 1，checkpoint/preprocessing 容器版本仍为 2。
- 新输入契约为 3。旧输入 contract 1/2 的架构仍可重建，但不能与 contract 3 混作公平比较。
- 新 manifest 要求 `design_fingerprint_version=2`。旧 manifest 不会被自动重写，必须在新输出目录重建实验；旧 checkpoint 的架构可读取不等于旧分组可以继续训练或评估。
- 比较输出 schema 为 3。`costs.csv` 每个可信 run 一行；`evaluation_costs.csv` 保存分 split 信息；存在实测采集成本时另有 `op_acquisition_costs.csv`。旧报告来源不全时 `complete=false`，即使模式和 seed 都齐全。
- 大型导入示例：`python -m device_jepa import_analog_agent --input /path/to/circuit --output runs/data.json --max-table-rows 1200000`。取消行数保护可显式使用 `--max-table-rows none`。这不保证任意规模的完整导入或训练能装入内存。

## 验证范围

完整命令与机器可读结果见 [verification/audit_fixes/results.json](../verification/audit_fixes/results.json)。在 Python 3.12.14、CPU PyTorch 2.14.1 环境完成全部 313 项单元回归（含显式启用的 1,020,000 行读取测试，零跳过）、7 项独立探针、7 个 shell 脚本语法及 wheel 构建/隔离安装检查。相比原有 241 项测试新增 72 项回归。

合成验证使用 5 个 toy 电路、120 个样本、seed 17/29，五模式每阶段 1 epoch。train/val/test/OOD 为 80/8/8/24，labeled_train=80；20 份评估报告聚合为 10 个训练成本记录，严格完整性通过。

真实验证使用原始 Opamp2 的 32 个样本、seed 17，五模式每阶段 1 epoch。train/val/test/OOD 为 26/3/3/0，labeled_train=26；5 份评估报告和 5 个训练成本记录，严格完整性通过。全部 21,120 个 OP 标量位置、288 个性能位置、23 个缺失性能值、5,725 个负 OP 值、93 个 saturation_below 和 32 个 feasible=false 样本均保留。ZIP 和 SQLite 的 SHA-256 与修复前一致。

这些检查证明回归行为和短流程贯通，不证明预测优势、跨电路泛化或全量 GPU 容量。真实数据仍仅一个电路，没有真实 OOD；未运行 SPICE、未重新采样、未提交集群作业。大表测试覆盖 reader 的限制与流式消费，未声称完整 60,000×17 器件×44 OP 字段的训练容量已经验证。OP 采集成本仍未知，不计作零。
