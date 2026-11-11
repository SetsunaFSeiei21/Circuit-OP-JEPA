# 仅融合层的分阶段引脚 OP 门控

## 新增内容

新增三种模型，与已有三种分阶段门控使用相同训练顺序、训练目标和冻结规则，区别是最终门控注入范围。

| 新模型 | 对应原模型 | 前置训练 | 最后阶段更新 |
|---|---|---|---|
| `device_op_distill_gate_fusion_pin` | `device_op_distill_gate_pin` | OP 预测器、真实 OP 教师、学生 encoder 蒸馏、学生 decoder | 只更新门控 |
| `device_op_distill_gate_decoder_fusion_pin` | `device_op_distill_gate_decoder_pin` | 复用同一蒸馏学生基模型 | 更新门控与 decoder |
| `device_op_supervised_gate_fusion_pin` | `device_op_supervised_gate_pin` | OP 预测器、关闭门控监督训练 encoder 与 decoder | 只更新门控 |

前置训练阶段完全绕过门控。部署时三个新模型均使用冻结的器件级 OP 预测器，随后使用引脚级 encoder 和 decoder，不需要实测 OP 或教师模型。

## 门控插入位置

| 网络部分 | 原分阶段门控 | 新 `_fusion_pin` 门控 |
|---|---|---|
| 结构前缀的 Self-Attention | 最终门控阶段启用 | 不加入门控 |
| 融合块的 Self-Attention | 最终门控阶段启用 | 最终门控阶段启用 |
| 融合块的 Cross-Attention | 不加入门控 | 不加入门控 |
| 性能 decoder | 不加入门控偏置 | 不加入门控偏置 |
| 器件级 OP 预测器 | 不加入门控 | 不加入门控 |

实际作用层数为 `model.cross_layers`，并非代码固定后 3 层。默认 `encoder_layers=6`、`cross_layers=3` 时，前 3 层结构编码不受 OP 门控影响，后 3 层融合块的 Self-Attention 受门控影响。门控不直接改变 Q/K/V 的定义，仍在 Softmax 前加入原有的有限对数偏置。

OP 向量及训练支持掩码经 MLP 映射到 `model.dim`，与有序 query/key 引脚角色 embedding 组合，输出每个注意力头的分数。各头输出不同分数；同一套门控网络及生成的偏置在融合层间复用，不为每一层增加独立的 OP 预测器。

门控仍只软抑制同一个 MOS 内、分属不同 net 的引脚对。真实同 net 连接、自连接、外部端口和非 MOS 节点保持保护。门控偏置之外的原有拓扑掩码、终端偏置和局部／全局注意力交替保持原定义。输入 OP 特征仍参与融合块的特征读取；“结构前缀不接入门控”不等于移除融合块的 OP 输入。

## 配置与预算

新配置：`final_exam/configs/staged_op_gate_fusion_pin.json`。除默认模型列表外，与 `staged_op_gate_pin.json` 使用相同参数：

| 配置项 | 默认最大轮数 | 用途 |
|---|---:|---|
| `op_transfer.epochs.op_predictor` | 30 | 器件级 OP 预测器 |
| `op_transfer.epochs.teacher` | 60 | 蒸馏路径使用的真实 OP 教师 |
| `op_transfer.epochs.predicted_op_distill_encoder` | 30 | 学生 encoder 蒸馏 |
| `op_transfer.epochs.predicted_op_distill_decoder` | 30 | 冻结学生 encoder 后训练 decoder |
| `op_transfer.epochs.readout` | 60 | 无教师路径的监督基模型 |
| `op_transfer.epochs.gate` | 30 | 最后的门控或门控＋decoder 阶段 |

这是最大预算；各阶段沿用验证集早停，`patience=10`，完成阶段恢复验证最佳权重。门控阶段是追加训练，不将完整流水线声称为总计 60 轮。默认 batch size 为 32、学习率为 `1e-4`。门控强度、下限与初始门控仍由 `op_gate_pin.strength`、`epsilon`、`initial_gate` 控制。

在同一个任务中同时选择两组模型，且对应模型有效配置一致时，器件级 OP 预测器、教师与关闭门控的基模型可以共享，成本按阶段 ID 去重。不同作业没有跨目录自动共享，不会默默拿旧实验的权重作为新实验初始化。

## 启动方式

在脚本顶部编辑电路、模型和 seeds：

```bash
CIRCUITS=(two_stage_folded_opamp)
MODELS=(
  device_op_distill_gate_fusion_pin
  device_op_distill_gate_decoder_fusion_pin
  device_op_supervised_gate_fusion_pin
)
SEEDS=(17 29 43)
```

执行：

```bash
cd /share/home/202411094934/python_project/Circuit-OP-JEPA
bash final_exam/single_circuit_pin/train_staged_op_gate_fusion_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_staged_op_gate_fusion_pin.sh
```

`--dry-run` 仅预览，不创建结果目录、不提交作业。`--smoke` 可以与入口一起使用，缩短训练轮数；样本数由配置的 `dataset.max_samples_per_circuit` 控制，默认 `null` 为全量，见 [数据抽样说明](../docs/DATA_SUBSAMPLING_ZH.md)。默认每个电路提交一个 Slurm 作业，8 CPU、1 GPU、80G 主机内存，账户 `b_phzhwu`、分区 `ex01A800`、conda 环境 `newbase`。数据仍读取 `runs/raw/full_data/<电路名>/`。

原 `train_staged_op_gate_pin.sh` 和原配置的默认模型保持不变。需要六模型同作业对照时，在新脚本的 `MODELS` 中另外加入原三个名称。

## 结果和日志

新实验 run ID 形如 `<电路>_staged_op_gate_fusion_<UTC时间>_<进程号>_<随机数>_pin`，目录仍在当前项目内：

| 内容 | 路径 |
|---|---|
| Slurm 输出、错误及提交配置快照 | `final_exam/single_circuit_pin/hpc_logs_pin/<run_id>/` |
| 启动、训练、评估日志 | `final_exam/single_circuit_pin/logs_pin/<run_id>/` |
| 数据、配置及启动记录 | `final_exam/single_circuit_pin/results_pin/<run_id>/` |
| checkpoints、历史、测试报告 | `final_exam/single_circuit_pin/results_pin/<run_id>/training_pin/` |
| MAE、RMSE、R² 与跨 seeds 标准差 | `final_exam/single_circuit_pin/results_pin/<run_id>/training_pin/summary_pin.xlsx` |

模型名保留 `_fusion_pin` 后缀，checkpoint recipe 和报告记录 `attention_scope=fusion_self_attention`。共享教师与 predictor 仍保存在 `training_pin/seed_<seed>_pin/shared_pin/`；门控前后比较、配置、训练成本与恢复记录沿用现有统计流程。输出和日志目录继续被 Git 忽略。

## 中断恢复

这三个新模型使用同一个恢复入口：

```bash
bash final_exam/single_circuit_pin/resume_staged_op_gate_pin.sh \
  final_exam/single_circuit_pin/results_pin/<失败的fusion_run_id>
```

保留源目录与数据 JSON。恢复自动继承原模型、seeds、配置和划分，已完成阶段不再优化，新格式 checkpoint 从完整 epoch 边界继续。恢复到一个新的结果目录，不覆盖原目录。旧全层门控模型不能通过改名称变成融合层模型续训；两者是不同实验。完整规则见 [RESUME_STAGED_OP_GATE_pin.md](RESUME_STAGED_OP_GATE_pin.md)。

## 验证边界

本次新增的是门控位置对照代码。它不预先声称融合层门控精度更高或能解决验证曲线振荡；这些结论需要在相同数据划分、seeds 和预算下对照实验后判断。实际执行过的检查见 [STAGED_OP_GATE_VERIFICATION_pin.md](STAGED_OP_GATE_VERIFICATION_pin.md)，不能把 CPU 小模型或模拟 Slurm 测试表述为已通过 A800 全量训练。
