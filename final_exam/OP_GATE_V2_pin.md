# 门控 v2：原 OP 与样本内相对 OP

v2 检验的假设是：在保留现有 OP 绝对尺度信息的同时，显式提供同一电路内部的相对器件状态，是否有助于门控学习。代码实现本身不证明低电流差异已导致 v1 性能退步，也不预设 v2 优于 v0、v1。版本演进和工程边界统一记录在 [record.txt](record.txt)。

## 六个新模型与对照

| 模型 ID | 最终训练策略 | 门控位置 |
|---|---|---|
| `device_op_gate_pin_v2` | 联合训练下游 encoder、decoder、门控 | 结构层＋融合层 Self-Attention |
| `device_op_gate_fusion_pin_v2` | 同上 | 仅融合层 Self-Attention |
| `device_op_distill_gate_pin_v2` | 真实 OP 教师→学生 encoder 蒸馏→冻结 encoder 训练 decoder→冻结二者，仅训练门控 | 结构层＋融合层 Self-Attention |
| `device_op_distill_gate_fusion_pin_v2` | 同上 | 仅融合层 Self-Attention |
| `device_op_supervised_gate_pin_v2` | 关门控监督训练 encoder＋decoder→冻结二者，仅训练门控 | 结构层＋融合层 Self-Attention |
| `device_op_supervised_gate_fusion_pin_v2` | 同上 | 仅融合层 Self-Attention |

这六个模型都先训练器件级 OP 预测器，再将其冻结。表中的“联合训练”不包含 OP 预测器。下游保持引脚级 ZeroSim 编码器框架。

默认还选择已有的 `device_op_cascade_pin` 无门控监督基线。蒸馏路线会自动保存、评估 `device_op_distilled_base_pin` 无门控基座，不需要额外写入 `MODELS`。默认七个选择入口对应 Excel `GateStudy` 中八个去重模型/基座结果；相同依赖的训练成本只计一次。

## 门控输入和统计范围

设 `z[b,d,f]` 是现有预处理标准化后的预测 OP。原预处理的 asinh 尺度及均值/方差仍只由 ID 训练集拟合；它们不会被样本内统计替代。

对每个样本 `b`、每个字段 `f`，在该样本全部有效 MOS 器件中独立计算：

```text
mu[b,f]  = mean(z[b,d,f] over valid MOS d)
var[b,f] = mean((z[b,d,f] - mu[b,f]) ** 2 over valid MOS d)
relative[b,d,f] = (z[b,d,f] - mu[b,f]) / sqrt(var[b,f] + norm_eps)
```

- NMOS 与 PMOS 一起统计；不同电路样本、不同 batch 成员互不混合。
- 所有训练支持的 OP 字段都参与；不同字段分别计算统计量，不将电压、电流等混成一个分布。
- 有效集合由真实器件槽位、MOS 类型和训练字段支持确定，排除 padding、R/C、端口和不支持字段。支持 mask 不拼接给 MLP。
- 不读取测试真实 OP 或其缺失模式来构建预测 OP 模型的统计量。
- 有效器件不足两个、字段恒定时，相对通道为零；无效槽位也为零。总体方差、float64 矩计算和独立正数 `norm_eps` 用于数值保护，计算保持在输入所在设备上。
- 默认 `norm_eps=1e-5`，作用于标准化 OP 空间的方差。它与注意力偏置公式中的 `epsilon=1e-4` 是两个不同设置。
- 没有 BatchNorm 的跨 batch 统计或运行均值；也没有引入完整 GraphNorm 的可学习中心化参数。

门控网络为：

```text
concat(safe_original_z, safe_relative_z, one_hot(query_role,4), one_hot(key_role,4))
    -> Linear(2F+8, model.dim) -> GELU -> Linear(model.dim, model.heads)
    -> sigmoid
```

例如 `F=41, dim=128, heads=8` 时为 **90→128→8**。两个 OP 通道先在器件级计算，再扩展到有序 D/G/S/B 引脚对。只有一个门控 MLP，产生各头分数，并在模型指定的 encoder Self-Attention 层复用；不为每层创建独立门控。主编码器原有预测 OP 特征通道不改变。

门控物理作用范围继续限制为同一 MOS、不同实际网络的引脚对。R/C、端口、graph token、自环、跨器件和同网短接的附加偏置保持零。偏置公式仍为 `strength * log(epsilon + (1-epsilon) * gate)`，默认 `initial_gate=0.99`；它是学习到的注意力调节，不是经过校准的导通概率。

## 配置与启动

编辑 [single_circuit_pin/train_op_gate_pin_v2.sh](single_circuit_pin/train_op_gate_pin_v2.sh) 顶部的 `CIRCUITS`、`MODELS`、`SEEDS`。默认电路 `two_stage_folded_opamp`，seed 为 17/29/43。

训练参数和抽样数量位于 [configs/op_gate_pin_v2.json](configs/op_gate_pin_v2.json)。若每电路先随机抽 20,000 条再划分，在已有 `dataset` 中修改：

```json
"max_samples_per_circuit": 20000,
"sampling_seed": 17
```

模板默认 `null` 使用全量；同一实验的模型和训练 seeds 共用样本名单。详细抽样规则见 [数据量设置](../docs/DATA_SUBSAMPLING_ZH.md)。

```bash
bash final_exam/single_circuit_pin/train_op_gate_pin_v2.sh --dry-run
bash final_exam/single_circuit_pin/train_op_gate_pin_v2.sh
```

`--dry-run` 不提交、不创建输出目录；`--smoke` 提交短训练，样本数量仍由配置控制。Slurm 默认申请 8 CPU、1 GPU、80G，账户 `b_phzhwu`、分区 `ex01A800`、环境 `newbase`。

默认最大阶段预算保持与 v1 一致：OP 预测器 30，教师 60，蒸馏 encoder 30，冻结 encoder 后 decoder 30，直接监督/联合下游 60，独立门控 30；`batch_size=32`、`lr=1e-4`、`patience=10`。早停后的实际轮数和各阶段成本写入结果，不宣称所有方法总成本相等。对比 v1/v2 时应使用相同样本名单、划分、预算与训练 seed；新增通道也会增加门控第一层参数量。

## 结果、日志、教师与续训

每个电路独立作业，run ID 形如 `<circuit>_op_gate_v2_<timestamp>_..._pin`：

| 内容 | 路径 |
|---|---|
| 实验结果 | `final_exam/single_circuit_pin/results_pin/<run_id>/` |
| 实验日志 | `final_exam/single_circuit_pin/logs_pin/<run_id>/` |
| HPC stdout/stderr、提交配置快照 | `final_exam/single_circuit_pin/hpc_logs_pin/<run_id>/` |
| 自动汇总 Excel | `<结果目录>/training_pin/summary_pin.xlsx` |
| 每模型配置、权重、测试报告 | `<结果目录>/training_pin/seed_<seed>_pin/<完整模型ID>/` |
| 真实 OP 教师 | `<结果目录>/training_pin/seed_<seed>_pin/shared_pin/design_op_oracle_pin/checkpoint_pin.pt` |
| 器件级 OP 预测器 | `<结果目录>/training_pin/seed_<seed>_pin/shared_pin/device_op_predictor_pin/checkpoint_pin.pt` |

模型目录保存 `checkpoint_pin.pt`、`config_pin.json`、`test_pin.json` 和门控审计；分阶段模型另有 `gate_comparison_pin.json`，对照同一无门控基座。Excel 保留完整 `_pin_v2` 模型名称和版本、层范围，并分别呈现 MAE、RMSE、R² 的 mean±std；一个 seed 不能说明跨 seed 稳定性。

```bash
# 重新整理已有结果
bash final_exam/single_circuit_pin/summarize_pin.sh <结果目录>

# 恢复失败的 v2 实验，使用源实验的模型、配置和样本名单
bash final_exam/single_circuit_pin/resume_staged_op_gate_pin.sh <失败实验目录>
```

恢复写入新目录，已完成阶段复用；中断阶段使用最近完整 epoch 的优化器、随机数及早停状态。v0/v1 权重不会被当作 v2 门控加载，不能通过改模型名或 `norm_eps` 接续旧门控优化器状态。无门控依赖可以在同一实验、同一 seed 内共享，版本属于各自的门控模型及配方，而不是改写无门控模型名称。
