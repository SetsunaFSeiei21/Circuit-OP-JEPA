# 分阶段引脚 OP 门控：失败恢复与断点续训

本功能用于单电路的三个分阶段门控模型：

- `device_op_distill_gate_pin`
- `device_op_distill_gate_decoder_pin`
- `device_op_supervised_gate_pin`

当前没有把所有 baseline、器件级实验或 OOD 实验都改成支持续训。恢复任务读取原实验的模型、seeds、有效配置和数据划分，在同一仓库内创建新的结果与日志子目录；不覆盖原失败目录，也不重新导入 SQLite。

## 恢复这次失败的实验

在仓库更新到包含本功能的版本后，先停止原来的训练作业，再提交恢复任务。原作业已经失败退出时，无需再次取消。Bash 入口在能使用 `squeue` 且源日志记录了作业 ID 时，会检查原作业是否仍然活跃；活跃时拒绝重复提交。它不会代替用户取消正在运行的作业。

本次失败目录对应的命令：

```bash
cd /share/home/202411094934/python_project/Circuit-OP-JEPA

bash final_exam/single_circuit_pin/resume_staged_op_gate_pin.sh \
  "$PWD/final_exam/single_circuit_pin/results_pin/two_stage_folded_opamp_staged_op_gate_20261110T122711Z_3250385_23234_pin"
```

需要先查看命令时，在末尾加 `--dry-run`：

```bash
bash final_exam/single_circuit_pin/resume_staged_op_gate_pin.sh \
  "$PWD/final_exam/single_circuit_pin/results_pin/two_stage_folded_opamp_staged_op_gate_20261110T122711Z_3250385_23234_pin" \
  --dry-run
```

`--dry-run` 只读取小型元数据并打印提交命令，不提交作业、不创建结果和日志目录，也不会在登录节点导入 torch、解析大型 canonical JSON 或分配 GPU。

路径既可以指向原 run 目录，也可以指向其 `training_pin` 子目录。也可以把路径写进 `resume_staged_op_gate_pin.sh` 顶部的 `RESUME_FROM` 后无参数运行。集群设置在同一 Bash 文件顶部，默认账户 `b_phzhwu`、分区 `ex01A800`、8 CPU、1 GPU、80G 主机内存和 `newbase` 环境。

恢复入口沿用原模型列表、seed 列表、配置和 manifest，不从现在的 `configs/default.json` 或其他新配置取值。提交时保存恢复来源的配置与 manifest 摘要，计算节点开始执行时再次检查。修改模型超参数、训练预算、数据或划分属于新实验，应使用普通训练入口；恢复入口不会默默接受这些变化。

## 旧实验具体能恢复多少

程序按每个 seed、每个阶段实际存在的 checkpoint 判断，不把某一个 seed 的训练进度用于其他 seed。

| 源阶段保存情况 | 行为 | 是否恢复优化器状态 |
|---|---|---|
| `best_pin.pt` 标记 `complete=true`，且配置、状态和成本校验通过 | 加载已完成阶段选中的权重，复用历史；不执行该阶段的优化器更新 | 不需要继续该阶段，因此不加载优化器 |
| 新格式 `last_pin.pt` 完整 | 从最后一个完成训练和验证的 epoch 边界继续剩余预算 | 是，同时恢复 RNG、早停和最佳权重状态 |
| 旧格式只有未完成的 `best_pin.pt` | 加载旧最佳权重，重置优化器与早停，以原配置的完整阶段预算重新训练该阶段 | 否，明确标记 `legacy_weights_restart` |
| 阶段未开始，或没有可恢复 checkpoint | 从该阶段正常初始化开始训练 | 否 |
| checkpoint 损坏或数据、配置、依赖不匹配 | 报错并保留源目录；不静默换用不匹配的权重 | 不适用 |

本次错误发生在最后 gate 阶段。若此前保存文件完整，可复用器件级 OP 预测器、真实 OP 教师、学生 encoder 蒸馏和学生 decoder 训练等已完成阶段。程序重新加载这些权重、重建必要的内存缓存，再执行尚未完成的门控训练；没有开始的后续模型和 seeds 正常训练。

旧版本没有保存 AdamW 状态与随机数状态，不能从一个旧 `best_pin.pt` 推导出这些信息。因此，旧中断阶段的权重恢复与新格式的完整状态续训会使用不同的策略名称和日志，不声称两者等价。

## 新训练保存的续训状态

更新后的三个分阶段模型及其前置训练阶段，会在每个完整的训练＋验证 epoch 结束后保存：

```text
training_pin/seed_<seed>_pin/stages_pin/<stage_name>/last_pin.pt
```

该文件包含当前 epoch 的模型权重、AdamW 状态、Python/NumPy/PyTorch CPU/CUDA 随机数状态、已完成 epoch 历史、优化器步数、累计时间、早停状态，以及截至该 epoch 的验证最佳权重和阶段元数据。当前权重与验证最佳权重分开保存；恢复后继续训练使用当前权重，阶段结束用于测试的仍是验证选中的最佳权重。

保存采用同目录临时文件、文件刷新与 `fsync`、原子替换；支持时也对目录执行 `fsync`。真实存储设备和共享文件系统的持久性仍受其实现约束，不把软件的原子写入表述为对任何存储故障的保证。

这是 **epoch 边界续训**，不是 batch 中间续训。若在某个 epoch 的训练或验证中途断电，该 epoch 尚未完成的工作会重跑。已保存为完整阶段的结果可以直接复用；若最后 epoch 已保存但阶段最终导出尚未完成，则恢复后进行最终选模与保存，不重复优化已完成的 epoch。

恢复时会使用保存的训练集归一化统计、引脚词表和教师 latent scaler，不重新拟合这些量。模型的冻结策略、门控开关、性能监督与验证选模规则保持原定义。CUDA RNG 恢复要求设备数量匹配；不保证不同 GPU、PyTorch 版本或非确定性内核之间逐位一致。CPU 的确定性小样本测试验证了中断恢复和不中断训练的一致性，尚未在实际 A800 上验证。

## 数据、结果和日志放在哪里

新 run ID 形如：

```text
<电路>_resume_staged_op_gate_<UTC时间>_<进程号>_<随机数>_pin
```

| 内容 | 仓库内位置 |
|---|---|
| Slurm 标准输出、错误和提交时来源快照 | `final_exam/single_circuit_pin/hpc_logs_pin/<新run_id>/` |
| 启动、阶段训练和评估日志 | `final_exam/single_circuit_pin/logs_pin/<新run_id>/` |
| 新配置、数据引用和来源记录 | `final_exam/single_circuit_pin/results_pin/<新run_id>/` |
| 恢复后的模型、历史、成本、测试结果与 Excel | `final_exam/single_circuit_pin/results_pin/<新run_id>/training_pin/` |

不会复制出另一个 `Circuit-OP-JEPA_*` 工程目录。新的实验目录通过 `data_source_pin.json` 引用原来的 canonical `data_pin.json`，不复制这份大文件、不再次导入原始 SQLite。**请保留该原始数据文件及路径**；移动或删除它会使后续恢复无法加载数据。恢复任务再次失败时，可以把这个新 run 目录作为下一次 `--resume-from` 的来源，数据引用会继续指向已有 canonical JSON。

原失败目录的模型和日志不被修改。新的训练目录中会重新保存已复用阶段的 checkpoint 和历史，后续恢复不必依赖更早目录里的阶段 checkpoint；canonical 数据文件仍保留引用关系。

## 如何确认没有重训前置阶段

日志会明确区分以下消息：

- `reused completed stage ...; no optimization`：复用完成阶段。
- `resume after completed epoch ... with optimizer, RNG and early-stopping state`：新格式完整状态续训。
- `legacy best weights ... restarting the full ...-epoch stage budget`：旧权重热启动，重训该阶段。
- `no usable checkpoint ... restarting this stage`：有历史痕迹但没有可恢复权重，重训该阶段。

`training_pin/resume_pin.json` 记录每个阶段的恢复策略和来源。新的完整 `summary_pin.xlsx` 会增加 `Resume` 页，展示恢复 epoch、是否恢复优化器、本次新增 epoch/优化器步数/时间、保留的累计训练量，以及无法精确计数的中断工作。

`TrainingCost` 仍保留形成最终模型所需的历史训练投入，并对共享阶段去重；已完成阶段不会因为本次只加载权重就变成“零训练成本”。`Resume` 页中的新增量用于查看本次作业真正新增的训练。GPU/RAM 中的 OP 预测缓存会重建，已保存的历史缓存前向成本与本次重建成本累计保留；`resume` 字段和 Excel `Resume` 页另列本次新增缓存耗时。这些前向计算不会被当作重新训练 OP 预测器，也不会因为恢复而抹去已知历史缓存成本。

旧版未记录的中断工作，以及断电时未完成 epoch 的工作，无法精确还原优化器步数与耗时；报告会显式标注，不编造数值。测试集仍只用于最终报告，不参与恢复策略或 checkpoint 选择。

## 开发验证入口

```bash
python -m unittest experiments.op_transfer.test_resume -v
python -m unittest final_exam.tests.test_staged_resume_pin -v
python -m unittest final_exam.single_circuit_pin.test_resume_staged_op_gate_launch_pin -v
```

这些是 CPU 小数据、故障注入与模拟 Slurm 测试，不会提交实际集群作业。完整验证记录见 [STAGED_OP_GATE_VERIFICATION_pin.md](STAGED_OP_GATE_VERIFICATION_pin.md)。
