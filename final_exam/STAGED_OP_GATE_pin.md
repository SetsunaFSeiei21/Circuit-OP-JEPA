# 分阶段 OP 门控：端口级单电路实验

本实验保留已有模型，新增三个独立模型选项。它们使用相同的器件级 OP 预测器、引脚级 ZeroSim 编码框架和性能 decoder，通过训练顺序与最后阶段的可训练模块区分。新方案尚不代表门控必然提高精度或消除验证损失波动；需要与保存的无门控基模型及既有联合训练模型比较。

## 三个模型的定义

| 模型名称 | 关闭门控时的前置训练 | 最后阶段更新 | 最后阶段冻结 |
|---|---|---|---|
| `device_op_distill_gate_pin` | 真实 OP 教师 → 学生编码器蒸馏 → 学生 decoder 训练 | 门控 | OP 预测器、学生编码器、decoder |
| `device_op_distill_gate_decoder_pin` | 与上一个模型复用同一个已完成训练的学生 | 门控和 decoder | OP 预测器、学生编码器 |
| `device_op_supervised_gate_pin` | 不使用教师，直接监督训练编码器和 decoder | 门控 | OP 预测器、编码器、decoder |

三个模型的部署输入都是拓扑和设计参数。器件级预测器根据这些输入产生近似 OP；近似 OP 既作为下游引脚特征，也作为门控输入。真实 OP 仅用于预测器监督和前两个模型的教师训练/蒸馏，部署时不读取真实 OP、真实 OP 缺失模式或教师。

这是在原 A 组“OP 特征＋门控”基础上增加训练顺序对照，没有把原 A/B 两组的全部八种组合重复扩展。原有 `device_op_gate_pin` 等八个选项和旧版 `student_predicted_op_distill_pin` 均保持可用。

## 数据粒度与结构

- OP 预测器用真实器件节点进行 ZeroSim 编码，每个器件输出一组 OP。它不是先用引脚编码器再平均池化得到的旧版 `OPPredictorPin`。
- 下游教师、学生编码器和 decoder 使用引脚节点。MOS 的 D/G/S/B 分别作为节点，命名外部端口也有独立节点。
- 预测器输出的器件 OP 广播到该器件的引脚；外部端口不接收器件 OP。
- OP 的输入变换、预测目标变换和教师 latent 尺度仅由 ID 训练集拟合；验证集用于阶段选模，测试集不参与训练或选模。
- 门控沿用 `OPAttentionGatePin`：OP 值与支持掩码经 MLP 映射到 `model.dim`，结合有序引脚角色，输出每个 attention head 的门控分数。并不是为每个 head 重新训练一个 OP 预测器。
- 门控作用于编码器的自注意力，只对同一 MOS 内、不同网络之间的合格引脚对添加有限偏置。同网络短接、自环、外部端口及非 MOS 对保持原有处理。它是软注意力重加权，不是物理连通性标签，也不会自动减少稠密 attention 的计算量。

## 训练顺序及冻结边界

### 公共 OP 预测器

每个 seed 训练一次器件级预测器，按验证 OP 损失选择最佳 checkpoint，之后冻结且保持 `eval()`。可缓存预测值，缓存绑定对应预测器权重摘要。三个新模型复用同一预测器与缓存，成本只在实验总计中记录一次。

### 蒸馏基模型：前两个选项共同使用

1. 训练 `design_op_oracle_pin` 教师：拓扑＋设计参数＋真实 OP 输入，监督性能预测。恢复验证最佳 checkpoint 后冻结。
2. 学生复制教师的编码器和 decoder 权重；学生始终使用预测 OP。门控完全旁路，attention 不添加门控偏置。
3. 仅训练学生编码器，对齐教师的 graph token 和合格引脚 latent；使用训练集拟合的共同尺度和缺失值掩码。该阶段不使用性能损失更新 decoder。
4. 冻结学生编码器，训练学生 decoder 预测性能。门控仍完全旁路。
5. 保存完整的无门控学生。两个蒸馏门控选项分别从同一份学生、同一份门控初始状态复制出发，不会让第一个门控实验的更新影响第二个实验。

### 直接监督基模型：第三个选项使用

1. 复用冻结的器件级 OP 预测器。
2. 门控完全旁路，联合训练引脚编码器和 decoder，监督性能预测。
3. 恢复验证最佳 checkpoint，保存无门控基模型。

只选第三个模型时，不会为了它训练教师或蒸馏学生。

### 最后门控阶段

- `device_op_distill_gate_pin` 和 `device_op_supervised_gate_pin`：只更新门控，编码器和 decoder 参数全部冻结、保持 `eval()`。
- `device_op_distill_gate_decoder_pin`：只更新门控和 decoder；编码器冻结并保持 `eval()`，训练阶段 decoder 正常启用训练模式。
- 冻结的是参数，不是经过模块的计算图。最后阶段仍需让性能损失的梯度穿过 decoder 和编码器传给门控，不能将这段前向包进 `torch.no_grad()`。
- 原始 OP 预测器一直冻结。其前向可以用 `no_grad()`，因为门控学习不需要更新 OP 预测器。
- 每阶段独立按验证集损失选最佳权重，独立重置早停计数。最后使用选中的门控模型评估测试集；保存的无门控基模型也在训练完成后评估，作为对应参考。

门控关闭时的偏置严格为零。门控开启后的初始化沿用 `initial_gate=0.99`，因此初始偏置接近零，但不是与关闭门控严格相同；不能把这次初始化描述为完全恒等映射。

## 配置位置与默认轮数

配置文件：`final_exam/configs/staged_op_gate_pin.json`。

```json
"op_transfer": {
  "epochs": {
    "teacher": 60,
    "op_predictor": 30,
    "predicted_op_distill_encoder": 30,
    "predicted_op_distill_decoder": 30,
    "readout": 60,
    "student_predicted_op": 60,
    "gate": 30
  },
  "batch_size": 32,
  "lr": 0.0001,
  "patience": 10
}
```

| 配置字段 | 控制内容 |
|---|---|
| `op_transfer.epochs.op_predictor` | 公共器件级 OP 预测器 |
| `op_transfer.epochs.teacher` | 前两个模型的真实 OP 教师 |
| `op_transfer.epochs.predicted_op_distill_encoder` | 学生编码器单独蒸馏 |
| `op_transfer.epochs.predicted_op_distill_decoder` | 学生 decoder 单独训练 |
| `op_transfer.epochs.readout` | 第三个模型的无门控监督训练 |
| `op_transfer.epochs.gate` | 三个模型最后的门控阶段；第二个同时更新 decoder |
| `op_transfer.batch_size`、`lr` | 各阶段 batch size 与学习率 |
| `op_transfer.patience` | 每阶段早停容忍轮数；0 关闭早停 |

`training.decoder_epochs` 不控制这些阶段。`student_predicted_op` 仍是旧版联合学生训练预算，并参与现有下游预算一致性检查：`readout == student_predicted_op == predicted_op_distill_encoder + predicted_op_distill_decoder`。`gate` 是独立追加预算，不包含在这条等式中。

在没有触发早停的情况下，每个模型的依赖链为：

| 模型 | 最大轮数构成 | 含前置依赖的最大轮数合计 |
|---|---|---:|
| `device_op_distill_gate_pin` | OP 30＋教师 60＋学生 encoder 30＋decoder 30＋门控 30 | 180 |
| `device_op_distill_gate_decoder_pin` | OP 30＋教师 60＋学生 encoder 30＋decoder 30＋门控和 decoder 30 | 180 |
| `device_op_supervised_gate_pin` | OP 30＋监督训练 60＋门控 30 | 120 |

这些合计用于说明依赖链，不能把三行相加作为一次实验实际训练成本。三个模型在同一个 seed 中一起运行时，共享 OP 预测器、教师与蒸馏基模型，最大唯一训练阶段合计为 **300 轮**；缓存生成另计耗时，不计训练 epoch。不同阶段可训练参数和每轮计算量不同，轮数相同不代表算力成本相同。最终以 `TrainingCost` 中的实际轮数、优化器步数和耗时为准。

最后阶段比已有一次性联合训练多出训练预算，因此不能仅凭最终分数声称改进完全来自训练顺序。严格的等计算量结论还需额外对照，例如给无门控模型相同的追加训练预算。尤其是“门控＋decoder”方案：若要将收益单独归因于门控，应另外比较“不开门控、只给 decoder 相同追加训练轮数”的模型；本次三个新增选项没有自动包含该额外控制。

## 如何启动

在仓库根目录编辑 `final_exam/single_circuit_pin/train_staged_op_gate_pin.sh` 顶部：

```bash
CIRCUITS=(two_stage_folded_opamp)
MODELS=(
  device_op_distill_gate_pin
  device_op_distill_gate_decoder_pin
  device_op_supervised_gate_pin
)
SEEDS=(17 29 43)
```

模型、电路与 seeds 都可以直接写在这个 Bash 文件中，不需要设置 `SOURCE_RUN`，也不需要先提供旧结果目录。从原始数据重新训练：

```bash
bash final_exam/single_circuit_pin/train_staged_op_gate_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_staged_op_gate_pin.sh
```

每个电路提交一个 Slurm 作业，默认 8 CPU、1 GPU、80G 主机内存，账户 `b_phzhwu`、分区 `ex01A800`、Conda 环境 `newbase`。GPU 用于训练；SQLite/JSON 读取和静态数据准备仍需 CPU。`--smoke` 缩短训练；样本数由配置的 `dataset.max_samples_per_circuit` 控制，默认 `null` 为全量，见 [数据抽样说明](../docs/DATA_SUBSAMPLING_ZH.md)。`--dry-run` 只预览，不提交也不创建输出目录。

脚本在提交时保存配置快照，并通过 `JEPA_PROJECT_ROOT` 传递当前仓库根目录，避免 Slurm `/var/spool/` 副本引起路径错误。不会创建另一个仓库副本。程序仅尝试记录 Git commit；计算节点没有 Git 时会继续训练并将版本标为 `unavailable`。

## 日志、模型与结果

本次 run ID 为 `<电路>_staged_op_gate_<UTC时间>_<进程号>_<随机数>_pin`。相对仓库根目录：

| 内容 | 位置 |
|---|---|
| HPC 标准输出、错误及提交配置 | `final_exam/single_circuit_pin/hpc_logs_pin/<run_id_pin>/` |
| 导入、总实验、阶段训练及评估日志 | `final_exam/single_circuit_pin/logs_pin/<run_id_pin>/` |
| 数据及运行配置 | `final_exam/single_circuit_pin/results_pin/<run_id_pin>/` |
| 模型、成本、测试结果和 Excel | `final_exam/single_circuit_pin/results_pin/<run_id_pin>/training_pin/` |

在 `training_pin/seed_<seed>_pin/` 内：

- `shared_pin/device_op_predictor_pin/checkpoint_pin.pt`：冻结的器件级 OP 预测器。
- `shared_pin/design_op_oracle_pin/checkpoint_pin.pt`：真实 OP 教师，仅蒸馏路线需要。
- `shared_pin/device_op_distilled_base_pin/checkpoint_pin.pt`：前两个模型共同使用的无门控学生。
- `shared_pin/device_op_supervised_base_pin/checkpoint_pin.pt`：第三个模型的无门控监督基模型。
- `<模型名>/checkpoint_pin.pt`：独立部署 checkpoint，包含所需预测器、编码器、decoder 和门控；推理不需要另外加载教师文件。
- `<模型名>/config_pin.json`、`dependencies_pin.json`：该模型的有效配置与依赖阶段。
- `<模型名>/test_pin.json`：最终模型的测试指标。
- `<模型名>/gate_comparison_pin.json`：无门控基模型和最终模型的测试结果及基模型 checkpoint 路径。
- `stages_pin/`：各阶段历史、配置、实际成本及最佳 checkpoint。

完成全部选定模型和 seeds 后自动生成 `training_pin/summary_pin.xlsx`。其中：

- `MAE`、`RMSE`、`R2`：各模型按指标比较，跨 seeds 为均值 ± 样本标准差；单 seed 不捏造标准差。
- `GateBeforeAfter`：逐 seed 对比对应的无门控基模型与最终门控模型；变化量为“最终－基模型”。MAE/RMSE 负值、R² 正值表示改善。该页使用原始物理单位，UGF 为 Hz。
- `ModelInputs`：粒度、预测 OP 特征、门控来源、基模型训练方式和最后阶段可训练模块。
- `TrainingCost`：共享阶段仅计一次，包含门控追加成本。
- `Configuration`、`PerSeed`、`Statistics`：可检查的配置、每 seed 原始数值和汇总值。

前两个模型在 `GateBeforeAfter` 中重复引用同一个无门控学生，只是方便逐模型比较，不代表两个独立 seed，也不重复计入训练成本。测试结果只用于报告，不参与 checkpoint 选择。

手动重新生成 Excel：

```bash
bash final_exam/single_circuit_pin/summarize_pin.sh \
  final_exam/single_circuit_pin/results_pin/<run_id_pin> \
  --output final_exam/single_circuit_pin/results_pin/<run_id_pin>/review_pin.xlsx
```

默认拒绝覆盖现有工作簿，未完成实验默认拒绝导出。`--allow-incomplete` 只生成明确标注的临时表，不进行最优高亮。三个新模型的完整导出要求有对应的基模型比较文件。

## CUDA 门控反向传播兼容性

部分 PyTorch CUDA memory-efficient SDPA 实现在 Q/K/V 均不需要梯度、只有浮点 attention mask 需要梯度时，不保存反向传播所需的 logsumexp（LSE）。这会导致 `LSE is not correctly aligned (strideH)`，某些形状也可能报非法内存访问。上游问题见 [PyTorch #148476](https://github.com/pytorch/pytorch/issues/148476)，上游修复见 [#188302](https://github.com/pytorch/pytorch/pull/188302)。

分阶段门控的第一个冻结自注意力块恰好满足这个条件：结构 embedding 与 Q/K/V 投影被冻结，只有 OP 门控偏置在训练。普通联合训练时，Q/K/V 也需要梯度，因此不一定触发同一缺陷。这个报错与 Git 元数据警告、loss 震荡不是同一个问题。

`device_jepa/model.py` 中的 `EncoderBlock` 现在只在以下条件同时成立时，局部使用 `sdpa_kernel(SDPBackend.MATH)`：

1. 当前允许记录梯度；
2. 自注意力 mask 需要梯度；
3. 输入与 Q/K/V 投影权重、偏置都不需要梯度。

这条路径仍在输入所在设备执行，CUDA 张量不会转到 CPU。编码器继续冻结，门控梯度穿过编码器和 decoder；不会 `detach` 门控或重新训练主干。上下文结束即恢复调用方的注意力后端设置，后续有 Q/K/V 梯度的层、正常联合训练、验证和推理仍沿用原来的后端选择。不同后端可能有浮点舍入差异；局部 math attention 的显存与速度开销需要在实际 GPU 上测量。

无需修改模型名称或 epoch。已有完成阶段的 checkpoint 可以通过新增的 `resume_staged_op_gate_pin.sh` 恢复入口复用；新训练另保存包含优化器、随机数和早停状态的 epoch 边界 checkpoint。普通 `train_staged_op_gate_pin.sh` 仍用于从头开始的新实验。恢复命令、旧版 checkpoint 的限制和日志路径见 [失败恢复与断点续训说明](RESUME_STAGED_OP_GATE_pin.md)，请保留失败目录中的模型、日志及原 canonical 数据文件。

CPU 回归与 CUDA 专用回归入口：

```bash
python -m unittest discover -s tests -p 'test_attention_mask_grad.py' -v
python -m unittest final_exam.tests.test_staged_op_gate_models_pin -v
```

CPU 环境会明确跳过 CUDA 用例。在已分配 GPU 的计算节点运行这两条命令，才能验证实际 CUDA 内核路径；这些测试使用合成小样本，不读取完整 60,000 点数据。新增 CUDA 用例覆盖冻结 Q/K/V 的偏置反向传播，以及三个分阶段门控模型的优化器更新。
