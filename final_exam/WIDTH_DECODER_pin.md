# 多 PE 宽读出：性能端与 OP 预测端

本组实验新增 `width_decoder_pin` 和 `width_decoder_op_pin`，对照为原有 `device_op_cascade_pin`。两种新模型都不使用 OP 门控，也不进行教师 latent 蒸馏。原有门控版本和模型 ID 保留原行为。

## 1. 三个模型分别改变哪里

| 模型 ID | 器件级 OP 预测器 | 引脚级性能预测器 |
|---|---|---|
| `device_op_cascade_pin` | 原始设计编码器最后一个 PE 的器件特征 → MLP → OP 向量 | 最后一个 PE → 一组指标查询 Decoder → MLP |
| `width_decoder_pin` | 与对照完全相同，同一 seed 在本次实验内共享冻结 checkpoint | 每个 PE → 独立指标查询 Decoder；拼接同一指标的各层特征 → MoE-MLP |
| `width_decoder_op_pin` | 每个 PE → 独立器件查询 Decoder；拼接同一器件的各层特征 → MoE-MLP → OP 向量 | 与 `width_decoder_pin` 相同的宽性能读出，使用本模型自己的冻结宽 OP 预测器 |

这里的“宽”指并列读取多个 PE 的输出，不是仅增加最后一个 Decoder 的 hidden dimension。两端仍使用仓库既有的 ZeroSim SE/PE 编码框架：拓扑结构编码、设计参数/OP 融合、拓扑掩码、端口身份、物理器件身份都沿用原实现。

## 2. 性能端的具体结构

设 `L=model.cross_layers`、`d=model.dim`、`M` 为性能指标数量：

1. SE 正常执行一次，PE1…PEL 顺序执行一次。
2. 在每个 PE 输出处取得一个分支。读取时复用既有 `output_norm` 和 `output_projection`，再交给该分支的 Decoder；归一化后的分支张量不会替代下一 PE 的原始输入。
3. 每个 Decoder 有自己独立的指标 query 和 Transformer 参数，深度为 `model.decoder_layers`，FFN 宽度为 `model.decoder_ffn_dim`。
4. 每个 Decoder 输出 `[B,M,d]` 的特征，不先压成 M 个标量。
5. 按 PE1…PEL 顺序，在最后一维拼接同一指标的特征，得到 `[B,M,L*d]`。
6. MoE-MLP 逐指标融合，输出 `[B,M]`，在最终预测上计算原有性能损失。

默认 3 个 PE、128 维表示、9 个性能指标时，拼接尺寸为 `[B,9,384]`。所有 decoder 分支、下游 SE/PE 与 MoE 在同一性能训练阶段更新。没有逐层额外损失、router balancing loss、门控 loss 或新的蒸馏目标。

PE 分支数由 **`cross_layers`** 决定。当前编码器的结构层数为 `encoder_layers-cross_layers`。例如 `encoder_layers=7,cross_layers=3` 是 4 个结构层和 3 个 PE，仍然有 3 个宽读出分支。

## 3. 宽 OP 端如何保持器件对应关系

OP 端不能直接沿用“9 个电路性能 query”，因为目标是每一个物理器件自己的 OP 字段：

1. 输入由 `device_input_pin` 从 canonical owner 拓扑与设计参数构造。编码的是物理器件，不是重复的 D/G/S/B 引脚。
2. 每个 PE 分支的 memory 包含 GRAPH 和所有器件。以该 PE 的第 j 个器件表示作为第 j 个 decoder query，query self-attention 排除 padding，cross-attention 读取该层有效 memory。
3. 每个分支得到 `[B,D,d]`。GRAPH 只作为 memory，不作为 OP 目标。
4. 对相同物理器件按层拼接为 `[B,D,L*d]`，MoE-MLP 输出 `[B,D,F]`，F 为保存的 OP 字段数。
5. 继续使用原有训练集支持规则、适用器件和逐字段缺失掩码。无训练支持/不适用的位置输出零，损失只计算真实观测且受支持的器件字段，每个物理器件只计一次。

器件 query 是本层设计特征，不包含真实 OP、性能标签、实测缺失标志或下一层信息。输出之后由原有 owner 索引广播到器件所属引脚，外部电路端口不因此获得 OP。同一器件的多个引脚不会各自训练一份 OP 目标。

OP 端和性能端的 Decoder/MoE 都是独立模块，两端不共享参数。这里使用同一类宽读出结构，但 query 的物理语义不同：OP 按器件，性能按指标。

## 4. MoE-MLP 定义

配置位置为 `width_decoder_pin`，默认：

```json
"width_decoder_pin": {
  "n_experts": 4,
  "hidden_dim": 128,
  "dropout": 0.1
}
```

每个专家是 `Linear(L*d, hidden_dim) → GELU → Dropout → Linear(hidden_dim, output_dim)`。router 是 `Linear(L*d, n_experts) → Softmax`，根据当前指标/器件的拼接特征生成权重，对所有专家输出加权求和。性能端 `output_dim=1`，OP 端 `output_dim=F`。

这是稠密 MoE：所有专家都计算，没有 top-k 路由、跨样本路由、额外均衡损失。它是普通完整 MLP 专家的混合，与门控 V3 及后续版本中逐线性层混合的 M-MLP 分开实现。无需安装新的第三方 MoE 包。

## 5. 训练、早停与预算

三个模型都是先 OP、后性能：

| 阶段 | 配置项 | 默认最多轮数 | 监督 | 更新范围 |
|---|---|---:|---|---|
| OP 预测器 | `op_transfer.epochs.op_predictor` | 30 | 训练集真实 DC OP | OP encoder＋原 MLP 或宽 Decoder/MoE |
| 性能模型 | `op_transfer.epochs.readout` | 60 | 训练集性能指标 | 下游 pin encoder＋其 Decoder/MoE |

每阶段用对应验证损失选择最佳 checkpoint。`op_transfer.patience=10` 表示连续 10 个 epoch 没改善时早停，0 禁用早停；保存实际执行轮数。默认 batch size=32、lr=1e-4。

OP 阶段结束后加载其验证最优权重，冻结全部 OP 参数，并在下游 `train()` 时保持 OP 分支 `eval()`。性能损失不反传进 OP predictor。下游训练、验证、测试都使用预测 OP；真实 OP 仅用于 OP 预测器的训练和验证监督。标准化继续只从 ID 训练集拟合。

同一 seed 内，前两个模型共享一次原始 OP 训练，第三个模型另训一次宽 OP predictor；不是为三个模型重复训练三份相同 OP predictor。每个模型的性能训练是独立的。共享阶段成本按阶段 ID 去重。宽读出增加参数和计算，**相同 epoch 上限不代表相同 FLOPs 或 wall time**，应同时查看 Excel 中的 TrainingCost。

## 6. 启动、数据量与目录

编辑 `final_exam/single_circuit_pin/train_width_decoder_pin.sh` 顶部：

```bash
CIRCUITS=(two_stage_folded_opamp Opamp1)
MODELS=(
    device_op_cascade_pin
    width_decoder_pin
    width_decoder_op_pin
)
SEEDS=(17 29 43)
```

脚本默认只列 `two_stage_folded_opamp`；上例展示两个电路，每个电路独立提交一个单电路实验，不是 OOD。配置为 `final_exam/configs/width_decoder_pin.json`。`dataset.max_samples_per_circuit=20000` 表示先按 `sampling_seed` 抽取最多 20,000 条，再划分；默认 `null` 使用全量。

```bash
bash final_exam/single_circuit_pin/train_width_decoder_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_width_decoder_pin.sh
```

也支持 `--smoke` 缩短 epoch；它不自动截断数据，要缩小数据量请设置上述样本上限。Slurm 默认沿用 8 CPU、1 GPU、80G、账户 `b_phzhwu`、分区 `ex01A800`、conda `newbase`。文件读取/准备在 CPU 上，模型张量训练由 `--device cuda` 指定 GPU。

所有目录在当前仓库 `final_exam/single_circuit_pin/` 内：

| 内容 | 相对路径 |
|---|---|
| HPC stdout/stderr、提交配置 | `hpc_logs_pin/<电路>_width_decoder_<时间戳>_pin/` |
| 导入、逐阶段训练、评估日志 | `logs_pin/<相同run_id>/` |
| 数据与结果根目录 | `results_pin/<相同run_id>/` |
| 最终 Excel | `results_pin/<run_id>/training_pin/summary_pin.xlsx` |
| 各模型配置、checkpoint、test 报告 | `results_pin/<run_id>/training_pin/seed_<seed>_pin/<model>/` |
| 原始 OP predictor | `.../seed_<seed>_pin/shared_pin/device_op_predictor_pin/` |
| 宽 OP predictor | `.../seed_<seed>_pin/shared_pin/width_device_op_predictor_pin/` |

最终模型 checkpoint 内也包含自身的冻结 OP predictor、完整架构 recipe、字段顺序和标准化参数；部署不需要从其他实验路径重新寻找 predictor。独立保存的 shared checkpoint 用于审计与复用。`width_audit_pin.json` 记录分支结构、原始 predictor checkpoint 路径、状态摘要和 decoder 参数量。

Excel 保留各模型 MAE/RMSE/R² 的跨 seed 均值与标准差、每个 seed 明细、配置和成本，并增加 `WidthDecoder` 表，明确两端的读出结构、分支数、拼接维数、专家数和冻结状态。仅运行一个 seed 时不能把结果解释为跨 seed 稳定性证据。

## 7. 中断恢复

```bash
bash final_exam/single_circuit_pin/resume_width_decoder_pin.sh \
  final_exam/single_circuit_pin/results_pin/<失败实验的run_id>
```

恢复沿用原始配置、样本/划分、模型和 seeds，输出到新目录，不改原结果。完成阶段直接复用；未完成阶段从最近完整 epoch 保存的 optimizer、RNG、早停状态继续。当前 epoch 中途未保存的进度会重跑。两个 predictor 家族分别恢复，不交叉使用 checkpoint 或预测缓存。不能把普通 predictor 改名当宽 predictor，或者改 PE 层数、专家数后直接接着旧优化器训练。

## 8. 实现位置与检查范围

- `device_jepa/model.py`：增加只读多 PE 输出接口。默认 `forward` 行为、旧参数名和旧 checkpoint 不变。
- `final_exam/models/width_decoder_pin.py`：分支 decoder、稠密 MoE、两种宽模型及独立架构保存/重建。
- `final_exam/width_registry_pin.py`：模型名、配置校验和轻量报告契约校验。
- `final_exam/run_pin.py`：按选择构建依赖、冻结 predictor、分离预测缓存、模型保存、阶段恢复与报告。
- `final_exam/summarize_pin.py`：原有指标表加宽读出结构表。
- 新测试覆盖 PE 顺序/梯度、器件与引脚边界、标签隔离、冻结权重、checkpoint 重建、模型子集、混跑缓存、完整阶段复用和中断 epoch 的精确恢复。

这些测试用于验证实现和工程流程；不代表完整 60,000 样本 GPU 实验已经执行，也不预先断言宽读出会提高预测精度。
