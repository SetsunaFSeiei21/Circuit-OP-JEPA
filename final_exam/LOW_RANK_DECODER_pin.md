# 宽 decoder 的低秩 attention / FFN 四组消融

本实验针对多 PE 宽读出的参数量：保持每个 PE 的独立性能 decoder 和最终 MoE 融合，分别替换 decoder 内 attention 投影和 FFN 线性层。它是从头训练的低秩架构实验，不是给旧 checkpoint 添加 LoRA，也不是先对旧权重做 SVD 再恢复训练。

## 1. 模型与修改范围

| 模型 ID | Decoder attention | Decoder FFN |
|---|---|---|
| `width_decoder_pin` | 原始完整投影 | 原始完整线性层 |
| `width_decoder_lowrank_attention_pin` | 独立低秩 Q/K/V/输出投影 | 原始完整线性层 |
| `width_decoder_lowrank_ffn_pin` | 原始完整投影 | 两个线性层独立低秩化 |
| `width_decoder_lowrank_both_pin` | 独立低秩 Q/K/V/输出投影 | 两个线性层独立低秩化 |

这里的“原模型”是已有 `width_decoder_pin`，不是最终 PE 单分支读出的 `device_op_cascade_pin`。四组都保留多个 PE 的读出，因此四组差异用于分析低秩化本身。

只修改**引脚级性能端各个 decoder 的每一层**，包括 query self-attention、读取 PE memory 的 cross-attention 以及 FFN。每个 PE 的 query 和 decoder 参数仍独立，不跨分支共享低秩因子。下列部分不变：

- 器件级 OP predictor 使用原始设计编码器加最终器件 MLP，不采用宽 OP decoder，也不做低秩化。
- 下游 SE、PE 的所有 attention、FFN、拓扑掩码、归一化和端口身份保留原实现。
- 性能 query、残差、dropout、层归一化以及 MoE 的专家和 router 保留。
- 不添加 OP 门控，不使用教师 latent 蒸馏，推理不需要真实 OP。

## 2. 线性层如何分解

对一个 `Linear(in_features, out_features)`，将完整权重替换为两个因子的乘积：

`Linear(in_features, rank, bias=False) → Linear(rank, out_features, bias=True)`。

两个因子之间不插入激活函数或额外 dropout。输出端只保留原线性层的一份 bias。两因子均可训练；不存在保留原完整权重的 `W0 + BA` 残差分支。因此这是模型权重低秩化，不是 LoRA 增量适配。

低秩因子从头随机初始化。相同 seed 下，未修改模块的初始权重保持一致；低秩投影不与原始完整投影的初始权重或初始函数逐点相同，因此比较包含低秩参数化及其优化过程的影响。

attention 中 Q、K、V 和输出投影分别分解。attention 的 softmax、头数、分头维度、padding 和 attention mask 的含义不变。这里不是 Linformer 式的 token 数压缩，不把 N×N attention 矩阵近似为少量 token；矩阵乘法实现仍保留原输入输出维度。

FFN 原来为 `128 → 2048 → ReLU → 128`，rank 64 时为 `128 → 64 → 2048 → ReLU → 64 → 128`，原有 dropout 仍位于原位置。2048 维非线性层和 128 维输出保留。FFN 的两个权重使用各自独立的因子。

## 3. 配置位置与秩的限制

配置文件：`final_exam/configs/low_rank_decoder_pin.json`。

```json
"low_rank_decoder_pin": {
  "attention_rank": 32,
  "ffn_rank": 64
}
```

模型名决定哪些部分启用低秩化。原始 `width_decoder_pin` 不启用；attention-only 不压缩 FFN；FFN-only 不压缩 attention；both 同时启用。配置中的数字是因子瓶颈的秩，不是全模型 `dim`，也不是 attention head 数。

rank 必须是正整数。启用的投影必须严格减少权重参数：

- attention 完整权重为 `dim×dim`，要求 `2×attention_rank < dim`。`dim=128` 时 rank 64 并不省参数，所以不允许。
- FFN 权重为 `dim×decoder_ffn_dim`，要求 `ffn_rank×(dim+decoder_ffn_dim) < dim×decoder_ffn_dim`。当前 128 / 2048 配置下 rank 64 和 32 都有效。

当前模板默认 `dim=128`、`heads=8`、`cross_layers=3`、`decoder_layers=1`、`decoder_ffn_dim=2048`。这是按最近的三电路宽 decoder 实验设置的一层 decoder 对照。若改为两层，四组需一起使用两层，参数量和计算量也会改变。`width_decoder_pin` 配置块继续管理未改动的 MoE。

以一个完整 `width_decoder_pin` 约 509 万参数、三个一层性能 decoder 为例，以下计数保留其他组件和原 bias：

| 设置 | Attention 投影权重合计 | FFN 权重合计 | 整个模型参数约值 |
|---|---:|---:|---:|
| 原始 | 393,216 | 1,572,864 | 509 万 |
| 仅 attention rank 32 | 196,608 | 1,572,864 | 489 万 |
| 仅 FFN rank 64 | 393,216 | 835,584 | 435 万 |
| attention rank 32 + FFN rank 64 | 196,608 | 835,584 | 416 万，减少约 18.3% |
| attention rank 32 + FFN rank 32 | 196,608 | 417,792 | 374 万，减少约 26.6% |

不同电路的身份嵌入等会使总参数略有变化；实际值以保存模型和成本记录为准。参数减少不保证 GPU 耗时同比下降，两个较小的矩阵乘法可能改变内核调用开销。

## 4. 训练预算与比较口径

每个 seed 先训练一次原始器件级 OP predictor，最多 30 轮。恢复其验证集最优权重后冻结，并在同一作业内由四组共享该 predictor 和预测 OP 缓存。各组分别训练自己的性能模型，最多 60 轮；下游 pin encoder、decoder 以及 MoE 按原宽读出训练方式共同更新。

配置项：

- `op_transfer.epochs.op_predictor=30`：OP 阶段上限。
- `op_transfer.epochs.readout=60`：四组性能阶段上限。
- `op_transfer.batch_size=32`、`lr=0.0001`：本组阶段设置。
- `op_transfer.patience=10`：验证损失连续 10 轮不改善触发早停。0 禁用。

全部模型共用当前实验的样本选择、划分、训练集归一化、seed 列表和性能指标。test 使用各自验证集最优 checkpoint。早停导致实际 epoch 不同，成本表保存实际轮数和时间；相同最大 epoch 不代表相同计算预算。

本组配置会在测试评估后，额外运行 `final_exam/benchmark_decoder_pin.py` 中的固定批次推理诊断。四组都使用本次实验确定的第一个 test batch，batch size 来自 `op_transfer.batch_size`；不足一个完整 batch 时记录实际样本数。每个模型先预热 3 次，再对同一个已经驻留设备的 batch 测量 10 次完整前向，记录平均延迟、跨测量标准差和按该批次计算的 samples/s。

计时输入只含拓扑与设计信息，**完整执行冻结的 OP predictor 和性能模型**，不读取训练时的预测 OP 缓存，不使用真实 OP 或性能标签。因此它测的是未缓存的级联推理，不能与只测性能分支的时间混在一起。数据准备、CPU/GPU 搬运、指标计算均在计时范围外；CUDA 路径在预热后和每次计时结束前同步。诊断不选择 checkpoint，不更新权重，并恢复模型的 train/eval 标志和 RNG 状态。

CUDA 记录预热后已分配显存、计时期间的峰值，以及峰值减预热后基线的增量。**这些是当前进程的显存统计**，可能包含驻留数据、共享 OP 缓存和其他模型，不是一个模型独立运行的显存占用。增量峰值也只表示相对于当前基线的新增分配，不能当成完整模型显存。CPU 执行时 CUDA 显存字段留空。

这是一个重复驻留批次的微基准，不代表整个 test 集的吞吐量，也不包含文件读取、数据转换或实际部署请求传输。训练阶段时间另外保存在成本表中，包括训练与验证，且会受早停轮数影响。当前工程验证不等于已运行完整 60,000 样本 GPU 实验；精度、实际集群 GPU 速度和显存表现仍以运行结果为准。

## 5. 选择电路、模型与运行

编辑 `final_exam/single_circuit_pin/train_low_rank_decoder_pin.sh` 顶部：

```bash
CIRCUITS=(two_stage_folded_opamp Opamp1 Peng_ACBC)
MODELS=(
    width_decoder_pin
    width_decoder_lowrank_attention_pin
    width_decoder_lowrank_ffn_pin
    width_decoder_lowrank_both_pin
)
SEEDS=(17 29 43)
```

模板默认只列二级折叠运放；上例会为三个电路分别提交单电路作业，不是 OOD。可以删减 MODELS，只为选择的模型训练和测试。Slurm 默认 8 CPU / 1 GPU / 80G、账户 `b_phzhwu`、分区 `ex01A800`、conda `newbase`。

在仓库根目录执行：

```bash
bash final_exam/single_circuit_pin/train_low_rank_decoder_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_low_rank_decoder_pin.sh
```

`--dry-run` 只打印 sbatch 命令，不创建输出目录。`--smoke` 缩短轮数，但不会自动抽少量数据。要先跑小样本，在新配置中设置 `dataset.max_samples_per_circuit=2000`；正式比较可改为 20000 或 `null`。抽样发生在 train/val/test 划分之前，同一个作业内四组共用抽出的数据。

原始数据仍为 `runs/raw/full_data/<电路>/`。宽度/长度单位不一致的原始输入仍会被严格拒绝；不要通过低秩配置或关闭检查绕过该错误。

若 `check_raw_geometry.py` 显示 `geometry_scale_to_m=1.0`、W/L 全部不一致、OP/input 比值为 `1e-6` 且 m 一致，可使用以下修复工具：

```bash
python scripts/run/fix_raw_geometry_scale.py \
  --input runs/raw/full_data/Leung_NMCNR runs/raw/full_data/Fan_SMC
```

它只处理缺失微米尺度声明的情况，修改前保存原网表的精确备份，添加或替换 `.option scale=1u`，随后重新检查；检查失败则恢复本次修改的全部网表。数据库不改。默认前后各检查 100 条，可用 `--max-samples 60000` 检查当前两个电路的全量样本。默认抽查通过不等于全量通过，训练导入仍严格检查所有选中的样本。

## 6. 结果、配置、日志与 Excel

所有产物都在当前仓库的 `final_exam/single_circuit_pin/` 中：

| 内容 | 路径 |
|---|---|
| 结果根目录 | `results_pin/<电路>_low_rank_decoder_<时间戳和唯一编号>_pin/` |
| HPC 输出、错误、提交配置快照 | `hpc_logs_pin/<相同run_id>/` |
| 导入、实验、训练与评估日志 | `logs_pin/<相同run_id>/` |
| 自动指标 Excel | `results_pin/<run_id>/training_pin/summary_pin.xlsx` |
| 模型配置、checkpoint、test 报告 | `results_pin/<run_id>/training_pin/seed_<seed>_pin/<模型名>/` |
| 每模型固定批次推理诊断 | `.../training_pin/seed_<seed>_pin/<模型名>/benchmark_pin.json` |
| 本 seed 的共享原始 OP predictor | `.../training_pin/seed_<seed>_pin/shared_pin/device_op_predictor_pin/` |

每个模型保存完整架构信息和适用的低秩设置，并在重建 checkpoint 时使用相同因子结构。Excel 保留 MAE、RMSE、R² 的 mean/std、PerSeed、配置、成本与宽读出架构。`LowRankDecoder` 表按模型/seed 记录 attention/FFN 是否分解、有效 rank、总参数与可训练参数、实际 readout 轮数和时间，并展示固定批次的级联延迟、延迟标准差、samples/s、CUDA 进程峰值和增量峰值。推理诊断协议同时写入 `benchmark_pin.json` 与 `test_pin.json`；没有测量的字段留空，不当作 0。

指标表中只有一个 seed 时，跨 seed 的 std 留空。`LowRankDecoder` 中的 latency std 是同一模型、同一批次重复前向 10 次的时间标准差，不是跨 seed 精度标准差；二者不能混用。

完成后若要重新整理表格：

```bash
bash final_exam/single_circuit_pin/summarize_pin.sh \
  final_exam/single_circuit_pin/results_pin/<run_id> \
  --output final_exam/single_circuit_pin/results_pin/<run_id>/summary_review_pin.xlsx
```

源码被 Git 管理，以上结果、checkpoint 与日志按已有 `.gitignore` 保持本地。不要在作业运行期间更新同一个 checkout 的代码。

## 7. 中断恢复与旧模型兼容

```bash
bash final_exam/single_circuit_pin/resume_low_rank_decoder_pin.sh \
  final_exam/single_circuit_pin/results_pin/<失败run_id>
```

恢复复用原样本、划分、配置、模型列表和 seeds，在新的结果目录继续，不修改失败目录。已完成的 OP predictor 和性能阶段复用已保存状态，未完成阶段使用最近完整 epoch 的 optimizer、RNG 和早停状态继续。一个 epoch 中尚未保存的部分会重新执行。

不能通过修改 rank、decoder 层数或模型名把原始完整权重 checkpoint 当成低秩模型续训。改变结构需要新实验。旧 `width_decoder_pin`、`width_decoder_op_pin` checkpoint 的结构和原有行为保持兼容；本组未新增“OP predictor 也低秩化”的实验变体。
