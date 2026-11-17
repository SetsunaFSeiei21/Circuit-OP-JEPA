# LASD：宽 Decoder 的逐 Head 注意力分数读出

## 1. 最终规格与模型

本实现遵循确认后的 **1A、2B、3A**，不是最初文字中的“共享节点 MLP → Sum Pooling → 图级分支”版本。

| 模型 | 每个 PE 的 Decoder memory | Decoder 之间 |
|---|---|---|
| `width_decoder_pin` | 原来的该 PE 节点表示 | 独立性能 Query；所有分支输出拼接后进入原 MoE |
| `width_decoder_score_pin` | 节点表示 + 每个 Head 的 R、T，经过该 PE 独立 MLP | 同上 |
| `width_decoder_lasd_pin` | 节点表示 + 每个 Head 的 R、T、ΔR、ΔT，经过该 PE 独立 MLP | 同上 |
| `width_decoder_score_detach_pin` | 与 Score 模型相同；浅层表示与 R/T 在 MLP **之前** detach | 所有 MLP、Decoder、MoE 继续训练 |
| `width_decoder_lasd_detach_pin` | 与 Dynamics 模型相同；浅层输入 detach，差分的上一层 Score 停止梯度 | 同上；最深层表示/当前 Score 正常回传 |

新名称中的 score 指分数输入，不是“仅靠分数、丢弃节点表示”。LASD 指 Layer-wise Attention Score Dynamics。
第一版不加入门控、低秩参数化、共享下基、Attention Rollout、特征流量估计、强化学习或新的 Transformer Decoder。
原宽 Decoder、宽 OP Decoder 和低秩/共享下基模型的 ID、checkpoint 与默认行为保留。

## 2. 仓库实际架构与必要适配

这里的序列是 **GRAPH token + 原生引脚/外部端口 token**。MOS、其他器件引脚和外部端口均在原有序列内，不新增原始输入字段。
主干先经过 SE，再经过多个 PE；每个 PE 的执行顺序是：

1. 带原始拓扑 mask 和 terminal-role bias 的 self-attention；
2. 节点对本身设计参数、预测 OP token 的 cross-attention；
3. 原有 FFN/残差/LayerNorm；
4. 用原有共享 output_norm/output_projection 形成该 PE 的 Decoder memory。

本次只观察 **PE self-attention** 的 Q/K 输入与 mask。没有把示例里的简化 Encoder 直接替换进主干，因此保留了设计参数/OP 的融合。
SE 不参与 Score 差分；PE1 的差分直接填零，不使用最后一个 SE 的分数。第一层 PE self-attention 在该层参数融合之前，所以在固定拓扑下，PE1 的分数可能主要反映拓扑及训练时的 dropout，而不是样本的尺寸差异。

当前 mask **局部/全局交替且允许自连接**。例如 3SE+3PE 时，PE 的 mask 顺序为 global、local、global。
因此相邻层差分包含 mask 可连接范围变化的影响，不可以把 Δ 直接解释成纯粹的物理角色变化。每个 checkpoint 和诊断文件都会保存 mask 顺序。
本模型没有跨 PE 的节点池化、采样或重排，token 编号贯穿各层保持一致。

## 3. 数学定义和融合位置

记第 l 个 PE、第 h 个 Head 的 pre-dropout attention 为 A[l,h]，其中 A[i,j] 表示 Query i 对 Key j 的注意力。

```
R[l,h,i] = sum(j != i) A[l,h,i,j]
T[l,h,i] = sum(j != i) A[l,h,j,i]
dR[l,h,i] = R[l,h,i] - R[l-1,h,i]
dT[l,h,i] = T[l,h,i] - T[l-1,h,i]
```

第一层 dR/dT = 0，明确属于边界填充。R/T 是注意力权重统计，不是实际传输的特征量或电路物理电流。
按所选 2B，**不对 Head 求平均**，直接按相同 Head 编号相减。不同层同编号 Head 不保证语义相同，这是该变体的实验假设，不作语义对齐保证。

拼接顺序是：`[R0,T0,R1,T1,...,R(H-1),T(H-1)]`，Dynamics 在其后继续拼接 `[dR0,dT0,...]`。
每个 PE 的 adapter 独立训练：

```
Score Only: concat(PE_memory, R/T)        -> Linear -> ReLU -> Dropout -> Linear -> decoder memory
Dynamics:   concat(PE_memory, R/T, dR/dT) -> Linear -> ReLU -> Dropout -> Linear -> decoder memory
```

默认 D=128、H=8、adapter hidden=128：Score Only 为 **144→128→128**，Dynamics 为 **160→128→128**。
两层 Linear 之间有非线性；padding 输入在 MLP 前清零，输出在 MLP 后再次清零，避免 bias 让 padding 复活。
各 Decoder 仍使用各自独立的可学习性能 Query，不将上一 Decoder 输出用作下一 Query。
各 PE 的 Decoder 输出按同一指标对齐拼接，交给原来的 DenseMoEMLPPin。

Score adapter 的输出仅送入对应 Decoder，不写回下一 PE。原节点主干前向更新完全保留。
原 Score/Dynamics 模型的所有 Score 分支都有梯度，会通过 Q/K 和原 mask 中的可学习 bias 更新 Encoder。
新增 detach 变体仅切断浅层读出边，保留最深读出的梯度；精确规则见第 9 节。

## 4. Softmax、mask、图规模与成本

- 统计包含 GRAPH 和外部端口；GRAPH 在导出的 token_id 中单独标为 `GRAPH`。
- 对角线在 softmax **之后**排除，padding 的行和列均排除；不对剩余边重新归一化。
- 保留自连接时，在有效 query 行上 R=1-Aii；若某种 mask 不允许自连接且仍有有效 key，R=1。实现不人为制造 Receive 的变化，Transmit 仍保留。
- 不先对原始 R/T 或 Δ 做全图求和再预测；本变体用节点级非线性 MLP 和现有指标 Query Decoder。
- 保留原始分数、不做图规模归一化。诊断保存有效 token 数、逐 Head 的 R/T 标准差与全图 R/T 守恒误差，供检查大小偏差。
- Attention dropout 之后的权重不再严格满足行和为 1。为了保留原来的 dropout 和消息路径，Score observer 独立重算同一 Q/K/mask 的 **dropout 前** softmax，原 MHA 继续按原设置计算消息。不会简单使用训练时 `need_weights=True` 返回的 dropout 后权重作为守恒分数。
- 仓库实际使用 dense masked MHA，不是基于 edge_index 的稀疏算子。统计按 query 分块；默认开启 checkpoint，在反向时重算 Score softmax，减少保存的 N×N 中间量。没有保存完整 attention 矩阵。
- 分数计算增加 Q/K/softmax 计算和 adapter 参数，不能声称零成本或一定更快。GPU 训练、验证、统计和梯度重算均跟随输入设备，CSV 序列化才转 CPU。

## 5. 配置与启动

修改 `final_exam/single_circuit_pin/train_lasd_pin.sh` 顶部：

```bash
CIRCUITS=(two_stage_folded_opamp Opamp1)
MODELS=(
    width_decoder_pin
    width_decoder_score_pin
    width_decoder_lasd_pin
    width_decoder_score_detach_pin
    width_decoder_lasd_detach_pin
)
SEEDS=(17 29 43)
```

配置为 `final_exam/configs/lasd_pin.json`。新命名空间：

```json
"lasd_pin": {
  "enabled": true,
  "hidden_dim": 128,
  "dropout": 0.0,
  "score_chunk_size": 128,
  "checkpoint_scores": true,
  "diagnostic_samples": 8
}
```

`enabled=false` 时所有 Score 模型完整旁路分数提取与 adapter，不创建闲置 adapter 参数。
原 Score/Dynamics 模型此时等价于 `width_decoder_pin`；两个 detach 变体则等价于 `width_decoder_detach_pin`，仍保留浅层 memory 断梯度。
相同初始化下参数/前向/相应梯度策略一致。`width_decoder_pin` 始终是原对照，不受此开关启用影响。
`diagnostic_samples=0` 只关闭训练结束后的逐 Head 导出，不关闭用于预测的 Score 分支。
`score_chunk_size` 控制每次统计的 Query 数；`checkpoint_scores` 控制反向重算。两者不改变数学定义。
原始 Encoder、Decoder FFN 和 MoE 配置保持原宽模型默认值：Decoder FFN=2048、每个 PE Decoder 深度=2。

数据量继续用 `dataset.max_samples_per_circuit` 控制，例如 20000 表示先抽总量 20000 再划分；`sampling_seed` 控制抽样。
默认 OP 预测器最多 30 轮、性能模型最多 60 轮、batch_size=32、patience=10；实际轮数可能因早停缩短。
同一 circuit/seed 下这些模型复用同一冻结器件级 OP 预测器；性能端从头训练，数据划分、预处理、优化器、epoch 上限和验证选模规则一致。
新增 adapter 在原模块初始化之后构造，保证相同 seed 下原 Encoder、各 Decoder Query/FFN 和 MoE 的初始权重一致。

```bash
bash final_exam/single_circuit_pin/train_lasd_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_lasd_pin.sh
```

每个电路一个独立 Slurm GPU 作业，在同一仓库目录内运行；不创建外部 worktree。`--smoke` 缩短训练，数据量仍由配置控制。
可复用现有宽模型续训入口：

```bash
bash final_exam/single_circuit_pin/resume_width_decoder_pin.sh /完整路径/旧结果目录
```

续训读取原模型列表、seed、配置、manifest 和保存的训练状态，输出到新目录。已完成阶段复用，未完成阶段从保存的 epoch/optimizer/RNG/early-stopping 状态恢复；未保存的当前 epoch 部分重跑。修改 LASD 配置后须开启新实验，不能作为同一次实验续训。

## 6. 结果、日志与诊断

所有路径相对于本仓库：

```
final_exam/single_circuit_pin/results_pin/<电路>_lasd_<时间>_<标识>_pin/
  config_pin.json
  data_pin.json
  training_pin/
    summary_pin.xlsx
    seed_17_pin/
      shared_pin/device_op_predictor_pin/checkpoint_pin.pt
      width_decoder_lasd_pin/
        config_pin.json
        checkpoint_pin.pt
        test_pin.json
        width_audit_pin.json
        benchmark_pin.json
        lasd_scores_pin.csv
        lasd_diagnostics_pin.json
final_exam/single_circuit_pin/logs_pin/<运行名>/
final_exam/single_circuit_pin/hpc_logs_pin/<运行名>/
```

Score Only 目录有相同诊断文件。CSV 每行是一个 sample / PE / token / head，含 R、T、dR、dT、mask 类型、原 token 编号和第一层边界标志。
Score Only 为诊断也计算差分，但不把差分输入模型。诊断使用验证集选出的最终 checkpoint，在测试集上确定性选择至多指定数量样本，不参与 checkpoint 选择。
原 Score/Dynamics 模型仅在诊断序列化边界 detach；新增 detach 变体按第 9 节切断指定训练边。
诊断返回的原始 Score 保留计算图，CSV 序列化统一 detach；诊断数值和分数定义不因 detach 改变。
Excel 保留 MAE/RMSE/R² 的多 seed 均值±样本标准差，并增加 `LASD` 页，列出输入维度、adapter 参数、总参数、实际训练轮数/耗时以及整条部署级联的延迟诊断。
延迟只重复计时一个驻留测试 batch，包含冻结 OP 预测器、不使用训练时 OP 缓存；不含导入、主机传输和指标计算，不等价于整数据集吞吐量。CUDA 峰值是该进程含驻留数据的口径。

## 7. 修改范围与验证

- `device_jepa/model.py`：增加可选的 PE attention observer；默认路径不变。
- `final_exam/models/attention_score_pin.py`：独立 R/T 统计、分块 pre-dropout 提取、跨 PE 差分、节点 adapter。
- `final_exam/models/width_decoder_pin.py`：保留独立 Decoder/Query/MoE，在 memory 前插入新 adapter；checkpoint recipe 包含完整配置。
- `final_exam/lasd_registry_pin.py`：模型名、配置解析和结构元数据校验。
- `run_pin.py` / `stage_resume_pin.py` / `summarize_pin.py`：模型选择、配置保存、续训、参数与耗时统计、Excel 导出。
- `final_exam/lasd_diagnostics_pin.py`：限制样本数的逐 Head CSV 和元数据导出。
- `configs/lasd_pin.json` / `single_circuit_pin/train_lasd_pin.sh`：原对照、Score/Dynamics 及对应 detach 变体的同协议入口。

测试覆盖行/列方向、对角线排除、padding、批量隔离、全图守恒、不允许 self 时的 Receive 退化、逐 Head 拼接、第一层边界、节点重排拒绝、跨层梯度、Q/K 与可学习 bias 梯度、bool/float mask、全掩码行、不同分块大小、checkpoint 重算、原主干 dropout/RNG/输出/梯度不变、冻结 OP、标签/真实 OP 无泄漏、保存重建和中断续训、配置一致性及 Excel/脚本入口。
GPU 专用测试仅在有 CUDA 时执行；小规模验证不代替真实电路上的精度、运行时间和显存实验。

## 8. 结果解释边界

如果新模型提升，首先能支持“节点注意力统计辅助宽 Decoder 读出有效”；不能直接据此证明物理信息流、解决过平滑或每层 Head 语义相同。
Score Only 与 Dynamics 均增加节点 MLP；前者相对 baseline 的提升也可能包含新增容量/读出变换的贡献。后续可加入相同 MLP 但将 score 固定为零的容量对照。当前第一版未加入该额外模型。
当前没有 GPU 实测精度结论，后续比较应以固定划分、多 seed、同电路/同指标的汇总为准。

## 9. Score / Dynamics 的浅层读出 detach 变体

### 改什么，以及为什么放在这里

沿用 `width_decoder_detach_pin` 的定义：有 L 个 PE 时，只切断 PE1 至 PE(L−1) 各自的读出捷径，PE(L) 保持连接。
Score 模型不能只 detach 节点表示 h：R/T 是由 Encoder 的 Q/K 计算的，仍有另一条梯度路径。
也不能直接 detach Score MLP 的输出：那会同时停止浅层 Score MLP 的训练。
因此同时 detach 浅层 h 和分数输入，然后再执行**可训练的** Score MLP 和 Decoder。

记 `sg(x)` 为前向数值不变、反向梯度为零，`s_l` 为按原顺序展开的逐 Head R/T：

```
width_decoder_score_detach_pin:
  l < L: adapter_l(concat(sg(h_l), sg(s_l))) -> decoder_l
  l = L: adapter_L(concat(h_L, s_L))         -> decoder_L

width_decoder_lasd_detach_pin:
  delta_1 = 0  # 边界填充
  delta_l = s_l - sg(s_(l-1)), l >= 2
  l < L: adapter_l(concat(sg(h_l), sg(s_l), sg(delta_l))) -> decoder_l
  l = L: adapter_L(concat(h_L, s_L, delta_L))            -> decoder_L
```

LASD 最后一层差分使用 `s_L - sg(s_(L-1))`，从而不通过差分项直接更新上一层的 Score 路径。
这不阻止 Encoder 的正常深度反传：最深的节点表示和当前层 Score 仍依赖前面的 SE/PE，早期 Encoder 仍可沿主干收到梯度。
一共只有一个 PE 时没有浅层读出、没有非零差分；两个新模型分别与对应非 detach 模型的前向和反向一致。

OP 预测器仍用原器件级 `device_op_predictor_pin`，先训练再冻结。没有给 OP 预测器加 LASD、宽 Decoder 或 detach。
新模型参数量、初始化顺序、Attention mask、dropout、Score 定义、Decoder Query 和 MoE 与各自的非 detach 对照相同。
相同权重与相同 RNG 下前向输出相同；训练后的权重和结果可以因为梯度路径不同而改变。
这不是冻结浅层 Encoder，也不是让浅层 Decoder 不训练；所有 Score MLP、性能 Decoder 和 MoE 都训练。

### 怎么比较

仍修改同一个 `train_lasd_pin.sh`，不新增 Bash 入口。例如只比较 Dynamics 的 detach：

```bash
MODELS=(width_decoder_lasd_pin width_decoder_lasd_detach_pin)
```

若比较四个带分数模型：

```bash
MODELS=(
    width_decoder_score_pin
    width_decoder_score_detach_pin
    width_decoder_lasd_pin
    width_decoder_lasd_detach_pin
)
```

可在同一列表加入 `width_decoder_pin` / `width_decoder_detach_pin` 形成不带分数的配对对照。
使用原 `lasd_pin.json`，不需要额外 detach 超参数；模型名明确决定梯度策略。
默认 30 轮 OP predictor + 每个下游模型 60 轮 readout 的上限和早停保持不变，且共享阶段每个电路/seed 只训练一次。
新增模型从头训练；旧的非 detach 实验不能直接当作该模型的同阶段断点恢复。

### 配置、结果和恢复

原有 `results_pin` / `logs_pin` / `hpc_logs_pin` 结构与运行名规则保持不变。
每个模型仍保存有效配置、checkpoint、测试指标、逐 Head CSV、耗时/参数量和完整结构说明。
checkpoint 的 `performance.memory_gradients` 与 `performance.lasd.score_gradients` 记录切断的 PE 序号、切断位置、差分的上一层梯度策略，以及 adapter/decoder 是否冻结。
Excel 的 `WidthDecoder` 页记录 memory 策略；`LASD` 页新增 `Readout gradient policy` 和 `Previous score in delta`，便于识别对照。
禁用 Score 时会记录实际使用的普通 memory detach 策略。
重建、汇总和断点恢复均验证模型名/配置/梯度策略的一致性，旧模型的结构元数据格式不变。

### 验证覆盖

新增测试覆盖多个 PE 深度、相同初始化/参数量/前向/dropout RNG、一层 PE 的反向等价性、浅层 memory 与 Score 梯度为零、差分不能绕回上一层、Score MLP 与 Decoder 仍训练、完整 Encoder 深度链仍可反传、OP predictor 冻结、禁用 Score 后等价于普通 detach、checkpoint 严格重建、诊断/Excel、完成阶段复用与中断 epoch 精确恢复。
CUDA 专项测试在有 GPU 的环境执行；CPU 测试通过不代表实际电路精度或 GPU 性能已经验证。

2026-11-17 本次验证环境为 PyTorch 2.6.0 CPU：新增梯度/训练恢复测试与启动器检查 28 项通过、2 项 CUDA 测试跳过；已有 LASD、宽 Decoder、普通 detach、低秩及汇总/计时相关回归测试 122 项通过、3 项 CUDA 测试跳过。共 150 项通过，5 项因无 CUDA 跳过。尚未运行完整电路训练。
