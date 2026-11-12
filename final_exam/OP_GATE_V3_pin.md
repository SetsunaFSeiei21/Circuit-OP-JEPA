# 门控 v3：样本内相对 OP 与 FlowNet M-MLP

v3 保留 v2 的门控输入与归一化，把评分网络改成三层混合线性层（MoLinear）。它研究的是更大、具有输入依赖专家混合的评分网络是否有助于性能预测和有序引脚对区分。实现本身不证明门控震荡已经解决，也不把门控分数解释成物理导通概率。

## 来源与适配边界

参考 FlowNet 官方实现，固定到提交 `57ad6128c2a1980d0aa73325a844659262271797`：

- [MoLinear 定义](https://github.com/Yutong-Feng/FlowNet/blob/57ad6128c2a1980d0aa73325a844659262271797/src/layers/mo_linear.py)。
- [FlowBlock.build_mixer](https://github.com/Yutong-Feng/FlowNet/blob/57ad6128c2a1980d0aa73325a844659262271797/src/blocks/hyper_flow.py)。
- [官方模型配置](https://github.com/Yutong-Feng/FlowNet/blob/57ad6128c2a1980d0aa73325a844659262271797/conf/model/flownet.yaml)。

这里使用独立 PyTorch 实现表达其数学运算，不引入整个 FlowNet、einops、时间序列数据接口或其 flow conservation 模块。每一层的计算为：

```text
alpha(x) = softmax(router(x), dim=experts)
expert_e(x) = x @ W_e + b_e
MoLinear(x) = sum_e alpha_e(x) * expert_e(x)
```

每层路由器读取该层自己的输入；所有专家参与软混合，没有 top-k、容量丢弃或负载均衡辅助损失。三层各有独立路由器与专家参数。专家权重之和为 1；专家数与 attention head 数是两个不同设置。

官方公开代码在 MoLinear 中使用 Xavier 初始化，虽然论文附录文字写的是 Kaiming。本实现以固定版本代码的数学结构和隐藏层初始化为依据，另对最后一层实施门控用途的初始化：每个专家 bias 设为 `logit(initial_gate)`，权重为小幅、独立、非零随机值，使初始门控接近 `initial_gate=0.99`，且保留专家差异。这属于门控适配，并非不加修改地复现整个 FlowNet。

## 输入、结构与作用范围

设训练支持的 OP 字段数为 F，`D=model.dim`，`H=model.heads`：

```text
x = concat(global_normalized_OP, sample_relative_OP,
           one_hot(query_role, 4), one_hot(key_role, 4))
x: 2F+8

MoLinear(2F+8, D, E)
  -> Dropout(p) -> GELU
  -> MoLinear(D, D, E)
  -> Dropout(p) -> GELU
  -> MoLinear(D, H, E)
  -> sigmoid
```

以 F=41、默认 D=128、H=8、E=16 为例，网络为 `90→128→128→8`。F 从保存的字段列表读取；D、H 随模型配置变化。门控隐藏维度不取 `model.ffn_dim`。Dropout 仅在门控训练模式启用，验证和推理时关闭。

相比 v2 的 `Linear(90,128)→GELU→Linear(128,8)`，v3 同时增加线性专家、一个隐藏层及 Dropout。默认评分网络参数约 472,656，v2 为 12,680；这是门控网络的参数数，不是完整模型大小或训练时间比。因此直接 v2/v3 对比不能将全部差异只归因于专家混合，后续可补层数、宽度或参数量匹配对照。

输入与使用规则延续 v2：

- 全局 OP 预处理仅用训练集拟合；样本内相对 OP 在每个样本的全部有效 MOS 上逐字段计算，NMOS/PMOS 一起统计，不混合不同 batch 样本。
- 真实器件槽位、MOS 类型、训练字段支持决定相对统计的有效集合。支持位不作为 M-MLP 额外输入；无效或不支持值安全填零，恒定字段或不足两个有效器件时相对通道为零。
- OP 预测器仍是器件级 ZeroSim，预训练后冻结；下游仍为引脚级 ZeroSim 框架。普通 v3 模型不读取验证/测试真实 OP 来替代预测值或构造缺失特征。
- 查询、键角色分别使用 D/G/S/B 独热编码，有序拼接；D→S 与 S→D 不强制相等。
- 同一个模型只有一个共享门控网络，一次产生 `[batch, device, 4, 4, H]` 的 logits，然后复用于选定的 encoder Self-Attention 层。16 个专家并不代表 16 个注意力头，也不代表每层各自训练一个门控。
- 只有同一 MOS 内、不同实际电气网络的有序引脚对接收附加偏置。R/C、不同器件、外部端口、graph token、自环和同网连接的附加偏置为零；这不删除它们原有注意力。
- 偏置仍为 `strength * log(epsilon + (1-epsilon) * sigmoid(logits))`。结构/融合块的 Cross-Attention 和性能 decoder 不插入门控。

## 八个模型与无门控对照

| 模型 ID | 训练方法 | 门控层范围 |
|---|---|---|
| `device_op_gate_pin_v3` | 冻结 predictor，联合训练 encoder、decoder、门控 | 结构层＋融合层 Self-Attention |
| `device_op_gate_fusion_pin_v3` | 同上 | 仅融合层 Self-Attention |
| `device_op_distill_gate_pin_v3` | 真实 OP 教师→蒸馏 encoder→冻结 encoder 训练 decoder→冻结二者只训练门控 | 结构层＋融合层 Self-Attention |
| `device_op_distill_gate_fusion_pin_v3` | 同上 | 仅融合层 Self-Attention |
| `device_op_supervised_gate_pin_v3` | 关闭门控监督训练 encoder＋decoder→冻结二者只训练门控 | 结构层＋融合层 Self-Attention |
| `device_op_supervised_gate_fusion_pin_v3` | 同上 | 仅融合层 Self-Attention |
| `device_op_distill_gate_decoder_pin_v3` | 教师→蒸馏 encoder→训练 decoder→冻结 encoder，同时训练门控＋decoder | 结构层＋融合层 Self-Attention |
| `device_op_distill_gate_decoder_fusion_pin_v3` | 同上 | 仅融合层 Self-Attention |

默认同时选择已有 `device_op_cascade_pin`，不创建虚假的 `device_op_cascade_pin_v3`。蒸馏路线还会自动保存并评估 `device_op_distilled_base_pin`；监督分段使用 `device_op_supervised_base_pin`。相同 seed、配置下的共享阶段去重，Excel 中门控前后比较使用各自真实无门控基座。

新增的 `distill_gate_decoder` 最后阶段使用 `op_transfer.epochs.gate`（默认 30），与只调门控的变体共用同一个蒸馏无门控基座；OP predictor 和 encoder 冻结，decoder 保留基座权重后继续训练，不重新初始化。前面的 decoder 预训练仍由 `predicted_op_distill_decoder` 控制。完整流程、六个新增名称及续训规则见 [GATE_DECODER_VERSIONS_pin.md](GATE_DECODER_VERSIONS_pin.md)。

## 配置和启动

编辑 `final_exam/single_circuit_pin/train_op_gate_pin_v3.sh` 顶部的 `CIRCUITS`、`MODELS`、`SEEDS`。默认一个电路 `two_stage_folded_opamp`、上述八个 v3 加无门控基线、三个 seeds `17 29 43`。需要只训练部分模型时直接删减 `MODELS`，依赖阶段自动补齐。

训练配置为 `final_exam/configs/op_gate_pin_v3.json`。JSON 不支持 `//` 或 `#` 注释。

| 配置位置 | 设置内容 |
|---|---|
| `model` | 编码器维度、heads、层数与 decoder 架构 |
| `op_transfer` | epoch、batch size、学习率、早停和进度频率 |
| `dataset.max_samples_per_circuit` | 每电路抽样总量；`null` 全量，`20000` 先抽最多 20,000 条再划分 |
| `dataset.sampling_seed` | 抽样种子；与模型训练 seed 区分 |
| `op_gate_pin.n_experts` | 每个 MoLinear 的专家数，默认整数 `16` |
| `op_gate_pin.mlp_dropout` | 两个门控隐藏层后的 Dropout 概率，默认 `0.1` |
| `op_gate_pin.norm_eps` | 样本内相对 OP 方差的数值保护，默认 `1e-5` |
| `op_gate_pin.epsilon/strength/initial_gate` | attention 偏置变换与初始门控强度 |

默认保持 `encoder_layers=6`、`cross_layers=3`，即 3 层结构编码＋3 层融合；融合层数量由 `cross_layers` 决定。主模型的 `dropout` 与门控的 `mlp_dropout` 分开配置。

```bash
bash final_exam/single_circuit_pin/train_op_gate_pin_v3.sh --dry-run
bash final_exam/single_circuit_pin/train_op_gate_pin_v3.sh
```

`--dry-run` 仅预览，不提交、不创建输出目录。`--smoke` 会提交短训练，样本量仍由 JSON 控制。每电路独立提交，默认 8 CPU/1 GPU/80G、账户 `b_phzhwu`、分区 `ex01A800`、Conda 环境 `newbase`。使用已有的 Slurm wrapper，通过提交端导出的项目根目录解析路径，避免从 `/var/spool` 拼错项目位置。训练张量使用请求的 CUDA，数据 IO 和静态准备仍需要 CPU。

最大阶段预算与 v2 相同：predictor 30、teacher 60、学生 encoder 蒸馏 30、decoder 30、直接监督/联合下游 60、独立门控 30。早停 patience=10、batch_size=32。教师和 predictor 是额外成本，四种训练方法并非总计算预算相等；结果记录实际轮数和去重阶段成本。

## 结果、HPC 日志与实验日志

run ID 含 `<circuit>_op_gate_v3_<timestamp>_..._pin`，所有目录仍在当前 repo 内：

| 内容 | 路径 |
|---|---|
| 实验结果 | `final_exam/single_circuit_pin/results_pin/<run_id>/` |
| 实验日志 | `final_exam/single_circuit_pin/logs_pin/<run_id>/` |
| HPC stdout/stderr | `final_exam/single_circuit_pin/hpc_logs_pin/<run_id>/<job_id>_pin.out`、`<job_id>_pin.err` |
| 提交配置快照 | `final_exam/single_circuit_pin/hpc_logs_pin/<run_id>/submitted_config_pin.json` |
| 自动 Excel | `<结果目录>/training_pin/summary_pin.xlsx` |
| 每模型配置、权重、报告 | `<结果目录>/training_pin/seed_<seed>_pin/<完整模型ID>/` |
| 教师 | `<结果目录>/training_pin/seed_<seed>_pin/shared_pin/design_op_oracle_pin/checkpoint_pin.pt` |
| 器件 OP 预测器 | `<结果目录>/training_pin/seed_<seed>_pin/shared_pin/device_op_predictor_pin/checkpoint_pin.pt` |

提交时复制配置，因此修改原 JSON 不改变已提交作业的配置。这个机制不复制代码或原始数据快照。每模型保存完整配置和门控结构配方，区分 v2/v3、专家数、Dropout、输入维度、共享范围；Excel 继续展示 MAE、RMSE、R² 的 mean±std 和实际训练成本。

```bash
# 重新汇总结果
bash final_exam/single_circuit_pin/summarize_pin.sh <结果目录>

# 恢复中断的同一个 v3 实验
bash final_exam/single_circuit_pin/resume_staged_op_gate_pin.sh <失败实验目录>

# 检查已完成模型的门控：先在脚本顶部修改 SOURCE_RUN、CIRCUIT、MODELS、SEED、N_SAMPLES
bash final_exam/single_circuit_pin/inspect_op_gate_pin.sh
```

恢复沿用源实验配置、模型身份和样本名单，写入新目录并保留原实验。已完成阶段复用；中断阶段恢复最近完整 epoch 的权重、优化器、随机数和早停状态，未完成 epoch 重跑。v2 门控 checkpoint 或 optimizer 不能改名为 v3 续训；改变专家数、Dropout 或层数属于新实验。

门控检查工具继续通过实际模型前向收集 `gate.score_head` 输出，保存逐 head 分数、实际 bias、相对 OP 与模型结构元数据。评估 v3 时应使用匹配的样本名单、划分、训练 seed 和阶段预算，并比较单个 head、不同引脚对及样本间变化，不能只看跨 head 平均热力图。

## 交付验证

2026-11-12 在 PyTorch 2.6.0 CPU 环境运行 v3、v2、v1、原始/分阶段门控、恢复、探针和启动脚本的组合回归：134 项测试，其中 132 项通过、2 项 CUDA 专用测试因没有 GPU 跳过。`bash -n` 和 `git diff --check` 通过。

验证包括独立专家循环的双精度前向/梯度等价、CPU BF16、非零路由梯度、实际注意力作用边界、六模型冻结规则、测试 OP 污染后预测不变、七选项小样本训练及 Excel 汇总。中断恢复测试恢复优化器与 Dropout 随机数后，与连续训练权重完全一致；改专家数、Dropout 或版本会在恢复写入前报错。门控探针直接检查实际网络输出及器件/引脚映射。

以上是软件正确性验证，未在 A800 上执行训练，也未获得 v3 的真实电路性能结果。
