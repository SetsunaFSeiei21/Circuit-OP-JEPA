# LASD Graph v8 / v9：多阶分数 + PE Hyper-Connection

2026-11-25。复用 v7 的多阶统计与 v5 的 HC，不改变旧模型定义。

| 模型 | 性能 PE | 分数路由 | Decoder 输入 |
| --- | --- | --- | --- |
| `lasd_graph_layer_headwise_detach_pin_v7` | 原生 PE | A..A^K，各阶 MLP 后求和 | H_last + F |
| `lasd_graph_layer_headwise_detach_pin_v8` | v5 Hyper-Connection PE | 与 v7 相同 | H_last + F |
| `lasd_graph_layer_headwise_detach_pin_v9` | 与 v8 相同 | 与 v8 相同 | LayerNorm(H_last + F) |

F 是逐节点、逐 Head 的 PE memory 加权融合。所有融合 memory 副本均 detach，
最后 PE 的原 memory 提供 live 残差。所有阶分数 detach；SE→PE 主干不 detach。
两版都只有原来的一个 Metric Decoder，无 Final Graph Encoder。

## HC 的位置和统计来源

直接复用 `HyperConnectionPEBlockPin`：每个性能 PE 的 Self-Attention、参数/OP
Cross-Attention、FFN 各自拥有独立 HC mixer。保留多条 D 维残差流，在子层前动态混合读出，
子层后写回并沿用原 post-norm。默认 4 条流，与 v5 一致；不同 PE 之间继续传递全部流。
每个 PE 对流求和，再经过原共享 output norm/projection，得到该层的 H_l。

分数观察器接收的是 HC **实际送入该 PE Self-Attention 的读出向量**，使用同一组 Q/K、
同一 mask/偏置计算 dropout 前 A。不是用流的总和或原 SE 输出另算一份不对应的 Attention。
v7 的矩阵幂、自环、GRAPH、padding 和逐阶局部/全局统计规则完整保留，详见
[v7 定义](GRAPH_V7_ATTENTION_POWERS_pin.md)。

HC 仅位于性能 PE。SE、冻结器件 OP predictor、打分图 Encoder、逐节点层 Encoder、
Metric Decoder 都保持原模块。打分分支仍是各阶独立 MLP 后相加，再共用一套图/层 Encoder。

v9 的额外 LayerNorm **位于残差相加之后、Decoder 之前**，与 v6 一致：
`LayerNorm(model.dim, eps=1e-5, elementwise_affine=True)`。只沿单节点的特征维归一化，
不混合样本或节点；归一化后再次清零 padding。不是分别归一化 H_last 和 F 后再相加。
相同 seed 下 v8/v9 的公共权重初始化相同，v9 额外增加 2×model.dim 个参数。

K=1 时，v8 的参数、前向和梯度对齐 v5；v9 则是 v5 再加上述 LayerNorm。
K=1 的 v9 不等于 v6，因为 v6 没有 PE HC。

## 配置与启动

继续修改 `final_exam/configs/lasd_moe_pin.json` 中已有的 `lasd_moe_pin` 区块：

```json
"aggregation_distance": 2,
"hyper_connection_streams": 4
```

两个值都必须是正整数。K 控制 A..A^K 的阶数和独立输入 MLP 数，streams 控制 HC 残差流数；
它们与 `model.cross_layers`（性能 PE 数）及 `model.heads` 相互独立。

在原 `train_lasd_moe_pin.sh` 顶部设置 CIRCUITS、SEEDS 和集群环境后运行：

```bash
# 比较两版；每个电路一个作业，内部按原流程执行模型和 seeds。
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --graph-layer-detach-v8-v9 --dry-run
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --graph-layer-detach-v8-v9
```

单独运行用 `--graph-layer-detach-v8` 或 `--graph-layer-detach-v9`。
也可以直接把完整模型名写入 MODELS，无 preset 参数运行原脚本。
原 HPC/实验日志、结果目录、采样/划分、学习率、epoch 和早停流程不变。

多电路：在 `final_exam/multi_circuit/train_multi_pin.sh` 的 MODELS 选择新模型。
`configs_pin/models` 和 `configs_pin/search_spaces` 已提供两版同名配置；默认 K=2、streams=4，
grid 为 lr=[1e-4,3e-4] × dim=[64,128] × K=[1,2,3]，每版12组，与 v7 相同。

## 结果、恢复和诊断

配置、checkpoint、test/val JSON 记录完整 K、HC streams/作用位置及 LayerNorm 契约。
Excel `LASD_MoE` 页同时记录聚合阶数、MLP 数、残差流数、HC 位置和 Decoder 归一化。
逐阶 R/T、差分、全局统计及最终 H×L 权重沿用 v7 的诊断 CSV。
梯度日志包含性能 PE（含 HC）、各阶 MLP、两类打分 Encoder；v9 另含 `DECODER_INPUT_NORM1`。

恢复原实验仍使用：

```bash
bash final_exam/single_circuit_pin/resume_width_decoder_pin.sh /absolute/path/to/saved_run
```

原配置、模型、K、streams、seed 和 optimizer 状态一起恢复。v7/v8/v9 或不同 K/streams
是不同架构，不能改 checkpoint 名冒充另一版续训。完整 GPU 成本与精度由实际实验测量；
多阶稠密乘法与 HC 都会增加计算或存储，建议先用 `--smoke` 检查集群运行。
