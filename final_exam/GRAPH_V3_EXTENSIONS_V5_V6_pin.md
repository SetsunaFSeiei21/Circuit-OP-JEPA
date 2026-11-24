# Graph-layer v5 / v6：基于 v3 的两个独立变体

2026-11-25。两者都从 `lasd_graph_layer_headwise_detach_pin_v3` 扩展，分别测试性能 PE 内的 Hyper-Connection 和 Decoder 输入的 LayerNorm。v5 采用 **PE 内部 HC**；不在 Decoder 前引入新的 HC 融合器。

## 模型定义

| 模型 | 性能 PE 内部 | Decoder 前处理 | Final Graph Encoder |
|---|---|---|---|
| `lasd_graph_layer_headwise_detach_pin_v3` | 原有残差 | `H_L + F` | 无 |
| `lasd_graph_layer_headwise_detach_pin_v5` | 已有 Hyper-Connection | `H_L + F` | 无 |
| `lasd_graph_layer_headwise_detach_pin_v6` | 原有残差 | `LayerNorm(H_L + F)` | 无 |

`H_L` 是最后一个性能 PE 的读出 memory，`F` 是图/层上下文路由得到的逐头加权融合表示。默认三个 PE 时，`H_L=H3`。实际实现按 `model.cross_layers` 取最后 PE，不硬编码三层。

三个版本的分数输入都 detach；参与加权融合的全部 PE memory 副本都 detach；最后 PE 的残差原张量不 detach。SE→PE1→…→PE_L 的串行主干从不 detach。路由的输入/输出 MLP、图上下文 Encoder、层上下文 Encoder 和 Decoder 正常训练。图路由继续使用原来的 local mask/可学习 terminal-role bias。

```python
logits = graph_layer_router(score_features.detach(), original_local_mask)
fusion_memories = [h.detach() for h in pe_memories]
F = headwise_weighted_fusion(fusion_memories, logits.softmax(dim=-1))

# v5：pe_memories 已由 PE 内部 HC 生成。
decoder_memory = pe_memories[-1] + F

# v6：原有 PE；只新增下面这一层。
decoder_memory = layer_norm(pe_memories[-1] + F)
decoder_memory = decoder_memory.masked_fill(padding[..., None], 0.)
prediction = original_metric_decoder(decoder_memory, padding)
```

## v5：HC 的具体接入位置

直接复用 `models/hyper_connection_pin.py` 的 `HyperConnectionPEBlockPin` 与 `HyperConnectionPin`，与已有 `lasd_moe_headwise_detach_hc_pin` 的 HC 规则一致，但保留 **v3 的图/层上下文路由、全部融合副本 detach、无 Final Graph Encoder，以及最后 PE 残差**。

- 只包装性能模型的 PE。每个 PE 的 self-attention、参数/预测 OP cross-attention、FFN 分别有独立 HC 参数。
- SE、器件级 OP 预测器、分数图上下文 Encoder、层上下文 Encoder 和 Metric Decoder 不增加 HC。
- `lasd_moe_pin.hyper_connection_streams` 控制全维残差流数量，默认4；每条流宽度仍是 `model.dim`，不是把128维切成4份。
- 复用静态系数加缩放 `tanh` 动态系数、初始 identity carry、旋转读取流、初始 write=1、动态投影零初始化、动态系数初值0.01。系数不经过 softmax。
- 每个 PE 内按原规则保留各流 post-norm。PE 的读出是各流求和，再经过原共享 output norm/projection，得到同一节点顺序的 `H_l`。这里的 **sum readout** 与已有 HC 版本一致。
- Attention Score 统计来自该 PE 实际执行 self-attention 的 Q/K 和原 mask，不额外平均残差流的虚构 Attention，也不改变 score 统计的 head 轴。
- 当仅设置1条流时，初始化对应原 PE 残差；默认4条流不是与 v3 完全相同的前向初始化。

HC 参数通过最后 PE 的 live 残差以及主干串行路径训练；分数和融合值的 detach 边界不会被绕过。没有新增数据监督或训练阶段。

## v6：LayerNorm 的具体定义

`nn.LayerNorm(model.dim, eps=1e-5, elementwise_affine=True)`，位置严格在 **残差求和之后、原 Metric Decoder 之前**。每个样本、每个节点独立在特征维归一化，例如128维，不跨节点或样本计算统计量。初始化 gamma=1、beta=0。归一化后重新将 padding 置零，避免训练后的 beta 使填充节点出现非零 memory。

这不是对两个分支分别归一化，也不是额外的 PE。D=128 时，相比 v3 只增加256个可训练参数。分数统计、路由和 OP 预测器不变。

## 单电路启动

继续使用 `single_circuit_pin/train_lasd_moe_pin.sh` 和 `configs/lasd_moe_pin.json`。修改脚本顶部的 `CIRCUITS`、`SEEDS` 和集群配置后：

```bash
# 核对提交参数，不提交、不创建结果目录。
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --graph-layer-detach-v5-v6 --dry-run
# 每个电路提交一个作业，依次训练两个新模型。
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --graph-layer-detach-v5-v6
```

也可分别使用 `--graph-layer-detach-v5` 或 `--graph-layer-detach-v6`。所有 preset 互斥；仅覆盖模型列表，不覆盖电路、seed、配置路径或集群设置。旧 preset 的含义不变。

若要把 v3 一并作为对照，顶部设为下列模型后**无 preset 参数**启动：

```bash
MODELS=(
    lasd_graph_layer_headwise_detach_pin_v3
    lasd_graph_layer_headwise_detach_pin_v5
    lasd_graph_layer_headwise_detach_pin_v6
)
```

```bash
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh
```

训练阶段仍由 `op_transfer.epochs.op_predictor/readout` 控制。相同 run 和 seed 的普通器件级 OP 预测器及冻结缓存可复用；不会自动把 v3 已训练权重当作新版本初始化。

## 多电路、保存和恢复

`multi_circuit/configs_pin/models/` 与 `search_spaces/` 已提供两个新模型的同名 JSON，默认配置、网格和训练预算复制自 v3。修改 `multi_circuit/train_multi_pin.sh` 的 MODELS 即可进行联合训练和网格搜索；协议、采样、数据划分及验证集选参方式沿用原流程。

结果与日志继续使用原 `results_pin`、`logs_pin`、`hpc_logs_pin` 目录，完整模型名作为子目录。每个模型保存配置、预处理、架构契约、checkpoint、MAE/RMSE/R² 和训练成本；checkpoint 包含所需的 HC 或 LayerNorm 状态。

`summary_pin.xlsx` 的 LASD_MoE 表记录 PE 残差流数量、HC 的作用范围、分数/融合 detach、Final Encoder 数和 Decoder 输入归一化。v5 的梯度历史中 PE block 包含 HC 参数；v6 另列 `DECODER_INPUT_NORM1`；两者均无 `FINAL1`。

恢复使用原脚本：

```bash
bash final_exam/single_circuit_pin/resume_width_decoder_pin.sh /绝对路径/原结果目录
```

架构契约版本9描述两个新增模型；之前16个 LASD 模型的契约保持不变。完成阶段可以跳过，未完成阶段恢复 optimizer/RNG/epoch。新版本只能恢复同名同配置的 checkpoint；不把 v3 checkpoint 冒充 v5/v6。

## 验证范围

测试覆盖：HC 仅包装性能 PE、1流初始化与 v3 的前向/RNG 一致、3/4流与不同 PE 深度、准确的 score/fusion detach 边界、最后 PE 残差梯度、v6 的 post-sum LayerNorm、padding 置零、模型导出重建、统计表、单电路/多电路训练与续训、启动 preset 和旧版本兼容性。小型合成数据测试不代表真实电路精度收益。
