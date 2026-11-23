# Graph-layer v4：分数保留梯度，仅 detach 浅层 PE 融合输入

模型名：`lasd_graph_layer_headwise_detach_pin_v4`。版本后缀沿用已有命名；名字包含 `detach`，但本版本 **不 detach 分数**，detach 的是除最后一层以外、送入加权融合的 PE 表示副本。

## 1. 与已有版本的区别

| 模型 | 分数输入 detach | 参与融合的 PE 表示 detach | Final Graph Encoder | 额外最后 PE 残差 |
|---|---|---|---|---|
| `lasd_graph_layer_headwise_pin` | 否 | 无 | 保留一层 | 无 |
| `lasd_graph_layer_headwise_detach_pin` | 是 | 无 | 保留一层 | 无 |
| `lasd_graph_layer_headwise_detach_pin_v1` | 是 | 全部 | 保留一层 | 无 |
| `lasd_graph_layer_headwise_detach_pin_v2` | 是 | 全部 | 移除 | 无 |
| `lasd_graph_layer_headwise_detach_pin_v3` | 是 | 全部 | 移除 | 有，保留梯度 |
| `lasd_graph_layer_headwise_detach_pin_v4` | **否** | **只 detach 前 L−1 层，最后一层保留梯度** | **保留一层** | **无** |

v4 以原始 graph-layer 架构为基础，不继承 v3 的残差或移除 Final Graph Encoder 的设置。旧模型和旧 preset 保持原含义。

## 2. 前向与梯度路径

设 PE 输出经原有共享输出 norm/projection 后为 `h_1,…,h_L`。分数分支仍接收每个节点各 PE、各 head 的 R、T、ΔR、ΔT，以及每图各 PE/head 的 mean R、std R、std T 与它们的差分。只统计 PE；沿用既有注意力 mask、有效节点规则、对角排除和首层差分零填充。

图上下文 Encoder、逐节点跨层 Encoder、输出 MLP 生成逐节点逐头的 L 个 logits，沿 PE 层维度 softmax 得到融合权重。输入分数不执行 stop-gradient，打分网络也保持可训练。

对每个节点和通道组（head），融合为：

\[
\widetilde h_i^{(h)}=
\sum_{l=1}^{L-1}\alpha_{i,l}^{(h)}\operatorname{sg}(h_{i,l}^{(h)})
+\alpha_{i,L}^{(h)}h_{i,L}^{(h)}.
\]

核心梯度边界可表述为：

```python
routing_input = score_features                       # 不 detach 分数
fusion_inputs = [h.detach() for h in pe_outputs[:-1]] + [pe_outputs[-1]]
fused = headwise_weighted_fusion(fusion_inputs, layer_weights)
memory = final_graph_encoder(fused, original_graph_mask)
prediction = metric_decoder(memory)
```

上面是结构示意，实际实现复用已有模块和 mask/padding 处理。没有新增单独的 Decoder，也没有在 Decoder 前额外相加最后一层 PE 输出。

三类梯度路径：

1. **保留分数路径**：性能损失 → Final Encoder → 融合权重 → 图/层打分网络 → 分数 → 各层注意力及主干。
2. **保留最后 PE 表示路径**：性能损失 → Final Encoder → 最后 PE 的融合输入 → 最后 PE → 前序 PE/SE 主干。
3. **截断浅层表示的直接融合路径**：性能损失 → 融合 → `h_1,…,h_(L−1)` 的副本停止反传。浅层输出的数值仍参与融合。

因此这不是冻结浅层 Encoder：浅层仍可通过第一、第二类路径获得梯度。PE1→PE2→…→PE_L 的主干从不 detach。`L=1` 时没有浅层副本可 detach，唯一的 PE 融合输入保留梯度。

## 3. 单电路启动与配置

沿用原脚本，不新增 `.sh`：

```bash
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --graph-layer-detach-v4 --dry-run
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --graph-layer-detach-v4 --smoke
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --graph-layer-detach-v4
```

`--graph-layer-detach-v4` 只选择 v4；保留脚本顶部的 CIRCUITS、SEEDS、CONFIG 和集群设置，与其他模型 preset 互斥。原 `--graph-layer-detach-variants` 仍只运行 v1/v2/v3。也可直接修改顶部：

```bash
MODELS=(lasd_graph_layer_headwise_detach_pin_v4)
```

然后无 preset 参数运行脚本。单电路默认配置仍为 `final_exam/configs/lasd_moe_pin.json`：

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `model.cross_layers` | 3 | PE 层数；前 L−1 层融合副本 detach |
| `model.dropout` | 0.1 | 主干、性能 Decoder、打分分支中的图/层 Encoder、Final Graph Encoder |
| `lasd_moe_pin.dropout` | 0.0 | 图/层打分分支的输入和输出投影 MLP |
| `op_transfer.epochs.op_predictor` | 30 | 器件级 OP 预测器训练上限 |
| `op_transfer.epochs.readout` | 60 | v4 性能训练上限；沿用早停 |

器件级 OP 预测器仍先训练、随后冻结；v4 性能阶段只使用预测 OP，不要求真实 OP。没有新增预训练阶段或额外标签。

输出目录沿用：

```text
final_exam/single_circuit_pin/results_pin/<run_id>/training_pin/seed_<seed>_pin/lasd_graph_layer_headwise_detach_pin_v4/
final_exam/single_circuit_pin/logs_pin/<run_id>/
final_exam/single_circuit_pin/hpc_logs_pin/<run_id>/
```

继续保存 config、checkpoint、test 指标、逐层梯度历史、分数/融合权重诊断及汇总 XLSX。三层 PE 时，架构契约记录 `score_detach=false`、`hidden_state_detach_scope=shallow_fusion_inputs_only` 和 `fusion_memory_detach=[true,true,false]`；XLSX 的 `Fusion detach by PE` 列逐层列出边界，不能仅凭模型名字判断。可使用现有 `resume_width_decoder_pin.sh` 对 **v4 自己的未完成 run** 续训；不同模型的梯度策略不是可互换的恢复配置，旧版本 checkpoint 不能当作 v4 的中断进度。

## 4. 多电路联合训练

在 `final_exam/multi_circuit/train_multi_pin.sh` 的 MODELS 中加入同一模型名。已提供独立完整配置和搜索空间：

```text
final_exam/multi_circuit/configs_pin/models/lasd_graph_layer_headwise_detach_pin_v4.json
final_exam/multi_circuit/configs_pin/search_spaces/lasd_graph_layer_headwise_detach_pin_v4.json
```

默认配置与原 graph-layer 模型保持相同的维度、训练预算及 dropout；默认网格仍为 `op_transfer.lr=[0.0001,0.0003]` × `model.dim=[64,128]`。其余联合数据划分、单 seed 搜索、正式多 seed 评估和汇总方法不变，见 [多电路实验说明](multi_circuit/README.md)。

## 5. 本次修改范围

- 独立注册 v4 架构契约，并在既有融合路径中指定浅层 detach 边界。
- 复用现有图/层打分 Encoder、Final Graph Encoder、性能 Decoder、训练与恢复逻辑。
- 为已有单电路脚本增加专用 preset，多电路增加独立配置和网格。
- 验证前向融合公式、分数/浅层/最后层梯度边界、可变 PE 深度、旧模型兼容性，以及启动参数传播。

v4 的研究变量是梯度路径；前向 detach 本身不改变数值。与相同权重的全梯度 graph-layer 模型相比，差异发生在浅层 PE 表示的直接融合反向路径，实际精度与训练稳定性需要实验结果判断。
