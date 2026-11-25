# Graph v7：基于 v3 的多阶 Attention 统计

模型：`lasd_graph_layer_headwise_detach_pin_v7`。2026-11-25。

## 定义与梯度

对每个性能 PE 的每个 Head，取原始 Q/K、原始 Attention mask 和偏置生成的
softmax 权重 A（attention dropout 之前）。聚合距离 K 使用 A, A², …, A^K。
幂是**同一 PE、同一 Head 的矩阵乘法**，不是逐元素幂，也不是不同 PE 的 rollout。

- 保留自环、GRAPH token 和原有局部/全局可达关系计算矩阵幂。
- 对各阶结果统计 R/T 时才排除对角线；padding 的行和列在乘法前置零。
- 不对 A^k 重新 softmax、行归一化，也不重新施加一阶电路 mask。
- 每阶分别计算每节点各 Head 的 R/T、同阶跨 PE 差分，以及每图/PE/Head 的
  mean R、std R、std T 和这些统计的跨 PE 差分。第一 PE 的差分为边界零。
- global 的口径与 v3 一致：总体方差除以有效节点数，std=sqrt(var+epsilon)，
  包含 GRAPH、不包含 padding、不混合图。没有 mean T 输入或高斯分布拟合。
- 每阶节点输入宽度仍为 10LH；独立 MLP_k 映射到 model.dim。默认 L=3、H=8
  时每阶 240 维输入、128 维输出。每阶 MLP 在节点和样本之间共享。
- 各阶投影**直接相加**，不求平均、不 concat、不额外 LayerNorm，再进入一套
  原图上下文 Encoder、逐节点层 Encoder 和 H-logit MLP。不是复制 K 套 Encoder。
- 最终仍对每节点、每 Head 的 L 个分数做 softmax，融合原性能 PE 的 memory。

```python
features = []
for k in range(1, K + 1):
    # 各 PE 各 head 分别计算；实现逐图、分块观察，不保留全部幂矩阵。
    features.append(node_and_global_dynamics([A_l.matrix_power(k) for A_l in pe_attention]))
e = sum(MLP_k(z.detach()) for MLP_k, z in zip(order_mlps, features))
logits = original_graph_and_layer_encoders(e, original_local_circuit_mask)
alpha = logits.softmax(dim=-1)  # [B,N,H,L]
F = headwise_fusion([h.detach() for h in pe_memories], alpha)
prediction = original_metric_decoder(pe_memories[-1] + F, padding)
```

所有阶 Score 均停止梯度；实际观察器在 no_grad 中计算，避免构建无用的幂矩阵反向图。
所有融合 memory 副本 detach，最后 PE 原张量作为 live 残差。原 SE→PE 串行主干
不 detach。各阶 MLP、图/层上下文 Encoder、最终打分 MLP 和 Decoder 正常训练。
不增加 Final Graph Encoder、HC 或 Decoder 前 LayerNorm；器件 OP 预测器保持普通冻结版本。
路由图 Encoder 本身仍沿用 v3 的原 local mask/terminal-role bias。

K=1 使用原有分块统计和同序初始化，参数形状与 v3 相同；在同权重、同输入及同随机状态下
预测值和梯度一致。新增 MLP 在原公共参数初始化后独立初始化，K>1 不改变公共权重初值。
原 v3 checkpoint 不改名冒充 v7；不同 K 的参数/训练身份不同，不能直接续训到另一 K。

这里的距离表示加权游走步数，不是物理电路的最短路径距离。自环、GRAPH token 和全局 PE
都会影响可达性。R/T 是注意力权重统计，不表示实际物理信息传输量。

## 单电路配置与启动

继续编辑 `final_exam/configs/lasd_moe_pin.json` 中已有的 lasd_moe_pin 对象，增加或修改：

```json
"aggregation_distance": 2
```

必须为正整数；默认 2。K=1 是 v3 对照，K=2 用一阶+二阶，K=3 用一阶+二阶+三阶。
该设置仅影响 v7；旧模型不会因此新增分支。学习率、batch、epoch、采样和 split 均沿用原设置。
`model.cross_layers` 控制 PE 数 L，`aggregation_distance` 控制矩阵幂数 K，两者独立。

脚本仍是 `final_exam/single_circuit_pin/train_lasd_moe_pin.sh`，修改顶部 CIRCUITS、SEEDS 和集群设置后：

```bash
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --graph-layer-detach-v7 --dry-run
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --graph-layer-detach-v7
```

也可在原 MODELS 中填写下列对照，无 preset 参数运行同一脚本：

```bash
MODELS=(
    device_op_cascade_pin
    lasd_graph_layer_headwise_detach_pin_v3
    lasd_graph_layer_headwise_detach_pin_v7
)
```

继续使用原 results_pin/logs_pin/hpc_logs_pin。模型目录为新模型全名；config、checkpoint
和报告记录 K、全部阶数、求和位置、梯度与 mask 口径。summary_pin.xlsx 的 LASD_MoE
页增加聚合距离、阶数、投影 MLP 数和融合方式；MAE/RMSE/R²、参数量和训练时间沿用原流程。

续训仍用现有脚本，恢复原 K、配置、模型、seed、split 和优化器状态：

```bash
bash final_exam/single_circuit_pin/resume_width_decoder_pin.sh /absolute/path/to/old_run
```

## 多电路与网格

将新模型名加入 `final_exam/multi_circuit/train_multi_pin.sh` 的 MODELS 即可。
同名 JSON 已提供：

- `multi_circuit/configs_pin/models/lasd_graph_layer_headwise_detach_pin_v7.json`：默认 K=2。
- `multi_circuit/configs_pin/search_spaces/lasd_graph_layer_headwise_detach_pin_v7.json`：
  lr=[1e-4,3e-4]、dim=[64,128]、`lasd_moe_pin.aggregation_distance`=[1,2,3]，共12组。

每个 trial 的训练预算继承 v3；总 trial 数增加为 v3 模板的3倍。可缩减 grid。
验证集选超参数，选定配置再跑固定多 seed；不使用 test 挑 K。

## 诊断与计算开销

每个模型结果目录新增：

- `lasd_moe_power_scores_pin.csv`：全部阶的逐节点/PE/Head R/T 和差分。
- `lasd_moe_power_global_scores_pin.csv`：全部阶的逐图全局统计和差分。
- 两张表都用 `attention_order` 标记阶数。原 `lasd_moe_scores_pin.csv` 和
  `lasd_moe_global_scores_pin.csv` 保留一阶视图，便于与旧版本比较。
- `lasd_moe_weights_pin.csv`：求和后最终的一套 H×L 权重，不是每阶单独一套权重。
- diagnostics JSON 包含 `power_statistics`（含每阶 R/T 总量守恒误差）；history 的
  `SCORE_ORDER1..K` 记录各投影 MLP 的梯度范数。

仅导出有界验证样本，数量由 diagnostic_samples 控制；不存储全体样本的 N×N 矩阵。
K>1 逐图保留一份全 Head A，以 score_chunk_size 控制查询行块，递推行块@A。
空间不按 batch×K 堆积完整幂，但稠密乘法计算量仍随 K 和 N 增长
（高阶部分约 O(B H (K−1) N³)）。没有把多阶统计误写成廉价的逐元素平方。
建议先用 K=2 做 --smoke，观察实际显存和耗时，再扩大 K。
