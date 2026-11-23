# 图/层上下文逐头融合：融合表示 detach 的三个对照版本

2026-11-23。新增模型基于 `lasd_graph_layer_headwise_detach_pin`，复用器件级 OP 预测器、端口级 SE/PE、节点/全图分数、图/层上下文路由和原性能 Decoder。没有引入新数据输入、性能标签、门控、MoE 专家或额外训练阶段。

## 1. 模型名称和区别

| 模型 | 分数输入 detach | 融合使用的全部 PE 表示 detach | Final Graph Encoder | Decoder 前的最后 PE 残差 |
|---|---|---|---|---|
| `lasd_graph_layer_headwise_detach_pin`（原模型） | 是 | 否 | 保留一层 | 无 |
| `lasd_graph_layer_headwise_detach_pin_v1` | 是 | 是 | 保留一层 | 无 |
| `lasd_graph_layer_headwise_detach_pin_v2` | 是 | 是 | 移除 | 无 |
| `lasd_graph_layer_headwise_detach_pin_v3` | 是 | 是 | 移除 | 最后一层 PE 原始输出，保留梯度 |

编号按最终确认的顺序：v1 只改变融合输入的梯度；v2 在 v1 上移除 Final Graph Encoder；v3 在 v2 上增加残差。这里的版本号属于 **graph/layer 路由模型**，不等同于已有 `lasd_moe_headwise_detach_pin_v1` 或 OP gate 的版本。

原模型的所有模块、初始化顺序和架构契约不变。新增三个名字分别保存独立结果，不替换原名字的含义。

## 2. 共用的输入和路由

第一阶段仍训练普通器件级 `device_op_predictor_pin`。性能阶段冻结该预测器，把同一器件的预测 OP 广播给所属引脚，输入原端口级编码器。部署时不需要真实 OP。

SE 前缀后依次执行 PE1…PE_L。只收集各 PE 的 self-attention 分数和输出；不收集 SE 分数，不改变 SE→PE 或 PE→PE 的前向连接。PE 输出先经过原有共享输出 norm/projection，记作 `h_1…h_L`。

对每个有效 token、PE、head 保留 R、T、ΔR、ΔT；全图部分为每样本/PE/head 的 μR、σR、σT 和它们的跨 PE 差分。统计沿用原 mask、排除 attention 对角，不重新归一化，不跨样本混合。首层差分为边界零。GRAPH token 按原约定参与统计，padding 不参与统计。

完整分数为每节点 `10×L×H` 维，在进入输入 MLP 前统一 detach。默认 L=3、H=8，为240维。随后仍为：

1. 输入 MLP 将完整分数轨迹映射到 D 维。
2. L 个图上下文 Encoder 使用原 local 电路 mask，保留每层输出。
3. 按节点将 L 个图上下文输出组成层序列，一层层上下文 Encoder 做 L×L 注意力。
4. 共享输出 MLP 对每个层 token 生成 H 个 logits，重排为 `[B,N,H,L]`，沿 L 做 softmax。

路由部分没有 MoE 专家，目录/配置仍沿用 LASD-MoE 家族的名称。其两段 Encoder 与 MLP 在三个新版本中都正常训练。

## 3. 融合和残差的确切位置

以下是三个版本的核心差异，实际实现复用原模块：

```python
# 共用：仅分支取用的副本断梯度。
score_input = score_features.detach()
logits, _ = layer_fusion(score_input, valid, original_local_mask)
fusion_values = tuple(h.detach() for h in pe_memories)
fused, alpha = _fuse_pe_memories_pin(fusion_values, logits, padding)

# v1
decoder_memory = final_graph_encoder(fused, original_local_mask, None, None, valid)
# v2
decoder_memory = fused
# v3
decoder_memory = pe_memories[-1] + fused

prediction = decoder(decoder_memory, padding)
```

每一 head 的 alpha 在 L 个 PE 上归一化，分别加权其对应 D/H 维通道，再拼回 D 维。默认 H=8、D=128，每组16维。这里的连续通道组来自输出投影后的 memory，不保证等于原 attention 输出投影之前的原生 head 子空间。

v3 是逐元素相加，不是 concat；没有新增投影、缩放、LayerNorm 或图 Encoder。最后 PE 残差使用 `pe_memories[-1]` 的原张量，**不 detach**。该 PE 的另一份副本在融合支路中仍然 detach。

v2/v3 的 `final_graph_encoder=None`：不创建该层，不把它计入参数量或 state_dict，也不生成虚构的 `FINAL1` 梯度记录。所有版本仍只有一个原有 Metric Decoder，其深度由 `model.decoder_layers` 决定。

## 4. 梯度路径及训练含义

| 路径/模块 | v1 | v2 | v3 |
|---|---|---|---|
| 损失 → 权重 → 原始分数 → 原 PE Attention | 阻断 | 阻断 | 阻断 |
| 损失 → 加权融合值 → 原 PE memory | 阻断全部 PE tap | 阻断全部 PE tap | 阻断全部 PE tap |
| 新图/层上下文 Encoder、输入/输出 MLP | 训练 | 训练 | 训练 |
| Final Graph Encoder | 训练 | 不存在 | 不存在 |
| 原 Metric Decoder | 训练 | 训练 | 训练 |
| 损失 → 最后 PE 残差 → PE/SE 主干 | 无此支路 | 无此支路 | 保留 |
| 冻结的 OP 预测器 | 不更新 | 不更新 | 不更新 |

**v1/v2 没有性能损失通向原 SE/PE attention、FFN 和 PE 参数 cross-attention 的梯度路径。** 它们用于检验同时切断分数与融合表示梯度的影响，不是“主干仍然端到端训练”的版本。启动流程保持从头初始化性能主干，不会自动加载原 graph-detach 的已训练主干，也不会增加未声明的预训练。

这不等于整个 `encoder` 对象被设为 `requires_grad=False`：原 terminal-role bias 是 local mask 的可学习部分，新图路由仍直接使用它，v1 的 Final Graph Encoder 也使用它，所以该 bias 仍可收到梯度。原主干仍按正常 train/eval 模式执行，dropout 保持原规则。报告的 Trainable parameters 仍按 `requires_grad` 统计；实际哪些 block 收到梯度请查看梯度历史，不能仅用该参数数目判断。

v3 通过未 detach 的最后 PE 残差恢复主干训练。虽然 PE1/PE2 的融合 tap 被 detach，PE1→PE2→…→PE_L 的串行计算从未 detach，因而早期层仍能从最后 PE 回传梯度。这与在编码器各层之间执行 detach 不同。

默认 L=3；实现支持配置中的实际 L。L=1 时层 softmax 恒为1，层选择本身没有可学习空间，路由梯度可能为零。

## 5. 启动和配置

沿用 `final_exam/single_circuit_pin/train_lasd_moe_pin.sh`，没有新增训练 bash 或实验任务目录。先修改原脚本顶部的 `CIRCUITS`、`SEEDS` 和集群/Conda 设置。

一次运行新增三组：

```bash
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --graph-layer-detach-variants --dry-run
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --graph-layer-detach-variants
```

该 preset 只覆盖 MODELS 为 v1/v2/v3，不覆盖电路、seed、CONFIG 或集群设置；与其他模型 preset 互斥。`--dry-run` 不提交作业，不创建输出目录。正式命令按电路各提交一个作业，同一电路内按配置执行模型与 seed。

若要同时比较原模型，或仅选其中一组，在脚本顶部设置 MODELS，**无模型 preset 参数运行**：

```bash
MODELS=(
    lasd_graph_layer_headwise_detach_pin
    lasd_graph_layer_headwise_detach_pin_v1
    lasd_graph_layer_headwise_detach_pin_v2
    lasd_graph_layer_headwise_detach_pin_v3
)
```

```bash
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh
```

配置仍为 `final_exam/configs/lasd_moe_pin.json`，本次未修改其默认值：

| 配置 | 作用 |
|---|---|
| `model.cross_layers` | PE 数 L；同时决定图路由 block 数、层 token 数和融合槽位数 |
| `model.encoder_layers` | 原 SE+PE 总层数，必须大于 cross_layers |
| `model.dim / heads / ffn_dim / dropout` | 原编码器及新上下文 Encoder 的基本宽度/头数/FFN/dropout |
| `model.decoder_layers / decoder_ffn_dim` | 唯一 Metric Decoder 的层数和 FFN 宽度 |
| `lasd_moe_pin.hidden_dim / dropout` | 路由输入/输出 MLP 的中间宽度和 dropout |
| `op_transfer.epochs.op_predictor` | OP 阶段最多轮数；仓库模板30 |
| `op_transfer.epochs.readout` | 性能阶段最多轮数；仓库模板60，沿用验证早停 |
| `dataset.max_samples_per_circuit / sampling_seed` | 先按电路抽样，再划分；null 取全部 |

模型名决定 detach、Final Graph Encoder 和残差策略，不需要额外布尔开关。配置中的原 OP 预测器结构不因这三个模型名改变。修改共用 `model` 配置仍会同时影响 OP 预测器和性能模型。

## 6. 结果、日志、独立加载和续训

继续保存到：

```text
final_exam/single_circuit_pin/results_pin/<run_id>/training_pin/
    summary_pin.xlsx
    seed_<seed>_pin/<完整模型名>/
    seed_<seed>_pin/stages_pin/<完整模型名>/history_pin.json
    seed_<seed>_pin/shared_pin/device_op_predictor_pin/checkpoint_pin.pt
final_exam/single_circuit_pin/logs_pin/<run_id>/
final_exam/single_circuit_pin/hpc_logs_pin/<run_id>/
```

完整模型名的 `_pin_v1`、`_pin_v2`、`_pin_v3` 会原样保留，不再追加 `_pin`。各模型保存配置、预处理、checkpoint、test 指标和架构审计，仍输出 MAE、RMSE、R²、参数量、训练时间和验证诊断。OP 预测器在同一次实验同一 seed 内共享训练/cache，单个性能模型的 checkpoint 也包含部署所需状态。

`LASD_MoE` 表页已有字段直接记录 Fusion memories detached、Final encoder layers、Final PE residual、Hidden-state detach scope。架构版本7另外记录 `encoder_chain_detach=false`、`residual_position` 和 `final_pe_residual_detach`；v2/v3 的 final encoder/mask 都记录为 none。旧模型契约不改。

梯度 history 中 v1/v2 的 SE/PE block 应无梯度，v3 通常有梯度；三组都有 SCORE_GRAPH/SCORE_LAYER 记录。只有 v1 有 FINAL1。这些是整个 block 的梯度范数，不是路径贡献度的因果估计。

使用原恢复入口：

```bash
bash final_exam/single_circuit_pin/resume_width_decoder_pin.sh /绝对路径/原结果目录
```

恢复相同模型自己的状态：完成阶段跳过，未完成阶段恢复 epoch/optimizer/RNG/最佳权重。新版本契约和参数布局严格校验；本次没有实现把原版本 checkpoint 当作 v1/v2/v3 的热启动输入。

## 7. 修改位置和验证范围

| 文件 | 修改 |
|---|---|
| `lasd_moe_registry_pin.py` | 注册三组模型，保存各版本精确梯度边界、最终层和残差契约 |
| `models/lasd_moe_pin.py` | 复用原融合 detach 和残差路径，按契约实际构造/跳过 Final Graph Encoder |
| `lasd_moe_diagnostics_pin.py` | 只记录实际存在的 Final Graph Encoder |
| `summarize_pin.py` | 汇总说明兼容零层 final encoder 和 decoder 前残差，统计口径不变 |
| `single_circuit_pin/train_lasd_moe_pin.sh` | 新增 `--graph-layer-detach-variants`，其他 preset 和默认设置不变 |
| `tests/test_graph_fusion_detach_pin.py` | 融合方程、残差 Jacobian、梯度边界、可变 PE 深度、参数/初始化、训练/导出/完整恢复 |
| 原 LASD 测试 | 新注册模型进入部署/缺失 OP/中断恢复测试；原图模型测试继续检查原契约 |

检查重点：v1 与原模型相同权重/RNG 下前向一致；v2/v3 参数相同且确实移除 Final Graph Encoder；v3 最后 PE 残差的 Jacobian 为恒等；padding 不产生融合输出；路由和 decoder 可训练；冻结 OP 不被性能阶段更新；原12个 LASD 架构契约保持不变。训练和恢复使用小型合成电路验证，不代表真实电路上的精度或收敛收益，后者需运行用户实验。
