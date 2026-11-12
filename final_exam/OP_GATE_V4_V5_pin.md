# V4/V5：跳过 PE1 的双向调节门控

V4/V5 都只在 **第二个及后续参数融合块（PE2…PEn）的 Self-Attention** 上加入门控。结构编码层、PE1 Self-Attention、所有 Cross-Attention 和 decoder 不直接接收门控偏置。V4 在这些层之间共享一套 M-MLP；V5 每个启用门控的层各有一套独立 M-MLP。两个版本都能增强或抑制符合条件的注意力边。

这是端口级第一阶段单电路实现。旧版本和已有结果保留原义，不因拉取新代码变成 V4/V5。

## 为什么另开版本

V3 使用 `strength * log(epsilon + (1-epsilon) * sigmoid(z))`，其偏置非正，只能直接压低符合条件的边。V3 的 `fusion` 变体还会作用于 PE1。V4/V5 同时改变作用范围和输出映射，因此使用新模型 ID、新配置和新实验目录；不能把 V3 checkpoint 改名当成 V4/V5 的续训状态。

当 `encoder_layers=6`、`cross_layers=3` 时：

| 模块 | V4 | V5 |
|---|---|---|
| 前 3 层结构编码 | 无门控 | 无门控 |
| PE1（encoder 第 4 层）Self-Attention | 无门控 | 无门控 |
| PE2（encoder 第 5 层）Self-Attention | 共享门控 | 独立门控 1 |
| PE3（encoder 第 6 层）Self-Attention | 同一共享门控 | 独立门控 2 |

PE 编号是融合块内部的相对位置，不硬编码为 encoder 第 4/5/6 层。增加结构层不会移动“跳过 PE1”的语义。`model.cross_layers` 必须至少为 2，否则没有可训练的门控层，配置会被拒绝。报告中的作用范围为 `fusion_after_first_self_attention`。

## 门控输入、分数和初始化

器件 OP 预测器保持器件级并冻结。其预测 OP 广播到引脚，下游编码器仍为 pin 级 ZeroSim 框架。门控输入继续使用 V3 的：

```text
concat(训练集标准化 OP, 当前样本内逐字段 MOS 相对 OP,
       Query 引脚角色 one-hot[4], Key 引脚角色 one-hot[4])
```

输入维度为 `2F+8`。相对 OP 在当前样本有效 MOS 范围内按字段计算，不跨 batch 采样点做 BatchNorm，也不使用验证集/测试集拟合统计量。门控网络仍为三层 MoLinear，维度为 `2F+8 → dim → dim → heads`；默认每个 MoLinear 有 16 个专家，隐藏层 Dropout 为 0.1。每套网络为全部 attention heads 输出不同分数，不是每个 head 独立训练一套 OP 预测器。

新偏置为：

\[
b_{ij}^{(l,h)} = \alpha\tanh(z_{ij}^{(l,h)}),\qquad
A^{(l,h)}=\operatorname{softmax}(S^{(l,h)}+b^{(l,h)}).
\]

`S` 是原来的 QK 分数、角色偏置和拓扑掩码；`op_gate_pin.strength` 就是非负的 \(\alpha\)，默认 1，因而有限偏置落在 `[-1,1]` 内。正值提高该边的 logit，负值降低；未归一化 softmax 权重对应乘以 `exp(b)`。因为 softmax 同时归一化其他边，单边最终概率还取决于同行其他偏置。

最终专家层使用小随机权重和零偏置，从**接近无门控**开始，保留输入和路由的梯度。V4/V5 不接受旧的 `epsilon`、`initial_gate` 参数，避免把 V3 的 0.99 开放率误用于新公式。

V5 的各层整套参数（专家、router 等）互不共享。初始化时从同一份初始网络深复制，因此初值相同，之后各层独立更新。训练模式下，各层 Dropout 采样可以不同；评估时 Dropout 关闭。对于相同 `cross_layers`，V5 门控参数量约为 V4 的 `cross_layers-1` 倍；主干参数不因此增加。应按实际日志比较训练成本，而不是据参数量直接推断时间比例。

资格规则沿用旧版：仅调节**同一 MOS、不同实际网络**的有向引脚对，且器件有受支持的 OP 字段。D→S 与 S→D 独立评分；自环、实际短接、R/C、其他器件、外部端口和 GRAPH 保留原规则。有限增强偏置不能复活原拓扑硬掩码禁止的边。

## 四种训练方式

每版新增 4 个模型。下面的 `v4` 替换为 `v5` 即为另一版对应名称。

| 模型 ID | 前置过程 | 门控启用后的训练参数 |
|---|---|---|
| `device_op_gate_fusion_pin_v4` | 训练并冻结器件 OP predictor | encoder、decoder、门控联合优化 |
| `device_op_distill_gate_fusion_pin_v4` | 冻结真实 OP 教师 → 无门控学生 encoder 蒸馏 → 冻结 encoder 训练 decoder | 只训练门控 |
| `device_op_supervised_gate_fusion_pin_v4` | 无门控模型直接监督性能训练 | 只训练门控 |
| `device_op_distill_gate_decoder_fusion_pin_v4` | 同一个无门控蒸馏基座 | 门控＋decoder；encoder 仍冻结 |

OP predictor 在这些下游阶段都冻结，部署不依赖真实 OP 或教师。decoder 联合微调变体从已训练 decoder 继续训练，不随机重置。只训门控时虽然 encoder 参数冻结，门控经 encoder 影响最终输出的可微路径必须保留。

两种版本各自启动脚本默认含上述 4 个模型，再加 `device_op_cascade_pin` 无门控监督参考，共 5 个评测入口。同次运行内所需 predictor、教师和无门控基座共享，额外保存蒸馏基座的测试结果，避免把其收益错误归因于最后的门控阶段。

## 启动和配置

在项目根目录编辑对应脚本顶部的 `CIRCUITS`、`MODELS`、`SEEDS`：

```bash
# 只预览；不会提交或创建实验目录。
bash final_exam/single_circuit_pin/train_op_gate_pin_v4.sh --dry-run
bash final_exam/single_circuit_pin/train_op_gate_pin_v5.sh --dry-run

# 正式提交，每个电路一个独立 Slurm 作业。
bash final_exam/single_circuit_pin/train_op_gate_pin_v4.sh
bash final_exam/single_circuit_pin/train_op_gate_pin_v5.sh
```

默认电路 `two_stage_folded_opamp`，seeds 为 17/29/43，CUDA、8 CPU/1 GPU、80G 主机内存，使用原有 `newbase` 环境和集群配置。原始数据仍为 `runs/raw/full_data/<电路名>/`，不另建仓库副本。

配置分别是 `final_exam/configs/op_gate_pin_v4.json`、`op_gate_pin_v5.json`。门控配置仅有：

```json
"op_gate_pin": {
  "strength": 1.0,
  "norm_eps": 0.00001,
  "n_experts": 16,
  "mlp_dropout": 0.1
}
```

抽样数量仍由 `dataset.max_samples_per_circuit` 控制：`null` 全量，`20000` 表示先随机取本电路总计最多 20,000 条再划分。`dataset.sampling_seed` 控制抽样，与训练 seed 分开。

阶段最大轮数和 V3 模板一致：

| 配置项 | 默认上限 | 用途 |
|---|---:|---|
| `op_transfer.epochs.op_predictor` | 30 | 器件 OP 预测器 |
| `op_transfer.epochs.teacher` | 60 | 蒸馏需要的真实 OP 教师 |
| `op_transfer.epochs.readout` | 60 | 无门控监督基座及联合门控模型 |
| `op_transfer.epochs.predicted_op_distill_encoder` | 30 | 学生 encoder 蒸馏 |
| `op_transfer.epochs.predicted_op_distill_decoder` | 30 | 冻结学生 encoder 后训练 decoder |
| `op_transfer.epochs.gate` | 30 | 分阶段模型的最后门控/门控＋decoder 阶段 |

默认 `batch_size=32`、`lr=1e-4`、`patience=10`。上述是各阶段上限，早停可能提前结束；整条模型依赖链不一定与另一种训练方式总轮次相同，实际成本分阶段保存。`--smoke` 缩短轮次，不自动缩小数据量；小样本烟测请同时设置抽样数量。

## 结果、日志和恢复

三个目录均在 `final_exam/single_circuit_pin/` 内，run ID 分别包含 `_op_gate_v4_` 或 `_op_gate_v5_`，并以 `_pin` 结尾：

| 内容 | 路径 |
|---|---|
| HPC 标准输出、错误、提交配置快照 | `hpc_logs_pin/<run_id_pin>/` |
| 导入/训练/评估实验日志 | `logs_pin/<run_id_pin>/` |
| 数据、配置、训练产物 | `results_pin/<run_id_pin>/` |
| 最终 Excel | `results_pin/<run_id_pin>/training_pin/summary_pin.xlsx` |
| 每模型配置、checkpoint、测试报告 | `training_pin/seed_<seed>_pin/<model>/` |
| 共享 predictor、教师、无门控基座 | `training_pin/seed_<seed>_pin/shared_pin/` |

Excel 保留模型完整版本名、MAE/RMSE/R²、跨 seed 标准差和门控前后比较。配置、门控范围、版本及实际成本随结果保存。V5 checkpoint 保存每个启用层的独立门控参数。

失败恢复继续使用已有入口，保留源实验的模型、种子、配置与划分，在新目录中恢复：

```bash
bash final_exam/single_circuit_pin/resume_staged_op_gate_pin.sh \
  final_exam/single_circuit_pin/results_pin/<失败实验目录>
```

V4 和 V5 不能互相冒充恢复，同版本改变网络层数、专家数、门控范围等也不应继续旧优化器状态。新增模型请选择新实验。门控检查入口仍为 `inspect_op_gate_pin.sh`，选定新模型后应按实际应用的 PE 层读取分数；V5 各层分数不能合并成一套共享分数解释。

## 对照实验的解释边界

V3→V4 同时改变“跳过 PE1”和“单向抑制改双向偏置”，因此直接比较不能把全部差异只归因于其中一个因素。V4→V5 的主要差别是跨层共享与逐层独立，但参数量也随之变化。有限偏置、近零初始化和分段训练不是稳定收敛的保证；需要比较实际训练曲线、相同划分和多个 seed 的测试结果。
