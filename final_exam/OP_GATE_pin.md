# 器件级 OP 预测引导的引脚级 attention：单电路实验

本目录扩展在 `final_exam/single_circuit_pin` 中运行。实验分为 A、B 两组，每组包含四个模型，共八个模型。第一阶段预测器采用器件级 ZeroSim-family 编码器，第二阶段性能模型采用引脚级 ZeroSim-family 编码器。两个阶段使用同一份器件身份、设计参数字段、OP 字段和训练集拟合的归一化参数。

这是仓库现有 ZeroSim-family 实现上的扩展，不声称逐行复现原论文全部实验。上游来源及各模型改造记录保存在 `final_exam/models/SOURCES_pin.json`。

## 1. 为什么单独增加这组实验

原有 `predicted_op_pin` 的 OP 预测器先运行 pin 编码器，再按器件池化其端口表示，输出一份器件 OP。因此它与这次要求的“第一阶段直接使用器件级图编码器”不是同一个结构。本次新增独立模型名称，不改变原有模型、checkpoint 或脚本的含义。

原 pin 图允许同一器件的端口交换信息；这种候选邻接关系不是对端口真实导通状态的判断。新增的门控将预测工作点用于调节这些信息交互的强弱。它是软 attention 重加权，不是删除网表导线，也不自动减少 dense attention 的 FLOPs。

两组实验的区别是：

- **A：OP 同时作为特征和控制信号。** 当前样本的预测 OP 被广播至其器件的各个 pin，作为原有 OP 融合模块的数值特征；门控模型还用 OP 生成 attention 偏置。
- **B：OP 只作为控制信号。** pin 特征融合模块输入设计参数，不额外输入 OP 特征 token；门控模型通过 OP 偏置间接使用工作点信息。

两组都保留原有拓扑、器件类型、物理 model 身份、端口角色以及设计参数信息。跨组比较不表示输入通道和参数量相同；组内比较用于检查添加门控后的变化。

## 2. 八个模型

| 模型名称 | 组别 | pin 数值特征中的 OP | attention 门控来源 | 推理需要真实 OP |
|---|---|---|---|---|
| `device_op_cascade_pin` | A | 当前样本预测 OP | 无 | 否 |
| `device_op_gate_pin` | A | 当前样本预测 OP | 当前样本预测 OP | 否 |
| `device_op_gate_oracle_pin` | A | 当前样本预测 OP | 当前样本真实 OP，缺失字段回退预测值 | 是 |
| `device_op_gate_shuffle_pin` | A | 当前样本预测 OP | 其他训练样本的预测 OP | 否；诊断对照 |
| `device_op_gate_only_base_pin` | B | 无 | 无 | 否 |
| `device_op_gate_only_pin` | B | 无 | 当前样本预测 OP | 否 |
| `device_op_gate_only_oracle_pin` | B | 无 | 当前样本真实 OP，缺失字段回退预测值 | 是 |
| `device_op_gate_only_shuffle_pin` | B | 无 | 其他训练样本的预测 OP | 否；诊断对照 |

A 组改变门控来源时，节点输入的预测 OP 始终不变。Oracle 不把真实 OP 替换到节点特征里。B 组无门控基线没有 OP 预测器依赖，也不计入 OP 预训练成本。

Oracle 是测量信息诊断对照，不是可脱离仿真的部署模型；其性能不保证形成数值上的上界。Shuffle 是破坏样本与 OP 对应关系的诊断对照，不作为最终部署候选。

## 3. 第一阶段：真正的器件级 OP 编码器

输入为器件级拓扑和每个器件的设计参数。编码器保持仓库 ZeroSim-family 的结构编码与参数交叉注意力融合框架，输出器件表示；OP 预测头为每个物理器件输出一份归一化 OP 向量。

pin 数据批次另行保留原始器件轴上的 `owner_*` 拓扑和参数张量。预测器使用这些原始 D 轴张量，而不是将 P 轴 pin 张量改名后作为器件输入，也不是在 pin 编码之后再池化。

监督目标在器件轴上计算。一个器件的一个 OP 字段只计一次，不随该器件的端口数量重复计权。只使用训练集支持的 OP 字段和类型/物理模型支持信息。归一化、可用字段和模型支持关系均由训练集拟合。

每个 seed 独立训练一个器件级预测器。在同一 seed 内，其选定 checkpoint 供所选模型复用。第一阶段结束后冻结参数并设置 `eval()`；下游性能损失不回传到该预测器。正常模型训练、验证和测试都使用预测 OP。只有两个名称带 `oracle` 的门控对照会读取真实 OP。

## 4. 第二阶段：共享 OP embedding 和逐 head 门控

设编码器隐藏维度为 `model.dim = d`，OP 字段数为 F，attention head 数为 H。

1. OP 数值和训练支持 mask 进入共享 MLP，形成 `d` 维 OP embedding。
2. 查询端口和键端口分别拥有 G、D、S、B 角色 embedding，维度同为 `d`。
3. OP embedding 与有序端口角色 embedding 相加，进入打分 MLP。
4. 打分 MLP 为每个有向端口对输出 H 个数值，经过 sigmoid 得到逐 head 门控。

对应的结构为：

\[
z_d^{OP}=\operatorname{MLP}_{OP}([\hat o_d,m_d])\in\mathbb{R}^{d},
\qquad
u_{d,i,j}=z_d^{OP}+e^Q_{r_i}+e^K_{r_j},
\]

\[
g_{d,i,j}^{(h)}=\sigma\bigl(\operatorname{MLP}_{score}(u_{d,i,j})_h\bigr),
\qquad
b_{d,i,j}^{(h)}=\lambda\log\bigl(\epsilon+(1-\epsilon)g_{d,i,j}^{(h)}\bigr).
\]

例如 `dim=512, heads=8` 时，共享 OP 表示是 512 维，一对端口的打分输出是 8 个标量。每个 head 有自己的输出参数，但不会复制八份完整 OP 预测器。OP embedding 维度始终由 `model.dim` 决定，不能独立设置成不匹配的宽度。

默认门控配置：

```json
"op_gate_pin": {
  "epsilon": 0.0001,
  "strength": 1.0,
  "initial_gate": 0.99
}
```

- `epsilon` 必须位于 0 与 1 之间，用于防止门控变成负无穷硬断路。
- `strength` 是固定非负偏置缩放系数；设置为 0 可以检查关闭偏置后的行为。
- `initial_gate` 必须位于 0 与 1 之间；默认从接近原模型的弱抑制开始，同时保留对 OP 的非零依赖。
- 门控网络与下游编码器、指标读出头一起更新。OP 预测器保持冻结。

## 5. 门控作用范围和保留关系

偏置加入全部编码器 self-attention 层的 softmax 之前，包括结构阶段和融合阶段，也包括局部与全局 attention。前面的结构层因此成为 OP 条件化结构编码，不能再称为仅依赖静态拓扑。

原来的设计/OP 特征 cross-attention 连接约束不变，原生指标查询读出不变。G、D、S、B 的有向门控无需对称。

门控仅覆盖同时满足以下条件的 pin 对：

1. 属于同一个 MOS；
2. 两端均为真实器件端口；
3. 两端连接到不同 net；
4. 该器件具有训练支持的 OP 字段。

同 net 导线连接、自环、graph token、外部端口和非 MOS 器件不施加这种偏置。这样，二极管连接的 G=D、B=S 等真实短接不会被错误抑制。原有全局 attention 仍允许其他信息通路；本方法不声称通过偏置实现了严格的图断连。

## 6. Oracle 与 shuffle 的信息边界

### Oracle

只允许门控读取当前样本的真实器件 OP。与预测 OP 使用同一个训练集归一化尺度，并受同一份训练字段支持限制。真实 OP 缺失时使用当前样本预测值，不额外输入真实观测 mask 作为特征。运行记录应保留真实 OP 使用覆盖率。A 组的节点 OP 特征仍然来自预测器。

### Shuffle

替换来源只包含训练样本的预测 OP。每次替换整份样本的器件 OP 数组，器件位置和字段顺序保持不变。通过 seed 与目标 sample ID 确定替换来源，训练样本不能选到自己。

验证集和测试集使用同一训练来源库，不读取其他验证/测试样本作为替换来源。因此同一样本的结果不依赖 batch 大小、排序或同时输入哪些测试样本。来源库、身份映射和随机种子随 checkpoint 保存；不足两个训练样本时不能构造这种排除自身的对照。

Shuffle 只修改门控输入。A 组输入融合模块的 OP 始终是当前样本的预测 OP。它用于检查工作点与设计样本的正确对应是否带来收益，而不是改变整个模型的输入分布后直接与主方法混为一谈。

## 7. 训练预算、选模和保存

专用配置为 `final_exam/configs/op_gate_pin.json`，默认：

| 项目 | 默认值 |
|---|---:|
| OP 预测器最大 epoch | 30 |
| 每个下游性能模型最大 epoch | 60 |
| batch size | 32 |
| learning rate | 0.0001 |
| weight decay | 0.0001 |
| early stopping patience | 10 |
| seeds | 17、29、43 |
| encoder dim / heads | 128 / 8 |
| encoder layers / fusion layers | 6 / 3 |
| decoder layers | 2 |

修改 `op_transfer.epochs.op_predictor` 和 `op_transfer.epochs.readout` 分别控制两个阶段。原 runner 的预算一致性检查仍保留；如果改变 `readout`，也需要按其要求保持 `student_predicted_op` 与 `predicted_op_distill_encoder + predicted_op_distill_decoder` 相等，即使本次未选择旧的蒸馏模型。

第一阶段按验证 OP 损失选择 checkpoint，第二阶段按验证性能损失选择 checkpoint。测试集不参与选模、归一化或早停。所有 epoch 都是最大值，实际执行轮数可能因早停更少。

30+60 是单个依赖 OP 的模型所涉及的最大阶段预算，不表示与无 OP 基线计算成本相同。一份共享预训练只实际训练和计费一次；各模型记录依赖 stage ID，汇总按唯一 stage ID 去重。B 组无门控基线只执行自己的下游训练。

每次运行保存有效配置、manifest、训练集归一化参数、器件级 OP 预测器 checkpoint、各下游模型 checkpoint、每轮历史、选中 epoch、实际 optimizer steps 和时间。模型文件包含重建所需的信息；共享依赖也独立保存，便于审查和复用。这里没有额外的性能蒸馏教师训练阶段。

冻结后的预测 OP 在每个 seed 下缓存一次，缓存按预测器权重摘要区分，训练、验证、测试分别按原始 sample ID/索引对齐。该过程只执行冻结模型前向，不读取 OP 真值；缓存准备时间单独作为零训练轮数的 `frozen_device_op_cache_pin` 阶段保存并去重。门控和下游编码器仍在每个训练批次重新计算，不能缓存会更新的门控结果。

器件级预测器独立保存在 `training_pin/seed_<seed>_pin/shared_pin/device_op_predictor_pin/checkpoint_pin.pt`。每个下游 checkpoint 同时包含自身重建需要的冻结预测器权重；两个 shuffle 对照还包含完整训练预测 OP 来源库，因此可脱离共享目录独立恢复。来源库会增加 checkpoint 体积和保存 I/O：例如 48,000 个训练样本、22 个器件、44 个 OP 字段，仅 float32 数值与布尔 support 约占 222 MiB，尚未包含模型权重。GPU 训练并不免除磁盘 I/O 成本。

## 8. 启动方式与文件位置

在项目根目录编辑：

```bash
final_exam/single_circuit_pin/train_op_gate_pin.sh
```

脚本顶部直接设置 `CIRCUITS`、`MODELS`、`SEEDS`、`CONFIG` 和集群参数。默认选取 Opamp1 的八个新模型，三个 seed。每个电路提交一个独立 Slurm 作业；只训练选中的模型及其必要依赖。

先预览，不提交作业，也不创建结果目录：

```bash
bash final_exam/single_circuit_pin/train_op_gate_pin.sh --dry-run
```

提交缩短训练预算的烟测：

```bash
bash final_exam/single_circuit_pin/train_op_gate_pin.sh --smoke
```

烟测只缩短训练轮数，仍然导入所选目录下的完整原始数据。要验证小数据流程，应把脚本的 `RAW_ROOT` 指向已准备好的小样本来源。

正式提交：

```bash
bash final_exam/single_circuit_pin/train_op_gate_pin.sh
```

默认申请 8 CPU、1 GPU、80G 内存，账户 `b_phzhwu`，分区 `ex01A800`，环境 `newbase`。CPU 仍用于原始数据读取、Python 控制和文件输出；数值训练使用指定 CUDA 设备，不声称所有 I/O 都在 GPU 上完成。

运行标识包含电路、`op_gate`、日期和唯一后缀，且以 `_pin` 结尾。对应位置：

| 内容 | 路径 |
|---|---|
| Slurm 标准输出/错误 | `final_exam/single_circuit_pin/hpc_logs_pin/<run_id_pin>/%j_pin.out`、`%j_pin.err` |
| 提交时配置快照 | 同一 HPC 目录的 `submitted_config_pin.json` |
| 导入与实验日志 | `final_exam/single_circuit_pin/logs_pin/<run_id_pin>/` |
| 原始导入结果与运行元数据 | `final_exam/single_circuit_pin/results_pin/<run_id_pin>/` |
| 训练结果、阶段、checkpoint | 上述结果目录的 `training_pin/` |
| 最终比较表 | `training_pin/summary_pin.xlsx` |

不会创建项目旁边的新 worktree 或复制一份项目再运行。以上运行输出目录沿用已有 gitignore 规则。

若需要重新导出表格：

```bash
bash final_exam/single_circuit_pin/summarize_pin.sh \
  final_exam/single_circuit_pin/results_pin/<run_id_pin> \
  --output final_exam/single_circuit_pin/results_pin/<run_id_pin>/comparison_pin.xlsx
```

## 9. XLSX 统计口径

输出包含 `MAE`、`RMSE`、`R2`、`PerSeed`、`Statistics`、`TrainingCost`、`Configuration`，选取新模型时额外包含 `ModelInputs`。

- MAE/RMSE/R² 按模型、指标、split 和电路分别统计，展示多 seed 均值和样本标准差（ddof=1）。单 seed 不伪造 std。
- `ModelInputs` 显示 A/B 组别、器件级 OP 预测器、OP 特征是否输入、门控来源、真实 OP 需求及诊断对照身份。
- Oracle 放在真实 OP 参考分组中，不能与无需真实 OP 的模型混用推理信息条件。
- Shuffle 列显式标注诊断对照，不参与最佳分数高亮。
- 其他高亮仅在 seed、标签数、推理输入条件可比时计算，不代表统计显著性。
- `TrainingCost` 按唯一 stage ID 展示实际成本，共享 OP 预测器不会重复加总。
- 有效配置、真实 OP 访问标记或成本依赖缺失时，正式完整汇总会报错；不把未完成实验伪装成完整比较。

## 10. 验证范围

运行前应验证原始几何单位一致，并确保 `model.max_devices` 容纳实际 pin 与命名外部端口的 token 总数；在 pin 模型中该上限不是物理器件数量。当前默认 200，超出时明确报错，不截断。

验证应覆盖：器件轴输入与规范 device batch 一致；真实 OP 与标签不进入普通模型；门控作用于全部指定 self-attention 层；同 net 连接不被抑制；OP 预测器冻结且门控有梯度；shuffle 不依赖 batch 顺序；oracle 缺失回退；checkpoint 独立重载；完整八模型运行与 XLSX 导出。

本说明不把合成数据单元测试等同于真实 60,000 点 GPU 实验验证，也不预先声称门控提高精度。实验结果需要由相同划分、seed 和预算的 A/B 对照给出。

本次执行的测试和限制见 [OP_GATE_VERIFICATION_pin.md](OP_GATE_VERIFICATION_pin.md)。
