# V6/V7：在 V4/V5 门控输入中显式加入 MOS 类型

V6 对应 **V4＋NMOS/PMOS 两维 one-hot**，V7 对应 **V5＋同一类型编码**。本次只扩展门控输入；评分网络结构、偏置公式、门控作用层、OP 处理和四种训练流程均沿用对应旧版。旧版模型、配置和启动脚本保留原义。

## 输入增加了什么

V4/V5 的门控通过器件类型判断“是否为 MOS、能否施加门控”，但评分网络本身没有显式接收 NMOS/PMOS 的区别。网络只能从 OP 数值间接学习极性线索。V6/V7 把所属物理器件的类型直接拼接到每个有向端口对的输入末尾：

\[
x_{d,i,j}=\operatorname{concat}\bigl(
z_d,\tilde z_d,\operatorname{onehot}(r_i),
\operatorname{onehot}(r_j),\operatorname{onehot}(t_d)\bigr).
\]

| 输入 | 维度 | 内容 |
|---|---:|---|
| `z_d` | F | 冻结器件级 OP predictor 输出的、训练集标准化空间中的 OP |
| `relative_op` | F | 当前样本所有有效 MOS 上逐字段计算的相对 OP |
| query 引脚角色 | 4 | D/G/S/B，有序端口对的第一项 |
| key 引脚角色 | 4 | D/G/S/B，有序端口对的第二项 |
| MOS 类型 | 2 | 固定语义：NMOS=`[1,0]`，PMOS=`[0,1]` |

类型依据保存的器件类型词表与当前物理 owner 对齐，不根据 OP 符号、器件 ID 字母或词表整数顺序猜测。类型编码不是可学习 embedding，同一 MOS 的所有端口对使用同一份类型编码。D→S 与 S→D 的 query/key 位置仍不同。

输入从 `2F+8` 扩为 `2F+10`；当前 41 个 OP 字段时从 90 维变成 92 维。M-MLP 为 `2F+10 → dim → dim → heads`，每个 MoLinear 的专家数和 Dropout 沿用旧配置。只增加第一层因输入变宽所需的参数。

**类型 one-hot 不参与标准化。** 相对 OP 继续对当前样本全部有效 MOS 一起统计，不按 NMOS/PMOS 分组，不引入 BatchNorm，不使用验证集或测试集拟合统计量。OP 字段集合、训练支持规则、缺失处理、原始 OP 通道和相对 OP 通道均沿用 V4/V5。新增类型输入不要求仿真 OP，部署仍只依赖拓扑、设计参数和预测 OP。

## V6 与 V7 的区别

| 项目 | V6 | V7 |
|---|---|---|
| 对应旧版 | V4 | V5 |
| 结构层 Self-Attention | 不加门控 | 不加门控 |
| 第一融合层 PE1 Self-Attention | 不加门控 | 不加门控 |
| PE2 及后续融合 Self-Attention | 共用一套门控 | 每层独立一套门控 |
| 各层门控初始化 | 同一套参数 | 同一初始网络深复制，初值相同、存储独立 |
| 输出 | 逐头的有符号偏置 | 逐头的有符号偏置 |

默认 `encoder_layers=6`、`cross_layers=3` 时，前三层为结构编码，后三层为 PE1/PE2/PE3，只有 PE2/PE3 直接接收门控偏置。PE 编号是融合块的相对位置；改变结构层数量不会改变此规则。`cross_layers` 至少为 2。

偏置保持：

\[
b_{d,i,j}^{(l,h)}=\alpha\tanh\bigl(f_{\theta_l}(x_{d,i,j})_h\bigr),
\qquad A=\operatorname{softmax}(S+b).
\]

`strength` 是 \(\alpha\)，默认 1，偏置范围为 `[-1,1]`。正值增强、负值抑制 softmax 归一化前的权重；最终 attention 概率还取决于同行其他边。输出保持近零初始化，隐藏维度、专家数和 attention heads 不变。V6/V7 继续不使用旧的 `epsilon` 或 `initial_gate` 配置项。

连接保护继续沿用旧版：只直接调整**同一 MOS、不同实际网络**的有向引脚对，要求有受支持 OP；实际短接、自环、R/C、其他器件、外部端口、GRAPH 不因此获得额外偏置。有限正偏置不能打开原拓扑硬掩码禁止的边。所有 Cross-Attention 和 decoder 均不直接注入该门控偏置。

## 八个新模型

| 训练方式 | V6 模型 ID | V7 模型 ID |
|---|---|---|
| encoder、decoder、gate 联合监督训练 | `device_op_gate_fusion_pin_v6` | `device_op_gate_fusion_pin_v7` |
| 蒸馏基础模型后，只训练 gate | `device_op_distill_gate_fusion_pin_v6` | `device_op_distill_gate_fusion_pin_v7` |
| 无门控监督基础模型后，只训练 gate | `device_op_supervised_gate_fusion_pin_v6` | `device_op_supervised_gate_fusion_pin_v7` |
| 蒸馏基础模型后，训练 gate＋decoder | `device_op_distill_gate_decoder_fusion_pin_v6` | `device_op_distill_gate_decoder_fusion_pin_v7` |

所有策略先训练器件级 OP predictor，然后在下游冻结。蒸馏策略需要真实 OP 教师训练、学生 encoder 对齐、冻结 encoder 的 decoder 训练；这些步骤继续先完成无门控基础模型。最后的只训 gate 策略冻结 encoder 和 decoder，gate＋decoder 策略冻结 encoder、继续微调已有 decoder。教师仅在训练需要时使用，部署不读取教师或真实 OP。

每个启动脚本默认选本版本四个新模型，加已有 `device_op_cascade_pin` 无门控监督参考，共五个评测入口。原有蒸馏无门控基座仍在汇总中报告。同次运行的共享 predictor、教师和基座按原流程复用。无门控模型没有新增 `_v6`/`_v7` 别名。

## 启动与设置

在项目根目录，修改相应脚本顶部的 `CIRCUITS`、`MODELS`、`SEEDS` 和 `CONFIG` 后运行：

```bash
# 预览：不提交、不创建输出目录。
bash final_exam/single_circuit_pin/train_op_gate_pin_v6.sh --dry-run
bash final_exam/single_circuit_pin/train_op_gate_pin_v7.sh --dry-run

# 正式提交：每个电路一个 Slurm 作业。
bash final_exam/single_circuit_pin/train_op_gate_pin_v6.sh
bash final_exam/single_circuit_pin/train_op_gate_pin_v7.sh
```

默认电路 `two_stage_folded_opamp`，seeds 为 17/29/43，设备为 CUDA，Slurm 配置沿用 8 CPU、1 GPU、80G 主机内存、`ex01A800`、账户 `b_phzhwu` 和 `newbase` 环境。原始数据仍为 `runs/raw/full_data/<电路名>/`，所有输出都留在当前仓库内部。

| 设置 | V6 | V7 |
|---|---|---|
| 配置文件 | `final_exam/configs/op_gate_pin_v6.json` | `final_exam/configs/op_gate_pin_v7.json` |
| 提交脚本 | `final_exam/single_circuit_pin/train_op_gate_pin_v6.sh` | `final_exam/single_circuit_pin/train_op_gate_pin_v7.sh` |
| run ID 中的版本标识 | `_op_gate_v6_` | `_op_gate_v7_` |

配置内容与对应 V4/V5 模板只在模型列表版本名上不同。类型编码由版本决定，不新增一个会改变旧模型语义的开关。当前门控配置仍为：

```json
"op_gate_pin": {
  "strength": 1.0,
  "norm_eps": 0.00001,
  "n_experts": 16,
  "mlp_dropout": 0.1
}
```

数据量由 `dataset.max_samples_per_circuit` 设置：`null` 为全量，`20000` 表示先随机选本电路最多 20,000 条再划分。抽样由 `dataset.sampling_seed` 控制，与训练 seed 分开；不按器件类型重采样。`--smoke` 只缩短训练轮次，小数据烟测请同时设置样本上限。

各阶段默认最大轮数如下，均沿用对应旧版本：

| 配置项 `op_transfer.epochs.*` | 上限 | 用途 |
|---|---:|---|
| `op_predictor` | 30 | 器件 OP predictor |
| `teacher` | 60 | 蒸馏需要的真实 OP 教师 |
| `readout` | 60 | 无门控监督基座、联合门控模型 |
| `predicted_op_distill_encoder` | 30 | 无门控学生 encoder 蒸馏 |
| `predicted_op_distill_decoder` | 30 | 冻结学生 encoder 后训练 decoder |
| `gate` | 30 | 最后的 gate 或 gate＋decoder 阶段 |

默认 batch size 为 32，学习率为 `1e-4`，早停 patience 为 10。阶段上限不是整条依赖链的总成本；各模型仍记录实际训练轮数、阶段成本和验证最优 checkpoint。

## 输出、检查与恢复

以下目录都位于 `final_exam/single_circuit_pin/`，run ID 继续以 `_pin` 结尾：

| 内容 | 路径 |
|---|---|
| HPC 标准输出/错误、提交配置快照 | `hpc_logs_pin/<run_id_pin>/` |
| 导入、训练、评估日志 | `logs_pin/<run_id_pin>/` |
| 数据、运行配置和训练产物 | `results_pin/<run_id_pin>/` |
| 自动生成的最终 Excel | `results_pin/<run_id_pin>/training_pin/summary_pin.xlsx` |
| 各模型配置、checkpoint、测试报告 | `results_pin/<run_id_pin>/training_pin/seed_<seed>_pin/<model>/` |
| 共享 predictor、教师、无门控基座 | `results_pin/<run_id_pin>/training_pin/seed_<seed>_pin/shared_pin/` |

报告保留完整模型名和版本、门控作用层、共享方式、输入结构、参数量、MAE/RMSE/R² 与跨 seed 标准差。分阶段模型保留 `gate_comparison_pin.json`，用于与同一基础模型比较最后门控阶段的收益。V7 checkpoint 保存各层独立门控参数。

已训练模型仍使用 `inspect_op_gate_pin.sh` 检查：填写原结果根目录、电路、样本数、完整新模型名称及 seed，按实际 PE 层和 head 查看门控。探针从实际评分网络的输入中读取新增类型坐标，并输出 `gate_device_types_pin.csv`：每个模型、样本和物理 MOS 一行，记录 `device_type`、`nmos_one_hot`、`pmos_one_hot`。HTML 的 MOS 输入详情同时显示这两个坐标；原有 `gate_op_inputs_pin.csv` 继续只记录 OP 字段，避免把类型坐标误当 OP。类型与原 OP 输入一起解释，门控分数仍不是校准后的物理导通概率。

失败恢复沿用：

```bash
bash final_exam/single_circuit_pin/resume_staged_op_gate_pin.sh \
  final_exam/single_circuit_pin/results_pin/<失败的V6或V7实验目录>
```

恢复应保留源实验的版本、模型、配置、seed 和划分。**V4/V5 的输入宽度与 V6/V7 不同，不能改名当作 V6/V7 checkpoint 续训；V6 与 V7 也不能互相冒充。** 同版本改变网络层数、专家数、输入字段等，也不能直接继续旧优化器状态。新增版本请从新实验启动。

## 对照关系

V4→V6、V5→V7 是“是否显式加入 MOS 类型”的对应对照；V6→V7 比较门控跨层共享与逐层独立。前两组也增加了少量输入层参数，后一组有完整门控网络数量差异，应随结果报告实际参数与成本。新增输入提供了明确极性条件，本实现不预先断言其提升性能、改善稳定性或恢复真实器件连接。
