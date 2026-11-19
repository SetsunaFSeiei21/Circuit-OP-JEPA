# LASD-MoE：逐节点的 Encoder 层融合

## 新增：逐头全图统计对照（2026-11-19）

这次在普通逐头模型上扩展 MoE 输入。新增两个模型，原有模型的名称、配置默认值与 checkpoint 契约保持不变。

| 模型 | MoE 输入 | Score 梯度 | PE 隐藏表示梯度 |
|---|---|---|---|
| `lasd_moe_headwise_pin` | 节点 R、T、ΔR、ΔT | 回传 | 回传 |
| `lasd_moe_headwise_global_pin` | 节点特征 + 全图 μR、σR、σT 及其差分 | 两部分均回传 | 回传 |
| `lasd_moe_headwise_detach_pin` | 节点 R、T、ΔR、ΔT | detach | 回传 |
| `lasd_moe_headwise_detach_global_pin` | 节点特征 + 全图 μR、σR、σT 及其差分 | 两部分一起 detach | 回传 |

### 统计口径与数学定义

对每个样本、每个 PE、每个 attention head 独立计算。设 V 为该图的有效 token 集合，N=|V|。**V 包括原始 index 0 的 GRAPH token 和有效引脚/外部端口 token，排除 padding**；不是只统计 MOS，也不跨样本聚合。沿用原模型在原 mask 后、attention dropout 前的 softmax；R/T 仍排除对角项，不重新归一化。

\[
\mu_R=\frac1N\sum_{i\in V}R_i,\quad
\sigma_R=\sqrt{\frac1N\sum_{i\in V}(R_i-\mu_R)^2+\varepsilon},\quad
\sigma_T=\sqrt{\frac1N\sum_{i\in V}(T_i-\mu_T)^2+\varepsilon}.
\]

均值、方差是当前图上的描述统计，不需要高斯分布假设，不拟合训练集统计，也不使用 BatchNorm 或 running mean。方差除以 N；这里没有 N−1 的样本估计修正。μT 只用于计算 σT，不作为输入，因为同一有效方阵的非对角元素总行和等于总列和，故 μR 与 μT 在浮点误差范围内相等。

令 `g[l,h] = [μR, σR, σT]`，`Δg[l,h] = g[l,h] - g[l-1,h]`。第一 PE 的 Δg 为边界填充 0；不借用 SE 或其他样本作为前一层。使用当前统计和差分表达当前及前一层的变化，不重复拼接上一层统计。各层相同 head 索引仍不意味着语义已对齐。局部/全局 attention mask 的交替也会影响这些统计。

全图统计原始值不会替代或归一化节点分数。共享给同一图的每个有效节点后，MoE 输入是：

```text
Concat(
  Flatten_PE_head([R_i, T_i, ΔR_i, ΔT_i]),
  Flatten_PE_head([μR, σR, σT, ΔμR, ΔσR, ΔσT])
)
```

`L=model.cross_layers`，`H=model.heads`，输入宽度由 `4LH` 变为 `10LH`。默认 3 PE × 8 heads 时，96+144=240 维。输出仍是每节点 H×L 个 logits：8 个通道组各自对 3 个 PE 做 softmax、混合对应 D/H 维表示，再拼回 128 维。全图统计对同图节点相同，但保留的局部特征使各节点仍可得到不同权重。

`global_score_pin.py` 是无可训练参数的独立 PyTorch 模块，直接聚合已提取的 `[B,N,L,H,2]`，不构建额外 N×N attention 矩阵。半精度输入在 float32 中计算均值/方差；double 保持 double。默认 ε=1e-8，因此有效图中完全相同的分数对应 σ=1e-4，这是数值稳定项，不是观测波动。零有效 token 的独立模块测试返回全零；实际模型总有 GRAPH token。输出 padding 特征清零。

### 梯度和对照范围

带 `detach` 的新增版本在节点特征与全图特征拼接完毕后执行一次 stop-gradient，避免全图分支绕过原有分数 detach。**PE1、PE2、PE3 的隐藏表示均不 detach**，Encoder 主链、MoE Router、Expert、融合权重、Final Graph Encoder 和性能 Decoder 正常训练。两种新模型的参数形状、相同 seed 下的初始化和前向完全一致。

本次不加入 HC、不加入 `_v1` 的最后 PE 加法残差、不改变 OP 预测器、原始 attention、图拓扑或损失。统计模块自身无参数，但 MoE 的 Router/Expert 第一层因输入增加而增加参数。默认 hidden_dim=128 时增加 `(4+4×128)×144=74,304` 个参数；参数量和时间继续由现有 TrainingCost 表记录。模型效果需要在相同 split、sampling seed 和训练 seed 下运行验证；此修改不预设精度收益。

### 运行、配置、输出与续训

在原来的 `final_exam/single_circuit_pin/train_lasd_moe_pin.sh` 顶部设置 CIRCUITS、SEEDS、CONFIG、集群环境。无需新目录或新 bash：

```bash
# 同时运行上表四个对照；一个电路一个 Slurm 作业。
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --global-stats --dry-run
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --global-stats
```

`--global-stats` 只覆盖 MODELS，保留脚本的 CIRCUITS、SEEDS、CONFIG；与其他模型 preset 互斥。只跑新增两个模型时，将脚本 MODELS 改为下面的内容，并**不传任何模型 preset**：

```bash
MODELS=(lasd_moe_headwise_global_pin lasd_moe_headwise_detach_global_pin)
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh
```

这里第一行是脚本内的设置示例，不是临时环境变量。配置继续使用 `final_exam/configs/lasd_moe_pin.json`。可选在现有 `lasd_moe_pin` 对象增加 `"global_std_epsilon": 1e-8`；不填写就使用该值。非正数、NaN、Infinity、布尔值会被拒绝。未给旧配置增加新默认键，以兼容既有 checkpoint 的精确配置校验。OP 训练预算仍由 `op_transfer.epochs.op_predictor` 控制，性能训练由 `op_transfer.epochs.readout` 控制，沿用原有早停。

结果、实验日志、HPC 日志分别继续在 `single_circuit_pin/results_pin/<run_id>/`、`logs_pin/<run_id>/`、`hpc_logs_pin/<run_id>/`。模型结果位于 `training_pin/seed_<seed>_pin/<model>/`，继续输出 checkpoint、配置、test 指标、梯度 history 和 summary_pin.xlsx（MAE/RMSE/R²、参数量、时间等）。LASD_MoE 工作表增加全图统计名称、维度、ε，保留完整架构契约。

诊断仍只用配置指定数量的验证样本（`diagnostic_samples`，默认 8；0 关闭）。新增 `lasd_moe_global_scores_pin.csv`，每样本/PE/head 一行，保存 μR、稳定化 σR、稳定化 σT、三项差分、有效 token 数、mask、ε、首层边界标志。原有 scores/weights/experts CSV 保留。原诊断 JSON 的 std 字段仍是未加 ε 的描述性标准差；新增 CSV 才是实际送入 MoE 的稳定化全图值。记录时才 detach；训练阶段按模型版本确定梯度边界。

使用已有续训入口，不重导入原始数据：

```bash
bash final_exam/single_circuit_pin/resume_width_decoder_pin.sh /绝对路径/原结果目录
```

续训保留原运行的模型、seed、配置、数据与 manifest。新增模型可恢复 optimizer、已完成 epoch、随机状态和最佳验证权重；不能把旧 96 维模型 checkpoint 当作新 240 维模型直接续训。架构契约记录统计定义与 ε，配置不一致时拒绝加载。

### 验证项目

单元测试覆盖人口方差/标准差、逐样本隔离、padding NaN 排除、节点重排不变性、首层零填充、逐层差分、常数/单节点/空图、半精度累积与数值梯度。模型测试单独隔离全图分数路径，检查 detach 阻断 Q/K 的额外梯度、隐藏路径仍回传，配对模型前向/RNG/参数相同。端到端小数据 CPU 测试覆盖训练、独立 checkpoint 重建、诊断与 XLSX、完成阶段跳过、模拟中断后的精确续训；实际集群精度和耗时仍需实验测量。

## 1. 本次实现范围

同一个单电路训练入口支持以下对照。

| 模型名称 | 层融合方式 | Score 分支梯度 | 节点隐藏表示梯度 |
|---|---|---|---|
| `device_op_cascade_pin` | 原始对照，仅使用最终 Encoder memory | 无新增分支 | 原始传播路径 |
| `lasd_moe_pin` | 所有 PE 层的 Score Dynamics → Top-2 MoE → 逐节点层权重 | 正常回传 | 所有层正常回传 |
| `lasd_moe_detach_pin` | 与 `lasd_moe_pin` 完全相同 | MoE 输入前执行一次 detach | 所有层正常回传 |
| `lasd_moe_headwise_pin` | 每个节点的每个头独立学习一组 PE 层权重 | 正常回传 | 所有层正常回传 |
| `lasd_moe_headwise_detach_pin` | 与 `lasd_moe_headwise_pin` 相同 | MoE 输入前执行一次 detach | 所有层正常回传 |
| `lasd_moe_headwise_pin_v1` | 最后一层 PE 输出 + 逐头融合输出 | 正常回传 | 残差主路与融合分支均正常回传 |
| `lasd_moe_headwise_detach_pin_v1` | 与 `lasd_moe_headwise_pin_v1` 相同 | MoE 输入前执行一次 detach | 所有 PE 输出进入融合时 detach，最后 PE 残差与层间主路正常回传 |
| `lasd_moe_headwise_hc_pin` | PE 内使用动态 Hyper-Connection，各 PE 的输出继续逐头融合 | 正常回传 | 所有 PE 与残差流参数正常回传 |
| `lasd_moe_headwise_detach_hc_pin` | PE 内使用 FlowNet 的动态 Hyper-Connection，各 PE 的输出继续逐头融合 | MoE 输入前执行一次 detach | 所有 PE 与残差流参数正常回传 |

模型继续使用现有级联输入：器件级 OP 预测器先从拓扑与设计参数学习训练集 OP，随后冻结。预测 OP 按 owner 对齐广播到原生引脚节点，与设计参数一起进入性能 Encoder。OP 预测器本身不采用 LASD-MoE，不增加真实 OP 推理输入，也不增加额外物理监督。

## 2. 与现有架构的对应关系

旧的 `width_decoder_lasd_pin` 是“每个 PE 配置独立 Decoder，再拼接所有 Decoder 输出”。这里实现的是新的读出拓扑：

1. 原始结构编码 SE 和参数融合 PE 顺序执行。
2. 仅收集每个 PE block 的节点隐藏表示与 self-attention 的逐 Head R/T；SE 只执行原始拓扑编码。
3. 在节点维度保留全部 PE 层、全部 Head 的 Score Dynamics。
4. 一个共享 Top-2 MoE 为每个节点输出所有 PE 层的融合权重。
5. 加权融合后的节点序列经过一个额外的 Graph Encoder block。两个 `_v1` 逐头模型在这一步先加上最后一层 PE 的输出。
6. 交给现有的一套 Metric Query Decoder 预测性能。

**L 是 `model.cross_layers`，仅包含参数融合 PE。** 例如 `encoder_layers=6, cross_layers=3`：先正常执行 3 个 SE，再收集 3 个 PE 的 Score 与输出，融合这三个 PE 层的表示，最后追加一个 Graph Encoder block。SE 不提取 Score、不参与层权重或隐藏表示融合；最终 block 也不进入 Score 历史。第一层 PE 的差分直接置零，不与最后一层 SE 比较。

每个 PE 读出的隐藏表示经过原有共享 `output_norm` 与 `output_projection`，复用已有 PE memory 输出接口。原始主干的下一层仍接收原来的 block 输出；收集与投影不会改写主干输入。

最终 Graph Encoder 复用 `EncoderBlock(cross=False)`：一层 self-attention、原有残差、LayerNorm 和 FFN，使用原来的 **local topology mask 与 terminal-role bias**。它不再次执行设计参数/OP cross-attention。

只有一套性能 Decoder；`model.decoder_layers=2` 表示这一套 Decoder 内含两个 Decoder block，不会为三个 PE 层各创建一套 Decoder。

## 3. 节点、mask 和 Attention 的实际定义

仓库的序列为 `GRAPH token + 原生引脚/外部端口 token`。所有层都保持相同节点编号、顺序和 padding，Graph token 也作为一个序列节点参与层融合，并在诊断中单独标识。

原有 self-attention 是 dense masked MultiheadAttention。SE/PE 沿深度交替使用 local/global mask，保留自连接和 terminal-role bias；例如 3SE+3PE 时，参与统计的三个 PE mask 顺序是 global、local、global。原来设计参数/OP cross-attention 的 mask 与计算过程继续使用现有实现。PE self-attention 在该层的参数 cross-attention 之前执行，因此第一层 PE 的分数主要来自之前的拓扑编码；不能默认它已经包含本层刚注入的设计参数或 OP。

Score 观察的是每层 self-attention 的实际 Q/K 输入、投影参数与原始 mask。为了保持 row-softmax 定义，统计使用 **attention dropout 之前**的 softmax 权重。原消息传递仍使用原来的 attention dropout，不用重新统计的权重替换消息路径，不消耗额外 dropout 随机数。

注意力分数提取复用已有的 query 分块实现，并支持 activation checkpoint 重算，避免为了导出诊断保留完整 N×N attention 矩阵。其计算仍然具有 dense attention 的计算成本；本实现不声称转成了稀疏图算子。

对于 A[i,j]（i 是 Query，j 是 Key）：

\[
R_i^{(l,h)}=\sum_{j\ne i}A_{ij}^{(l,h)},\qquad
T_i^{(l,h)}=\sum_{j\ne i}A_{ji}^{(l,h)}.
\]

对角线与 padding 的 query/key 在统计时排除，不对剩余注意力重新归一化。在正常有效行中 R=1−Aii；如果某种 mask 完全禁止 self，R 会退化为 1。即便 Receive 退化，Transmit 仍正常计算；不人为更改 Attention 来制造变化。

不同样本始终沿 batch 轴分离，不跨样本累加。保留所有 Head，既不求 Head 平均，也不提前全图求和。

## 4. Score Dynamics 和融合公式

逐层 Score 张量形状为 `[B,N,L,H,2]`。差分为：

\[
\Delta S^{(1)}=0,\qquad
\Delta S^{(l)}=S^{(l)}-S^{(l-1)}\quad(l>1).
\]

第一层填零是边界约定，不代表观测到零变化。不同层暂按相同 Head 索引相减；同编号 Head 不保证语义相同。另外，local/global mask 的交替也会造成差分，因此差分不是纯粹的电路物理状态变化。

每个节点保留全部 PE 层的 R/T 和 ΔR/ΔT，展开顺序为 `layer → head → [R,T,ΔR,ΔT]`，共 `4 × L × H` 个数。默认 L=3、H=8，MoE 输入是 **96 维**。这个展开顺序保存在 checkpoint 结构契约中，重建时会校验；同一节点的所有 PE 层、所有 Head 共同构成一个 Router 输入。

Router 为每个节点选出 4 个 expert 中得分最高的两个，并在这两个 expert 间归一化。每个 expert 都是参数独立的两层 MLP。基础版每个 expert 输出 L 个层 logits，逐头版每个 expert 输出 H × L 个 logits。同一个节点的所有融合头共享本次 Top-2 选择。

\[
u_i=\sum_{e\in\operatorname{Top2}(r_i)}p_{i,e}f_e(z_i),\qquad
\alpha_i=\operatorname{softmax}_{\text{layer}}(u_i).
\]

在基础版中，所有头的分数共同决定一组 L 维权重。有效节点的层权重和为 1，padding 节点的权重清零。

\[
\widetilde h_i=\sum_{l=1}^{L}\alpha_i^{(l)}h_i^{(l)},\qquad
\hat y=D_{\mathrm{metric}}\big(E_{\mathrm{final}}(\widetilde H)\big).
\]

本版不增加 capacity dropping、不引入 expert load-balancing loss。Expert 使用失衡应通过诊断观察，不能假定四个 expert 会自动等量使用。

### 4.1 逐头融合

以 3 个 PE、8 个头、128 维隐藏表示为例，每个引脚的 96 维 Score Dynamics 进入 MoE。两个选中 expert 各输出 24 个 logits，按路由权重混合，再重排为 `[8, 3]`。每一行只沿 PE1、PE2、PE3 做 softmax，得到 8 组独立的三层权重。

每个 PE 的完整节点输出经过已有的输出归一化与投影。将同一节点的 128 个投影通道按原顺序分成 8 组，头 0 使用各 PE 的第 0–15 通道，头 1 使用第 16–31 通道，以此类推。每个头用自己的三个权重加权求和得到 16 维，依次拼回 128 维，送入已有的最终 Graph Encoder 与 Metric Decoder。这里的头编号指投影后的融合通道组。原生 self-attention 中各个头的输出已经经过注意力输出投影和后续 PE 运算。

一般配置下，层权重形状为 `[B,N,H,L]`，通道组宽度为 `D/H`。每个有效节点、每个头的 L 个权重之和为 1。普通逐头版与对应的 detach 版参数量相同。相对基础版，四个 expert 的最后一层在默认配置下共增加 `4 × (128 + 1) × (24 − 3) = 10,836` 个参数。

### 4.2 PE 内的动态 Hyper-Connection

`lasd_moe_headwise_detach_hc_pin` 与 `lasd_moe_headwise_detach_pin` 配对，检验 **PE 内的残差连接**对相同逐头读出的影响。FlowNet 将输入复制为多条全维残差流，每个子层先混合这些流并读出一个向量做计算，随后将一次计算的结果按流写回。上游注意力与 FFN 各有独立的读写参数，默认保留 4 条流，在所有块结束时求和。[FlowNet 算子](https://github.com/Yutong-Feng/FlowNet/blob/57ad6128c2a1980d0aa73325a844659262271797/src/layers/hyper_connect.py#L7-L56) · [FlowBlock 调用位置](https://github.com/Yutong-Feng/FlowNet/blob/57ad6128c2a1980d0aa73325a844659262271797/src/blocks/hyper_flow.py#L201-L239) · [上游默认配置](https://github.com/Yutong-Feng/FlowNet/blob/57ad6128c2a1980d0aa73325a844659262271797/conf/model/flownet.yaml#L7)

对单个引脚，设输入为 `X[R,D]`。模块从逐流 LayerNorm 后的表示生成两组系数，在可学习的静态参数上叠加经 tanh 和可学习尺度缩放的动态项。读出系数与写回系数都允许取负值。初始化时动态投影为零，两个尺度均为 0.01。

1. **读出。** 计算 `Aᵀ X`，其中 `A` 的形状为 `[R,R+1]`。结果的首行作为子层输入，其余 R 行作为各流的旁路。初始化时读取编号为 `PE索引 % R` 的流（PE 索引从 0 起），旁路为单位映射。
2. **写回。** 给定一次子层计算及 dropout 后的更新 `u[D]`，第 r 条流得到 `旁路[r] + β[r] × u`。每个 β 的静态初值为 1。
3. **接入引脚模型。** 最后一层 SE 的输出在首个 PE 处展开。每个性能 PE 的 self-attention、设计参数/预测 OP 的 cross-attention、FFN 分别使用一套系数。FlowNet 在子层算子之前归一化分支输入。此变体沿用引脚 PE 原有的 post-norm，在每次写回后对每条流归一化。每个 PE 结束时，完整的 R 条流送给下一层，同时对 R 条流求和并经过原有的输出 LayerNorm 和投影，供 LASD 逐头融合。[引脚 PE 实现](models/hyper_connection_pin.py)

默认设置下，一个引脚在 PE 内有 **4 条 128 维残差流**。LASD MoE 仍从 3 个 PE 的真实 self-attention 输入提取 R/T 与差分，对这组分数执行 detach，再输出 **8 组 × 3 个 PE 权重**。这 8 组权重用来融合投影后的 8 组 16 维通道。3 个 PE 的 9 个残差混合模块共增加 **9,450 个可训练参数**。OP 预测器继续冻结，SE、最终 Graph Encoder 和 Metric Decoder 使用原有结构。

`lasd_moe_headwise_hc_pin` 是对应的无 detach 对照。它直接将 `score_features` 送入同一个 MoE，不执行 `score_features.detach()`。两个 HC 模型的结构、参数数量、初始化顺序、前向结果和 dropout 随机数消耗相同；架构契约仅 `mode` 与 `score_detach` 不同。无 detach 版恢复“性能损失 → 层权重 → MoE → R/T 与差分 → Attention Q/K 和 HC 读出”的梯度路径。两者的 PE 隐藏表示、层间主干、HC 参数、Router、Expert 和性能 Decoder 都保留正常训练，器件级 OP 预测器仍被冻结。

此配对不采用 `_v1` 的最后 PE 额外相加，也不 detach 任何 PE 融合输入。原有 detached HC 的契约保持不变，旧 checkpoint 可继续重建和续训。新模型使用独立名称及结果目录，不能将旧模型已完成的 stage 改名当作新模型的训练结果。

## 5. Detach 的准确边界

每对普通与 detach 模型使用相同参数形状、前向计算与初始化过程。下面先说明原有基础版、逐头版和 HC 版的 Score detach。

```python
# 普通版本
moe_input = score_features

# Detach 版本：Score 与 Delta 都已计算完成
moe_input = score_features.detach()
```

Detach 在 MoE 之前，因此 Router 和 Expert 仍训练，层权重和融合结果也保持可微。所有参与融合的 PE 隐藏表示不 detach，SE/PE 主干的层间连接也不 detach。SE 仍可通过正常主干获得梯度。

普通版本可同时通过 Hidden State Fusion 和 Score 分支影响原始 Encoder；Detach 版关闭直接的 Score 分支梯度，但仍通过融合隐藏表示更新原始 Encoder。Q/K 仍可能从正常 Attention 消息传递及 Hidden State Fusion 获得梯度。不能把 Detach 版解释为冻结 Attention。

旧的 `width_decoder_lasd_detach_pin` 切断浅层 memory/Score 通往对应 Decoder 的读出边，并对差分上一层 Score 断梯度。原有的 `lasd_moe_headwise_detach_pin` 只在完整 Score 特征送入 MoE 前 detach 一次，各 PE 层的 hidden-state 融合边都保留梯度。

### 5.1 最后 PE 残差与融合分支 detach 的 v1 模型

设 `h1, h2, h3` 是原有接口返回的 PE memory，即每个 PE block 输出经过共享 `output_norm` 和 `output_projection` 后的表示。两个 v1 模型都使用 `x = h3 + fused`，然后将 x 送入现有 Final Graph Encoder 和性能 Decoder。层数可配置，残差自动取最后一个 PE memory。直接相加的系数为 1，此项操作增加 0 个参数。

```python
# lasd_moe_headwise_pin_v1
logits, _ = layer_fusion(score_features, valid)
fused, _ = fuse_pe_memories(memories, logits, padding)
x = memories[-1] + fused

# lasd_moe_headwise_detach_pin_v1
logits, _ = layer_fusion(score_features.detach(), valid)
fusion_inputs = tuple(memory.detach() for memory in memories)
fused, _ = fuse_pe_memories(fusion_inputs, logits, padding)
x = memories[-1] + fused
```

`lasd_moe_headwise_detach_pin_v1` 的损失通过最后 PE 的残差进入 Encoder，再沿 PE3 → PE2 → PE1 → SE 反传。所有层之间传递的隐藏表示都保留梯度。融合分支的数值仍取自同一次前向中的 PE1、PE2、PE3，detach 切断的是这三个读出输入与 Encoder 的梯度连接。MoE 的 Router、Experts 和逐头 softmax 权重仍通过 `fused` 学习。

两个 v1 模型的逐头权重仍沿 PE 层维归一化。最后 PE 还具有额外的系数为 1 的残差，因此完整的 `x` 包含两部分贡献。诊断 CSV 记录的是融合分支内的权重，相加残差由结构记录中的 `residual_connection` 描述。

## 6. 配置、预算与实验控制

配置文件：`final_exam/configs/lasd_moe_pin.json`。

```json
"lasd_moe_pin": {
  "n_experts": 4,
  "top_k": 2,
  "hidden_dim": 128,
  "dropout": 0.0,
  "score_chunk_size": 128,
  "checkpoint_scores": true,
  "diagnostic_samples": 8,
  "hyper_connection_streams": 4
}
```

首版固定四个 expert、Top-2；参数验证拒绝其他值，避免相同模型名对应不同实验定义。`hidden_dim` 是每个 Expert 的隐藏维度，`dropout` 是专家 MLP 的 dropout。`model.dropout` 继续控制原始 Encoder/Decoder。

`score_chunk_size` 和 `checkpoint_scores` 控制提取的计算方式，不改变数学定义；`diagnostic_samples=0` 关闭训练后样本级诊断，不关闭模型使用的 Score 分支。

`hyper_connection_streams` 是 HC 模型的残差流数量，取正整数，缺省为 4。单流初始化时与普通残差的前向计算一致，此后其读写系数也可学习。显存和训练时间按实际配置测量。

默认预算：

| 阶段 | 配置键 | 上限 |
|---|---|---|
| 冻结前的器件级 OP 预测器训练 | `op_transfer.epochs.op_predictor` | 30 epoch |
| 性能模型训练 | `op_transfer.epochs.readout` | 60 epoch |
| 批量 | `op_transfer.batch_size` | 32 |
| 早停耐心 | `op_transfer.patience` | 10 |

同一电路/seed 下复用同一个冻结 OP 预测器；性能模型各自训练，以原有验证目标早停和选择 checkpoint，最终在 test 上计算 MAE、RMSE、R²。实际 epoch、参数量与耗时使用原训练统计，不把上限当成真实训练成本。

`dataset.max_samples_per_circuit=null` 表示全部样本；设置 20000 表示先随机抽总量 20000 再按 train/val/test 比例划分。抽样由 `sampling_seed` 控制，划分由 `split_seed` 控制。预处理统计保持由 ID 训练集拟合。

## 7. 启动方式与两个集群

修改 `final_exam/single_circuit_pin/train_lasd_moe_pin.sh` 顶部：

```bash
CIRCUITS=(two_stage_folded_opamp Opamp1)
MODELS=(
    device_op_cascade_pin
    lasd_moe_pin
    lasd_moe_detach_pin
)
SEEDS=(17 29 43)
```

每个电路独立提交一个 Slurm 作业，一个作业内按 seed/model 依次执行并共享允许复用的阶段。不同电路不共享模型权重、预处理统计或训练数据。

只跑两个逐头版本时，在同一个脚本上加 `--headwise`。该参数覆盖 `MODELS`，沿用脚本顶部的 `CIRCUITS`、`SEEDS`、资源与配置。也可以直接在 `MODELS` 中按名称组合基础版和逐头版。

```bash
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --headwise --dry-run
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --headwise
```

训练这次新增的两个残差逐头版本，使用 `--headwise-v1`。它选择 `lasd_moe_headwise_pin_v1` 和 `lasd_moe_headwise_detach_pin_v1`，沿用脚本顶部的 `CIRCUITS`、`SEEDS`、资源设置与同一份 `configs/lasd_moe_pin.json`。

```bash
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --headwise-v1 --dry-run
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --headwise-v1
```

也可以在脚本顶部直接写 `MODELS=(lasd_moe_headwise_pin_v1 lasd_moe_headwise_detach_pin_v1)` 后无参数启动。若希望把上一版放在同一轮实验比较，可把四个逐头模型都写入 `MODELS` 后无参数启动。每次提交选择一个模型预设，脚本会拒绝同时传入 `--headwise`、`--headwise-v1`、`--hyper-connection`、`--hyper-connection-grad` 中的多个选项。

比较普通残差与 Hyper-Connection 时使用同一脚本的 `--hyper-connection` 预设。它选择 `lasd_moe_headwise_detach_pin` 和 `lasd_moe_headwise_detach_hc_pin`，使用脚本顶部的电路和 seed。两组复用同一电路、同一 seed 的冻结 OP 预测器。如只训练 HC 模型，可把脚本顶部设为 `MODELS=(lasd_moe_headwise_detach_hc_pin)`，然后无参数运行。

```bash
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --hyper-connection --dry-run
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --hyper-connection
```

只比较 HC 内的分数梯度开关，使用新增预设：

```bash
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --hyper-connection-grad --dry-run
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --hyper-connection-grad
```

它选择 `lasd_moe_headwise_hc_pin` 与 `lasd_moe_headwise_detach_hc_pin`，继续使用脚本顶部的 `CIRCUITS`、`SEEDS` 和 `CONFIG`。无参数启动仍使用顶部 `MODELS`。只训练新模型时填写 `MODELS=(lasd_moe_headwise_hc_pin)` 并无参数运行。旧的 `--hyper-connection` 预设含义不变。

两者共用配置 `lasd_moe_pin.hyper_connection_streams`（默认 4），分别使用 `op_transfer.epochs.op_predictor` 和 `op_transfer.epochs.readout` 控制 OP 预训练和性能训练上限。默认 30 和 60，实际轮数由原有验证早停决定。结果、实验日志和 HPC 日志仍位于 `single_circuit_pin/results_pin`、`logs_pin` 和 `hpc_logs_pin` 的同一个 run ID 下。checkpoint、结构契约、诊断 CSV、配置快照和 Excel 汇总均保留完整模型名及分数梯度设置。

```bash
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh
```

`--dry-run` 只显示命令、不创建实验目录。`--smoke` 缩短训练 epoch，默认 OP 预测器 1 轮、性能读出 2 轮；数据量仍由 JSON 配置控制。要做 128 条烟测，应将烟测使用的配置中 `max_samples_per_circuit` 设为 128，完整实验使用独立配置恢复所需样本数。

新集群 `login1` 可使用环境覆盖，无需修改旧集群默认路径：

```bash
JEPA_CONDA_HOME=/public/home/202411094934/anaconda3 \
JEPA_CONDA_ENV=newbase \
SLURM_ACCOUNT=b_phzhwu \
SLURM_PARTITION=ex01A800 \
bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh
```

默认请求 8 CPU、1 GPU、80G 主机内存，训练设备为 CUDA。代码与数据以当前项目根目录定位，不创建旁边的新 worktree。提交时复制 JSON 配置到对应 HPC 日志目录，运行时继续保存有效配置，防止后来修改默认配置改变已提交作业的参数。

### 7.1 从中断实验恢复

各 LASD-MoE 模型以及 `device_op_cascade_pin` 都使用现有的 `resume_width_decoder_pin.sh` 恢复。参数可以是原实验根目录，也可以是其中的 `training_pin` 目录。

```bash
LASD_SOURCE_RUN="/path/to/results_pin/two_stage_folded_opamp_lasd_moe_..._pin"
bash final_exam/single_circuit_pin/resume_width_decoder_pin.sh "$LASD_SOURCE_RUN" --dry-run
bash final_exam/single_circuit_pin/resume_width_decoder_pin.sh "$LASD_SOURCE_RUN"
```

在新集群使用时，给恢复命令添加上面的 `JEPA_CONDA_HOME`、`JEPA_CONDA_ENV`、`SLURM_ACCOUNT`、`SLURM_PARTITION` 环境设置。确认原作业已退出，再提交恢复任务；脚本会在能够查询到原 Slurm 作业时阻止重复运行。

恢复任务保留旧实验的模型列表、seed、配置和数据划分，在新的 `*_resume_width_decoder_*_pin` 结果与日志目录执行。它复用原有 canonical JSON，不重新导入原始 SQLite，不修改旧结果目录。已完成阶段直接复用，未完成阶段从已保存的 epoch 边界恢复模型、优化器、随机状态和早停状态；中断时尚未完成的 epoch 会重新执行。若某阶段没有恢复 checkpoint，则该阶段从头开始。原 canonical JSON 和 checkpoint 必须继续保留，不能在恢复后删除来源目录。

## 8. 结果、日志和可复现记录

目录沿用现有单电路任务布局，运行名使用独立的 `lasd_moe` 标记：

```text
final_exam/single_circuit_pin/results_pin/<电路>_lasd_moe_<时间和标识>_pin/
  config_pin.json
  data_pin.json
  training_pin/
    summary_pin.xlsx
    seed_17_pin/
      shared_pin/device_op_predictor_pin/checkpoint_pin.pt
      lasd_moe_pin/
        config_pin.json
        checkpoint_pin.pt
        test_pin.json
        lasd_moe_audit_pin.json
        lasd_moe_scores_pin.csv
        lasd_moe_weights_pin.csv
        lasd_moe_experts_pin.csv
        lasd_moe_diagnostics_pin.json
      lasd_moe_detach_pin/
        config_pin.json
        checkpoint_pin.pt
        test_pin.json
      stages_pin/<模型名>/history_pin.json
final_exam/single_circuit_pin/logs_pin/<运行名>/
final_exam/single_circuit_pin/hpc_logs_pin/<运行名>/
```

保留模型结构契约、数据划分、有效配置、冻结 OP 预测器 checkpoint、所用 seed 与实际阶段成本。汇总沿用 MAE/RMSE/R² 与多 seed 的均值±标准差。

模型按既定顺序从**验证集**选取至多 `diagnostic_samples` 条样本，在 checkpoint 选择完成后导出诊断。

| 文件 | 内容 |
|---|---|
| `lasd_moe_scores_pin.csv` | 每个 sample/node/PE/head 的 R、T、ΔR、ΔT，保留节点身份和层位置 |
| `lasd_moe_weights_pin.csv` | 基础版按节点、PE 输出一行。逐头版按节点、`head`（从 0 起）、`pe_layer`（从 1 起）输出 `layer_weight` |
| `lasd_moe_experts_pin.csv` | 节点的 Router、选中 expert 和混合权重 |
| `lasd_moe_diagnostics_pin.json` | 验证集 sample IDs、有效节点数、结构契约、分数统计、层权重与 expert 使用汇总，以及文件清单 |
| `stages_pin/<模型名>/history_pin.json` | 每个 epoch 的训练/验证 loss 与 `encoder_gradients` 梯度统计 |
| `stages_pin/<模型名>/cost_pin.json` | 阶段训练成本，包括实际 epoch 数与累计训练耗时 |

逐头版的 `layer_weight_summary.per_head` 按 `[H,L]` 保存各头的层权重统计。外层摘要先在同一节点内对头取均值，便于与基础版对照。Expert 使用次数按节点统计。工作簿 `summary_pin.xlsx` 的 `LASD_MoE` 页标出层权重轴顺序、expert 输出维度、每个融合头的通道数和每个 PE 的残差流数。HC 模型的 `lasd_moe_weights_pin.csv` 记录的是各头对各 PE 的融合权重。残差流的混合模块参数随 checkpoint 保存。

梯度在 `backward()` 之后、梯度裁剪之前统计。每个 SE、PE 和额外 Final Encoder block 独立记录 L2 范数的 batch 平均值、最大值和 batch 数。SE 梯度诊断用于观察原主干传播，**不表示 SE 被加入 Score 提取或隐藏表示融合**。参数梯度范数也不等同于节点激活梯度范数，不能互换解释。

padding 不参与节点分数、层权重和 expert 使用率统计。诊断保留全部 Head，不跨样本混合；只有序列化时将张量 detach 并转到 CPU。诊断结束后恢复原训练模式与随机状态。当前不额外加入诊断损失、专家负载均衡 loss 或 test 选模规则。

## 9. 验证与结果解释

核心测试检查 R/T 方向、self/padding 排除、分块与原 Attention 一致性、第一层差分零填充、节点对齐、Top-2 路由和层权重归一化、每对模型相同权重下的前向一致性，以及各版本规定的梯度边界。v1 测试还检查最后 PE 残差、三个融合输入 detach 与 MoE 梯度。工程检查覆盖保存/重建、配置与模型名一致性、结果汇总、Slurm 参数与目录隔离。

相同参数规模与相同初始化不意味着两模型优化过程相同；其梯度路径是本次实验变量。该结构没有将 Attention Score 等同于真实物理信息流，也没有证明低秩、过平滑改善或 Expert 专门化。精度、收敛速度与稳定性必须由真实电路的多 seed 实验给出；小规模 CPU/GPU 烟测只验证代码运行路径。
