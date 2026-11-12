# 无门控 ZeroSim：PE 阶段的 MOS 引脚自注意力探测

## 对象与边界

本入口实现方案 A：加载原本没有门控的已训练 checkpoint，在 `eval()` 和禁用梯度的方式下执行真实推理。默认 `device_op_cascade_pin` 仍使用训练好的 OP 预测器提供下游输入；若要检查纯拓扑＋设计参数输入，选择 `zerosim_supervised_pin`。

范围限定为下游**引脚编码器的 PE 融合块 Self-Attention**，不收集 SE、Cross-Attention、decoder、前置 OP 预测器或教师的注意力。拒绝带门控的模型，不会运行时关闭门控来代替无门控基线。支持以下已保存模型：

```text
device_op_cascade_pin
zerosim_supervised_pin
device_op_gate_only_base_pin
predicted_op_pin
student_predicted_op_pin
student_predicted_op_distill_pin
op_oracle_pin
design_op_oracle_pin
```

后两种需要真实 OP，报告明确记录输入条件。只能选原结果中实际保存 checkpoint 的模型和 seed；共享目录中的中间阶段不自动作为模型别名加载。

新版门控对照实验中的 `device_op_cascade_pin` 可能以共享监督基线的封装保存，包含未使用的门控参数。本入口接受 checkpoint 原本保存的 `supervised` 无门控阶段，不调用 `set_phase()`；额外 hook 确认这些闲置门控在推理中没有被执行。已经训练到 `gate_only` 或 `gate_decoder` 的模型仍拒绝。

## 配置与运行

编辑 `final_exam/single_circuit_pin/inspect_pe_attention_pin.sh` 顶部：

```bash
SOURCE_RUN="/absolute/path/to/results_pin/<run_id_pin>"
CIRCUIT="two_stage_folded_opamp"
N_SAMPLES=20
MODELS=(device_op_cascade_pin)
SEED=17
SPLIT="test"
SAMPLE_SEED=17
BATCH_SIZE=8
VIEWER_SAMPLES=20
```

`SOURCE_RUN` 下面有 `training_pin/` 即可，也支持直接填到 `training_pin/`。可将结果路径作为脚本的第一个位置参数传入。运行：

```bash
bash final_exam/single_circuit_pin/inspect_pe_attention_pin.sh --dry-run
bash final_exam/single_circuit_pin/inspect_pe_attention_pin.sh
```

默认 Slurm：8 CPU、1 GPU、80G 主机内存、1 小时、`b_phzhwu` / `ex01A800` / `newbase`。`--dry-run` 不创建目录或提交；`--local` 在当前计算环境执行。CUDA 不可用时不自动退回 CPU。JSON/报告 IO 在 CPU，模型和 QK/softmax 在指定设备。

不训练、不重新拟合统计量。样本从 checkpoint 保存的指定 split cohort 无放回抽取，超过可用数则取全部，多个模型共用同一批样本。核对数据/manifest 指纹和 checkpoint 的模型、seed、cohort。当前为核验身份仍读取整份 canonical JSON，然后只为抽取样本准备缓存；`N_SAMPLES` 不代表 JSON 的内存开销同比下降。

## 输出

三个目录使用相同的 `<circuit>_pe_attention_probe_<timestamp>_<id>_pin` 标识：

| 内容 | 位置 |
|---|---|
| 结果 | `final_exam/single_circuit_pin/results_pin/<run_id>/` |
| 实验日志 | `final_exam/single_circuit_pin/logs_pin/<run_id>/probe_pin.log` |
| HPC 日志 | `final_exam/single_circuit_pin/hpc_logs_pin/<run_id>/%j_pin.out` 和 `%j_pin.err` |

- `attention_viewer_pin.html`：离线页面，选择模型、样本、MOS 和 head 后，**所有 PE 的 Self-Attention 各显示一张 4×4 热图**，按层号排列；下方可选一层查看详情和 Query 权重去向。可选 scaled QK、原偏置、logit、softmax 权重。同一次展示的所有层共用色标：权重固定 0–1，其余得分使用所有 PE 的共同对称范围，禁止逐图独立缩放造成视觉误判。
- `attention_scores_pin.xlsx`：`SampleMatrices` 是预览样本的 head 平均矩阵；`MeanMatrices` 是所有抽取样本平均矩阵；`ScoreSummary` 保留各 head 的统计；`MassSummary` 是权重去向；`DetailPreview` 为前 4000 行原始明细；`Checkpoints` 记录权重身份与重建误差；`ReadMe` 解释统计口径。
- `attention_scores_pin.csv`：完整逐样本/PE/head/MOS/有向端口对得分，含对角项和同网引脚。
- `attention_mass_pin.csv`：每个 MOS Query 分配给本器件、其他器件、外部端口、图 token、padding 的概率质量及总和。
- `attention_summary_pin.csv`、`attention_mass_summary_pin.csv`：跨本次抽取样本的统计；总体标准差 `ddof=0`，不是多 seed 实验标准差。
- `probe_pin.json`、`selection_pin.json`：配置、checkpoint SHA-256、源指纹、样本 IDs、实际层数/head 数、是否使用真实 OP、重建误差。

`VIEWER_SAMPLES` 仅限制 HTML 与 Excel `SampleMatrices` 的样本数。完整 CSV、均值和标准差覆盖所有抽取样本。每个 head 独立保存，额外 `head=mean` 为算术平均。所有原实验数据、权重、配置和日志保持原样，输出必须使用单独的新目录。

### 已有探测结果直接生成所有 PE 热图

如果已经生成过 `attention_viewer_pin.html`，在项目根目录执行：

```bash
python -m final_exam.attention_probe_report_pin \
  --source /absolute/path/to/results_pin/<pe_attention_probe_run_id>
```

`--source` 是**注意力探测结果目录**，也可以直接填旧 HTML 的路径。默认在同目录新增 `attention_pe_overview_pin.html`；旧页面、CSV 和 Excel 保持原样。不读取 checkpoint、不需要 Torch/GPU、不重新推理，仅使用旧页面已保存的样本和所有 PE/head 数据。若该输出已存在，可用 `--output /path/to/new_name.html` 指定另一个新文件，不会覆盖已有文件。新运行探测时，默认 `attention_viewer_pin.html` 已直接包含总览。

## 分数与索引

只读 hook 获取真实下游 MHA 输入和掩码，使用保存的 Q/K/V 投影重建 `S=QKᵀ/√head_dim`、`L=S+原偏置+拓扑掩码`、`A=softmax(L)`。对应字段为 `qk_scaled`、`additive_bias`、`attention_logit`、`attention_weight`，不包含门控偏置。

Softmax 保留包含图 token、全部有效引脚、外部端口的完整 key 轴，**计算后才截取 MOS 子矩阵，不做器件内二次归一化**。每层还将 `A·V` 经过原输出投影，与原生 MHA 输出核对；允许融合核和显式运算的有限浮点差异，不替换正常 forward 输出。它是由实际输入重建的概率，不是直接读取 Flash Attention 内部的矩阵。

MOS 通过 `owner_index`、`pin_terminal_index` 和保存的 terminal schema 定位，并核对网络身份；不假定四个引脚恰好相邻。器件顺序按 resident batch 的 device_id 排序。`*_pin_index` 为不含图 token 的 0-based 下标，`*_sequence_index` 为加入前置图 token 的实际下标。PE、全编码器层号、head 从 1 起算。

## 解读

1. 行是 Query、列是 Key，顺序 D/G/S/B。D 行 S 列代表 D 的更新读取 S 的 value 权重，一般不要求与 S 行 D 列相等。
2. 子矩阵每行和通常小于 1，完整 key 轴行和才约为 1；质量表说明差额去了哪里。
3. Head 平均是 softmax **之后**求平均，不是平均 logits 后重新 softmax。跨样本同样先分别计算，再求平均。
4. 硬掩码的 `-inf` logit/bias 在 CSV/JSON 中用空值表示，并附 `hard_masked`；权重为零，HTML 显示 `−∞`。
5. 当前 PE 层数等于 `cross_layers`，SE 等于 `encoder_layers-cross_layers`。每个 PE 块先 Self-Attention 后 Cross-Attention，因此无门控固定拓扑下 PE1 尚未收到参数/OP 注入，跨设计样本得分相同是正常可能性。
6. 注意力是模型信息聚合权重，不是电气导通概率。原门控探测入口继续独立使用，门控 sigmoid 与 attention softmax 不混用。

## 本次验证

在 PyTorch 2.6.0 CPU 环境完成 31 项相关单元/集成测试，1 项 CUDA 数值测试因无 GPU 跳过。覆盖原生逐 head 权重比对、不同掩码、交错引脚索引、完整归一化质量、8 种无门控 checkpoint、原预测与源文件不变、异常后 hook 清理、拒绝门控模型、数据指纹检查、Slurm 预检及旧门控探测回归。离线页面通过 48 组模型/层/head/得分切换的 DOM 检查，无 JavaScript 错误；未在这里验证 GPU 集群或浏览器实际像素布局。

所有 PE 总览更新后，重新执行探测与启动入口测试：12 项通过、1 项 CUDA 跳过。使用真实小规模探测输出验证 216 组模型/样本/MOS/head/得分切换、6,912 个热图单元格与保存的数据一致，并核对共同色标和层详情选择。旧 HTML 更新在不加载 Torch 的环境中完成，保留全部原 payload，原文件摘要不变，已有输出拒绝覆盖。
