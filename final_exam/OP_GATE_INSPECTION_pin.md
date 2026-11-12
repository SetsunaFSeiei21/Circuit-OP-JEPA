# 已训练模型的门控分数检查

入口：`final_exam/single_circuit_pin/inspect_op_gate_pin.sh`。这个脚本只加载最终 checkpoint 做推理，从原电路实验的保存划分中抽样，记录模型实际前向计算中的门控输出。不会训练模型、重新拟合归一化统计量、重新划分数据或覆盖原实验。

## 1. 编辑脚本顶部

```bash
SOURCE_RUN="/share/home/202411094934/python_project/Circuit-OP-JEPA/final_exam/single_circuit_pin/results_pin/你的实验目录_pin"
CIRCUIT="two_stage_folded_opamp"
N_SAMPLES=20
MODELS=(
  device_op_gate_pin_v2
  device_op_supervised_gate_fusion_pin_v2
)
SEED=17
SPLIT="test"
SAMPLE_SEED=17
BATCH_SIZE=32
VIEWER_SAMPLES=20
DEVICE="${DEVICE:-cuda}"
```

| 设置 | 含义 |
|---|---|
| `SOURCE_RUN` | 原实验结果根目录，也可填其 `training_pin` 子目录；不是单个 `.pt` 文件 |
| `CIRCUIT` | 原实验的电路名，必须与保存记录一致 |
| `N_SAMPLES` | 本次检测样本数；超过原保存划分的实际样本数时取全部 |
| `MODELS` | 一个或多个已经在该实验训练完成、保存最终 checkpoint 的模型 ID |
| `SEED` | 要读取哪一个训练 seed 的 checkpoint，例如 `seed_17_pin` |
| `SPLIT` | `train`、`val` 或 `test`，默认 `test` |
| `SAMPLE_SEED` | 本次抽样种子；与训练 seed 独立，不启动新训练 |
| `BATCH_SIZE` | 推理 batch size，不改变模型权重 |
| `VIEWER_SAMPLES` | HTML 展示样本上限；完整 CSV 仍含所有选中样本 |

同一次检查的所有模型使用同一批 sample IDs。抽样范围是 checkpoint 保存的实际 cohort；若原实验为 smoke 或使用数据量上限，不会混入未参与该实验的样本。默认在 test 集无放回、固定随机抽样；ID 和顺序写入 `selection_pin.json`。如果改变模型排列，抽样结果不变。

支持现有 v0、v1、v2 的 device-OP pin 门控，包括联合、蒸馏后分段、监督后分段、仅融合层、oracle、shuffle 和仅门控系列。模型名称必须完整匹配原实验记录。可以加入 `device_op_cascade_pin` 或 `device_op_gate_only_base_pin` 无门控对照；对照会实际运行，但报告明确写“无门控”，不虚构分数。当前入口针对 **final_exam 单电路 pin 实验**，不解析旧器件级实验或任意外部 checkpoint。

## 2. 运行

在项目根目录执行：

```bash
# 只预览命令，不创建目录或提交作业。
bash final_exam/single_circuit_pin/inspect_op_gate_pin.sh --dry-run

# 提交只做推理的 Slurm 作业。
bash final_exam/single_circuit_pin/inspect_op_gate_pin.sh
```

也可以不在脚本中填 `SOURCE_RUN`，每次通过参数指定：

```bash
bash final_exam/single_circuit_pin/inspect_op_gate_pin.sh \
  "$PWD/final_exam/single_circuit_pin/results_pin/你的实验目录_pin"
```

默认集群设置沿用 `b_phzhwu`、`ex01A800`、8 CPU、1 GPU、80G 主机内存、`newbase`，诊断作业时限 1 小时。可在顶部调整。提交前在登录节点只读取小型元数据并检查最终 checkpoint 是否存在，不导入 PyTorch、不解析大数据。真正的数据读取和推理在计算节点执行。CUDA 不可用会报错，不会静默改用 CPU。

已经在分配好的 GPU 节点中时可以直接执行：

```bash
bash final_exam/single_circuit_pin/inspect_op_gate_pin.sh \
  /绝对路径/原实验目录_pin --local
```

`--local` 默认仍激活脚本设置的 Conda 环境。要使用当前 Python 环境而不激活 Conda，可显式设置 `JEPA_CONDA_ENV=''`；只做小规模 CPU 检查时再设置 `DEVICE=cpu`。

数据来自原实验的 `data_pin.json`；续训目录中的 `data_source_pin.json` 引用也支持。脚本会核对保存的配置、数据指纹、manifest 和 checkpoint cohort。无需另填 raw SQLite 路径；原实验的数据文件必须仍存在。

**当前仍会在 CPU 完整解析 canonical JSON 并验证数据指纹**，之后只把选中的样本构造成推理缓存放到指定设备。`N_SAMPLES=20` 不代表只从 JSON 文件读取 20 条，因此开始时会有数据加载时间和主机内存占用。不会重新导入 SQLite 或训练 OP predictor。

## 3. 输出与日志

每次生成独立的 `<circuit>_gate_probe_<时间和随机标识>_pin`，以下路径均相对 `final_exam/single_circuit_pin/`：

| 内容 | 路径 |
|---|---|
| 检查结果 | `results_pin/<检查运行ID>/` |
| 详细推理日志 | `logs_pin/<检查运行ID>/probe_pin.log` |
| Slurm stdout / stderr | `hpc_logs_pin/<检查运行ID>/<job_id>_pin.out`、`<job_id>_pin.err` |
| 提交命令 | `hpc_logs_pin/<检查运行ID>/submission_pin.txt` |

结果目录包含：

| 文件 | 用法 |
|---|---|
| `gate_viewer_pin.html` | 下载到本地浏览器直接打开，离线使用；选择模型、样本、MOS、head，查看 D/G/S/B 的 4×4 热力图 |
| `gate_scores_pin.xlsx` | `Models` 模型汇总、`ByDeviceHead` 固定器件/方向/head 的跨样本变化、`ScorePreview` 原始行预览、`SelectedSamples` 抽样清单 |
| `gate_scores_pin.csv` | 全部样本、全部 MOS、有序引脚对、全部 head 的原始分数与实际偏置，保留数值精度 |
| `gate_summary_pin.csv` | 固定模型/器件/query/key/head 的均值、总体标准差、极值和分数阈值占比；仅统计实际生效的连接 |
| `gate_op_inputs_pin.csv` | 实际送入门控的归一化 OP；v2 另有样本内相对 OP 通道，支持标志仅作解释 |
| `probe_pin.json` | checkpoint 路径与 SHA256、原实验代码提交号、各模型完整配置、门控版本/来源/层范围、检查参数与状态 |
| `selection_pin.json` | 划分和选中的 sample IDs |

Excel 原始行预览最多 4,000 行，完整明细使用 CSV。HTML 最多展示 `VIEWER_SAMPLES` 个样本，避免检查大量样本时页面过大；通常先用 20 个样本看热力图，再扩大 `N_SAMPLES` 看跨样本统计。所有推理输入样本会建立设备缓存，因此不要把检测数量无意间设为全量。

原实验目录始终只读。新的结果/实验日志目录必须不存在，并且不得放在原实验目录内部。检查失败会在新目录记录 `status=failed` 和错误，不修改任何训练结果。

## 4. 三种分数如何理解

设门控 MLP 的输出为 logit，原始门控为：

\[
g_{d,r_q,r_k,h}=\operatorname{sigmoid}(\text{logit}_{d,r_q,r_k,h}).
\]

对实际允许门控的引脚对，附加 attention 偏置为：

\[
b=\text{strength}\cdot\log\bigl(\epsilon+(1-\epsilon)g\bigr).
\]

| 列 | 含义 |
|---|---|
| `gate_score` | 原始 sigmoid 输出，范围 [0,1] |
| `attention_bias` | 实际加到 self-attention logits 上的偏置；受保护连接为 0 |
| `effective_multiplier` | `exp(attention_bias)`，作用于 softmax 归一化前权重；受保护连接为 1 |
| `applied` | 该连接是否真的施加门控 |
| `protection_reason` | `self` 自连接、`same_net` 物理同网短接、`no_supported_op` 无支持 OP |

分数不是最终 softmax attention 权重，也不是经过校准的物理导通概率。`exp(bias)=0.5` 不能解释为最终 attention 权重精确减半，因为同一 query 的全部 key 还要共同归一化。

**方向按 attention 的 query / key 定义**：D→S 表示 query=D、key=S，S→D 单独记录。这是模型中的有序引脚对，不直接等同于电流方向。每个 head 分别输出；HTML 默认可看各头平均，但模型实际不取这个平均。层间共享门控，因此相同样本/器件/方向/head 不需要按层重复导出，具体作用层号写入报告。仅融合层版本只标记后面的融合 self-attention 层。

同网短接、同一引脚自连接等保护对，MLP 可能仍有原始输出，但实际偏置为 0；热力图用灰色标注，汇总均值/方差不把这些保护对算入。电阻、电容、外部端口、跨器件连接、graph token 和 padding 不属于这个 MOS 内门控的生效范围，不伪造它们的门控分数。

预测 OP 模型仍只使用其冻结 predictor 的输出；oracle 模型才使用真实 OP，并沿用原模型的缺失回填规则；shuffle 模型仍使用 checkpoint 保存的打乱对照。检查器不会为了观察更明显的变化，擅自把真实 OP 替换进预测模型。

## 5. 实现和验证边界

`gate_probe_pin.py` 复用 `load_checkpoint_pin` 和保存的 preprocessing，使用 forward hooks 截取实际 MLP logits、门控输入和已应用的偏置；同时检查重新映射的偏置与模型实际输出完全一致、graph token 保护成立、推理前后模型权重/缓冲区摘要不变。hook 离开单次前向时移除。报告和离线页面分别在 `gate_probe_report_pin.py` 与 `templates/gate_viewer_pin.html` 中实现；训练代码和门控公式不变。

自动测试覆盖小电路真实训练 checkpoint 回放、六个 v2 模型、v0/v1/oracle/shuffle、无门控对照、有向引脚/head、保护对、原始/过滤 OP schema、禁止重新拟合/优化、源文件不变、样本固定选择、超量取全部、续训数据引用、数据指纹变化拒绝，以及无 PyTorch 的登录节点预检查和 Slurm spool 路径。

这里的自动验证在 CPU 小型合成数据上执行；真实集群 CUDA 和大规模 checkpoint 的耗时/内存需要实际运行确认。
