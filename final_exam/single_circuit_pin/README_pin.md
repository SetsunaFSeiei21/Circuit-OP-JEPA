# 端口级单电路实验

**LASD-MoE 层融合**使用 `train_lasd_moe_pin.sh` 顶部的 `CIRCUITS`、`MODELS`、`SEEDS`。默认比较 `device_op_cascade_pin`、`lasd_moe_pin` 和 `lasd_moe_detach_pin`。运行 `bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh --headwise` 会选择 `lasd_moe_headwise_pin` 与 `lasd_moe_headwise_detach_pin`。每个模型目录的 `lasd_moe_weights_pin.csv` 可按引脚和头查看各 PE 的权重。结果、实验日志和 HPC 日志分别位于本目录的 `results_pin`、`logs_pin`、`hpc_logs_pin`。[LASD-MoE 结构与训练说明](../LASD_MOE_pin.md)

**共同下基模型训练**：在 `train_shared_decoder_pin.sh` 顶部设置 `CIRCUITS`、`MODELS`、`SEEDS`，运行 `bash final_exam/single_circuit_pin/train_shared_decoder_pin.sh`。默认比较原始 `width_decoder_pin` 与仅 cross-K/V、全部 decoder attention、FFN、attention＋FFN 四种共同下基模型。共享仅发生在一个模型内部对应 PE decoder 的相同投影处，采用从头训练的 `W_i=B_i A`，各电路独立提交。配置为 `final_exam/configs/shared_decoder_pin.json`；结果、实验日志和 HPC 日志仍分别放在本任务的 `results_pin`、`logs_pin`、`hpc_logs_pin`。失败后使用 `resume_shared_decoder_pin.sh`，详见 [共同下基训练说明](../SHARED_DECODER_TRAINING_pin.md)。

**同一模型内部的 decoder 共享下基诊断**：编辑 `inspect_shared_decoder_pin.sh` 顶部的 `CIRCUITS` / `SOURCE_RUNS` 配对、`MODELS`、`SEEDS`、候选 `RANKS` 和训练/验证样本数，运行 `bash final_exam/single_circuit_pin/inspect_shared_decoder_pin.sh`。每个 checkpoint 内分别对多个 PE decoder 的对应 Q/K/V/O 或 FFN 拟合共同下基、独立上基；多个电路独立分析，不混合权重。导出各 checkpoint 和汇总 Excel、CSV、图及可选验证误差，不训练、不修改来源 checkpoint。见 [共享下基诊断说明](../SHARED_DECODER_BASIS_pin.md)。

同一入口现已自动包含 decoder 两两右奇异子空间重叠、主角度、双向交叉基重建与向量绝对余弦热图。`RANKS` 控制比较秩，`VECTOR_LIMIT=64` 只限制向量图每轴根数。只补看权重/子空间相似度时设 `N_SAMPLES=0`、`N_VAL_SAMPLES=0`；Excel 新增 `SubspaceOverlap`、`PrincipalAngles`、`CrossBasis`、`VectorCosines`，输出与日志目录规则不变。

低秩宽 decoder 四组消融：编辑 `train_low_rank_decoder_pin.sh` 的 `CIRCUITS` / `MODELS` / `SEEDS`，用 `final_exam/configs/low_rank_decoder_pin.json` 设置 attention rank 与 FFN rank。保留原宽读出，新增仅 attention、仅 FFN、两者同时低秩化三组；只改性能端 decoder，原始器件级 OP predictor、SE/PE、MoE 保留。训练、日志、自动 Excel 和恢复命令见 [低秩 decoder 说明](../LOW_RANK_DECODER_pin.md)。

低秩/共享基底诊断：编辑 `inspect_spectrum_pin.sh` 的电路与结果目录配对、模型、seed 和校准样本数，运行 `bash final_exam/single_circuit_pin/inspect_spectrum_pin.sh`。读取已有 checkpoint，默认在 GPU 上分析 FFN 权重谱、ID train 激活误差和跨电路原坐标子空间；自动导出 Excel、CSV 和 PNG，不训练、不覆盖原结果。见 [谱分析使用说明与边界](../SPECTRAL_PROBE_pin.md)。

宽读出实验：编辑 `train_width_decoder_pin.sh` 的 `CIRCUITS` / `MODELS` / `SEEDS`，运行 `bash final_exam/single_circuit_pin/train_width_decoder_pin.sh`。支持 `width_decoder_pin`（只扩展性能读出）和 `width_decoder_op_pin`（OP 与性能两端均扩展）；默认同时运行 `device_op_cascade_pin` 对照。配置、自动 Excel、日志和恢复命令见 [宽读出说明](../WIDTH_DECODER_pin.md)。

该脚本也包含 `width_decoder_detach_pin`：只切断浅层 PE memory 通往对应 Decoder 的梯度，最后一个 PE 正常回传，Encoder 层间连接和各 Decoder 不冻结。只做此对照时设置 `MODELS=(width_decoder_pin width_decoder_detach_pin)`；配置仍是 `configs/width_decoder_pin.json`，续训仍用 `resume_width_decoder_pin.sh OLD_RUN`。

**门控 V6/V7：在 V4/V5 输入末尾显式拼接 NMOS=`[1,0]` / PMOS=`[0,1]`。** 输入由 `2F+8` 扩为 `2F+10`（41 个 OP 字段时 92 维）；类型不参与 OP 标准化，相对 OP 仍由同一样本全部有效 MOS 一起统计。V6 跨 PE2 及后续层共享门控，V7 每层独立；结构层和 PE1 不加门控，仍使用 `strength*tanh(z)`。每版四种训练策略和已有无门控参考，入口为 `final_exam/single_circuit_pin/train_op_gate_pin_v6.sh` / `train_op_gate_pin_v7.sh`，配置为 `final_exam/configs/op_gate_pin_v6.json` / `op_gate_pin_v7.json`。详见 [V6/V7 输入、模型、预算和恢复说明](../OP_GATE_V6_V7_pin.md)。

**门控 V4/V5：只调节 PE2 及后续融合 Self-Attention，并支持增强和抑制。** V4 跨层共享一套 V3 M-MLP；V5 每个启用层独立一套（初值相同、独立更新）。结构层和 PE1 不加门控。默认偏置 `strength*tanh(z)` 位于 `[-1,1]`，近零初始化；新配置不使用旧 `epsilon`/`initial_gate`。每版四种训练方式加无门控参考，入口分别为 `final_exam/single_circuit_pin/train_op_gate_pin_v4.sh` 和 `train_op_gate_pin_v5.sh`，配置为 `final_exam/configs/op_gate_pin_v4.json`/`op_gate_pin_v5.json`。需要 `model.cross_layers >= 2`。完整模型、阶段预算、路径与恢复说明见 [V4/V5 说明](../OP_GATE_V4_V5_pin.md)。

**无门控 PE 注意力检查**：编辑 `inspect_pe_attention_pin.sh` 顶部的结果路径、电路、样本数、模型和 seed，运行 `bash final_exam/single_circuit_pin/inspect_pe_attention_pin.sh`。默认加载 `device_op_cascade_pin`，按样本、PE 层、head、MOS 输出 D/G/S/B 的 4×4 QK/logit/softmax 矩阵及权重去向，生成离线 HTML、Excel 和完整 CSV。只读取原本没有门控的 checkpoint，不重训，不在门控模型上临时关门控；详见 [PE 注意力检查说明](../PE_ATTENTION_PROBE_pin.md)。

HTML 顶部同时展示所有 PE 的**全电路 N×N 引脚大图**，一层一张，包含跨器件注意力，各层共用色标，支持每张图下载 PNG；下方保留 MOS 4×4 局部图。完整矩阵另存 `full_attention_pin/` 的 NPZ。旧版只有 MOS 子矩阵，缺少跨器件数据，需要重新运行同一探测脚本（无需重新训练）。`python -m final_exam.attention_probe_report_pin --source /path/to/pe_attention_probe_result` 仅刷新已有页面数据，不能补出旧报告缺失的完整矩阵。

**门控＋decoder 分段微调**：v1/v2/v3 各增加全层和仅融合层两个变体，最终冻结 OP predictor 与 encoder，同时更新门控和 decoder。完整名称、轮数与续训规则见 [版本化门控＋decoder 说明](../GATE_DECODER_VERSIONS_pin.md)。

**门控 v3（FlowNet M-MLP）**见 [OP_GATE_V3_pin.md](../OP_GATE_V3_pin.md)。入口 `final_exam/single_circuit_pin/train_op_gate_pin_v3.sh`，配置 `final_exam/configs/op_gate_pin_v3.json`。保留 v2 的双 OP 通道，将评分网络改为三层 MoLinear，默认每层 16 个专家；`op_gate_pin.n_experts` 和 `mlp_dropout` 可独立设置。八个 `_pin_v3` 模型对应四种训练方法与两个层范围（含冻结 encoder、联合调整门控＋decoder），默认搭配已有无门控基线，旧版本继续保留。结果、实验日志和 HPC 日志仍分别放在本任务的 `results_pin/`、`logs_pin/`、`hpc_logs_pin/`，run ID 含 `_op_gate_v3_`。

## 检查已训练模型的门控分数

编辑 `inspect_op_gate_pin.sh` 顶部的 `SOURCE_RUN`、`CIRCUIT`、`N_SAMPLES`、`MODELS`、`SEED`，然后运行：

```bash
bash final_exam/single_circuit_pin/inspect_op_gate_pin.sh
```

默认提交只做推理的 Slurm 作业，从原实验 test 集抽样，不重训、不覆盖原结果。输出离线 HTML 热力图、XLSX 与完整 CSV；各模型共用选中样本，D→S/S→D 分开，逐 head 记录原始门控和实际 attention 偏置。结果、实验日志、HPC 日志分别在本任务的 `results_pin/`、`logs_pin/`、`hpc_logs_pin/` 新目录。路径参数、`--dry-run`、`--local` 及分数解释见 [门控检查说明](../OP_GATE_INSPECTION_pin.md)。

## 训练入口

门控 **v2** 的独立入口为 `train_op_gate_pin_v2.sh`，默认选择八个 `_pin_v2` 门控模型和原有无门控基线。编辑该脚本的 `CIRCUITS`、`MODELS`、`SEEDS`，在 `final_exam/configs/op_gate_pin_v2.json` 中设置训练预算、`op_gate_pin.norm_eps` 及 `dataset.max_samples_per_circuit`。运行：

```bash
bash final_exam/single_circuit_pin/train_op_gate_pin_v2.sh --dry-run
bash final_exam/single_circuit_pin/train_op_gate_pin_v2.sh
```

v2 将原标准化 OP 与当前样本内所有有效 MOS 的逐字段相对 OP 拼接，再拼接有序引脚 one-hot，送入一个共享门控 MLP。输出 run ID 含 `op_gate_v2`，结果、实验日志与 HPC 日志仍分别保存在 `results_pin/`、`logs_pin/`、`hpc_logs_pin/`。教师、OP 预测器、各模型配置和自动 Excel 保留。完整定义和续训命令见 [v2 说明](../OP_GATE_V2_pin.md)，历史版本记录见 [record.txt](../record.txt)。

门控 **v1** 的独立入口为 `train_op_gate_pin_v1.sh`，默认包含八个 `_pin_v1` 门控模型和 `device_op_cascade_pin` 无门控对照。编辑脚本顶部的 `CIRCUITS`、`MODELS`、`SEEDS` 后运行：

```bash
bash final_exam/single_circuit_pin/train_op_gate_pin_v1.sh --dry-run
bash final_exam/single_circuit_pin/train_op_gate_pin_v1.sh
```

配置为 `final_exam/configs/op_gate_pin_v1.json`；结果、实验日志、HPC 日志仍分别位于本目录的 `results_pin/`、`logs_pin/`、`hpc_logs_pin/`，run ID 含 `op_gate_v1`。默认九个评测入口之外，蒸馏无门控基座也会在 Excel `GateStudy` 中比较。各版本输入结构、八个门控模型的训练流程、预算与 checkpoint 兼容规则集中记录在 [../record.txt](../record.txt)。原有脚本继续启动原有版本。

本入口与器件级 `single_circuit/` 并列。每个被观测到的器件引脚是一个节点，命名外部端口也有独立节点；MOS 的 D/G/S/B 不合并成一个学习节点。图上保留同器件关系、同网络连接及端口接入关系。编码、输入权重和教师均使用端口级版本。

真实 OP 仍属于器件。原有 `predicted_op_pin`/学生蒸馏系列的 OP predictor 对同一器件的引脚表示聚合，预测一组器件 OP，再广播到所属引脚；OP 监督按器件计算一次，不因 MOS 拆成四个引脚就重复加权四次。外部端口无所属器件，不参与 OP 监督。部署级联模型不读取真实 OP 或其缺失模式。

编辑 `train_pin.sh` 顶部的 `CIRCUITS`、`MODELS`、`SEEDS`、`CONFIG` 与集群设置。默认 Opamp1、13 个 `_pin` 模型、seeds 17/29/43，每个电路提交一个独立作业。作业内依次运行所选模型；仅为级联学生自动训练必需的教师与 OP predictor。

```bash
bash final_exam/single_circuit_pin/train_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_pin.sh --smoke
bash final_exam/single_circuit_pin/train_pin.sh
```

`--dry-run` 不提交、不创建输出目录。`--smoke` 提交短轮次训练，不自动改变样本数。在所用配置中设置 `dataset.max_samples_per_circuit: 20000` 可先按电路抽取最多 20,000 条再划分；默认 `null` 为全量，见 [数据抽样说明](../../docs/DATA_SUBSAMPLING_ZH.md)。原始输入继续使用 `runs/raw/full_data/<电路名>/`，无需复制或重命名 SQLite/网表。输入单位检查和器件级入口相同，不自动猜测几何单位。

`train_pin.sh`、`train_op_gate_pin.sh`、`train_staged_op_gate_pin.sh` 和 `train_staged_op_gate_fusion_pin.sh` 均通过 `JEPA_PROJECT_ROOT` 传递仓库绝对路径。Slurm 会将 `run_pin.sh` 复制到 `/var/spool/...` 执行，因此计算节点入口不再直接用该副本的位置推算项目根目录；它依次检查显式路径、`--chdir` 对应的工作目录和 `SLURM_SUBMIT_DIR`，最后才尝试本地脚本相对路径。找不到 `final_exam/common/launch_pin.py` 时提前报错，启动日志打印 `Project root:`。这同样适用于新增的 OP 门控实验，结果仍保存在当前仓库。修复后需要重新提交失败的作业。

模型列表为：

```bash
MODELS=(
  mlp_pin xgboost_pin gat_pin deepgen_supervised_pin deepgen_op_pretrained_pin gtn_pin
  zerosim_supervised_pin accformer_pin
  predicted_op_pin student_predicted_op_pin student_predicted_op_distill_pin
  op_oracle_pin design_op_oracle_pin
)
```

配置模板为 `final_exam/configs/default_pin.json`。`op_transfer` 管理训练预算、batch size、学习率及早停；`final_exam_pin` 管理模型选择、模型覆盖及 XGBoost 设置。虽然为兼容模型接口仍使用 `model.max_devices` 这个字段名，在端口级入口中它表示**引脚节点与独立外部端口节点总数上限，不含额外 graph token**；超限时报错，不截断。不要直接把器件数量当成端口级容量。

默认神经网络下游最大预算保持普通/联合模型 60 轮，顺序蒸馏 encoder 30 + decoder 30；教师 60 轮、OP predictor 30 轮作为额外依赖成本记录。每阶段验证集早停并恢复验证最佳权重。XGBoost 单独按 boosting rounds 记录，不冒充 epoch。

## 日志和结果

所有产物位于当前仓库 `final_exam/single_circuit_pin/`，每次使用以 `_pin` 结尾的唯一 run ID；人工传入的 run ID 也必须以 `_pin` 结尾。

| 内容 | 相对任务目录的路径 |
|---|---|
| Slurm 标准输出、错误 | `hpc_logs_pin/<run_id_pin>/<job_id>_pin.out`、`<job_id>_pin.err` |
| 提交配置快照、记录 | `hpc_logs_pin/<run_id_pin>/submitted_config_pin.json`、`submission_pin.txt` |
| 数据导入和命令日志 | `logs_pin/<run_id_pin>/launch_pin.log` |
| 实验总日志 | `logs_pin/<run_id_pin>/experiment_pin.log` |
| 各 seed/模型训练、评估日志 | `logs_pin/<run_id_pin>/seed_<seed>_pin/<model_pin>/train_pin.log`、`evaluate_pin.log` |
| 本次选中的输入数据及启动配置 | `results_pin/<run_id_pin>/data_pin.json`、`config_pin.json` |
| 作业状态与路径关联 | `results_pin/<run_id_pin>/run_pin.json` |
| 模型结果与训练状态 | `results_pin/<run_id_pin>/training_pin/` |
| 自动生成 Excel | `results_pin/<run_id_pin>/training_pin/summary_pin.xlsx` |

共享教师与 predictor 保存在 `training_pin/seed_<seed>_pin/shared_pin/`；每模型独立保存 `config_pin.json`、`checkpoint_pin.pt`（XGBoost 为原生模型文件及 JSON）、`test_pin.json`。每阶段保存历史、成本和最佳检查点，名称均带 `_pin`。结果和日志不会覆盖器件级实验，也不提交到 Git。

脚本默认账户 `b_phzhwu`、分区 `ex01A800`、环境 `newbase`，每作业 8 CPU/1 GPU/80G 主机内存。正式作业最长 10 天，烟测 30 分钟。CUDA 不可用时报错；原始数据 IO、JSON 解析和静态图准备仍需 CPU，不能解释为整个实验没有 CPU 工作。

配置在提交时保存快照。正在排队或运行的作业不应共享一份不断修改的代码或原始数据；此入口没有复制代码/data 快照。所有选定模型与 seeds 完成且 Excel 成功生成，才标记实验 `complete`。

## 重新整理 Excel

按模型挑选最好/最差 seed：编辑 `summarize_selected_seeds_pin.sh` 顶部的 `SOURCE_RUN`、`K`、`MODELS_BETTER`、`MODELS_WORSE`，运行 `bash final_exam/single_circuit_pin/summarize_selected_seeds_pin.sh`。按测试集平均 R² 选择完整 seed 后，复用现有 MAE/RMSE/R²、配置、成本和门控对照汇总；未列出模型保留全部 seed。新表显式标注测试集选 seed，附完整候选排名和未筛选统计，原结果不覆盖。详见 [选择 seed 的规则、路径与使用说明](../SEED_SELECTION_pin.md)。

```bash
bash final_exam/single_circuit_pin/summarize_pin.sh \
  final_exam/single_circuit_pin/results_pin/<run_id_pin> \
  --output final_exam/single_circuit_pin/results_pin/<run_id_pin>/summary_review_pin.xlsx
```

输出文件名必须以 `_pin.xlsx` 结尾。默认拒绝覆盖，明确指定 `--overwrite` 才替换。未完成实验默认拒绝导出；`--allow-incomplete` 可生成明确标注的临时表，并禁用最优高亮。

若所有训练/测试完成，但最后导出失败，`training_pin/run_pin.json` 的状态是 `failed_export`，可在**新 Python 进程**中只恢复 Excel：

```bash
bash final_exam/single_circuit_pin/summarize_pin.sh \
  final_exam/single_circuit_pin/results_pin/<run_id_pin> --recover-export
```

此操作仍严格检查原计划的模型、seed、split、每模型配置、训练成本及门控前后对照；不能跳过缺失结果。只写 `training_pin/summary_pin.xlsx`，不读取完整数据、不加载 checkpoint、不训练，也不修改原始失败状态或 HPC 日志。Excel 显式注明导出已恢复。普通 `failed`/`running` 状态不能用这个参数代替训练恢复。版本混用排查及详情见 [导出恢复说明](../EXPORT_RECOVERY_pin.md)。

Excel 含 MAE、RMSE、R2、PerSeed、Statistics、TrainingCost、Configuration。跨 seeds 使用样本标准差；只有一个 seed 时留空标准差。真实 OP 参考与无需真实 OP 的模型分组比较；样本数或 seeds 不匹配时不高亮最优。每个模型名称保留 `_pin`，不与器件级结果混合统计。

本入口完成的是第一阶段单电路流程，不代表多电路/OOD、消融或迁移学习已经实现。

## 分阶段门控（三个新增模型）

已有八个门控模型继续使用 `train_op_gate_pin.sh`。新增的三个训练顺序对照使用独立入口：

```bash
bash final_exam/single_circuit_pin/train_staged_op_gate_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_staged_op_gate_pin.sh
```

在该脚本顶部直接修改 `CIRCUITS`、`MODELS`、`SEEDS`，默认二级折叠运放、seeds 17/29/43。配置为 `final_exam/configs/staged_op_gate_pin.json`。模型名称及最后阶段如下：

| 模型 | 前置训练 | 最后阶段 |
|---|---|---|
| `device_op_distill_gate_pin` | 教师 → 学生编码器蒸馏 → decoder | 只训练门控 |
| `device_op_distill_gate_decoder_pin` | 复用同一个蒸馏学生 | 门控＋decoder |
| `device_op_supervised_gate_pin` | 关闭门控直接监督训练 | 只训练门控 |

这三个模型与原八个门控模型都使用**器件级 OP 预测器＋引脚级下游编码器**；不是上文原有级联系列的“引脚编码后按器件池化”预测器。部署时不需要真实 OP 或教师。

最终门控轮数由 `op_transfer.epochs.gate` 独立控制，默认 30；前置训练之外会追加训练成本，不声称完整预算相等。保存两类无门控基模型和真实 OP 教师，结果仍放在本任务的 `results_pin/`，实验和 HPC 日志仍分别放在 `logs_pin/`、`hpc_logs_pin/`。自动 Excel 增加 `GateBeforeAfter` 比较页，展示逐 seed 的无门控基模型与最终模型差值。

完整阶段、冻结规则、配置、成本和路径参见 [分阶段 OP 门控说明](../STAGED_OP_GATE_pin.md)。

## 仅融合层门控（三个对照模型）

新增模型沿用上述三种训练顺序，但门控只加到融合块的 Self-Attention；结构前缀、融合块 Cross-Attention 和 decoder 不接收门控偏置。默认 `encoder_layers=6`、`cross_layers=3`，因此作用于后 3 层；数量随 `cross_layers` 配置改变。

```bash
MODELS=(
  device_op_distill_gate_fusion_pin
  device_op_distill_gate_decoder_fusion_pin
  device_op_supervised_gate_fusion_pin
)
```

新入口默认只运行这三个模型，原入口仍运行原三个模型：

```bash
bash final_exam/single_circuit_pin/train_staged_op_gate_fusion_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_staged_op_gate_fusion_pin.sh
```

在新脚本顶部编辑 `CIRCUITS`、`MODELS`、`SEEDS`；默认配置为 `final_exam/configs/staged_op_gate_fusion_pin.json`，训练预算与旧分阶段门控模板一致。新 run ID 含 `_staged_op_gate_fusion_`，结果和两类日志仍在本任务现有目录内。可在同一个 `MODELS` 数组中列出两组共六个模型，以便共享相同配置下的 OP 预测器、教师和无门控基模型；单独提交的作业分别训练依赖。

六个分阶段门控模型都支持原恢复入口。范围、训练顺序和具体路径见 [仅融合层门控说明](../FUSION_STAGED_OP_GATE_pin.md)；恢复规则见 [断点续训说明](../RESUME_STAGED_OP_GATE_pin.md)。

## LASD 宽 Decoder 分数读出

新增 `width_decoder_score_pin`（逐 Head R/T）和 `width_decoder_lasd_pin`（再加逐 Head ΔR/ΔT）。
每个 PE 的独立节点 MLP 改变 Decoder memory；Encoder 消息路径、独立性能 Query 和最终 MoE 保留。
原 `width_decoder_pin` 作对照。配置：`final_exam/configs/lasd_pin.json`。
配对断梯度模型为 `width_decoder_score_detach_pin` 和 `width_decoder_lasd_detach_pin`：浅层 memory/Score 在可训练节点 MLP 之前 detach，最深层正常回传；LASD 差分的上一层 Score 也停止梯度。OP 预测器不变。

```bash
bash final_exam/single_circuit_pin/train_lasd_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_lasd_pin.sh
```

在脚本顶部编辑 CIRCUITS / MODELS / SEEDS，每个电路一个 GPU 作业。原结果与日志目录结构不变。
`summary_pin.xlsx` 新增 LASD 页；各新模型目录导出 `lasd_scores_pin.csv` 和 `lasd_diagnostics_pin.json`。
断点续训复用 `resume_width_decoder_pin.sh OLD_RUN`。完整数学定义、mask/dropout 适配、已知解释边界见
[LASD_pin.md](../LASD_pin.md)。
