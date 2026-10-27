# Circuit-OP-JEPA

器件级结构编码、OP 表示对齐，以及可按需选择的八个模式：

- `zerosim_pin`：独立复现的 **pin-node ZeroSim-style** 监督基线，训练 encoder + metric-query decoder。
- `zerosim_device`：器件节点 ZeroSim-style 监督基线，训练 encoder + decoder。
- `jepa_zerosim`：器件节点共享结构 OP 预训练，然后冻结整个输入 encoder，只训练 decoder；性能推理不读取 OP。
- `zerosim_pin_op`：在匹配的 pin-node 监督基线上，加入逐器件电气 OP 与观测 mask；训练和推理均使用 OP。
- `zerosim_device_op`：在匹配的 device-node 监督基线上，加入逐器件电气 OP 与观测 mask；训练和推理均使用 OP。
- `zerosim_pin_op_only`：沿用 `zerosim_pin_op` 的拓扑编码、融合层和 decoder，仅输入拓扑及 DC OP，移除设计参数输入分支。
- `zerosim_device_op_only`：沿用 `zerosim_device_op` 的对应架构，仅输入拓扑及 DC OP，移除设计参数输入分支。
- `jepa_zerosim_ema`：独立结构编码器与 OP 交叉注意力目标分支；目标结构/融合主干用 EMA，设计与 OP 字段 embedding 各自梯度更新，随后冻结输入 encoder 训练 decoder，性能推理不读取 OP。

`zerosim_pin` 是本仓库的 clean-room 架构复现，不是直接导入官方代码，也不声称逐算子、权重或论文结果完全一致。各模式都用同一份原始样本、持久化划分、标签预算和目标归一化，JEPA 的额外预训练成本单独记录。两个 `*_op` 模式是加入 OP 的信息增强对照；两个 `*_op_only` 模式用于检验固定拓扑与 OP 后设计参数输入的作用。四个直接 OP 模型在训练和推理时均需要 OP。OP 采集成本未测量，报告标为 `not_measured`/`null`，不默认为零。

默认名单仍为原有五个模型。新 EMA 模型须显式选择 `--models jepa_zerosim_ema`，或与原模型一起选择 `--models jepa_zerosim jepa_zerosim_ema`；`--models all` 显式选择八个模型。现有 Bash 启动脚本中的 `MODELS` 数组保持不变，使用新模型时自行加入名称。架构、更新规则和用法见 [EMA JEPA 说明](docs/EMA_JEPA_ZH.md)。

实验汇总现在自动生成 `comparison/model_comparison.xlsx`，并排展示各模型 MAE、RMSE、R² 的均值 ± std。旧结果可用 `python scripts/run/export_results_excel.py --input 实验目录` 补导出，无需重新训练或安装额外依赖，见 [Excel 整理说明](docs/EXCEL_RESULTS_ZH.md)。

新版完整交接与独立复审提示见 [docs/REVIEW_HANDOFF_ZH.md](docs/REVIEW_HANDOFF_ZH.md)；本轮有限验证与历史三模式证据分开见 [verification/five_mode/RESULTS.md](verification/five_mode/RESULTS.md)。

## 快速开始

Python 3.10+、稳定版 PyTorch 2.6+，建议独立环境。旧版 PyTorch 的 `weights_only=True` 存在已知安全缺陷，运行时也会拒绝加载 checkpoint；仅加载可信来源的模型文件。CPU 示例：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
python -m device_jepa smoke --output runs/synthetic_jepa
```

真实 Opamp2 的短流程检查：

```bash
python -m device_jepa smoke_real \
  --input /path/to/tmp_data/Opamp2 \
  --config configs/opamp2_smoke.json \
  --output runs/opamp2_smoke
```

真实数据不包含在仓库中。目前收到的 Opamp2 只有一个电路，32 个设计样本，每个 15 个 MOS + 1 个电流源，44 个原始 OP 字段、9 个性能指标。配置明确 `ood_circuits: []`，**该 smoke 不能验证跨电路 OOD 泛化，也不是预测效果实验**。不会因为原始 `feasible=false` 就丢掉样本。

通过 `--models jepa_zerosim zerosim_device` 选择本次要训练、测试和汇总的模型，也可只选一个；`--list-models` 列出可用名称。选择规则、配置和提交示例见 [模型选择说明](docs/MODEL_SELECTION_ZH.md)，多个随机种子及成对结果处理见 [scripts/README.md](scripts/README.md)。脚本分为 `scripts/git/`、`scripts/smoke_test/`、`scripts/run/`，与 Analog-Agent 的组织方式对应。先做 smoke，再运行正式实验。

大型原始数据可直接通过 `scripts/run/submit_raw_comparison.sh --input 原始电路目录 --max-table-rows none --device cuda` 提交；导入和所选模型训练均在计算节点执行，默认仍为预览，实际提交需设置 `DRY_RUN=0`。省略 `--output` 时，结果统一放在原仓库的 `runs/gpu_full/电路名_YYYYMMDD_HHMMSS/`，从 Git worktree 提交也使用此位置。该入口自动从 SQLite 读取单电路身份并生成独立实验配置，完整双电路示例及资源要求见 [运行说明](scripts/README.md#slurm)。

单电路可用 `bash scripts/run/train_single_circuit.sh`；跨电路联合训练与 OOD 可用 `bash scripts/run/train_ood.sh`，后者自动在计算节点合并原始数据。两者均可在 Bash 文件开头设置模型、种子和配置，默认正式提交，加 `--dry-run` 预览。OOD 默认用 5t_ota/Opamp0/Opamp1 训练，留出整个 two_stage_folded_opamp；调整方法见 [OOD 运行说明](docs/OOD_RUN_ZH.md)。

## 数据处理流程

```bash
python -m device_jepa import_analog_agent \
  --input /path/to/Opamp2 \
  --config configs/opamp2_smoke.json \
  --output runs/opamp2/data.json

python -m device_jepa split \
  --data runs/opamp2/data.json \
  --config configs/opamp2_smoke.json \
  --output runs/opamp2/manifest.json
```

导入器只读 SQLite 和 SPICE 文本，不运行仿真、网表、外部 Python 或 SQL 脚本。它会：

1. 提取 DUT 器件、类型、模型、D/G/S/B 或 P/N、端口和参数表达式。
2. 按对应样本的 design vector 解析 W/L/m 等表达式。采用有界语法树白名单，不使用 `eval`；不支持的语法明确报错，不静默漏器件。
3. 按 `(sample_id, device_name)` 精确匹配 OP，保留全部 44 个原始字段及符号，性能 NaN 转为 `null`。
4. 将电流源等非 MOS 保留在输入图中，但不把它们当作 MOS OP 对齐目标。
5. 保存原始 reasons、check scope、阈值来源和运行 provenance。多来源合并通过 source_id + circuit_id + run_id 关联检查记录，不能仅用不同电路可能重复的 run_id。collector 工作检查类别只进入 JEPA 的 OP 目标分支，不进入共享结构、设计输入或四个直接 OP 监督模型。

预训练启用状态特征时，会从原始证据重新推导并核对已存编码与配置；不匹配或缺少 provenance 时要求重新导入，不能把缺失编码静默当作全未知。显式 `features: []` 可关闭状态输入。

当前参数采用明确单位：`W_m`、`L_m`、`m`、`R_ohm`、`C_F`、`dc_current_A`、`dc_voltage_V`；原始 OP 的 `w_um`、`l_um` 等不改写。默认训练 OP 字段排除 `w_um/l_um/multiplicity`，避免从 OP 几何字段直接抄回设计参数。详细契约见 [docs/data_schema.md](docs/data_schema.md)。

## 异构器件和缺失语义

具体 NMOS/PMOS/R/C/源类型保留，并增加 `active/passive/source/unknown` 粗分类；pin 图有额外 `port` 类，粗分类不替代具体类型。所有新训练模式都使用同一输入契约。

- `parameter_mask` 表示实际观测，`parameter_applicability` 表示字段是否适用于该器件。
- 适用但缺失的参数保留命名 token 与 observed=0；不适用字段不进入参数注意力；真实零是 observed=1。
- R/C 和源使用自己的真实参数字段，不被统一塞入 W/L 槽位。未知类型可显式声明适用字段。
- 当前 collector 只有 MOS OP，非 MOS 仍保留在图中，但不会编造其 OP；这不是声称无源器件没有物理 DC 状态。

两个 `*_op` 模式把命名电气 OP 值及观测位加入后段 own-node 输入融合；pin 模式只复制到父器件自己的 pins，端口无伪 OP。新增 OP token 会增加参数量，所以配对比较是信息增强消融，并非严格参数量匹配的纯因果实验。

两个 `*_op_only` 模式移除全部设计参数 token、观测/适用性 mask 和设计参数 embedding/投影，仅保留拓扑身份信息与电气 OP。W/L/m、R/C、源 DC 设定值均不进入网络；自定义 OP 字段也不得包含 `w_um/l_um/multiplicity`。其余骨干结构相同，移除输入分支后总参数量会减少。原始设计参数仍用于完整性检查和设计分组，见 [模型选择说明](docs/MODEL_SELECTION_ZH.md#仅拓扑与-dc-op-的消融模型)。

## 划分与归一化

配置明确指定：

```json
{
  "dataset": {
    "train_circuits": ["CircuitA", "CircuitB", "CircuitC", "CircuitD"],
    "ood_circuits": ["CircuitE"],
    "split_ratios": [0.8, 0.1, 0.1],
    "split_seed": 17,
    "finetune_label_fraction": 1.0
  }
}
```

先按电路分出 OOD，再在每个 ID 电路内按 `design_key/design_id/group_id` 和精确重复设计分组划分。标签预算只抽取 train 内的完整设计组，绝不从验证/测试补标签。OOD 不参与 OP 预训练、任何统计量拟合、模型选择或标签抽样。

manifest 保存版本、样本集合、指纹和随机种子。再次运行只验证；增删数据或换配置会报错，必须显式选择新实验目录，不会偷偷重新划分。阶段一可以没有性能标签，阶段二可只补充/更改性能值；输入/OP/身份/划分仍必须完全一致。

按物理字段拟合变换，默认每个训练电路等权：

- 电压：线性 z-score。
- 带符号、跨数量级的电流/电导/电容/比值：`asinh(x/s)` 后 z-score。
- `s` 可固定，也可只用 ID train 拟合；默认是电路均衡的非零绝对值加权中位数。
- `0` 是有效测量，missing/padding 不参加拟合。不对负值取 log，不 blanket abs，不裁剪极值。
- OP/输入统计使用 ID train，性能统计只使用 `labeled_train`。所有变换/尺度/字段顺序保存到 checkpoint，后续评估与冻结输入复用。

## 共享结构与两阶段

本节描述原模型 `jepa_zerosim`；新增 `jepa_zerosim_ema` 的独立结构与 EMA 规则见 [EMA JEPA 说明](docs/EMA_JEPA_ZH.md)。

默认 `encoder_layers=6, cross_layers=4`，切分点是 **第一次参数融合之前**：

- 前 2 层完整纯结构 prefix：器件 embedding、位置、端子连接及 local/global attention。
- 后 4 层设计专属：local/global attention + own-parameter cross-attention。

这不是“只共享最初 embedding”，也不是“6 层全部共享”。两个分支引用同一个结构模块对象，结构 forward 只执行一次，复用同一份 dropout 结果；两条梯度路径都回到共享参数，没有 EMA、detach 或复制后同步。

```text
                  同一结构模块 S(G)，只算一次
                    /                   \
设计参数 -> 后4层参数融合 -> h       concat(S, OP, mask, states) -> shared OP MLP -> z
                         直接对齐 h 与 z
```

已移除独立 predictor。目标是 MSE + variance/covariance anti-collapse 正则，不是原版 I-JEPA 的 EMA/masked-context 复现。共享结构也可能形成 shortcut，必须在正式实验检查同拓扑不同设计的表示差异、OP 打乱消融与下游迁移，不能只看 loss 下降。

阶段二对整个输入 encoder `requires_grad=False + eval + no_grad`，优化器只有 metric-query decoder；每个 epoch 检查 encoder 参数和 buffers 的摘要完全不变。decoder 读取 `[graph token, 全部器件 token]`，不是 pooled 单 token。

## OP 工作检查类别

这里编码的是 collector 的**检查结果**，不是 cutoff/triode/saturation 三种互斥工作区。

- current：missing / below threshold / pass。
- saturation margin：missing / below threshold / pass。

两个组可以同时失败。`:saturation` 是“不满足饱和裕量要求”的 violation，不是“处于饱和区”。默认 `collector_checked`，没有检查证据的组全部 mask=false；“没有 reason”绝不等于正常。`all_mos` 是需显式选择的派生重算模式，不冒充原 collector 的原始检查。类别保持 one-hot，不进行数值 z-score。

## 单模式命令

```bash
python -m device_jepa pretrain --data DATA --manifest MANIFEST --config CONFIG --output RUN
python -m device_jepa train_decoder --data DATA --pretrained RUN/pretrain_best.pt --config CONFIG --output RUN
python -m device_jepa train_baseline --mode zerosim_pin --data DATA --manifest MANIFEST --config CONFIG --output RUN_PIN
python -m device_jepa train_baseline --mode zerosim_device --data DATA --manifest MANIFEST --config CONFIG --output RUN_DEVICE
python -m device_jepa train_baseline --mode zerosim_pin_op --data DATA --manifest MANIFEST --config CONFIG --output RUN_PIN_OP
python -m device_jepa train_baseline --mode zerosim_device_op --data DATA --manifest MANIFEST --config CONFIG --output RUN_DEVICE_OP
python -m device_jepa evaluate --data DATA --checkpoint CHECKPOINT --split test --output report.json
```

正式多电路实验请修改配置，不要把单电路重复复制成五个电路。默认训练配置在 `configs/default.json`，模型结构参数在 `model` 下，训练参数在 `training` 下。JEPA decoder 与四种非 JEPA 模型的轮数独立设置：

| 配置项 | 默认值 | 控制范围 |
| --- | ---: | --- |
| `training.batch_size` | 32 | 每次训练迭代的样本数 |
| `training.pretrain_epochs` | 30 | JEPA 表示预训练 |
| `training.decoder_epochs` | 50 | JEPA 冻结 encoder 后的 decoder 训练 |
| `training.baseline_epochs` | 80 | 四种非 JEPA 模型各自的端到端监督训练 |

五模式比较按**最大总训练轮数**对齐：`pretrain_epochs + decoder_epochs == baseline_epochs`，默认是 JEPA 最多 30＋50、其他模型各最多 80 轮。非 JEPA 训练不再借用 `decoder_epochs`；已有自定义配置需要显式添加 `baseline_epochs`。修改 JEPA 两阶段轮数后，应同时调整基线总上限，比较入口会在输出写入和训练前拒绝不匹配的预算。`--smoke` 将 JEPA 设为 1＋1 轮，其他模型各 2 轮。相同轮数上限不代表 FLOPs、参数量、实际耗时或 OP 信息条件相同；报告继续分别记录阶段成本。

0.4.3 默认配置为所有模型加入独立早停：`training.early_stopping` 默认 `enabled=true, patience=10, min_delta=0.0, min_epochs=1`。监督阶段监控 val 归一化性能 MSE，JEPA 预训练监控 val OP 对齐损失；test/OOD 只在选好 checkpoint 后评估。报告包含 R²、MAE、RMSE，R² 不作为当前选模或早停目标。早停后实际轮数可能不同，报告分别记录上限、执行轮数、最佳轮数和停止原因。关闭早停才能要求实际总轮数相同；旧自定义配置缺少该对象时保持固定轮数。配置与报告字段见 [早停说明](docs/EARLY_STOPPING_ZH.md)。

smoke 配置使用小模型与极少 epoch。0.4.0 的 checkpoint/preprocessing 容器保持 v2，输入契约升级为 `input_contract_version=3`：加入仅从训练集拟合的器件模型名和命名端口关联。新 manifest 要求 `design_fingerprint_version=2`，修复模型身份与正负零分组；旧 manifest 必须从原始 canonical 数据新建实验。旧输入架构仍可重建，旧容器 v1 仍不兼容；不同输入契约不能混作一组公平比较。

独立审查发现的问题、逐项修复和验证范围见 [修复说明](docs/INDEPENDENT_AUDIT_FIXES_ZH.md)。0.4.1 的[边界修复](docs/BOUNDARY_FIXES_ZH.md)使 `smoke` / `smoke_real` 在任何输出写入前拒绝非空目录；训练和验证选模在标签归一化前就排除训练中没有观测的指标。正式评估仍保留全部指标。直接训练命令也会拒绝覆盖已有阶段产物；JEPA 第二阶段保留真实预训练协议与种子。导入大型 SQLite 时可显式提高 `--max-table-rows`，或用 `none` 取消行数限制；canonical 数据仍需放入内存。

0.4.2 将五模式的训练与验证改为设备常驻数据路径。指定 `--device cuda` 时，数值归一化、训练集统计拟合、图关系构建、pin 特征展开、batch 选择及模型计算在 GPU 上执行；同拓扑的图只缓存一份。JEPA 第二阶段提前计算并缓存冻结 encoder 的输出，各轮只训练 decoder。数据文件解析、字符串身份校验、checkpoint/JSON 写入和最终报表统计仍由 CPU 处理。CUDA 不可用或显存不足时会报错，训练不会静默转到 CPU。

`training.progress_every` 控制 batch 进度间隔，默认 100；日志显示阶段、epoch、batch 比例、loss、耗时和 ETA，准备数据时也显示已解析样本数及常驻显存数据大小。完整实验应使用统一代码版本和新输出目录；旧任务需停止后重新提交，比较器会拒绝混合旧标量数据路径与新常驻路径的报告。总训练预算仍为 JEPA 30＋50、各基线 80 轮。实现和验证边界见 [GPU 数据路径说明](docs/GPU_PIPELINE_ZH.md)。

## 代码说明与建议审查顺序

见 [docs/review_order.md](docs/review_order.md)。重点顺序：adapter → states → data/split/normalization → tensors → shared model → 两阶段冻结 → baselines → comparison → scripts。

## 边界

- 不是无条件通用 SPICE parser；不支持的 DUT、表达式或未知模型会报错。
- 当前 context 要求固定；不同工艺角/温度/供电/负载需显式建模后才能混训。
- 学习位置编码按稳定器件 ID 排序，不保证器件重命名/图同构不变。
- 精确重复检测不求解 net 重命名图同构；跨来源重复应提供可靠 group_id。
- 标准 checkpoint 支持阶段衔接与评估，不支持任意中断恢复；现有数据只做短 smoke，不宣称性能或 OOD 优势。
- 来源、原实现与本实现差异见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[docs/architecture.md](docs/architecture.md)。

数值评估使用 float64 稳定反变换，物理单位真值直接读取原始标签，不经 float32 归一化往返。CUDA 确定性要求在初始化前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`（代码在设种子时默认设置，允许显式 `:16:8`）；若在 notebook 中已初始化 CUDA，请在启动进程前设置。当前验证为 CPU，不声称 GPU 实测。PyTorch 安全依据：[CVE-2025-32434 官方公告](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)。
