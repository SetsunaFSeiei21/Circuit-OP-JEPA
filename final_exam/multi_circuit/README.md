# 多电路联合训练（端口级）

**Graph v7 多阶统计**：`lasd_graph_layer_headwise_detach_pin_v7` 在 v3 基础上统计 A…A^K，每阶独立 MLP 后求和，再送入原图/层 Encoder。`lasd_moe_pin.aggregation_distance` 默认2，1对应v3。原单电路脚本增加 `--graph-layer-detach-v7`；多电路模型配置和距离网格已提供。[完整定义、配置与启动](../GRAPH_V7_ATTENTION_POWERS_pin.md)

**Graph v5 / v6** 可加入 MODELS：`lasd_graph_layer_headwise_detach_pin_v5` 和 `lasd_graph_layer_headwise_detach_pin_v6`。同名模型配置和网格已提供，预算与 v3 保持一致。v5 在性能 PE 内加入 HC，v6 在 Decoder 前对残差和做 LayerNorm。[详细说明](../GRAPH_V3_EXTENSIONS_V5_V6_pin.md)

**Graph-layer v4** 可直接加入 MODELS：`lasd_graph_layer_headwise_detach_pin_v4`。已配套 `configs_pin/models/` 和 `configs_pin/search_spaces/` 下的同名 JSON，默认网格与原 graph-layer 模型一致。[结构、梯度及配置说明](../GRAPH_FUSION_SHALLOW_DETACH_V4_pin.md)

这里训练的是**一个模型同时学习所有指定电路**，再分别评估各电路及合并测试集。每个电路内部划分 train/val/test；这是已见电路的联合 ID 实验，不是留出拓扑的 OOD 实验。单电路入口和原有模型名称保持兼容。

## 1. 启动

依赖与单电路实验相同，新增 Excel 导出依赖 `openpyxl>=3.1`：

```bash
python -m pip install -r final_exam/requirements.txt
bash final_exam/multi_circuit/train_multi_pin.sh --dry-run
bash final_exam/multi_circuit/train_multi_pin.sh --smoke
bash final_exam/multi_circuit/train_multi_pin.sh
```

`--dry-run` 不创建目录、不导入数据、不提交作业，显示每个模型的合法 trial 数、正式训练次数、并发上限与完整 sbatch 命令。`--smoke` 每电路最多抽 100 个样本，缩短阶段预算；保留搜索和五 seed 流程，结果明确标记 smoke，不能当正式结果。

脚本顶部修改：

```bash
CIRCUITS=(two_stage_folded_opamp Opamp1 Peng_ACBC)
MODELS=(zerosim_supervised_pin device_op_cascade_pin lasd_graph_layer_headwise_detach_pin_v3)
SEARCH_SEED=11
SEEDS=(17 29 43 59 71)
MODE="all"
MAX_CONCURRENT=2
```

默认 MODELS 是全部八个 baseline 加 `device_op_cascade_pin`。每个模型默认四个候选点，因此默认一次提交包含 36 次搜索、45 次正式训练，依赖模型的 OP/teacher 阶段另计。先检查 dry-run 的预算。

同一个数组任务负责一个模型的所有候选和正式 seeds，顺序执行；最多 `MAX_CONCURRENT` 个模型任务同时使用 GPU。每个试验启动独立 Python 子进程，结束后释放数据和模型内存。

login1 设置 Conda 路径后直接运行同一脚本：

```bash
export JEPA_CONDA_HOME=/public/home/202411094934/anaconda3
export JEPA_CONDA_ENV=newbase
bash final_exam/multi_circuit/train_multi_pin.sh --dry-run
```

login04 默认 `/share/software/anaconda3`，环境默认 `newbase`。账户、分区、CPU、内存、时限均在 bash 顶部；默认 `b_phzhwu / ex01A800`。不修改另一集群的设置。`--local` 在当前计算节点顺序执行整个流程，不申请额外资源；提交阶段 Python 可用 `JEPA_SUBMIT_PYTHON` 指定。

## 2. 配置在哪里

| 内容 | 文件 |
|---|---|
| 采样上限、sampling_seed、split_seed、划分比例、归一化方式 | `configs_pin/protocol_pin.json` |
| 某个模型的完整默认模型/训练配置 | `configs_pin/models/<模型名>.json` |
| 某个模型的独立网格 | `configs_pin/search_spaces/<模型名>.json` |
| 模型、电路、搜索 seed、正式 seeds、集群、并发 | `train_multi_pin.sh` |

`protocol_pin.json` 的 dataset 和 normalization 覆盖各模型文件中的对应部分，避免模型意外使用不同数据协议。bash 中的 CIRCUITS、SEARCH_SEED、SEEDS 覆盖 protocol 的同名设置。模型文件直接修改 `model`，禁止再隐藏一层 `model_overrides`。

数据量仍是**每电路先随机抽总量，再划分**：

```json
"max_samples_per_circuit": 20000,
"sampling_seed": 17
```

`null`/`"all"` 取全量；正整数超过该电路的可用总数时取全量。分组划分沿用原有 design identity/fingerprint 规则，避免同一设计泄漏，实际分组数可能导致划分比例有小幅偏差。

独立模型网格示例：

```json
{
  "grid": {
    "op_transfer.lr": [0.0001, 0.0003],
    "model.dim": [64, 128]
  }
}
```

按笛卡尔积展开。拒绝未知字段、重复候选和未生效字段，例如 MLP 的 heads、ACCFormer 的 decoder_ffn_dim。维度不能整除 heads 或 PE 深度超过 encoder 深度的组合记为 skipped。其它架构约束在共享准备任务中检查，失败则模型数组不启动。每模型最大 trial 数由 `max_trials_per_model` 限制。

MLP 网格默认搜索 `ffn_dim` 和学习率；XGBoost 默认搜索 eta 和 max_depth；其余 baseline 默认搜索 dim 和学习率。它们保留各自原生 readout，没有统一替换成同一个 decoder。

阶段轮数在每个模型文件的 `op_transfer.epochs` 中；仍遵守原来的 readout / student / distill budget 检查。OP 预测、teacher、latent alignment 使用原有监督目标。性能预测阶段用验证集 circuit-macro R² 选 checkpoint/早停；性能 loss 仍用于反向传播，并保留在历史中。多电路的额外验证前向时间计入阶段耗时。

## 3. 可选模型与自己的模型

八个 baseline 均有独立配置和网格：

- `mlp_pin`、`xgboost_pin`、`gat_pin`、`gtn_pin`。
- `deepgen_supervised_pin`、`deepgen_op_pretrained_pin`。
- `zerosim_supervised_pin`、`accformer_pin`。

同时提供配置：`device_op_cascade_pin`，`width_decoder_pin`，`width_decoder_op_pin`，`predicted_op_pin`，`student_predicted_op_pin`，`student_predicted_op_distill_pin`，`lasd_graph_layer_headwise_pin`，以及 `lasd_graph_layer_headwise_detach_pin` 原版和 `_v1`、`_v2`、`_v3`。

其它已经注册到 `final_exam/run_pin.py` 的 pin 模型可生成独立配置后加入 MODELS：

```bash
python final_exam/multi_circuit/launch_pin.py --init-model lasd_moe_headwise_detach_pin
```

生成过程拒绝覆盖已有文件。默认给新模型两点学习率网格；按该模型实际结构扩展，不自动假定全部配置都有效。新的网络代码仍接入原有工厂/registry，然后使用这个入口，不需要再写一套联合训练循环。

单拓扑 `_shuffle_` 诊断模型不支持联合 ID，显式报错。true-OP oracle 如选择运行，报告标记 `requires_measured_op`；它们不是无仿真推理 baseline。`deepgen_op_pretrained_pin` 和 OP cascade 有额外 OP 训练监督，论文表中应保留这种区别。

## 4. 数据与工程约束

- 各电路的导入、采样和划分只在共享准备任务中做一次，保存 canonical 数据、manifest、preprocessing、pin schema。
- 所有 trial、model、seed 共用此 manifest。归一化只拟合 train；性能统计只用 labeled_train。先按器件拟合物理字段统计，再扩展到引脚。
- 各电路保留自己的 canonical 引脚顺序；不同电路不假定第 k 个 pin 是相同物理器件。MLP 维持 flatten 结构，加显式有效位，补齐到 max_devices。XGBoost 同样补齐、保存并校验每个电路的拓扑 schema，每个性能指标一个共享 booster。
- 图模型按拓扑组批；不同图的 mask/节点身份保持隔离。设计输入保持原有白名单，OP 真值不会进入 design-only baseline。
- 常驻数值表放在 **CPU**，当前 batch 才进入 GPU。多电路关闭整套 frozen latent/OP 的 GPU 缓存，冻结的预测器按 batch 运行，避免随样本总量占满显存。
- CPU 仍需容纳 canonical 对象和数值表；这不是磁盘流式训练。14 个电路全量数据的主存需求需要在实际集群测量，10G 内存不适合默认大规模任务。可先用每电路 20000 或 smoke 检查。XGBoost 展平特征使用磁盘 memmap 中间表，但 DMatrix 构造和训练仍消耗主存。
- 数据 fingerprint 改为按样本流式哈希，**摘要与旧实现一致**；JSON 导出改为流式写，减少超大临时字符串。
- 每次提交复制实际工作区源文件到不可写的 `source_pin/`，记录逐文件 SHA256、整体源码摘要和 Git commit。它包含尚未提交的本地源代码修改，不能仅用 commit 代替摘要。计算节点不需要 Git。worker 启动前验证 snapshot 和搜索计划，后续 pull 不影响已提交作业。
- 不跨 trial/model 自动复用 OP/teacher，避免不同维度、预算、seed、preprocessing 的误用。单次模型训练内仍复用原有共享阶段；续训只复用严格身份验证通过的阶段。全部 teacher/OP checkpoint 保存在该模型/seed 的 shared_pin 下。

## 5. 搜索与正式训练

搜索只物化 train/val 的特征缓存；请求 test/ood batch 会报错。测试数据的身份/标签参与完整性校验、存在于 canonical 文件中，但不会计算测试预测或用于选择配置。

选参目标：每电路先对有定义的性能 R² 等权平均，再对所有指定电路等权平均，最大者胜出。常数目标、少于两个观测的 R² 保留 null、不补零；每个电路至少有一个可用验证指标，否则准备/选择失败。并列按 trial ID 固定排序，保证复现。XGBoost 每指标 booster 也按该指标的验证电路宏 R² 早停；没有可用验证 R² 的指标使用固定 boosting 轮数，不看 test。

整个预定网格必须成功才锁定最佳配置，失败 trial 不静默丢弃。`selected_pin/<模型>.json` 保存完整获胜配置、候选 ID、验证分数、源码摘要。正式五 seed **从新初始化重新训练**，不从搜索权重 warm start。搜索 seed 与正式 seed 必须不同。

`MODE=search` 只搜索；同 campaign 之后 `MODE=train` 使用已锁定配置。跳过 HPO：新 campaign 设置 `MODE=train`、`FIXED_CONFIG=1`，直接按 models JSON 训练五 seed，明确标记 user fixed。控制消融时可让各变体使用相同固定配置；方法独立调参则各自网格搜索。

## 6. 断电续训

在原提交脚本中填写 `CAMPAIGN` 为已有结果根目录，保持所需 MODE 后重跑：

```bash
bash final_exam/multi_circuit/train_multi_pin.sh --campaign /绝对路径/results_pin/multi_..._pin
```

已有 campaign 使用其保存的电路/模型/seed/配置/源码，当前 bash 中其它实验参数不会覆盖它；集群资源可调整。完成任务直接跳过；失败任务新建 attempt 目录，原目录保持不变。神经网络从最近完整 epoch 恢复模型、AdamW、RNG、早停和 best 权重；已完成 OP/teacher 阶段不重复优化。未完成 epoch 重跑。XGBoost 中断的 trial 重新拟合，完整 trial/seed 不重跑。

使用系统文件锁阻止同一个任务并发写；进程崩溃自动释放锁。每个任务原子写 status，保留所有 attempt 及返回码。一个 trial/seed 失败后继续其余可运行任务；网格不完整则阻止该模型选参，其它模型不受影响。

## 7. 输出和统计口径

目录均位于本任务目录：

- `results_pin/<campaign>/data_pin/`：一次准备的 canonical 数据、manifest、preprocessing、schema。
- `results_pin/<campaign>/search_pin/<模型>/<trial>/`：完整配置、status、各次 attempt。
- `results_pin/<campaign>/selected_pin/<模型>.json`：锁定的配置。
- `results_pin/<campaign>/training_pin/seed_<seed>_pin/<模型>/attempt_..._pin/training_pin/`：模型、依赖、阶段 checkpoint、历史、预测、报告。
- `results_pin/<campaign>/reports_pin/summary_pin.xlsx` 和 `summary_pin.json`：统一统计。
- `logs_pin/<campaign>/`：每个 trial/seed/attempt 的 worker、阶段 train/evaluate 日志。
- `hpc_logs_pin/<campaign>/`：共享准备、模型数组、汇总的 Slurm stdout/stderr 和 submission JSON。

这些结果/log 路径由现有 `.gitignore` 规则覆盖，不上传 GitHub。源码、配置、说明进入版本管理。

| Sheet | 含义 |
|---|---|
| PerCircuit | 每模型、每电路、每指标跨 seed 的 MAE/RMSE/R²/normalized RMSE mean±std |
| PooledTest | 每 seed 先拼接全部电路的物理单位预测和真值，重算 MAE/RMSE/R²，然后跨 seed 汇总 |
| CircuitMacro | 每 seed 对各电路指标等权平均，再跨 seed 汇总；不等于 pooled |
| SeedDetails | 各 seed 原始指标、有效标签计数、eligible circuit 数 |
| SearchTrials / BestConfigs | 搜索结果、获胜参数及配置路径 |
| TrainingCost | 阶段参数量、可训练参数量、真实轮数、预算、耗时、恢复信息 |
| RunStatus | 所有预定正式 seed 的状态，未完成也显式列出 |
| PairedComparison | 相同 seed 下相对 protocol.baseline 的差值与探索性配对 sign-flip 检验 |
| Protocol | 数据协议、seed、源码、smoke、聚合口径 |

标准差采用样本标准差 `ddof=1`；只完成一个 seed 时 std 为 NA。缺失指标/未完成 seeds 不补零，必须看 completed_seeds 和有效计数。本入口不挑最好/最差 seed。五个配对 seed 的双侧精确 sign-flip 检验最小 p=0.0625，不能保证得到 p<0.05；同时报告跨本工作簿比较的 Holm 修正，不将电路、指标或测试样本伪装成独立训练重复。

每个模型保存 `val_predictions_pin.npz`（搜索）或 `test_predictions_pin.npz`（正式）：physical prediction/truth、mask、normalized prediction/truth、circuit_id、sample_id、metric_names。`np.load(..., allow_pickle=False)` 可独立检查。

重生成 Excel：

```bash
CAMPAIGN=/绝对路径/results_pin/multi_..._pin bash final_exam/multi_circuit/summarize_pin.sh
```

只有汇总进程写 Excel；模型任务不并发写同一个工作簿。

## 8. 代码和验证

`run_pin.py` 仅增加显式 joint/validation-only 分支，复用原模型工厂、teacher/student 协议、优化器与 checkpoint 格式；单电路默认仍严格只允许一个电路。

`context_pin.py`：CPU resident、拓扑组批、test 缓存防护和 flat schema。`metrics_pin.py`：验证宏平均与物理单位 pooled 计算。`tree_pin.py`：多拓扑固定顺序 XGBoost。`config_pin.py`：独立配置与确定性网格。`launch_pin.py`/`worker_pin.py`：快照、Slurm DAG、子进程隔离、resume。`summarize_pin.py`：seed 汇总和 Excel。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q final_exam/multi_circuit/tests
```

测试使用不同 pin 数的两个合成电路，覆盖所有 baseline、OP cascade、width 和 graph detach 四版本的联合训练/重载，验证搜索不读测试缓存、真实 pooled 指标、五 seed 汇总、半程中断恢复、源码快照和 Slurm 依赖。CPU 检查不能代替实际 CUDA/Slurm 环境的 smoke；未在本环境提交真实集群作业或运行全量论文实验。
