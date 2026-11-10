# final_exam：器件级受控实验

本目录提供第一阶段的**单电路训练、测试和 Excel 对比**。每个电路独立训练，不把多个电路合并进同一个单电路作业。`multi_circuit/`、`ablation/`、`transfer_learning/` 已建立任务位置；目前只有范围说明，没有把单电路脚本冒充为这些任务的实现。

## 运行入口

在项目根目录、已安装 CUDA PyTorch 的 `newbase` 环境中安装新增依赖：

```bash
python -m pip install -r final_exam/requirements.txt
```

编辑 `final_exam/single_circuit/train.sh` 顶部的 `CIRCUITS`、`MODELS`、`SEEDS` 和集群资源。默认 Opamp1、13 个模型、3 个 seed（17、29、43），默认每个电路一个 Slurm 作业，8 CPU、1 GPU、80G 内存。

```bash
# 只打印命令，不提交、不创建结果目录
bash final_exam/single_circuit/train.sh --dry-run
# 实际提交短轮次烟测
bash final_exam/single_circuit/train.sh --smoke
# 正式提交
bash final_exam/single_circuit/train.sh
```

烟测只缩短训练轮次，仍会导入完整输入；大数据的导入时间和内存需求不会因为 `--smoke` 消失。输入默认 `runs/raw/full_data/<电路名>/`，需要 SQLite 和对应网表。原始 W/L 单位必须已经正确，导入仍严格校验与 OP 几何的一致性，不自动猜单位或改写原始文件。

只运行需要的模型，直接在 bash 的数组中删减。仅选择学生模型时，会自动训练必需的教师和 OP predictor；它们保存在共享依赖目录，不会因此运行其他无关 baseline。

```bash
MODELS=(zerosim_supervised predicted_op student_predicted_op_distill)
SEEDS=(17 29 43)
CIRCUITS=(Opamp1 two_stage_folded_opamp)
```

每个作业保存提交时的配置快照；排队期间修改模板不会改变已提交配置。输入数据和代码本身没有复制快照，运行中的实验应使用固定版本，避免在同一 checkout 更新代码或修改原始数据。

已有 canonical JSON 可以直接运行，不必重新导入：

```bash
python final_exam/run.py \
  --data runs/example/data.json \
  --config final_exam/configs/default.json \
  --output final_exam/single_circuit/results/my_run/training \
  --log-dir final_exam/single_circuit/logs/my_run \
  --models zerosim_supervised predicted_op student_predicted_op_distill \
  --seeds 17 29 43 --device cuda
```

直接调用时，配置的 `dataset.train_circuits` 必须等于 JSON 中唯一电路名；Slurm launcher 会根据 `CIRCUITS` 自动填入。必须用新的结果目录，训练和日志入口会拒绝覆盖已有实验。

## 模型与预算

所有模型以完整器件为粒度。D/G/S/B 是边上的端口角色，不是四个独立学习节点。统一数据划分、训练集归一化、参数及标签缺失处理、验证选模和物理单位指标计算。各图模型保留自己的读出，不强行共用 ZeroSim decoder。MLP、XGBoost 用固定器件顺序展开设计参数；此阶段要求单一固定拓扑，不宣称其具有跨拓扑泛化能力。

| 名称 | 训练方式 | 默认最大训练预算 |
|---|---|---|
| `mlp` | 固定器件设计特征 → MLP → 性能 | readout 60 |
| `xgboost` | 每项性能指标单独的树回归 | 每指标最多 500 boosting rounds |
| `gat` | 器件图 GAT + mean/linear 读出 | readout 60 |
| `deepgen_supervised` | 器件图 DeepGEN + 原生 query pooling 读出 | readout 60 |
| `deepgen_op_pretrained` | DeepGEN 先预测完整受支持 OP，再监督性能 | OP 预训练 30 + readout 60 |
| `gtn` | 邻居/拓扑环双分支 attention + GTN 读出 | readout 60 |
| `zerosim_supervised` | 与级联方法匹配的设计侧编码器 + decoder | readout 60 |
| `accformer` | ACCFormer_CR 器件级适配及其逐指标 query decoder | readout 60 |
| `predicted_op` | 冻结 OP predictor，设计＋预测 OP → 性能 | predictor 30 + readout 60 |
| `student_predicted_op` | 教师初始化学生；对齐损失＋性能损失联合训练 | predictor 30 + teacher 60 + student 60 |
| `student_predicted_op_distill` | 教师初始化；先仅对齐 encoder，再冻结 encoder 训练 decoder | predictor 30 + teacher 60 + encoder 30 + decoder 30 |
| `op_oracle` | 真实 OP＋拓扑 → 性能 | teacher 60 |
| `design_op_oracle` | 真实 OP＋设计参数＋拓扑 → 性能 | teacher 60 |

配置在 `final_exam/configs/default.json`：

- `model`：默认宽度、heads、层数、dropout 等。
- `op_transfer`：本入口的 batch_size、lr、patience、进度频率及 epochs。旧入口的 `training.decoder_epochs`、`training.baseline_epochs` **不控制本入口**；模板已删除这些容易混淆的字段。
- `op_transfer.epochs.readout` 控制普通神经网络性能训练和 predicted_op 的下游；`student_predicted_op` 控制联合学生；`predicted_op_distill_encoder/decoder` 控制顺序学生的两阶段。
- 强制 `readout == student_predicted_op == predicted_op_distill_encoder + predicted_op_distill_decoder`。统一的是下游最大轮次；预训练依赖成本另计，不宣称总算力相同。
- `op_transfer.patience=10` 开启验证集早停，设 0 关闭提前停止。两种情况下均恢复验证最佳权重；最大轮次不是保证实际执行轮次。
- `final_exam.model_overrides`：按 baseline 名称覆盖 `model` 字段。`zerosim_supervised` 的覆盖同时用于教师、OP predictor 和级联学生，保证蒸馏结构匹配。DeepGEN 两种变体应使用相同覆盖才能作为预训练消融比较。
- `final_exam.xgboost`：原生树超参数；`num_boost_round` 是逐指标上限，不与神经网络 epoch 混算。

每个 seed、每种教师（`op_oracle` 或 `design_op_oracle`）与级联 OP predictor 只训练一次，供所选模型复用。两个学生依赖的是 `design_op_oracle`；`predicted_op` 不依赖性能教师。DeepGEN 的 OP predictor 属于不同架构，单独训练、保存和计费。神经网络阶段记录 planned/actual epochs、验证最佳 epoch、参数量、时间、optimizer steps；Excel 按 stage ID 去重保留这些独立成本。

`TrainingCost.Parameters` 是**该训练阶段整个模块对象**的参数总数，包含其中冻结的教师或 OP predictor；`Trainable parameters` 才是该阶段参与优化的参数数目。它们不能直接解释成最终部署模型大小，也不能把不同阶段的参数量相加。XGBoost 的 epoch 和神经网络参数量单元格留空，另列每项指标实际 boosting rounds 及其预算上限。

## 训练边界与教师保存

归一化、OP 字段支持范围、latent 尺度仅由 ID 训练集拟合。训练/验证使用各自标签；测试只在最终模型选择完成后评估。未在训练中出现的 OP/性能字段在进入变换前屏蔽。OP oracle 在推理时读取真实 OP，其结果作为额外输入信息的参考单独分组，不保证其分数一定优于其他模型。三个级联模型部署时只读取拓扑、设计参数及自身预测 OP，不读取真实 OP、真实 OP 缺失模式或性能标签。

顺序学生第一阶段的 decoder 和 OP predictor 冻结，仅通过教师 latent 对齐训练 encoder；第二阶段 encoder、OP predictor 冻结，仅训练 decoder。代码比较阶段前后的参数摘要，不满足冻结要求会报错。联合学生与顺序学生不是相同训练过程。

单个结果 `training/` 内包含：

- `run.json`：选定模型、seeds、设备、Git 提交、数据指纹、Slurm job/node、状态。
- `config.json`、`manifest.json`、`cohort.json`、`preprocessing.json`：有效配置、固定划分、使用样本和训练集统计量。
- `model_sources.json`：本次模型来源、固定上游版本和适配说明快照。
- `seed_<seed>/<model>/config.json`、`test.json`、`checkpoint.pt`：每模型配置、测试指标和可独立重建的最终模型。XGBoost 使用 `checkpoint.json` 和逐指标原生 `metric_*.ubj`，其 `history.json`、`cost.json` 同样位于 `seed_<seed>/xgboost/`。
- `seed_<seed>/shared/design_op_oracle/checkpoint.pt`：学生依赖的完整教师，包含 encoder、decoder、预处理、架构和配置。即使不选择教师作为评测模型也会保存。
- `seed_<seed>/shared/op_predictor/checkpoint.pt`：完整预测 OP 模型与训练集支持范围。
- `seed_<seed>/stages/<stage>/best.pt`、`history.json`、`cost.json`、`config.json`：每个神经网络阶段的检查点、真实训练历史和成本。
- 蒸馏学生额外保存 `alignment_scales.pt`、`distillation.json`，记录教师来源和冻结检查。
- `results.json`、`summary.csv`、`summary.json`、`stages.json`、`summary.xlsx`：逐 seed 结果、聚合结果和去重阶段成本。

Git 提交号属于可选的来源信息，计算节点无需安装 Git 才能训练。找不到或无法执行 Git、当前目录没有 Git 元数据、查询失败/无输出或超过 5 秒时，日志给出提示并将 `run.json` 的 `code_commit` 记录为 `unavailable`；配置、数据指纹和模型结果仍按正常流程保存，不伪造提交号。因旧版本查询代码而失败的实验可能已创建输出锁，修复后须选择新结果目录；重新运行提交脚本会自动创建新目录。

独立恢复不需要原教师文件和训练代码状态对象；模型配方及全部必要子模块权重已包含在最终 checkpoint：

```python
from final_exam.run import load_checkpoint
model, metadata = load_checkpoint(
    'final_exam/single_circuit/results/<run_id>/training/seed_17/student_predicted_op_distill/checkpoint.pt',
    device='cuda',
)
# 使用 metadata['preprocessing'] 构造与训练一致的 device_jepa resident batch。
# model(batch) 输出归一化后的性能值；计算物理指标可复用 evaluate_model。
```

教师/OP predictor 的独立 checkpoint 也由该接口恢复；冻结标志与预处理一起保存。该接口返回 `.eval()` 模型，不提供断点续训调度器或完整 optimizer-resume 语义。

## 日志、Excel 和状态

每个任务独立管理目录，同一次作业共享 `<run_id>`：

| 内容 | 路径（相对 `final_exam/single_circuit`） |
|---|---|
| HPC stdout / stderr | `hpc_logs/<run_id>/<job_id>.out` / `.err` |
| 提交配置与 job 记录 | `hpc_logs/<run_id>/submitted_config.json`、`submission.txt` |
| 导入及启动日志 | `logs/<run_id>/launch.log` |
| 实验总日志 | `logs/<run_id>/experiment.log` |
| 各 seed/模型或阶段日志 | `logs/<run_id>/seed_<seed>/<model>/train.log`、`evaluate.log`；分阶段学生有 encoder/decoder 子目录 |
| 共享依赖日志 | `logs/<run_id>/seed_<seed>/shared/<teacher_or_predictor>/train.log` |
| 所有数据、配置与模型 | `results/<run_id>/` |
| 最终可对比 Excel | `results/<run_id>/training/summary.xlsx` |

Excel 自动生成以下 7 个工作表：

| 工作表 | 内容 |
|---|---|
| `MAE`、`RMSE`、`R2` | 按电路、split、指标排列，每列一个模型，展示跨 seed 的 mean ± 样本标准差 |
| `PerSeed` | 每 seed 的原始物理单位指标、有效标签数量和依赖阶段 ID |
| `Statistics` | 未按显示精度取整的均值、标准差、有效 seed 数与 seed 列表 |
| `TrainingCost` | 按阶段去重的轮次/boosting rounds、时间、参数量、optimizer steps 和验证选模信息 |
| `Configuration` | 本次运行元数据、有效总配置和每模型配置，附原 JSON 路径 |

标准差使用 `ddof=1`。只有一个有效 seed 时，对比表只显示均值，`Statistics` 中 std 留空，不伪造 0；部分 seed 指标未定义时会注明有效 seed 数。UGF 的 MAE/RMSE 对比页显示为 MHz（均值和标准差均换算），`PerSeed` 和 `Statistics` 保留 Hz；R² 始终无量纲。

MAE/RMSE 越低越好，R² 越高越好。绿色只标记**推理输入条件相同**且 seed、有效标签数一致的组内最优未取整均值；一个组至少有两个模型且全部 seed 指标有定义才会高亮。真实 OP 参考模型置于表格右侧；`op_oracle` 不含设计参数，`design_op_oracle` 含设计参数，二者不会被当成同一输入条件组。高亮不是显著性检验。

全部所选模型×seed 完成且 Excel 写出成功才标记 `complete`。失败保留状态和已完成模型，不把部分结果当完整实验；汇总默认拒绝不完整矩阵或覆盖既有 xlsx。若需重新导出：

```bash
bash final_exam/single_circuit/summarize.sh \
  final_exam/single_circuit/results/<run_id> \
  --output final_exam/single_circuit/results/<run_id>/summary_review.xlsx
```

需要覆盖时显式 `--overwrite`。部分结果可用 `--allow-incomplete` 临时导出，文件带未完成标识，禁用最优高亮。临时导出仍要求已有结果、阶段成本等基础文件格式正确；它不会忽略损坏文件。

XLSX 的检查发生在导出前，不另建名为 `Validation` 的工作表。检查包括计划中的模型×seed×split 是否齐全、重复结果、非有限数值、指标字段一致性、每模型配置是否存在、依赖阶段成本是否缺失/重复、实际训练预算是否越界。写出失败不发布半个工作簿，默认也不覆盖旧文件。

## 来源、验证和限制

详细来源在 `models/SOURCES.json` 与 `models/README.md`。ZeroSim 官方公开版本没有发布其全部 baseline，因此这里如实区分：现有 ZeroSim-inspired 器件模型、原作者代码适配、基于论文的独立实现。ACCFormer 使用用户指定的 `SetsunaFSeiei21/ACCFormer_CR`。GTN 没有找到原作者公开实现，以 DAC 2024 原论文独立实现并列明差异；这些模型应在论文中称为器件级受控适配，不是原 benchmark 分数的精确复现。

```bash
python -m unittest discover -s final_exam/tests -v
python -m unittest final_exam.single_circuit.test_launch -v
```

其中 `final_exam/tests/test_summary.py` 检查生成 XLSX 内所有 XML 是否可解析、工作表名称、跨 seed 样本标准差、单 seed 空 std、MHz/Hz 单位、精确值与公式缓存、分组高亮、阶段去重、XGBoost 独立轮次字段，以及不完整或损坏输入和覆盖保护。这些是文件结构与数值检查，不是 Excel/LibreOffice 的人工排版验收。

本地机制验证使用小型合成数据和 CPU；并非完整真实数据精度、GPU 性能、集群资源或内存容量验证。CUDA 请求在不可用时直接失败，不静默改用 CPU。神经网络 resident 张量计算在指定 device；原始 SQLite/JSON、解析、静态图准备和文件 IO 仍使用 CPU。XGBoost 原生 hist 在指定 CUDA backend 训练，当前接口通过主机 DMatrix 传输数据；没有宣称整个实验零 CPU 工作。

运行产物被 `.gitignore` 排除；代码、配置模板、来源记录和测试进入 Git。旧的 `scripts/run/`、`experiments/op_transfer/` 入口保持兼容，新的实验统一由本目录入口管理。
