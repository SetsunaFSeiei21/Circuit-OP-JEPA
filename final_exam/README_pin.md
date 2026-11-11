# final_exam 端口级第一阶段

本次交付在器件级流程旁新增独立的端口级单电路实验。所有新增模型实现文件、模型 ID、脚本和生成的结果/日志使用 `_pin` 后缀。器件级代码与入口继续保留。后续多电路、消融、迁移学习不属于本次已完成范围。

**门控 v1 实验**使用 `single_circuit_pin/train_op_gate_pin_v1.sh` 和 `configs/op_gate_pin_v1.json`。新模型在原有完整名称后增加 `_v1`，例如 `device_op_gate_pin_v1`。输入改为归一化 OP＋两个有序引脚独热编码，经一个跨层共享 MLP 产生逐头门控。提供“联合优化、蒸馏后分段、监督后分段”×“全层、仅融合层”六个模型，默认另选无门控基线；自动 Excel 的 `GateStudy` 同时展示蒸馏无门控基座。完整版本差异、模型清单、预算与文件架构见 [record.txt](record.txt)。旧名称继续使用 v0，不会被静默替换。

新增的 **器件级 OP → pin attention 门控** 单电路对照见 [OP_GATE_pin.md](OP_GATE_pin.md)，独立提供 8 个模型（OP 特征＋门控、仅门控两组），使用 `single_circuit_pin/train_op_gate_pin.sh`。

另有 3 个**先训练基础模型、再训练门控**的选项，见 [STAGED_OP_GATE_pin.md](STAGED_OP_GATE_pin.md)：蒸馏后仅训练门控、蒸馏后训练门控＋decoder、直接监督训练后仅训练门控。入口是 `single_circuit_pin/train_staged_op_gate_pin.sh`；这三个选项和前述 8 个门控模型都使用真正的器件级 OP 预测器，下游仍为引脚级。下文的 13 个原有模型、默认配置和启动脚本保持原义；完整注册列表可通过 `python final_exam/run_pin.py --list-models` 查看。

## 已确定的表示规则

1. **节点**：每个已观测的物理器件引脚是一个节点；MOS 的 D/G/S/B 分别成节点，电阻、电容等按实际端子展开。每个命名外部端口也单独成节点。
2. **连接**：保留同器件引脚关系、同网络引脚关系、外部端口与同网络器件引脚的连接。两个外部端口不会仅因连接到同一个网络就额外直连。
3. **特征**：引脚包含端子类型、所属器件类别/物理 model、命名外部端口关联、所属器件设计参数及观测/适用掩码。外部端口没有器件设计参数或器件 OP。
4. **归一化**：先按原始器件的 ID 训练样本拟合参数/OP 统计量，再将标准化结果展开到引脚。拆成更多引脚不会改变某个器件在统计量中的权重。性能标签仍每电路样本一份。
5. **OP predictor**：端口编码器输出全部引脚表示；对同一个物理器件的实际引脚表示取均值，经过 OP head 预测一组器件 OP，再向所属引脚广播。同一器件各引脚接收到完全相同的预测 OP，外部端口不接收 OP。
6. **OP 监督**：在原器件轴上对受支持、有观测的 OP 字段计算损失。每器件每字段只计一次；不同字段沿用等字段权重的 MSE。不会对重复广播后的四份 MOS OP 再计算四次损失。
7. **latent 蒸馏**：教师、学生均为端口编码器，对齐符合条件的引脚 latent 与 graph latent。这里的引脚 latent 对齐与第 6 项物理 OP 预测损失是两个不同目标；外部端口不作为带 OP 的引脚对齐目标。

OP oracle 读取真实器件 OP 的引脚广播视图。三个级联模型在所有数据划分中的下游输入都来自 OP predictor，部署时不读取真实 OP、真实 OP 缺失模式或性能标签。归一化、字段支持和教师 latent 尺度仅由训练集拟合。

## 模型与最大轮次

| 模型 ID | 端口级版本的用途 | 默认最大预算 |
|---|---|---|
| `mlp_pin` | 固定顺序展开引脚及外部端口特征 | 性能训练 60 |
| `xgboost_pin` | 原生逐指标树回归；固定 pin 特征及拓扑列 | 每指标 500 boosting rounds |
| `gat_pin` | pin 图 GAT 与自己的 mean/linear 读出 | 性能训练 60 |
| `deepgen_supervised_pin` | pin 图 DeepGEN 与原生 query pooling 读出 | 性能训练 60 |
| `deepgen_op_pretrained_pin` | pin DeepGEN 先按器件预测完整受支持 OP，再性能训练 | OP 30 + 性能 60 |
| `gtn_pin` | pin 邻居/拓扑环双分支与 GTN 读出 | 性能训练 60 |
| `zerosim_supervised_pin` | 与级联方法匹配的 pin 设计编码器和 decoder | 性能训练 60 |
| `accformer_pin` | ACCFormer_CR 的 pin 适配及逐指标 query decoder | 性能训练 60 |
| `predicted_op_pin` | 冻结 predictor，训练设计＋预测 OP 的下游模型 | predictor 30 + 下游 60 |
| `student_predicted_op_pin` | 教师初始化，latent 对齐＋性能损失联合训练学生 | predictor 30 + teacher 60 + student 60 |
| `student_predicted_op_distill_pin` | 教师初始化，先仅对齐 encoder，再冻结 encoder 训练 decoder | predictor 30 + teacher 60 + encoder 30 + decoder 30 |
| `op_oracle_pin` | 拓扑＋真实 OP 的参考模型 | teacher 60 |
| `design_op_oracle_pin` | 拓扑＋设计参数＋真实 OP 的参考模型，同时可作学生教师 | teacher 60 |

各架构保留自己的读出结构。原生张量算子可以共享，但输入轴实际是 pin，不先压成器件图再冒充 pin 模型；OP head 的器件聚合只发生在 OP 输出分支。模型工厂、输入映射和 checkpoint 格式均区分两种粒度。

配置在 `configs/default_pin.json`，模型选择及覆盖位于 `final_exam_pin`，训练预算、batch size、学习率和 patience 位于 `op_transfer`。与器件级相同，强制下游 `readout == student_predicted_op == predicted_op_distill_encoder + predicted_op_distill_decoder`；依赖预训练成本另计。`patience=0` 关闭提前停止，但仍选验证最佳权重。

`model.max_devices` 沿用公共架构字段名，在此表示 **物理引脚数＋外部端口数** 的上限，不含 graph token。运行前检查所选模型实际容量，超限拒绝执行，不截断。默认 200。`final_exam_pin.model_overrides` 按 `_pin` 模型 ID 覆盖架构；`zerosim_supervised_pin` 的覆盖同时应用于相关教师、predictor 和级联学生。最终配置与容量记录到每个模型和运行元数据。

MLP/XGBoost 的单电路输入顺序固定；当前 runner 要求整个输入为同一拓扑。不能根据本阶段结果推断它们的跨拓扑能力。XGBoost 保存列语义和实际 pin 身份/拓扑，恢复后拒绝顺序或拓扑不匹配的输入。

## 使用

在项目根目录、已有 CUDA PyTorch 的环境中：

```bash
python -m pip install -r final_exam/requirements_pin.txt
```

修改 `final_exam/single_circuit_pin/train_pin.sh` 顶部的 `CIRCUITS`、`MODELS`、`SEEDS`。默认 Opamp1、全部 13 个 pin 模型、17/29/43 三个 seed。原始数据仍放在 `runs/raw/full_data/<circuit>/`，无需复制或修改原始文件名称。

```bash
bash final_exam/single_circuit_pin/train_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_pin.sh --smoke
# smoke 完成并检查后再提交正式实验
bash final_exam/single_circuit_pin/train_pin.sh
```

`--dry-run` 只打印，不提交或写目录。`--smoke` 实际提交、缩短训练轮次，但仍导入完整原始源；使用完整 60,000 点源时不会自动裁成小数据。默认 Slurm 8 CPU/1 GPU、80G、`b_phzhwu`/`ex01A800`、环境 `newbase`。

只选学生会自动训练必需的教师和 predictor，每个 seed 各一次，不训练其他无关模型。例如：

```bash
MODELS=(zerosim_supervised_pin predicted_op_pin student_predicted_op_distill_pin)
```

已有 canonical JSON 可直接复用，数据本身不必改为 pin JSON：

```bash
python final_exam/run_pin.py \
  --data runs/example/data.json \
  --config final_exam/configs/default_pin.json \
  --output final_exam/single_circuit_pin/results_pin/my_run_pin/training_pin \
  --log-dir final_exam/single_circuit_pin/logs_pin/my_run_pin \
  --models zerosim_supervised_pin predicted_op_pin student_predicted_op_distill_pin \
  --seeds 17 29 43 --device cuda
```

直接运行前须将配置的 `dataset.train_circuits` 改为 JSON 中的唯一电路名；Slurm 入口会自动填入。可用 `--manifest` 复用器件级划分，仍严格检查数据指纹、split_seed、比例和标签预算一致。保存时会写成 `manifest_pin.json`。训练结果目录和日志目录必须以 `_pin` 结尾，并使用新目录。

## 输出、教师与日志

任务目录为 `final_exam/single_circuit_pin/`。详细路径见 [任务说明](single_circuit_pin/README_pin.md)。

| 内容 | 相对任务目录的路径 |
|---|---|
| Slurm stdout / stderr | `hpc_logs_pin/<run_id_pin>/<job_id>_pin.out`、`<job_id>_pin.err` |
| 总日志、原始导入日志 | `logs_pin/<run_id_pin>/experiment_pin.log`、`launch_pin.log` |
| 模型和阶段日志 | `logs_pin/<run_id_pin>/seed_<seed>_pin/<model_pin>/`；分阶段学生有 `encoder_pin/decoder_pin/` |
| 原始导入结果、启动记录 | `results_pin/<run_id_pin>/data_pin.json`、`config_pin.json`、`run_pin.json` |
| 训练输出 | `results_pin/<run_id_pin>/training_pin/` |
| Excel | `results_pin/<run_id_pin>/training_pin/summary_pin.xlsx` |

训练输出内保存 `config_pin.json`、`manifest_pin.json`、`cohort_pin.json`、`preprocessing_pin.json`、`model_sources_pin.json`、`results_pin.json`、`stages_pin.json` 和 CSV/JSON/XLSX 汇总。运行元数据明确记录 `granularity=pin`、真实 pin/device 数量、代码提交、数据指纹、模型/seeds、作业及日志路径。

代码提交号查询是可选操作：计算节点找不到或无法执行 Git、没有 Git 元数据、查询失败/无输出或超过 5 秒时，日志给出提示，`run_pin.json` 中的 `code_commit` 写为 `unavailable`，继续正常训练及保存配置、数据指纹和结果。不要将该值当成实际提交号。旧版本可能在报错前已创建 `.run_pin.lock`，修复后不能直接复用该训练输出目录；重新运行提交脚本会自动生成新的实验目录。

每模型有 `config_pin.json`、`checkpoint_pin.pt`、`test_pin.json`；树模型使用 `checkpoint_pin.json` 及 `metric_<index>_pin.ubj`。神经网络每阶段在 `seed_<seed>_pin/stages_pin/<stage_pin>/` 保存 `best_pin.pt`、`history_pin.json`、`cost_pin.json` 和 `config_pin.json`；XGBoost 的 `history_pin.json`、`cost_pin.json` 位于 `seed_<seed>_pin/xgboost_pin/` 模型目录。

学生所需完整教师保存于 `seed_<seed>_pin/shared_pin/design_op_oracle_pin/checkpoint_pin.pt`，OP predictor 保存于 `shared_pin/op_predictor_pin/checkpoint_pin.pt`。即使教师不在所选评测名单中，仍会独立保存。DeepGEN OP 预训练依赖单独位于 `shared_pin/deepgen_op_predictor_pin/`，不与 ZeroSim predictor 混用。

最终学生 checkpoint 已包含所需 predictor、学生 encoder/decoder、pin 与原器件预处理、OP 支持、配方和冻结状态，恢复无需旁边的教师文件。教师 checkpoint 仍完整保留，便于复查和后续使用。接口不提供 optimizer 断点续训语义。

```python
from final_exam.run_pin import load_checkpoint_pin
model, metadata = load_checkpoint_pin('path/to/checkpoint_pin.pt', device='cuda')
# 由 metadata['preprocessing'] 和 PinResidentDataset 构造 pin batch。
# model(batch) 返回归一化后的性能预测；物理指标可复用 evaluate_model。
```

顺序学生还保存 `distillation_pin.json`、`alignment_scales_pin.pt`，并运行冻结摘要检查。OP 物理监督按器件计数；蒸馏历史字段使用 `pin_alignment_mse`、`raw_pin_alignment_mse` 明确表示引脚 latent。

Excel 按原规则展示 MAE、RMSE、R² 的跨 seed 均值与样本标准差（ddof=1），单 seed 时 std 留空。比较表中的 UGF 误差以 MHz 显示，PerSeed/Statistics 保留原始 Hz；R² 无量纲。高亮按“是否需要真实 OP”和“是否使用设计参数”分别分组，样本数/seeds 不匹配不高亮最优。TrainingCost 中每个 stage ID 只计一次，额外依赖不能当作零成本；神经网络记录 epoch，树模型记录逐指标 boosting rounds，epoch 留空。阶段总参数量包含该阶段包装器中的教师、predictor 等冻结模块，可训练参数量另列，不能把阶段总参数量直接当成部署模型大小。重新导出：

```bash
bash final_exam/single_circuit_pin/summarize_pin.sh \
  final_exam/single_circuit_pin/results_pin/<run_id_pin> \
  --output final_exam/single_circuit_pin/results_pin/<run_id_pin>/summary_review_pin.xlsx
```

## 来源与实现边界

来源和固定上游版本在 [SOURCES_pin.json](models/SOURCES_pin.json)，结构差异见 [模型说明](models/README_pin.md)。遵循此前来源优先级；ZeroSim 官方公开版本缺少部分 baseline 代码时，保留已核验的原作者来源或论文适配记录，不把它们冒称为 ZeroSim 官方 baseline 原码。ACCFormer 使用指定 ACCFormer_CR 架构，GTN 为论文基础上的受控适配。

GTN 在物理器件/网络二部图中发现拓扑环，再投影到环上的实际引脚；同网 clique、同器件 clique 不直接作为物理环。环结构不等于导通路径或严格电路方程求解。训练/验证拓扑预计算一次，测试阶段再准备测试缓存；训练 batch 不重复做 CPU 环发现。

为复用已验证的训练代码，公共 engine 只新增默认不启用的文件后缀、尺度拟合输入过滤接口；公共蒸馏类新增可覆盖的输入过滤钩子。器件级默认过滤和输出命名不变。端口级 factory 和 OP oracle 均严格要求 pin 映射，器件级 checkpoint 与 pin checkpoint 不能相互直接加载。

CUDA 不可用时拒绝静默退回 CPU；JSON/SQLite、静态图准备和文件 IO 仍在 CPU。XGBoost 使用请求的原生 backend，并保留 host DMatrix 传输。GPU 性能、真实完整数据内存和预测精度仍需要实际集群实验，见 [验证记录](VERIFICATION_pin.md)。
