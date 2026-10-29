# OP transfer experiments

本目录用于比较 OP 信息怎样转移到仅接收拓扑与设计参数的模型。代码、运行结果和日志分别在本目录、`results/` 和 `logs/`；不会另建一份仓库，也不会改写已有生产实验。

## 在 bash 中设置电路、模型和种子

单电路直接编辑 `experiments/op_transfer/train_single_circuit.sh` 开头的 `CIRCUIT`、`MODELS` 和 `SEEDS`，然后运行：

```bash
bash experiments/op_transfer/train_single_circuit.sh
```

OOD 直接编辑 `experiments/op_transfer/train_ood.sh` 开头的 `TRAIN_CIRCUITS`、`OOD_CIRCUITS`、`MODELS` 和 `SEEDS`，然后运行：

```bash
bash experiments/op_transfer/train_ood.sh
```

两个脚本均默认提交完整数据的 GPU 作业；可加 `--dry-run` 仅预览。原始数据目录 `RAW_ROOT` 和配置路径 `CONFIG` 也写在脚本开头。batch size、各阶段 epoch 和模型结构参数在 `CONFIG` 指定的 JSON 中设置。脚本自动完成导入和新训练，不需要填写旧实验目录。结果、日志和 Excel 的位置与下面的 raw 入口相同。

模型选择以两个 bash 脚本各自的 `MODELS` 为准，新增的分阶段学生也已加入其中。单电路脚本保留用户设置的 `Opamp1`、seed 17 和原五个对照模型；OOD 脚本保留自己的电路与种子设置。若只验证“预测 OP 的教师补偿”，可使用：

```bash
MODELS=(
  design_supervised
  predicted_op
  design_op_oracle
  student_predicted_op
  student_predicted_op_distill
)
```

只想运行新学生时也可写 `MODELS=(student_predicted_op)`；必要的真实 OP 教师和 OP 预测器会自动训练，无需手工指定 checkpoint。依赖阶段不等于额外选择了性能结果模型。

只运行分阶段学生则写 `MODELS=(student_predicted_op_distill)`，它独立从同一教师初始化，不依赖已经完成的 `student_predicted_op` 训练。

## 直接指定电路和模型重新训练

无需准备旧实验目录。原始数据放在 `runs/raw/full_data/<电路名>/`，包含 `sampling_history.sqlite3` 和对应 `.sp` 网表，然后执行：

```bash
bash experiments/op_transfer/submit.sh --submit \
  --circuit two_stage_folded_opamp \
  --models design_supervised op_oracle op_distill predicted_op dual_distill dual_design_control \
  --seeds 17 29 43 --device cuda
```

修改 `--circuit` 选择电路，修改 `--models` 选择模型。默认使用本目录 `example_config.json`；`--config` 可指定其他配置，电路角色由命令覆盖配置中的示例电路。此命令使用完整数据，从头训练；导入、创建新划分、必要前置阶段、性能训练、测试和 Excel 汇总都在作业内自动执行。新划分使用配置的 `split_seed`，相同数据与种子可复现划分；无需旧 checkpoint 或 manifest。默认资源仍是 1 GPU、8 CPU。

原始目录不在默认位置时加 `--raw-root /path/to/full_data`。`--max-table-rows` 默认 `none`，可显式设置导入安全上限。几何单位仍由原有严格 importer 检查，不修改原始网表或自动猜测单位。`--max-train/val/test` 仅限制训练/评估子集，不减少原始数据导入量。

直接指定 ID/OOD 电路也可：

```bash
bash experiments/op_transfer/submit.sh --submit \
  --train-circuits 5t_ota Opamp0 Opamp1 two_stage_opamp_otaf \
  --ood-circuits two_stage_folded_opamp \
  --models op_distill predicted_op dual_distill \
  --seeds 17 29 43 --device cuda
```

电路名须与各目录及 SQLite 中的 `circuit_name` 一致，ID 与 OOD 不可重叠。多个 `--train-circuits` 不带 `--ood-circuits` 时运行联合 ID 比较。把 `--submit` 改为 `--dry-run` 可只查看作业命令；已分配计算资源时可直接执行 `python experiments/op_transfer/run_raw.py` 加相同实验参数。

该入口输出为 `experiments/op_transfer/results/<电路名或OOD名>_<UTC时间>_<编号>/`：

- `data.json`、`config.json`、`raw_run.json`：新导入数据、本次配置、导入/训练状态。
- `results/summary.xlsx`：最终 Excel。
- `results/seed_<seed>/<model>/`：模型 checkpoint 和 test/OOD 指标。
- 作业 stdout/stderr 仍在 `experiments/op_transfer/logs/<时间编号>/`。

下面的 `--data` 用法是复用已经导入的 canonical 数据的可选入口；直接选电路时使用上面的命令即可。

## 模型选择

`--models` 只运行所选结果及其必要前置训练。例如只选 `op_distill` 时仍须训练教师；只选 `predicted_op` 时仍须训练 OP 预测器。前置阶段可在同一 seed 的相关变体间复用，其成本单独记录。

| 名称 | 训练方式 | 性能推理输入 |
| --- | --- | --- |
| `design_supervised` | 设计编码器直接接受性能监督 | 拓扑＋设计参数 |
| `op_oracle` | OP-only 编码器接受性能监督；也作为蒸馏教师 | 拓扑＋真实 OP |
| `design_op_oracle` | `zerosim_device_op` 架构接受性能监督；也是新级联学生的教师 | 拓扑＋设计参数＋真实 OP |
| `jepa_standardized` | 基础 EMA 对齐目标、本实验统一损失配置＋训练集拟合的标准化读出 | 拓扑＋设计参数 |
| `jepa_grouped` | 按同一拓扑、同一器件分组计算跨样本变化约束 | 拓扑＋设计参数 |
| `jepa_residual` | 在 grouped 变体基础上加入中心化、缩放后的残差对齐 | 拓扑＋设计参数 |
| `jepa_op_aux` | 在基础 JEPA 上加入设计 latent → 真实 OP 的辅助监督 | 拓扑＋设计参数 |
| `op_distill` | 完整冻结性能监督训练好的 OP 教师；直接对齐学生和教师编码器输出，再训练冻结学生的读出头 | 拓扑＋设计参数 |
| `op_distill_finetune` | 蒸馏后再用性能监督微调学生 | 拓扑＋设计参数 |
| `predicted_op` | 先训练设计→OP 预测器并冻结，再训练接收预测 OP 的性能网络 | 拓扑＋设计参数 |
| `predicted_op_joint` | 级联网络联合微调，同时保留真实 OP 辅助损失 | 拓扑＋设计参数 |
| `student_predicted_op` | 冻结 OP 预测器；教师权重初始化同结构级联学生，用真实 OP 教师表示与性能标签联合监督学生 | 拓扑＋设计参数 |
| `student_predicted_op_distill` | 教师权重初始化；先仅蒸馏 encoder，再冻结 encoder、仅训练 decoder | 拓扑＋设计参数 |
| `dual_distill` | 直接设计分支＋蒸馏得到的设计分支，融合读出 | 拓扑＋设计参数 |
| `dual_predicted_op` | 直接设计分支＋OP 预测器的设计编码分支，融合读出 | 拓扑＋设计参数 |
| `dual_design_control` | 相同双编码器及融合结构，但没有 OP 引导 | 拓扑＋设计参数 |

`op_oracle` 与 `design_op_oracle` 都是真实 OP 参考模型，不是数学意义的性能上界。前者不输入设计参数，供原 `op_distill` 路线使用；后者输入设计参数，供 `student_predicted_op` 使用，二者不混用。教师依赖性能标签，因此蒸馏路线不能称为纯无监督 JEPA。两种蒸馏都直接比较编码器表示，没有额外的 latent predictor；教师所有参数（包括 embedding）被冻结，不做 EMA 更新。

## 新增：真实 OP 教师指导预测 OP 学生

`student_predicted_op` 用于检验：真实 OP 条件下学到的表示，是否能帮助同结构编码器适应近似 OP 的误差。它增加了训练目标与初始化，不预先保证优于 `predicted_op`。

1. **训练真实 OP 教师。** 输入拓扑 `G`、设计参数 `x` 和真实 OP `o`，用性能标签训练 `zerosim_device_op` 架构的编码器与读出头。只用 ID 验证集性能 MSE 选取最佳教师，随后冻结完整教师。选择 `design_op_oracle` 时会额外导出其性能结果，它在推理时确实需要真实 OP。
2. **训练 OP 预测器。** 使用原有 `op_predictor`，由 `G,x` 预测归一化的器件 OP。只用 ID 训练 OP 监督，按 ID 验证集逐字段等权 MSE 选模，然后冻结完整预测器。它与 `predicted_op` 等模式共享同一 seed 的前置训练，避免重复训练。训练样本使用该预测器的 in-sample 预测，本实现没有声称采用交叉拟合。
3. **构造同结构学生。** 将选中教师的编码器与性能读出头深拷贝为新的可训练网络，输入改为 `G,x,预测 OP`。教师和学生参数不共享；教师保持冻结，学生编码器及读出头接受更新。OP 预测器始终冻结，因此补偿发生在学生表示中，不会修改预测 OP 数值。
4. **联合优化表示与性能。** 学生训练损失为下面两项的加权和；验证选模只看未加权的性能 MSE，不把 latent 对齐损失加进选模指标。教师目标只用于训练，不参与学生验证/test/OOD 的前向输入。

\[
h_T=E_T(G,x,o),\qquad \hat{o}=P(G,x),\qquad h_S=E_S(G,x,\hat{o})
\]

\[
\mathcal{L}=\lambda_{\mathrm{perf}}\mathcal{L}_{\mathrm{perf}}
             +\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}.
\]

其中性能损失继续使用有效性能标签的归一化 MSE。对齐前，只在 ID 训练数据上拟合教师 latent 的固定均值与尺度：器件 token 按“电路＋拓扑＋器件”分组，graph token 使用训练集 graph 统计量。学生与教师都经过**同一份教师统计量**变换，没有分别重新拟合学生尺度，没有按单个样本做归一化；性能读出头仍接收原始 latent，此标准化只用于对齐损失。

器件对齐只使用非 padding、适用于 OP，且真实 OP 与预测器训练支持字段至少有一项交集的对应器件。存在这类器件的样本才计算 graph 对齐。器件 MSE 与 graph MSE 等权平均形成 `alignment_mse`；没有有效 OP 对齐目标的 batch 仍可接受性能监督，没有有效性能标签的 batch 跳过。OP 缺失掩码只决定训练监督哪些位置，部署时学生使用预测器从训练支持与拓扑生成的掩码。

`student_alignment_weight` 默认 `1.0`，允许设为 `0.0`，便于做“教师初始化＋纯性能微调”的对照。`student_performance_weight` 默认 `1.0` 且必须为正。二者不是自适应权重，应根据验证集选择；不要依据 test/OOD 结果调整。将对齐权重置零仍保留相同教师初始化及依赖成本，不能将其误称为原版随机初始化 `predicted_op`。

冻结教师的训练目标在设备上缓存，避免每轮重复编码教师。逐轮 `history.json` 同时记录性能、原始/缩放后的对齐及加权项，可检查大 loss 是否来自尺度；损失大小不等于梯度贡献。最终 `student_predicted_op/checkpoint.pt` 只保存冻结的 OP 预测器、学生编码器和学生读出头，不需要载入教师或训练对齐尺度即可预测。

建议先保持原 `predicted_op` 与新学生的下游 epoch、batch size、数据划分一致，再查看 `stages.json` 报告教师预训练的额外成本。当前 `example_config.json` 显式设置教师 60、OP 预测器 30、联合学生 60 轮；未写这些键时的代码回退值仍分别为 30、30、50。早停按各阶段验证目标独立执行，因此不能声称整条路线与单阶段基线等计算量。

## 分阶段学生：student_predicted_op_distill

这是一项独立消融，保留 `student_predicted_op` 的联合训练行为。教师仍为 `design_op_oracle`（拓扑＋设计参数＋真实 OP），OP 预测器仍为共享的 `op_predictor`；二者各自训练后完全冻结。学生的 encoder 和 decoder 都复制验证最优教师的权重，包括 embedding，不共享参数存储。

1. **Encoder 阶段**：冻结 OP 预测器、教师和学生 decoder，只更新学生 encoder。输入始终为拓扑＋设计参数＋预测 OP，目标仅为教师的 graph/device 表示对齐 MSE，不使用性能损失，也不调用 decoder。对齐规则和固定的训练集教师尺度与联合版本相同。该阶段可使用有有效 OP 但没有性能标签的 ID 训练样本；无可用对齐目标的 batch 跳过。
2. **阶段切换**：恢复验证对齐损失最优的学生 encoder，完全冻结它并保持 eval 模式；清除旧梯度。decoder 保留复制教师时的权重，在第一阶段没有更新，随后解冻。两个阶段使用独立优化器。
3. **Decoder 阶段**：仅更新学生 decoder，优化有效标签的归一化性能 MSE。固定 OP 预测器和 encoder 的 train/val 特征在 GPU 上一次缓存，避免每轮重复前向。该阶段不需要教师或真实 OP 输入，不重新拟合任何统计量。

**选模与早停**：encoder 阶段监控验证集表示对齐损失，因此会使用验证集真实 OP 作为冻结教师的目标；验证集不参与归一化统计拟合。decoder 阶段监控验证集性能 MSE，完全不读取真实 OP 作为模型输入。两阶段各自应用 `op_transfer.patience`，每次阶段切换与最终测试都恢复该阶段 val-best。test/OOD 不用于拟合或选模。若 train 或 val 整个划分没有可用对齐目标，则明确报错，不静默退化为联合训练或性能监督。

两个独立轮数键在 `op_transfer.epochs` 中，均默认 60；可单独改成 30/60 等组合：

```json
"predicted_op_distill_encoder": 60,
"predicted_op_distill_decoder": 60
```

它们不读取 `student`、`student_predicted_op`、`readout` 或 `training.decoder_epochs`。`student_alignment_weight` 和 `student_performance_weight` 只控制联合版本，分阶段版本的各阶段都是单一、未加权的目标，不受这两个权重影响。

按当前示例配置，该模型的完整预算为教师 60＋OP 预测器 30＋encoder 60＋decoder 60＝210 轮上限，不等同于单阶段基线的 60 轮。与其他模型同一 seed 共用前置教师和 OP 预测器，`stages.json` 对共享成本去重。模型间不共享已经蒸馏或联合训练后的学生权重。

raw 入口的输出在 `experiments/op_transfer/results/<本次实验>/results/` 内：

- `seed_<seed>/stages/predicted_op_distill_encoder/`：encoder 阶段 `best.pt`、`history.json`、`cost.json` 和训练集 `scales.pt`。
- `seed_<seed>/stages/predicted_op_distill_decoder/`：decoder 阶段 `best.pt`、`history.json`、`cost.json`。
- `seed_<seed>/student_predicted_op_distill/`：可部署的 `checkpoint.pt`、`test.json`、存在 OOD 时的 `ood.json` 和 `distillation.json` 审计记录。
- `distillation.json` 记录复制教师的初始化、第一阶段 decoder 未改变、第二阶段 encoder 未改变、教师和 OP 预测器全程未改变的摘要，以及各阶段目标/配置键。
- 同一实验的 `summary.xlsx` 自动增加该模型；日志仍在 `experiments/op_transfer/logs/`。

最终 checkpoint 仅包含 OP 预测器、冻结的学生 encoder 和训练好的 decoder，沿用原有级联模型加载与 `predict.py` 推理，无需教师、真实 OP 或对齐尺度。没有额外的 latent predictor，不做 EMA；OP 数值预测器始终固定。

`jepa_residual` 相对 `jepa_standardized` 改变了不止一个因素；应先比较 `jepa_standardized → jepa_grouped`，再比较 `jepa_grouped → jepa_residual`。所有变体均为待验证的实验，不承诺自动优于直接监督基线。

四个 JEPA 变体在这里都会从头训练，不自动载入以前的 JEPA checkpoint。其 variance floor 默认继承 `loss.variance_floor`；显式设置 `op_transfer.variance_floor` 时覆盖该值，本目录示例设为 `1.0`。改变 floor 后，新旧结果的差异不能全部归因于读出标准化。已有 checkpoint 的严格冻结读出消融仍使用 `experiments/branch_probe/ablate.py`。

## 输入与划分

复用已有实验的 canonical `data.json`，不必重新导入原始 SQLite。建议同时传入那次实验的 `config.json` 和 `results/manifest.json`，固定电路角色和 train/val/test/OOD 划分。

仓库 `configs/default.json` 中的示例电路名可能是 `synthetic_*`。真实电路运行必须使用与数据匹配的配置；单电路和 OOD 由配置/manifest 决定，不由模型名称决定。提供已有 manifest 时，采用其原有划分。

本目录 `example_config.json` 是可直接检查/复制的完整配置：只训练 `two_stage_folded_opamp`、没有 OOD 电路，6 层编码器包含 3 层结构编码和 3 层融合。用于多电路/OOD 时，应修改 `dataset.train_circuits` 和 `dataset.ood_circuits`，或直接复用相应实验的配置和 manifest。

先在仓库根目录指定已有实验目录，例如：

```bash
cd /share/home/202411094934/python_project/Circuit-OP-JEPA
SOURCE_RUN="$PWD/runs/gpu_full/two_stage_folded_opamp_20261028_150115"
```

列出模型和参数：

```bash
python experiments/op_transfer/run.py --list-models
python experiments/op_transfer/run.py --help
```

## 先运行小规模比较

以下命令只预览，不提交作业，也不创建结果/日志目录：

```bash
bash experiments/op_transfer/submit.sh --dry-run \
  --data "$SOURCE_RUN/data.json" \
  --config "$SOURCE_RUN/config.json" \
  --manifest "$SOURCE_RUN/results/manifest.json" \
  --models design_supervised op_oracle op_distill predicted_op \
           dual_distill dual_design_control \
  --seeds 17 \
  --max-train 4096 --max-val 1024 --max-test 1024 \
  --device cuda
```

检查路径后，把 `--dry-run` 换成 `--submit` 即提交。同一命令省略三个 `--max-*` 参数即可使用完整划分。多个训练随机种子可以写成 `--seeds 17 29 43`。初期只选一个模型也可，例如 `--models predicted_op`。

`--max-test` 同时限制 test 和 OOD 各自的样本数；限制是整个划分的上限，不是每个电路各取该数量。样本选择固定，不随训练 seed 改变。全量结论必须使用完整划分重新评估。

JEPA 目标和读出变体可单独提交：

```bash
bash experiments/op_transfer/submit.sh --submit \
  --data "$SOURCE_RUN/data.json" \
  --config "$SOURCE_RUN/config.json" \
  --manifest "$SOURCE_RUN/results/manifest.json" \
  --models jepa_standardized jepa_grouped jepa_residual jepa_op_aux \
  --seeds 17 --max-train 4096 --max-val 1024 --max-test 1024 \
  --device cuda
```

已有 GPU 交互式分配时，可把 `bash experiments/op_transfer/submit.sh --submit` 换成 `python experiments/op_transfer/run.py`。默认要求 CUDA，CUDA 不可用会报错；`--device cpu` 用于小规模机制测试。

默认申请 1 GPU、8 CPU、80 GB 内存；可通过 `SLURM_CPUS`、`SLURM_GPUS`、`SLURM_MEM`、`SLURM_TIME`、`SLURM_ACCOUNT`、`SLURM_PARTITION` 和 `JEPA_CONDA_ENV` 覆盖。默认采用仓库已有集群设置 `b_phzhwu/ex01A800/newbase`。数据文件读取、JSON 解析和 Python 元数据处理仍在 CPU；模型与 resident 数值计算在指定设备执行。

## 单独配置阶段轮数

在实际运行配置 JSON 顶层添加 `op_transfer`。不要覆盖原来整个配置文件；`model`、`dataset`、`normalization` 等仍需保留。

可以先从已有实验配置生成单独的本地配置，保留原电路划分与模型设置，只补入本实验阶段参数：

```bash
mkdir -p experiments/op_transfer/results/configs
python - "$SOURCE_RUN/config.json" <<'PY'
import json
from pathlib import Path
import sys

config = json.loads(Path(sys.argv[1]).read_text())
example = json.loads(Path('experiments/op_transfer/example_config.json').read_text())
config['op_transfer'] = example['op_transfer']
destination = Path('experiments/op_transfer/results/configs/train.json')
if destination.exists():
    raise SystemExit(f'配置已经存在，请编辑该文件或换一个文件名：{destination}')
destination.write_text(json.dumps(config, indent=2) + '\n')
print(destination)
PY
```

之后编辑生成文件中的 `op_transfer`，运行时改用 `--config "$PWD/experiments/op_transfer/results/configs/train.json"`。下面的 JSON 仅展示新增部分：

```json
{
  "op_transfer": {
    "epochs": {
      "teacher": 30,
      "student": 30,
      "op_predictor": 30,
      "student_predicted_op": 50,
      "readout": 50,
      "dual": 50,
      "jepa": 30
    },
    "batch_size": 32,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "grad_clip": 1.0,
    "latent_floor": 0.0001,
    "patience": 10,
    "progress_every": 100,
    "op_aux_weight": 1.0,
    "student_alignment_weight": 1.0,
    "student_performance_weight": 1.0,
    "ema_decay": 0.99,
    "variance_floor": 1.0,
    "torch_threads": 1
  }
}
```

`patience: 0` 关闭早停，但仍选取验证集最优 checkpoint。`variance_floor` 是可调实验超参数，不是物理常量，也不保证每个 latent 维度都有可学信号。编码器维度、层数、head 数等继续使用 `model`。未设置 `op_transfer.batch_size/lr/weight_decay` 时使用原 `training` 的对应设置（lr 回退到 `decoder_lr`）。

`epochs.student` 仍是原 `op_distill` 的 latent 蒸馏轮数；新模型使用独立的 `epochs.student_predicted_op`。`epochs.teacher` 控制两个独立教师各自的预算，`epochs.op_predictor` 控制共享 OP 预测器；不要用原生产训练的 `decoder_epochs` 推断新模型总轮数。

总训练成本包括教师、学生、OP 预测器、读出和双分支阶段，不能仅以最终读出轮数声称计算量相等。`stages.json` 记录实际阶段成本；早停后实际训练轮数可能不同。

## 防止信息泄漏和错误解释

- 参数、OP、性能标签以及 latent 尺度只在对应训练集拟合；验证集用于选模和早停，test/OOD 不选模。
- latent 按训练时器件身份校准；新拓扑/新器件使用训练集全局后备尺度。后备不是 OOD 校准，跨拓扑效果必须单独验证。
- 级联下游训练、验证和测试均接收预测 OP；不会训练时使用真实 OP、测试时偷偷替换。
- 预测 OP 的可用掩码由拓扑适用性和训练支持字段决定，不读取待预测样本真实 OP 的缺失状态。归一化预测直接进入相同归一化空间，避免二次归一化。
- 学生、级联和双分支在性能推理时不需要真实 OP。真实 OP 仅在训练监督、教师参考或 OP 误差诊断中使用。
- 新 `student_predicted_op` 的教师使用真实 OP，学生始终使用预测 OP；教师目标与学生输入是独立构造的，最终部署 checkpoint 不包含教师。其验证选模不使用教师 latent。
- 双分支对照的结构相同；冻结/训练参数量、前置监督和总成本仍可能不同，需要连同结果报告。
- 一次 seed 的标准差没有定义，汇总中保留空值/null；多 seed 的标准差描述训练随机性，不代表样本级置信区间。

## 输出

默认结果目录为 `experiments/op_transfer/results/<UTC时间>_<随机后缀>/`。也可传 `--output` 指定该 `results/` 下的新子目录；已有非空目录会在写入前拒绝，防止覆盖旧数据/报告。

- `run.json`、`protocol.json`：运行设置及协议。
- `stages.json`：各训练阶段与实际成本。
- `summary.csv`、`summary.json`：各模型、划分和指标的多 seed 汇总。
- `summary.xlsx`：MAE、RMSE、R² 的模型对比表，展示 mean ± sample std，保留精确数值明细。
- `seed_<seed>/<model>/checkpoint.pt`、`test.json`、`ood.json`：可恢复的最终模型与有对应划分时的评估结果。
- `seed_<seed>/stages/<stage>/best.pt`、`history.json`、`cost.json`：前置/读出训练的验证最优状态、逐轮记录及成本；阶段文件不等同于最终可部署 checkpoint。
- `seed_<seed>/op_predictor/checkpoint.pt`、`test_op.json`、`ood_op.json`：可单独部署的 OP 预测器和逐字段物理单位误差；联合微调后的对应 OP 误差另存于 `predicted_op_joint/`。
- 新教师阶段为 `seed_<seed>/stages/design_op_teacher/`，新学生阶段为 `seed_<seed>/stages/student_predicted_op/`；各自保存 `best.pt`、`history.json`、`cost.json`。新学生的训练对齐统计另存为该阶段的 `scales.pt`，最佳阶段快照也携带对应 `stage_metadata`。
- `seed_<seed>/student_predicted_op/distillation.json`：记录教师/预测器冻结校验、学生初始化来源、损失权重与对齐掩码等协议。选择 `design_op_oracle` 时，其可推理教师 checkpoint 与性能报告保存在 `seed_<seed>/design_op_oracle/`；只选新学生时不自动增加教师性能结果行。
- `cohort.json`、`preprocessing.json`、`results.json`：实际使用的样本 ID、训练拟合变换以及逐模型逐 seed 报告。

JEPA 阶段的 `history.json` 同时记录原始和加权后的对齐、方差、协方差项，以及相应变体的残差/OP 辅助项；损失数值占比不等于梯度占比。OP 预测器按全验证集逐字段 MSE 等权选模，联合微调级联按性能验证 MSE 选模。冻结教师目标和冻结学生读出特征在设备上预先缓存，拓扑身份只准备一次；`run.json` 另记包含准备与评估的整体耗时。

验证改善时即保存阶段最佳权重，每轮保存历史，减少长作业中断损失；这不提供任意中断后的优化器续训。阶段 `best.pt` 的 `complete` 标明阶段是否完成。Excel 导出失败会把运行状态改为 `failed_export`，已生成的 JSON 可用于手动重导出。

## 无真实 OP 推理

```bash
TRANSFER_RUN="$PWD/experiments/op_transfer/results/<运行目录>"
python experiments/op_transfer/predict.py \
  --checkpoint "$TRANSFER_RUN/seed_17/op_distill/checkpoint.pt" \
  --data /path/to/canonical_design_inputs.json \
  --output "$TRANSFER_RUN/student_predictions.json" --device cuda
```

设计输入 JSON 保留 canonical 拓扑和参数 schema；OP/性能字典可缺省或为空。脚本只使用 checkpoint 保存的训练统计。级联 checkpoint 可加 `--include-predicted-op` 导出归一化及物理单位 OP；独立 `op_predictor/checkpoint.pt` 自动输出 OP 预测。已有输出拒绝覆盖。`op_oracle` 与 `design_op_oracle` 两个性能参考需要真实 OP，其余性能模型均不需要。

新学生使用相同推理入口，把 checkpoint 改为 `seed_17/student_predicted_op/checkpoint.pt` 即可。对于上面从原始 SQLite 启动的运行，`TRANSFER_RUN` 应指向 `<电路名或OOD名>_<UTC时间>_<编号>/results` 这个训练结果子目录，例如：

```bash
TRANSFER_RUN="$PWD/experiments/op_transfer/results/<运行目录>/results"
python experiments/op_transfer/predict.py \
  --checkpoint "$TRANSFER_RUN/seed_17/student_predicted_op/checkpoint.pt" \
  --data /path/to/canonical_design_inputs.json \
  --output "$TRANSFER_RUN/student_predicted_op_predictions.json" \
  --include-predicted-op --device cuda
```

这里导出的 OP 仍来自冻结预测器，新增教师监督补偿的是学生隐空间，不能用这些 OP 数值宣称预测器经过了联合修正。`test.json`/`ood.json` 是性能评估，`op_predictor/test_op.json`/`ood_op.json` 是 OP 字段误差，二者目标不同。

本目录输出格式与原生产 comparison 报告不同，使用配套 `summarize.py` 从 `results.json` 生成 Excel，避免把本目录的长格式 CSV 错当作生产 summary。完整运行自动生成 `summary.xlsx`；需要再次导出不同文件名时：

```bash
python experiments/op_transfer/summarize.py \
  experiments/op_transfer/results/你的运行目录 \
  --output experiments/op_transfer/results/你的运行目录/summary_copy.xlsx
```

已有文件不会覆盖。Excel 中 test/OOD 分开，多电路分数明确标记为 `pooled / ALL_SELECTED_CIRCUITS`，没有从汇总数据虚构逐电路结果；成本另见 `stages.json`。一个 seed 的 std 留空。UGF 的 MAE/RMSE 在展示页换算为 MHz，精确明细仍保留原始 Hz。该整理脚本只依赖 Python 标准库，可以在无 torch 的登录节点执行。

机制测试：

```bash
python -m unittest discover -s experiments/op_transfer -p 'test_*.py' -v
```
