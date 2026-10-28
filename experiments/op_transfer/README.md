# OP transfer experiments

本目录用于比较 OP 信息怎样转移到仅接收拓扑与设计参数的模型。代码、运行结果和日志分别在本目录、`results/` 和 `logs/`；不会另建一份仓库，也不会改写已有生产实验。

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
| `jepa_standardized` | 基础 EMA 对齐目标、本实验统一损失配置＋训练集拟合的标准化读出 | 拓扑＋设计参数 |
| `jepa_grouped` | 按同一拓扑、同一器件分组计算跨样本变化约束 | 拓扑＋设计参数 |
| `jepa_residual` | 在 grouped 变体基础上加入中心化、缩放后的残差对齐 | 拓扑＋设计参数 |
| `jepa_op_aux` | 在基础 JEPA 上加入设计 latent → 真实 OP 的辅助监督 | 拓扑＋设计参数 |
| `op_distill` | 完整冻结性能监督训练好的 OP 教师；直接对齐学生和教师编码器输出，再训练冻结学生的读出头 | 拓扑＋设计参数 |
| `op_distill_finetune` | 蒸馏后再用性能监督微调学生 | 拓扑＋设计参数 |
| `predicted_op` | 先训练设计→OP 预测器并冻结，再训练接收预测 OP 的性能网络 | 拓扑＋设计参数 |
| `predicted_op_joint` | 级联网络联合微调，同时保留真实 OP 辅助损失 | 拓扑＋设计参数 |
| `dual_distill` | 直接设计分支＋蒸馏得到的设计分支，融合读出 | 拓扑＋设计参数 |
| `dual_predicted_op` | 直接设计分支＋OP 预测器的设计编码分支，融合读出 | 拓扑＋设计参数 |
| `dual_design_control` | 相同双编码器及融合结构，但没有 OP 引导 | 拓扑＋设计参数 |

`op_oracle` 是真实 OP 参考模型，不是数学意义的性能上界。教师依赖性能标签，因此蒸馏路线也不能称为纯无监督 JEPA。默认蒸馏没有 predictor，教师所有参数（包括 embedding）被冻结，教师不做 EMA 更新。

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
    "ema_decay": 0.99,
    "variance_floor": 1.0,
    "torch_threads": 1
  }
}
```

`patience: 0` 关闭早停，但仍选取验证集最优 checkpoint。`variance_floor` 是可调实验超参数，不是物理常量，也不保证每个 latent 维度都有可学信号。编码器维度、层数、head 数等继续使用 `model`。未设置 `op_transfer.batch_size/lr/weight_decay` 时使用原 `training` 的对应设置（lr 回退到 `decoder_lr`）。

总训练成本包括教师、学生、OP 预测器、读出和双分支阶段，不能仅以最终读出轮数声称计算量相等。`stages.json` 记录实际阶段成本；早停后实际训练轮数可能不同。

## 防止信息泄漏和错误解释

- 参数、OP、性能标签以及 latent 尺度只在对应训练集拟合；验证集用于选模和早停，test/OOD 不选模。
- latent 按训练时器件身份校准；新拓扑/新器件使用训练集全局后备尺度。后备不是 OOD 校准，跨拓扑效果必须单独验证。
- 级联下游训练、验证和测试均接收预测 OP；不会训练时使用真实 OP、测试时偷偷替换。
- 预测 OP 的可用掩码由拓扑适用性和训练支持字段决定，不读取待预测样本真实 OP 的缺失状态。归一化预测直接进入相同归一化空间，避免二次归一化。
- 学生、级联和双分支在性能推理时不需要真实 OP。真实 OP 仅在训练监督、教师参考或 OP 误差诊断中使用。
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

设计输入 JSON 保留 canonical 拓扑和参数 schema；OP/性能字典可缺省或为空。脚本只使用 checkpoint 保存的训练统计。级联 checkpoint 可加 `--include-predicted-op` 导出归一化及物理单位 OP；独立 `op_predictor/checkpoint.pt` 自动输出 OP 预测。已有输出拒绝覆盖。`op_oracle` 是唯一需要真实 OP 的性能参考。

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
