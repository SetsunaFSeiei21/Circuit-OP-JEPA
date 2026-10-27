# 从原始数据启动跨电路 OOD 实验

在仓库根目录执行：

```bash
bash scripts/run/train_ood.sh
```

默认提交一个 Slurm 作业，申请 8 CPU、1 GPU、80 GB 主机内存。
作业在计算节点读取、合并原始数据，生成配置与固定划分，依次训练所选模型和 seed，
评估 ID test 与 OOD，再汇总 CSV 和 Excel。无需提前手动创建联合 data.json。

## 在 Bash 文件开头设置实验

`scripts/run/train_ood.sh` 中：

```bash
TRAIN_CIRCUITS=(5t_ota Opamp0 Opamp1)
OOD_CIRCUITS=("${OOD_CIRCUIT:-two_stage_folded_opamp}")
RAW_ROOT="${RAW_ROOT:-$PROJECT_ROOT/runs/raw/full_data}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/default.json}"
MODELS=(
  zerosim_pin
  zerosim_device
  jepa_zerosim
  zerosim_pin_op
  zerosim_device_op
  zerosim_pin_op_only
  zerosim_device_op_only
)
SEEDS=(17 29 43)
```

`TRAIN_CIRCUITS` 中各电路内部按配置划分 train/val/test；`OOD_CIRCUITS` 中各电路的全部样本
留作 OOD，不能进入预训练、训练统计量拟合、监督训练或验证选模。
模型从本次 ID 数据重新训练，不会读取之前单电路实验的 checkpoint。
OP 输入模型在 OOD 推理时仍读取该样本的 DC OP；JEPA 性能推理不输入 OP。

该脚本按 `RAW_ROOT/电路名/` 组装原始目录，名称必须与 SQLite 的 `dataset_schema.circuit_name` 一致。
单电路脚本中的 `CIRCUIT` 只控制单电路输入路径；OOD 脚本使用独立的两组电路名单。
临时指定一个 OOD 电路或训练配置可以使用：

```bash
OOD_CIRCUIT=two_stage_folded_opamp CONFIG=runs/configs/my_train.json \
bash scripts/run/train_ood.sh
```

OOD 目标不能同时在 `TRAIN_CIRCUITS` 中；如果更换目标，按需要调整训练名单。
支持在文件中把 `OOD_CIRCUITS` 改为多个明确名称。batch size、epoch、早停与网络大小仍由 CONFIG 设置。

## 预览、检查与输出

```bash
# 预览不提交、不读取原始数据，也不创建结果目录：
bash scripts/run/train_ood.sh --dry-run

# 可在提交前抽查四个原始目录的几何单位：
python scripts/run/check_raw_geometry.py \
  --input runs/raw/full_data/5t_ota runs/raw/full_data/Opamp0 \
          runs/raw/full_data/Opamp1 runs/raw/full_data/two_stage_folded_opamp \
  --max-samples 100
```

默认结果目录为当前仓库 `runs/gpu_full/two_stage_folded_opamp_OOD_日期时间/`，
包括 `config.json`、联合 `data.json` 和 `results/`。
`results/manifest.json` 保存共同划分，`results/comparison/model_comparison.xlsx` 分别展示 test 和 OOD。
日志在 `hpc_logs/gpu_full/`。现有 `JEPA_RESULTS_ROOT`、`JEPA_LOG_DIR` 绝对路径覆盖仍受支持。
单电路与 OOD 作业相互独立；反复启动会创建新实验目录，不会续跑旧结果。

原始数据导入仍严格检查 W/L/m、器件身份与单位；脚本不会自动猜测或修改单位。
联合实验需要保存全部导入对象及对应数值缓存，四个 60,000 点数据集的内存容量需在实际集群验证；
`--max-table-rows none` 只解除单表行数保护，不是分块训练。`--smoke` 只缩短 epoch，不减少导入样本。

## 底层原始数据入口

也可以用通用提交脚本，目录名与电路身份不同时显式提供路径：

```bash
source scripts/run/cluster.env.example
export JEPA_RESULTS_ROOT="$PWD/runs/gpu_full"
export JEPA_LOG_DIR="$PWD/hpc_logs/gpu_full"
DRY_RUN=0 bash scripts/run/submit_raw_comparison.sh \
  --input runs/raw/full_data/5t_ota runs/raw/full_data/Opamp0 \
          runs/raw/full_data/Opamp1 runs/raw/full_data/two_stage_folded_opamp \
  --train-circuits 5t_ota Opamp0 Opamp1 \
  --ood-circuits two_stage_folded_opamp \
  --config configs/default.json --models all \
  --seeds 17 29 43 --max-table-rows none --device cuda
```

多个输入必须同时指定 `--train-circuits` 与 `--ood-circuits`，两组名称需互斥、无重复且完整覆盖所有输入的 SQLite 电路身份。
单输入且不指定上述参数时保留原有独立单电路行为，自动设置 `ood_circuits=[]`。
已有联合 canonical 数据仍可使用 `submit_comparison.sh --data ... --config ... --output ...`。
