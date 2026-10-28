# 第一步：同一 JEPA 预训练 checkpoint 的两端冻结读出

这个目录独立保存诊断代码、日志和结果，不修改正式七/八模型比较流程，也不重跑预训练。
目标是区分：OP 端是否已经学到有用的工作点表示，以及设计端是否学到了可用于性能预测的表示。
支持 `jepa_zerosim` 和 `jepa_zerosim_ema` 的 `pretrain_best.pt`；不能用只含设计端的 `decoder_best.pt` 代替。

## 实验约定

- 从**同一个**完整预训练 checkpoint 恢复两端。结构、设计字段、OP/state embedding、融合主干全部冻结并使用 eval 行为；不执行任何 EMA 更新。
- `design_probe` 只消费拓扑与设计参数；`op_probe` 消费拓扑和 OP/可选状态，不消费设计参数。OP 端诊断仍需要仿真 OP，不能当作无 OP 推理模型。
- 两端使用相同架构、相同初始参数的 metric-query decoder，相同的训练样本、标签、batch 顺序、种子、优化器配置和最大训练轮数。各自通过 ID validation 选择最佳 checkpoint；启用早停时实际轮数可能不同。
- 沿用源 checkpoint 的 manifest，包括 `train`、`labeled_train`、`val`、`test`、`ood`。设计和 OP 归一化保持不变；性能归一化只拟合原 `labeled_train`。测试/OOD 不参与拟合或选模。
- **配对读出只使用两边都有的、拓扑上有 OP 资格的器件 token，不使用 graph token。** 原版 OP MLP 没有 graph token，因此两端统一排除。资格来自器件类别/显式 `op_target`，不会因某个样本 OP 是否缺失而改变读出 mask。缺失 OP 的样本仍保留并报告覆盖率；没有任何合格器件的样本会报错。
- 这与正式设计端 decoder 使用 graph/全部器件的读出协议不同，结果不能直接冒充正式 `jepa_zerosim`/`zerosim_device_op` 排名。旧版共享 SE 和新版独立 EMA SE 的来源身份都会保留。

默认只读出 **val**，用于方法诊断。val 同时参与 decoder 选模，因此它不是无偏的最终性能估计。结构方案确定后，再显式启用 `test`/`ood`；不要反复依据 OOD 选择结构。

## 直接运行

从项目根目录执行；替换为已经完成实验的实际路径：

```bash
python experiments/branch_probe/run.py \
  --data runs/gpu_full/你的实验目录/data.json \
  --pretrained runs/gpu_full/你的实验目录/results/jepa_zerosim/seed_17/pretrain_best.pt \
  --seeds 17 29 43 \
  --device cuda
```

EMA 版本只需把 `--pretrained` 指向 `results/jepa_zerosim_ema/seed_17/pretrain_best.pt`。
默认 decoder 超参数来自源 checkpoint 的训练配置，可以显式覆盖：

```bash
python experiments/branch_probe/run.py \
  --data runs/gpu_full/你的实验目录/data.json \
  --pretrained runs/gpu_full/你的实验目录/results/jepa_zerosim_ema/seed_17/pretrain_best.pt \
  --epochs 30 --batch-size 256 --lr 0.001 \
  --seeds 17 29 43 --splits val --device cuda
```

`--seeds` 是 **decoder 训练种子**。例如固定 `seed_17/pretrain_best.pt` 后训练 17、29、43 三个 decoder 种子，得到的是固定表示下读出训练的波动，**不是三个独立预训练实验的标准差**。

最终评估可显式传入：

```bash
python experiments/branch_probe/run.py \
  --data runs/gpu_full/你的实验目录/data.json \
  --pretrained runs/gpu_full/你的实验目录/results/jepa_zerosim/seed_17/pretrain_best.pt \
  --seeds 17 29 43 --splits val test ood --device cuda
```

若源 manifest 没有 OOD，不能请求 `ood`；不要用 ID test 冒充 OOD。

## 集群提交

默认资源与现有集群配置一致：8 CPU、1 GPU、80G 内存，`b_phzhwu/ex01A800`、`newbase` 环境。先预览：

```bash
bash experiments/branch_probe/submit.sh --dry-run \
  --data "$PWD/runs/gpu_full/你的实验目录/data.json" \
  --pretrained "$PWD/runs/gpu_full/你的实验目录/results/jepa_zerosim/seed_17/pretrain_best.pt" \
  --seeds 17 29 43 --device cuda
```

确认路径后，将 `--dry-run` 换成 `--submit` 即可提交。也支持 `DRY_RUN=0`。
`SLURM_CPUS`、`SLURM_GPUS`、`SLURM_MEM`、`SLURM_TIME`、`SLURM_ACCOUNT`、`SLURM_PARTITION`、`JEPA_CONDA_ENV` 和 `JEPA_CONDA_HOME` 可通过环境变量覆盖。
提交日志位于本目录的 `logs/<时间戳>/`。不会在测试中提交真实作业。

## 输出与汇总

默认结果位于本目录 `results/<自动生成的运行名>/`。可以用 `--output` 指定新子目录，但必须位于 `experiments/branch_probe/results/` 内，不允许覆盖已有实验。原始数据与 checkpoint 只读，无需复制到诊断目录。

```text
experiments/branch_probe/
  run.py / probe.py / summarize.py / submit.sh
  logs/<时间戳>/<jobid>.out、<jobid>.err
  results/<运行名>/
    seed_17/design/...
    seed_17/op/...
    comparison/
      summary.json
      summary.csv
      per_seed.csv
      paired_deltas.csv
      paired_summary.csv
      model_comparison.xlsx
```

每端保存 decoder 最佳 checkpoint、训练历史和所请求 split 的报告，记录预训练来源、冻结验证、实际训练成本、OP 覆盖率及选模依据。共享预训练成本只记录一次，不能把 test/OOD 或两端重复加总。

Excel 并排展示两端 MAE、RMSE、R² 的均值 ± 标准差，按 split 和电路分别展示，数值单位沿用数据集（UGF 展示换算为 MHz）。仅一个有效 decoder seed 时标准差显示 `n/a`；常量目标等导致 R² 未定义时保持缺失。
`paired_deltas.csv` 记录同 seed 的 `OP − design`：MAE/RMSE 为负、R² 为正表示 OP 端读出更好。不同物理单位的误差不做简单平均。

已有报告可以独立重新导出，不需要训练或 GPU：

```bash
python experiments/branch_probe/summarize.py \
  --input experiments/branch_probe/results/你的诊断目录
```

## 如何解释

若 OP 端弱，优先检查 OP 预训练目标与表示；若 OP 端强而设计端弱，优先检查跨模态预测/对齐是否有效。两端相近只能说明当前读出协议下的可用信息相近，不能据此证明完全迁移或没有丢失信息。
结构差异、缺失 OP、decoder 优化和种子波动都可能影响差距。这个第一步不包含打乱 OP 配对、加入 predictor、OP 重建辅助任务或重新训练 teacher。

运行 `python -m unittest discover -s experiments/branch_probe -p 'test_*.py'` 可检查小样本机制。
CPU 合成检查不能验证真实电路预测收益、GPU 吞吐或 30 万样本内存容量。数值张量和缓存可驻留 CUDA，JSON 读取与数据校验仍在 CPU 上执行。

## 第二步：检查同一器件的跨设计变化，换简单读出器

当两端冻结 decoder 都接近常数输出时，运行 `diagnose.py`。它读取同一个完整预训练 checkpoint，冻结两端，按器件 ID 对齐同一电路的样本，不重新预训练、不读 test/OOD 标签用于拟合或评估。

```bash
RUN="$PWD/runs/gpu_full/two_stage_folded_opamp_20261028_150115"
bash experiments/branch_probe/submit_diagnostics.sh --submit \
  --data "$RUN/data.json" \
  --pretrained "$RUN/results/jepa_zerosim_ema/seed_17/pretrain_best.pt" \
  --circuit two_stage_folded_opamp \
  --seed 17 --device cuda
```

默认仍为 **8 CPU + 1 GPU**。`--dry-run` 只预览，`--submit` 才提交；也可以在已分配 GPU 的计算节点用 `python experiments/branch_probe/diagnose.py` 加同样的 Python 参数直接运行。

默认从原 `labeled_train`、`val` 中按固定种子、与标签无关地抽取最多 **4096 / 1024** 个样本。`--max-train-samples none --max-val-samples none` 使用该电路的全部原划分样本。该子集诊断不能直接与全量正式训练的分数排名比较。完整 canonical JSON 仍须加载校验，抽样不会减少 JSON 加载峰值内存。

输出包括：

- `diagnostics.json`、`device_variation.csv`：分别统计 train/val 内**同一器件、同一维度跨设计样本的标准差**；另外记录器件均值之间的差异及两者合并后的波动，防止只看混合 token 的方差。标准差采用总体定义，不是多 seed 误差条。
- `readouts.json`、`readouts.csv`：设计端 latent、OP 端 latent、原始 OP 输入，各做一次线性岭回归和小型 MLP。另有仅输出所选训练集均值的基线；都报告 train/val 的 MAE、RMSE、R²、归一化 MSE、预测和真实值的跨样本标准差。
- `run.json`：来源 checkpoint 哈希、原划分指纹、实际抽样 ID、超参数与完成状态；`preprocessing.json` 保存诊断的归一化统计。

这里把**固定 ID 的器件表示拼接**后读出，不采用均值池化，不包含 graph token。要求同一电路的器件 ID、类型、物理 model、端口、连接和 OP eligibility 相同；多个 ID 电路时必须用 `--circuit` 选一个，拓扑不一致会明确报错。

原始 OP 对照使用与 OP 分支相同的连续 OP 字段、缺失 mask，以及 checkpoint 使用的 OP state 和 mask，不额外拼入设计参数。它是固定拓扑诊断对照，输入维数与 latent 不同，不能当作等参数量的新正式 baseline。

线性岭回归使用 float64 闭式求解，目标为 `sum(square_error) + alpha * ||W||²`，截距不惩罚，默认 `--ridge-alpha 1`。不用神经网络优化器，也不搜索正则系数。MLP 是 `Linear → GELU → Linear`，默认 hidden=128、batch=256、lr=0.001、最多 50 epochs、验证早停 patience=10；每轮打印进度。两端 latent 的 MLP 使用相同初始化和样本顺序。

预训练输入归一化保持不变；性能标签与读出特征统计只在**所选训练子集**拟合。特征逐坐标标准差下限默认 `--std-floor 1e-4`，用于限制对极小浮点噪声的放大；原始表示方差统计不经过此缩放。训练子集中完全没有标签的指标不进入验证标签转换、选模或读出评分，并保留 `count=0`。

只想快速看表示时加 `--stats-only`，会跳过全部读出训练。新结果仍在本目录 `results/<新时间戳>/`，日志在 `logs/`；不会覆盖正式实验或第一步诊断。结果不进入正式模型排行榜。

解释时先看两端的 `within_device_std_rms` 与 `between_device_std_rms`：前者极小、后者明显非零符合“保留器件差异、忽略设计差异”的现象。再看原始 OP 对照能否预测性能，以及简单读出是否优于常数基线。冻结表示方差非零不保证含有性能信息；所有读出都差也不能单凭此实验断定 OP 本身无效。先使用验证集定位，修复完成后再做独立 test/OOD 评估。
