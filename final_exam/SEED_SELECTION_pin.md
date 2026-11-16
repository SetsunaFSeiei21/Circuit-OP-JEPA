# 按模型选择最好/最差 seed 后统计

入口：`final_exam/single_circuit_pin/summarize_selected_seeds_pin.sh`。
仅读取已完成的端口级 `final_exam` 实验，不训练、不加载数据/checkpoint，不需要 GPU、PyTorch、pandas 或 openpyxl。可直接在登录节点使用 Python 标准库运行。

## 设置与运行

修改 Bash 顶部：

```bash
SOURCE_RUN="/share/home/202411094934/python_project/Circuit-OP-JEPA/final_exam/single_circuit_pin/results_pin/你的实验目录"
K=2
MODELS_BETTER=(
    width_decoder_pin
    width_decoder_op_pin
)
MODELS_WORSE=(
    device_op_cascade_pin
)
OUTPUT=""
RECOVER_EXPORT=0
```

然后运行：

```bash
bash final_exam/single_circuit_pin/summarize_selected_seeds_pin.sh
```

`SOURCE_RUN` 可以是包含 `training_pin/` 的实验根目录、`training_pin/` 本身，或其中的 `results_pin.json`。
不是某个模型的子目录，也不是 `summary_pin.xlsx`。必须保留原来的结果 JSON、run/config/stages JSON、每个模型的配置，以及门控模型的 `gate_comparison_pin.json`。

`SOURCE_RUN`、`K`、`OUTPUT` 也可用环境变量传入。模型数组仍在 Bash 顶部编辑。
某组不用时写成 `MODELS_BETTER=()` 或 `MODELS_WORSE=()`，两组不能同时为空。

## 排序和统计口径

用户选择的排序依据是**测试集各指标 R² 的等权平均**。

1. 对每个模型的每个候选 seed，计算测试集 macro R²；默认九指标，不按量纲或标签数量加权。
2. `MODELS_BETTER` 中每个模型保留分数最高的 K 个 seed。
3. `MODELS_WORSE` 中每个模型保留分数最低的 K 个 seed。
4. 未列入两组的模型保留全部 seed。各模型可以选择不同的 seed ID。
5. 同分时按整数 seed ID 从小到大取。K 必须为正整数；超过某个指定模型的候选数量时报错，不悄悄少取。K 等于候选数时保留全部。
6. 选中的是**完整 seed 结果**。MAE、RMSE、R² 都使用同一组 seed，不为每个指标单独挑 seed。
7. 沿用原统计器的样本标准差 `ddof=1`，K=1 时标准差留空，不伪造为 0。
8. 一个指标的 R² 若对所有候选都未定义，会从排序分数中排除并记录；只有部分候选缺失 R² 时拒绝排序，避免不同 seed 用不同指标集合。
9. 如果输入已包含多个测试电路，先在每个电路内等权平均指标，再等权平均电路；要求所有模型/seed 的测试电路集合相同。不能混合 pooled 总表和分电路表重复计分。
10. 如果结果同时含 OOD，选 seed 仍只看 test，OOD 沿用该模型选定的 seed，不再根据 OOD 分数另选一次。只有 OOD、没有 test 的输入不能使用此入口。这不增加任何 OOD 训练功能。

这是按测试表现挑选后的描述性统计，不能作为未筛选的多 seed 性能估计。所有筛选页显式标记 `TEST-SELECTED SEEDS`，保留完整候选和原始统计，不改写原实验结果。

## 输出

默认新文件：

```text
<SOURCE_RUN>/training_pin/summary_seed_selection_k2_pin.xlsx
```

若输入已是 `training_pin/`，直接放在该目录下；数字随 K 变化。
同名输出已存在时拒绝覆盖，修改 `OUTPUT` 为新的 `*_pin.xlsx` 即可。不能用此入口替换原 `summary_pin.xlsx`。

保留原汇总的页面和规则：

| 页面 | 内容 |
|---|---|
| MAE / RMSE / R2 | 筛选后均值 ± 样本标准差；UGF 误差显示 MHz |
| PerSeed / Statistics | 入选 seed 的原始指标和精确汇总；UGF 原始值为 Hz |
| Configuration | 原始实验及所有模型/seed 配置，保留原计划 |
| TrainingCost | **全实验**的实际成本，包含被丢弃的 seed，共享阶段只算一次 |
| Resume / WidthDecoder / ModelInputs | 按源结果类型保留相应页面；Resume 保留全实验成本来源 |
| GateBeforeAfter | 入选 seed 的门控前后逐 seed 对照 |
| GateStudy | 按每个门控模型分别匹配其入选 seed 的基座，不把多个模型的不同 seed 混成一份基座均值 |

附加审计页面：

| 页面 | 内容 |
|---|---|
| SeedSelection | 所有候选 seed、macro R²、从好到坏排名、best/worst/all 策略、是否入选、原结果路径及 SHA256、参与/排除排序的指标 |
| AllSeedStatistics | 所有原始 seed 的精确 mean/std，便于与筛选结果对照 |
| AllSeedPerSeed | 所有原始 seed 的逐指标数据，包括未入选 seed |
| SelectedTrainingCost | 入选结果依赖的阶段成本，共享依赖去重；**不代表完整实验成本** |

保持原来的高亮规则：只有模型 seed 集合、标签数和输入分组匹配时才标绿。各模型挑出的 seed 不同，可能没有绿色“最佳值”；数值仍正常统计。真实 OP 参考仍独立分组。

## 检查与导出恢复

先严格检查**完整原计划**的模型/seed/split、指标、配置、依赖成本及门控对照，然后筛选。丢弃某 seed 不会绕过该 seed 的缺文件、错误数值或不一致配置检查。

源运行必须 `complete`。如果所有训练/评估都已完成、仅最后生成 Excel 失败，状态为 `failed_export`，可设置 `RECOVER_EXPORT=1`，按原导出恢复逻辑严格检查后生成这个新表；不更改原失败状态。正在训练、训练失败或缺少候选结果时不能用这个入口补成“完整”结果。

也可直接使用 Python：

```bash
python final_exam/summarize_selected_seeds_pin.py \
  /path/to/saved/run \
  --k 2 \
  --models-better width_decoder_pin width_decoder_op_pin \
  --models-worse device_op_cascade_pin
```

原来的 `summarize_pin.sh` 默认行为和训练自动生成 `summary_pin.xlsx` 的行为保持不变。
