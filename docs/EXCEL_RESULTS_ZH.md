# 导出模型对比 Excel

新实验完成结果汇总后，会自动生成：

```text
runs/gpu_full/电路名_日期时间/results/comparison/model_comparison.xlsx
```

原有 CSV、JSON、checkpoint 和报告继续保留。Excel 只展示已有统计结果，不重新训练、测试或重新计算跨 seed 的统计量。

## 已完成的实验

在仓库根目录执行，将路径替换成已有实验目录：

```bash
python scripts/run/export_results_excel.py \
  --input runs/gpu_full/two_stage_opamp_otaf_你的日期时间
```

`--input` 可以是实验目录、`results/`、`results/comparison/`、`summary.csv` 或 `comparison.json`。
目录内优先读取 `comparison.json`，以展示来源记录的完整性状态；只有 CSV 时完整性标为未知。
默认在找到的汇总文件旁写入 `model_comparison.xlsx`。重复导出会原子替换这个派生 Excel，不修改源文件。
只需 Python 3.10+ 标准库，无需安装 Excel 库、PyTorch 或申请 GPU，可直接在登录节点运行。

筛选测试集、指定模型或另存文件：

```bash
python scripts/run/export_results_excel.py \
  --input runs/gpu_full/two_stage_opamp_otaf_你的日期时间 \
  --split test \
  --models zerosim_device zerosim_device_op zerosim_device_op_only \
  --output runs/gpu_full/device_comparison.xlsx
```

也可以使用安装包入口 `python -m device_jepa.excel`，参数相同。

## 表格含义

- **模型对比**：各模型并排展示 MAE、RMSE、R² 的“均值 ± 标准差”。默认展示各电路的物理尺度结果，test、OOD 和各电路分别成块，不混合统计。根据实际结果自动识别模型数量，不固定为五个；相同模型的不同 backend 使用不同列。
- **数据明细**：保留输入汇总的全部记录，包括未在对比页选中的模型、划分、归一化尺度和 pooled 统计。保留原始单位及数值，支持筛选；对比页的均值与标准差通过公式引用此页。
- 标准差来自原汇总的跨 seed **样本标准差（ddof=1）**。仅一个有效 seed 时显示 `均值 ± n/a`；R² 不可定义时显示 `n/a`，不会填成零。各分数的有效 seed 数可在明细中的 `*_defined_seeds` 查看。
- 物理尺度的 UGF 误差从 Hz 转为 MHz，**均值和标准差同时换算**；R² 不换算。原始 Hz 数据仍在明细中。其他指标保留原有单位。归一化结果使用 `--scale normalized`，不做物理单位换算。
- 显示默认保留 4 位小数，可用 `--decimals 6` 修改；很小的非零值及大数使用科学计数法，避免显示为零或挤满单元格。
- 绿色标出**导出时未舍入均值最优**的模型：MAE/RMSE 越低越好，R² 越高越好。仅在该行所有展示模型都有结果，且总 seed 数、有效 seed 数、标签数一致时标绿。这是描述性比较，不是显著性检验；手动编辑明细后需重新导出以刷新最优标记。
- 导出不会重新验证原始训练协议；完整性状态沿用 `comparison.json`。CSV 单独导出不能证明模型训练、划分或 seed 身份一致。

`--scope pooled` 可单独展示已有的跨电路汇总；不会把各电路指标直接平均。`--split all` 是默认值，各划分分别显示。导出不支持把多个独立实验目录直接混为一份公平对比；应先通过已有严格汇总入口验证并生成对应 summary。
