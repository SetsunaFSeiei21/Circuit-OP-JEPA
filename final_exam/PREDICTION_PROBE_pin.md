# 保存模型的逐样本预测诊断

需要进一步回查指定异常样本的设计参数、真实/预测 OP 和原始波形时，使用
[指定样本回查入口](ERROR_CASE_PROBE_pin.md)：`single_circuit_pin/inspect_error_cases_pin.sh`。
默认检查 `two_stage_opamp_otaf` 原始 ID 41074/38711/38196、seed 29。

入口：`final_exam/single_circuit_pin/inspect_predictions_pin.sh`。

修改脚本顶部：

```bash
SOURCE_RUNS=(
    "/public/home/202411094934/python_project/Circuit-OP-JEPA/final_exam/single_circuit_pin/results_pin/two_stage_opamp_otaf_lasd_moe_..._pin"
    "/public/home/202411094934/python_project/Circuit-OP-JEPA/final_exam/single_circuit_pin/results_pin/Opamp1_lasd_moe_..._pin"
)
MODELS=()   # 留空：每个目录中全部已完成的性能模型
SEEDS=()    # 留空：全部已完成的 seed
BASELINE="device_op_cascade_pin"
SPLITS=(val test)
```

只想检查指定模型时：

```bash
MODELS=(device_op_cascade_pin lasd_graph_layer_headwise_detach_pin_v3)
SEEDS=(17 29 43)
```

模型和 seed 筛选作用于每个目录；目录中尚未完成的组合记录为 `skipped_incomplete`，其他已完成组合正常分析。同一个电路可以给多个结果目录，以便比较不同批次训练的模型。目录可以精确到运行根目录或其 `training_pin`，无需指定 seed/模型子目录。

当前入口面向 `single_circuit_pin` 的保存格式，可一次处理多个电路，各电路独立分析；不接受 multi_circuit 网格搜索 campaign 目录。加载器复用仓库的 `load_checkpoint_pin`，支持已保存的 Cascade、LASD/graph 各版本、宽 decoder、低秩/共享 decoder、端口级神经网络 baseline 和原生 XGBoost checkpoint。Oracle 的真实 OP 输入需求单独记录；普通部署模型的 forward 不接收真实 OP、性能标签或其缺失掩码。

## 启动

```bash
bash final_exam/single_circuit_pin/inspect_predictions_pin.sh --dry-run
bash final_exam/single_circuit_pin/inspect_predictions_pin.sh
```

默认提交一个 Slurm 作业，顺序加载多个目录/模型/seed。默认 `b_phzhwu / ex01A800`，1 GPU、8 CPU、32G 内存、4 小时，可在顶部或通过环境变量修改。CPU 只驻留有限的 256 条预处理样本块，GPU 只接收当前 batch；原始 canonical JSON 仍需要在 CPU 完整读取，一次只读取一个源目录。

login1 运行前：

```bash
export JEPA_CONDA_HOME=/public/home/202411094934/anaconda3
export JEPA_CONDA_ENV=newbase
```

login04 沿用脚本默认 Conda 根目录，或设置为该机器实际的 Conda 根目录。已有计算节点分配时可以 `--local`；不要在登录节点直接运行 GPU 分析。

依赖沿用训练环境的 PyTorch/NumPy；绘图需要 `matplotlib`。Excel 使用项目自带写入器，不要求 openpyxl。检查 XGBoost 模型时需要训练环境中的 XGBoost。设置 `PLOTS=0` 可只导出 CSV/JSON/Excel。

也可直接运行：

```bash
python final_exam/prediction_probe_pin.py \
  --source-runs /path/to/run_a /path/to/run_b \
  --models device_op_cascade_pin lasd_graph_layer_headwise_detach_pin_v3 \
  --splits val test --device cuda
```

`--inspect-source` 在登录节点只检查小型 JSON 与文件存在性，不导入 PyTorch、不读取大数据文件、不生成结果。推理时才校验数据指纹和 checkpoint。

## 输出和统计口径

输出在新建的 `single_circuit_pin/results_pin/prediction_probe_..._pin`；实验日志在对应 `logs_pin`，Slurm 输出在对应 `hpc_logs_pin`。不修改任何原训练目录。

| 文件/表 | 内容 |
| --- | --- |
| `prediction_viewer_pin.html` | 离线可筛选的图索引；电路、模型、seed、指标、val/test 分开查看 |
| `plots_pin/*.png` | 真值-预测散点、残差-真值、绝对误差 CDF、最大残差的累计平方误差贡献 |
| `prediction_summary_pin.xlsx` | 下面各统计表和中文 ReadMe |
| `predictions_pin.csv.gz` | 全量逐样本真值、预测、物理残差、归一化残差、有效性、原数据库 sample_id |
| `sample_membership_pin.csv` | 每个源目录完整 train/val/test 名单及是否有监督训练标签 |
| `sources_pin/.../selection_pin.json` | 每个 checkpoint 实际使用的样本名单（含 smoke 子集）；源 manifest 另存 |
| `Metrics` / `Summary` | 每 seed 的 MAE、RMSE、R²、归一化 MSE、偏差、残差分位数、大误差占比；跨全部完成 seed 的 mean±std |
| `Aggregate` | 全指标按有效标签数加权的 normalized MSE；与原保存 test 报告的重放差值 |
| `ValTestGap` | 同一个保存 checkpoint 的逐指标 val/test 差值 |
| `Thresholds` / `Strata` | 使用有监督训练样本的 P1/P5/P95/P99，将 val/test 分为五个互斥标签区间 |
| `LargestErrors` / `PairedCases` | 最大绝对残差，以及相对基线平方误差增加/减少最多的样本 |
| `PairedSeeds` / `PairedSummary` | 同 seed、同样本的模型-基线差值；跨配对 seed 的区间和检验 |
| `PairStatus` / `Inventory` | 不可配对、缺少基线、未完成组合的明确记录 |

各表同时提供 CSV，完整统计在 `statistics_pin.json`。读取压缩预测文件可使用 `pandas.read_csv('predictions_pin.csv.gz')`，或 Python 标准库 `gzip.open` + `csv.DictReader`。

`MAX_PLOT_POINTS` 只影响散点显示，推理、CDF、误差贡献和统计一直使用保存划分的完整有效样本。`TOP_ERRORS` 只控制大误差明细的导出条数，不剔除任何样本。常量标签或不足两个观测时 R² 留空，负 R² 保留。标准差使用 ddof=1，单 seed 不伪造标准差。

## 配对与统计假设

以 `device_op_cascade_pin` 为默认基线，使用全部已完成 seed。只有数据指纹、train/val/test 和有监督训练名单、checkpoint 实际样本子集、性能归一化均一致时，才归为同一 cohort。之后按相同 seed 和相同样本 ID 配对，并复核标签一致性；不会只取两个不同测试集的交集。`cohort_id` 不同的记录独立展示，无法找到基线的模型仍保留自身所有图表与统计。

同一 cohort/model/seed 若在两个来源中出现不同或重复 checkpoint，会报错要求选择一个来源，不根据测试分数自动挑结果。同名模型不同 seed 的保存训练设置也会校验一致性（忽略 seed 和模型选择列表）；不同超参数实验应分两次诊断，避免将它们冒充 seed 重复。每个 checkpoint 的设置另存为 `training_protocol_pin.json`。

显著性检验的重复单位是配对训练 seed。使用双侧 sign-flip（<=16 seed 精确枚举，其余 Monte Carlo），假设零假设下差值符号可交换；同 cohort/电路/split 的所有模型×指标×统计量采用 Holm 校正。95% 区间用训练 seed 的 percentile bootstrap；它只描述固定数据划分下的训练随机性。三个配对 seed 的精确双侧 p 最小为 0.25，不能把数千个测试样本当作数千个独立训练重复。只有一个 seed 时不给 CI 或 p。

这些报告用于诊断 val/test 差异和定位高残差，不能据此替换验证集所选 checkpoint；本入口不会选择最好/最坏 seed。`Aggregate` 使用 `rtol=1e-4, atol=1e-6` 比较重放和原报告 MSE，同时核对测试样本数；不匹配会标记 `mismatch_check_before_interpreting` 并打印警告。`Metrics` 也列出每项 MAE/RMSE/R² 的重放差值。明显不匹配时先核查推理环境、模型版本和原记录，再解释误差分布。`probe_pin.json` 保存源 checkpoint 哈希、分析代码版本、Python/PyTorch 版本。

## 实现文件

- `prediction_probe_pin.py`：来源检查、checkpoint 重建、按原划分逐批推理、完整样本导出。
- `prediction_analysis_pin.py`：指标、训练分位数、配对 seed 分析和 Holm 校正。
- `prediction_report_pin.py`：图、离线 HTML、CSV、Excel。
- 原训练、模型实现、采样逻辑和 checkpoint 选择流程不变。
