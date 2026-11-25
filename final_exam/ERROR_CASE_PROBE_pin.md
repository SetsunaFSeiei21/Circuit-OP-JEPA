# 指定大误差样本回查

入口：`final_exam/single_circuit_pin/inspect_error_cases_pin.sh`。
这是只读诊断，不训练、不重新抽样、不改模型或原始数据库，也不自动启动 SPICE。

脚本默认检查上一轮报告中 `two_stage_opamp_otaf` 的原始样本 **41074、38711、38196**，
对比 seed **29** 的 `device_op_cascade_pin` 和 `lasd_graph_layer_headwise_detach_pin_v3`。
`SOURCE_RUN` 默认指向之前给出的 `two_stage_opamp_otaf_lasd_moe_20261124T161718Z_1748571_30854_pin`。
它根据当前仓库目录解析，所以 login1 的 `/public` 和 login04 的 `/share` 都可用，前提是该机器上存在原结果。

## 设置与启动

在脚本顶部修改：

```bash
SOURCE_RUN="/实际路径/原训练结果目录"  # 下面有 training_pin；也支持 training_pin 本身
CIRCUIT="two_stage_opamp_otaf"
SAMPLE_IDS=(41074 38711 38196)      # SQLite samples.sample_id，不是样本序号
MODELS=(device_op_cascade_pin lasd_graph_layer_headwise_detach_pin_v3)
SEEDS=(29)
SPLIT="test"
RAW_DIR="$PROJECT_ROOT/runs/raw/full_data/$CIRCUIT"
WAVEFORM_MANIFEST=""              # 没有保存波形时先留空
REFERENCE_SAMPLES=128             # 只作 OP 误差参照，不训练；0 关闭
```

`SOURCE_RUN` 是**原训练目录**，不是 `prediction_probe_*_pin` 诊断结果目录。
必须与产生先前报告的实验一致；脚本验证原 manifest、实际 checkpoint 样本集合和样本 ID。
不会从全量原数据补入一个本次训练没有选中的样本。

login1：

```bash
cd /public/home/202411094934/python_project/Circuit-OP-JEPA
export JEPA_CONDA_HOME=/public/home/202411094934/anaconda3
export JEPA_CONDA_ENV=newbase
bash final_exam/single_circuit_pin/inspect_error_cases_pin.sh --dry-run
bash final_exam/single_circuit_pin/inspect_error_cases_pin.sh
```

login04 用 `/share/home/202411094934/python_project/Circuit-OP-JEPA`，
`JEPA_CONDA_HOME` 指向该集群可用的 Conda；默认 `/share/software/anaconda3`，原包装器也会尝试当前 Conda。
新入口的登录节点预检查同样先激活指定环境，避免系统 Python 太旧。

默认提交一个 1 GPU / 4 CPU / 32G / 1 小时作业，依次检查所列模型。
若已有计算节点分配，使用 `--local`。`--dry-run` 不创建任何目录。
CPU 读取并验证完整保存数据；GPU 只接收三个指定样本和至多 128 个训练 OP 参照样本的当前 batch。
128 个参照只用于诊断冻结 OP 预测器的误差分布，不是重新选择训练集或重新训练。

## 输出

输出仍在现有任务目录下，启动后会打印准确路径：

```text
final_exam/single_circuit_pin/results_pin/<circuit>_error_cases_<时间>_pin/
```

对应日志位于同任务的 `logs_pin/` 与 `hpc_logs_pin/`。

| 文件/工作表 | 内容 |
|---|---|
| `case_summary_pin.xlsx` | 全部下列表和中文口径说明 |
| `case_details_pin.json` | 三个完整 canonical 样本：设计、器件/端口、真实 OP、标签、原状态与条件；附原 SQLite 指定行和可用的小型网表/参数模板 |
| `case_review_pin.html` + `plots_pin/` | SR 预测对照、OP 误差热图、真实 OP 的训练分位数热图、可用波形；需一起下载 |
| `case_statistics_pin.json` | Excel 各表的完整机器可读数据 |
| `Performance` | 每个模型/seed/样本全部性能指标的真实、预测、物理及归一化残差 |
| `Parameters` | 原始设计参数及每个器件物理参数；在实际 train 中的经验分位数、范围 |
| `OPComparison` | 每个器件/字段真实 OP、预测 OP、物理/标准化残差、训练分布位置和训练参照误差分位数 |
| `OPSummary` / `OPReference` | 每个案例 OP 总体误差、每器件字段的训练参照误差分布 |
| `PredictorChecks` | 下游模型是否使用相同的 OP 预测器权重和预处理 |
| `RawChecks` | 保存样本与原数据库的设计/标签/OP 是否一致 |
| `Waveforms` / `ArtifactCandidates` | 波形状态与原元数据中现存路径线索；候选路径不自动视作正确波形 |
| `probe_pin.json` | checkpoint 哈希、原预处理、实际划分及训练参照 ID、执行环境与完成状态 |

每张表同时导出 CSV。先把 **`case_summary_pin.xlsx` 和 `case_details_pin.json`** 发回来即可；
如果加载到了波形，再附对应 `waveforms_pin/*.csv` 或图。

OP 字段沿用保存模型支持的字段，按物理器件 `device_id` 对齐。没有对 PMOS 电流/电压取绝对值，
没有把未支持字段的填零当作实际预测；反变换严格沿用保存的 linear/asinh 参数。
训练分布按同一器件 ID、类型和物理模型计算，不把全电路 MOS 的数值混在一起。
分位数是经验统计，不假设高斯分布；极端值或高误差不自动判为坏数据。

## 瞬态波形没有保存时

`sampling_history.sqlite3` 的性能标量和 DC OP **无法还原瞬态波形**。
没有提供明确对应样本的波形文件时，`Waveforms.status=not_supplied`，其余回查仍正常完成。
脚本不臆造瞬态刺激、PDK、测量窗口或仿真结果；也不从 SR 标量画一条假波形。

若原仿真 workspace 仍在，先看 `ArtifactCandidates` 和 `case_details_pin.json` 里的路径。
如果旧集群有波形，需将对应 trace 复制过来；二进制 SPICE raw 需先按原仿真工具导出带表头的数值表。
要重新仿真则还需要原 testbench、PDK/模型库、仿真条件、激励和 SR 提取规则，不能仅凭本报告等价恢复。

输出自带 `waveform_manifest_template_pin.json`。复制到便于编辑的位置，填写明确映射：

```json
{
  "version": 1,
  "circuit": "two_stage_opamp_otaf",
  "waveforms": [
    {
      "source_sample_id": 41074,
      "path": "/实际路径/41074_tran.csv",
      "time_column": "time_s",
      "voltage_columns": ["vin_v", "vout_v"],
      "delimiter": ",",
      "time_scale_to_s": 1.0,
      "voltage_scale_to_v": 1.0,
      "provenance": "original_saved"
    }
  ]
}
```

如果时间列本身是微秒，`time_scale_to_s=1e-6`；若电压是毫伏，`voltage_scale_to_v=1e-3`。
支持表头 CSV、TSV（delimiter 为制表符）以及 `delimiter="whitespace"`。
一个样本可有多个不同路径文件，比如独立上升/下降激励。
每个文件最多 64 MiB，必须含至少三个有限数值点；时间倒退会拒绝，不跨仿真段拼接求导。
`provenance` 标明 `original_saved` 或 `rerun`，不要将后来重仿真冒充当时原始波形。

再设置脚本的 `WAVEFORM_MANIFEST` 指向填写后的 JSON 并运行，会生成新的诊断目录。
电压曲线和正时间间隔的相邻有限差分都会输出，但导数峰值**不等价于原 SR 指标**，
仍需原测量窗口和算法才能验证 SR 标签。

## 本地验证

```bash
python -m pytest -q final_exam/tests/test_error_case_probe_pin.py \
  final_exam/single_circuit_pin/test_error_case_launch_pin.py
```

测试使用小型合成电路训练并重载 Cascade/v3 checkpoint，覆盖器件乱序对齐、严格样本 ID、
训练参照边界、OP 反变换、无真实 OP 输入、原数据库不修改、错误 split 拒绝、波形单位和重复时间。
