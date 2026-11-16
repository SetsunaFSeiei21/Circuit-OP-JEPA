# 同一模型内部的多 decoder 共享下基诊断

本工具检查一个已训练模型内部、多个 PE 对应的 decoder 是否可以使用共同的下投影基，并为每个 decoder 保留独立的上投影。一次任务可以列出多个电路、模型和 seed；每个 checkpoint 都单独拟合、单独计算误差。最后的 Excel 只汇集结果，**不使用其他电路、其他模型或其他 seed 的权重拟合本 checkpoint 的共享基**。

这是分析工具，不会训练新的共享参数模型，不会修改来源 checkpoint，不会向来源结果目录写文件。已有 `width_decoder_lowrank_*_pin` 使用的是各 decoder 独立的低秩因子；本工具研究的共享下基是另一个约束。

## 1. 共享什么

按 PyTorch 线性权重的方向，`W_i` 的形状为 `[out, in]`，共享方案为：

`W_i ≈ B_i A`，其中 `A=[rank,in]` 跨 PE decoder 共享，`B_i=[out,rank]` 由各 decoder 独立保留。

它对应 `Linear(in, rank, bias=False) → Linear(rank, out, bias=原bias)`。两个因子之间没有激活函数。不同 PE 的输入不同，上投影也不同，所以共享下投影并不要求各 decoder 输出相同。

共享分组固定为**同一模型、同一组件、同一 decoder 内部层序号、同一投影角色**：

| 分组 | 跨多个 PE decoder 共享的对象 |
|---|---|
| query self-attention | Q、K、V、O 各自一个下基 |
| memory cross-attention | Q、K、V、O 各自一个下基 |
| FFN 第一层 | `linear1` 的下基 |
| FFN 第二层 | `linear2` 的下基 |

Q/K/V/O 之间不合并拟合，self-attention 与 cross-attention 之间不合并，FFN 两层不合并，decoder 内部第 1 层与第 2 层不合并。只跨相应的 PE 分支共享。普通单读出模型没有多个 PE decoder，不是本工具的分析对象。

`COMPONENT=performance` 只分析性能端宽 decoder；`op_predictor` 只分析宽 OP 预测器的 decoder；`both` 分别分析两端。两端仍不共用下基。`width_decoder_op_pin` 才有宽 OP 预测器；普通 `width_decoder_pin` 的 OP predictor 使用最终器件 MLP。

## 2. 如何拟合、如何比较

将同组原权重纵向拼接：

`W_stack = [W_1; W_2; ...; W_m] = U Σ Vᵀ`。

取 `A=Vᵀ[:rank]`，再令 `B_i=W_i Aᵀ`。这里对原始权重直接做 SVD，不预先将各分支缩放到单位 Frobenius 范数。这给出当前保存坐标下、总平方 Frobenius 误差最小的共同输入子空间。纵向拼接对应共享下基；横向拼接会对应另一种共享方向。

同一候选 rank 同时计算两个参考：

- **Independent**：每个 decoder 的同类矩阵独立做截断 SVD。
- **Shared down**：多个 decoder 的同类矩阵共同拟合下基，各自保留上基。

原始完整权重是 Dense 参考。若请求的 rank 超过矩阵或拼接矩阵的维度，报告分别记录实际秩，不能只看请求值。特别是 `linear2` 的单分支形状为 `[128,2048]` 时，三个分支的拼接矩阵可以具有高于 128 的秩；各自 rank≤128 并不意味着共同 rank≤128。

分析包含：

1. 共同矩阵的奇异值谱、累计平方奇异值能量及所需 rank。
2. 每个候选 rank 的完整权重、独立低秩、共享下基参数预算。
3. 各 PE 分支的权重重建相对误差，以及已捕获位置的真实输入激活输出误差。
4. 可选的验证集端到端输出与性能指标诊断，比较原模型、独立重建和共享重建。

权重误差是 `||W-W_hat||F / ||W||F`；投影输出误差是 `||(W-W_hat)X||F / ||WX||F`。它们不是 MAE、RMSE 或 R²。未捕获的激活位置留空并记录原因，不能当作 0 误差。

激活取保存划分中的训练样本，不重新拟合归一化。端到端误差取保存划分中的验证样本，不读取 test 做候选 rank 选择。网络输入仍遵守原模型的部署边界；不会把真实 OP 或性能标签输入无需真实 OP 的模型。

验证时只在内存中临时替换重建出的完整权重，完成后恢复原权重；它用于测量函数误差，**不是低秩因子实际部署的速度测试**。不会保存可部署的压缩 checkpoint，也不保证实际低秩实现会同比加速。

## 3. 参数预算口径

对 m 个同形状权重，忽略保持不变的 bias：

- 完整权重：`m × in × out`。
- 独立低秩：`m × rank × (in + out)`。
- 共享下基：`rank × in + m × out × rank`。

三分支示例：

| 投影 | 完整权重 | 独立低秩 | 共享下基 |
|---|---:|---:|---:|
| 128→128，rank32 | 49,152 | 24,576 | 16,384 |
| FFN 128→2048，rank64 | 786,432 | 417,792 | 401,408 |
| FFN 2048→128，rank64 | 786,432 | 417,792 | 155,648 |

这些只统计候选矩阵权重，不是整个模型参数量。完整模型、未分析组件、bias/embedding/normalization 等另有参数记录。已是低秩模型的输入 checkpoint 会按其有效权重进行谱分析，其现有存储参数和理论完整矩阵预算是不同概念。

多个分支输入不同，仍需要分别计算 `A x_i`。共享参数不代表该矩阵乘法可以只算一次，也不意味着运行速度按分支数量倍增。

## 4. 配置与启动

编辑 `final_exam/single_circuit_pin/inspect_shared_decoder_pin.sh` 顶部：

```bash
CIRCUITS=(Opamp1 two_stage_folded_opamp)
SOURCE_RUNS=(
    "/share/home/202411094934/python_project/Circuit-OP-JEPA/final_exam/single_circuit_pin/results_pin/Opamp1_width_decoder_..._pin"
    "/share/home/202411094934/python_project/Circuit-OP-JEPA/final_exam/single_circuit_pin/results_pin/two_stage_folded_opamp_width_decoder_..._pin"
)
MODELS=(width_decoder_pin)
SEEDS=(17)
RANKS=(8 16 32 64)
N_SAMPLES=128
N_VAL_SAMPLES=128
MAX_TOKENS=512
SAMPLE_SEED=17
BATCH_SIZE=16
SCOPE="all"
COMPONENT="performance"
VALIDATION_SCOPE="all"
MAX_PLOTS=12
```

`CIRCUITS` 与 `SOURCE_RUNS` 按位置对应。路径填某一次实验的结果根目录，或其 `training_pin` 子目录，不填某个模型目录。每个来源都必须包含请求的全部 MODELS 和 SEEDS；提交前检查所有来源，避免一部分分析完成后才发现其他来源缺文件。

支持宽 decoder 模型 `width_decoder_pin`、`width_decoder_op_pin`，以及已有的 `width_decoder_lowrank_attention_pin`、`width_decoder_lowrank_ffn_pin`、`width_decoder_lowrank_both_pin`。默认仅分析 `width_decoder_pin`，不要求每个实验保存所有模型。

| 设置 | 含义 |
|---|---|
| `N_SAMPLES` | 每个电路从保存的 train 划分抽取的激活诊断样本数；0 禁用激活捕获 |
| `N_VAL_SAMPLES` | 从保存的 val 划分抽取的端到端诊断样本数；0 禁用验证前向 |
| `MAX_TOKENS` | 每个矩阵最多保存的激活向量数，控制分析内存 |
| `RANKS` | 同一 checkpoint 内要比较的候选秩 |
| `SCOPE` | `all` / `attention` / `ffn`，限定 decoder 投影范围 |
| `COMPONENT` | `performance` / `op_predictor` / `both` |
| `VALIDATION_SCOPE` | `all` 同时替换所有选定分组；`group` 每次只替换一个分组；`both` 两类都做 |
| `MAX_PLOTS` | 每个报告最多绘制的分组数；0 不依赖 matplotlib |

`N_SAMPLES=0` 且 `N_VAL_SAMPLES=0` 时只读模型权重，不读取完整数据。只要启用样本诊断，现有 canonical JSON 仍需要读取和校验；原始文件解析在 CPU，模型前向与 SVD 使用所选 GPU，`DEVICE=cuda` 时不静默退回 CPU。

在仓库根目录运行：

```bash
bash final_exam/single_circuit_pin/inspect_shared_decoder_pin.sh --dry-run
bash final_exam/single_circuit_pin/inspect_shared_decoder_pin.sh
```

默认提交一个 Slurm GPU 作业，在此作业内逐个分析所列 checkpoint。默认 8 CPU / 1 GPU、账户 `b_phzhwu`、分区 `ex01A800`、环境 `newbase`。CPU 数量不表示会用 CPU 进行模型推理。`--dry-run` 只打印命令，不创建结果或日志目录。

已有 GPU 计算节点分配时可使用：

```bash
bash final_exam/single_circuit_pin/inspect_shared_decoder_pin.sh --local
```

`--local` 不申请 Slurm 资源，因此应在已有计算节点分配中运行。无需重新训练，也无需重新导入 raw SQLite 数据。

## 5. 输出文件与日志

任务目录仍为 `final_exam/single_circuit_pin/`：

| 内容 | 路径 |
|---|---|
| 结果根目录 | `results_pin/shared_decoder_probe_<时间戳和唯一编号>_pin/` |
| 汇总 Excel | `<结果根>/shared_decoder_summary_pin.xlsx` |
| 单 checkpoint Excel | `<结果根>/<电路>/seed_<seed>_pin/<模型>/shared_decoder_summary_pin.xlsx` |
| 单 checkpoint 来源与设置 | `<结果根>/<电路>/seed_<seed>_pin/<模型>/probe_pin.json` |
| 表格 CSV、HTML 和图 | 各报告目录下的 `*_pin.csv`、`report_pin.html`、`plots_pin/` |
| 实验日志 | `logs_pin/<相同run_id>/shared_decoder_probe_pin.log` |
| HPC 输出和错误 | `hpc_logs_pin/<相同run_id>/%j_pin.out`、`%j_pin.err` |
| 提交设置快照 | `hpc_logs_pin/<相同run_id>/submitted_inspect_shared_decoder_pin.sh` |

Excel 分别记录来源 checkpoint 的路径、哈希、配置、归一化来源，分组谱指标、参数预算、各分支误差、可选验证结果和局限说明。多个电路的表格只是放在同一个文件中，仍带有 circuit/model/seed 标识；不会跨电路平均误差后冒充某个模型的结果。

## 6. 怎样解释结论

- **个别矩阵低秩不等于共同下基低秩。** 应观察纵向拼接后的谱和 shared 相对 independent 的额外误差。
- **谱误差小不保证任务指标不变。** 还要看激活误差、多个分组同时替换后的验证误差。
- **旧权重直接共享失败不等于共享训练失败。** 各 decoder 独立训练时隐藏坐标可以不同；FFN 隐藏通道的同步排列可以保持原网络函数，但影响未经对齐的共同输入子空间分析。
- **本工具没有做神经元排列对齐，也没有微调。** 它测的是现有坐标下的可压缩性，不是共享参数架构从头训练的能力上限。
- **少量验证样本只用于初筛。** 最终应固定方案后比较原始 decoder、独立低秩、共享下基三组，多 seed 训练，再统一测试；不应使用 test 反复挑 rank。

当前分析不更改训练入口、模型命名或训练预算。若后续确定共享方案，需要另外实现真正参数绑定的训练模型，以及可重建该参数共享关系的 checkpoint 协议。
