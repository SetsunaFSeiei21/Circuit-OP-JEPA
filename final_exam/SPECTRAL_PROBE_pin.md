# 低秩与跨电路共享基底诊断

本工具直接读取已经完成的 `final_exam` 单电路 pin checkpoint，检查哪些矩阵可能适合低秩压缩，以及多个电路的同名算子是否在保存坐标中具有共同子空间。它不训练、不修改 checkpoint，也不启动压缩后模型的性能评测。

## 1. 修改入口脚本

文件：`final_exam/single_circuit_pin/inspect_spectrum_pin.sh`。

```bash
CIRCUITS=(Opamp1 two_stage_folded_opamp)
SOURCE_RUNS=(
    "/你的路径/Opamp1_width_decoder_日期_pin"
    "/你的路径/two_stage_folded_opamp_width_decoder_日期_pin"
)
MODELS=(device_op_cascade_pin width_decoder_pin width_decoder_op_pin)
SEEDS=(17)
RANKS=(8 16 32 64)
N_SAMPLES=128
MAX_TOKENS=512
SCOPE="ffn"
COMPONENT="both"
```

- `CIRCUITS` 与 `SOURCE_RUNS` 按位置对应；路径填原实验结果根目录（里面有 `training_pin/`），或直接填 `training_pin/`。
- 不是填写某个模型目录、某个 seed 目录或 `checkpoint_pin.pt` 文件路径。
- 每个源目录都必须包含所选模型和 seed；缺失会在提交作业前报错。支持上述三个模型和 `zerosim_supervised_pin`。若只检查一个电路，只保留一组电路/路径；单模型谱分析照常执行，共享基底列为空。
- `SEEDS` 指已训练 checkpoint 的 seed，不会重新训练。可以改为 `(17 29 43)`；不同 seed 分开分析，不拼成同一个共享基底。
- `N_SAMPLES` 是每个电路从原 manifest 的 **train** 集固定随机抽取的校准样本数，超过现有样本数取全部。设为 `0` 时只分析权重，完全不需要原始数据。
- `MAX_TOKENS` 是每个矩阵累计保留的真实激活向量上限，不是电路数量、每样本 token 数或训练 batch size。各批次分配 token 配额，随机选取有效 token，排除 padding。
- `SCOPE=ffn` 先看 encoder FFN 与 Decoder FFN；`all` 扩展到 Linear、MHA Q/K/V/输出投影和宽读出 MoE 专家。`COMPONENT` 可为 `both`、`performance`、`op_predictor`。
- `MAX_PLOTS=12` 只限制导出的图数量，CSV/Excel 保留所有矩阵；设为 0 可以不安装 matplotlib。

生成图片需要在实际作业使用的环境安装 matplotlib（无需安装 Excel 包）：

```bash
conda activate newbase
python -m pip install matplotlib
```

提交与预览：

```bash
bash final_exam/single_circuit_pin/inspect_spectrum_pin.sh --dry-run
bash final_exam/single_circuit_pin/inspect_spectrum_pin.sh
```

默认 8 CPU、1 GPU、80G、2 小时，账户 `b_phzhwu`，分区 `ex01A800`，conda `newbase`。GPU 执行模型前向、精确 float64 SVD、子空间和重构误差计算；checkpoint/JSON 读取、来源校验、Excel/图片写入在 CPU。CUDA 不可用时直接报错，不自动回退。已进入计算节点 allocation 时可用 `--local`。登录节点预检只读取小型 JSON 和文件存在性，不导入 torch、不加载模型/原始数据。

也可直接调用 Python：

```bash
python final_exam/spectral_probe_pin.py \
  --source "Opamp1=/实际路径/Opamp1结果根目录" \
  --source "two_stage_folded_opamp=/实际路径/折叠运放结果根目录" \
  --models width_decoder_pin width_decoder_op_pin \
  --seeds 17 --ranks 8 16 32 64 \
  --num-samples 128 --max-tokens 512 --scope ffn --device cuda
```

## 2. 输出路径与先看什么

每次创建一个新 run ID，原实验目录不会被覆盖：

| 内容 | 路径 |
|---|---|
| Excel、CSV、图片、HTML | `final_exam/single_circuit_pin/results_pin/spectral_probe_日期_编号_pin/` |
| 实验过程日志 | `final_exam/single_circuit_pin/logs_pin/相同编号/spectral_probe_pin.log` |
| Slurm stdout/stderr、提交脚本快照 | `final_exam/single_circuit_pin/hpc_logs_pin/相同编号/` |

结果目录中：

| 文件/工作表 | 用途 |
|---|---|
| `spectral_summary_pin.xlsx` / `Parameters` | 每个模型的 SE、PE、readout 等参数数目；包含冻结 OP predictor，按模型分别统计 |
| `Matrices` | 每个矩阵的形状、参数数、stable rank、能量有效秩、rank95/99/99.9、实际激活 token 数 |
| `IndependentLowRank` | 单矩阵不同 rank 的参数预算、相对权重误差、真实激活上的局部输出误差 |
| `SubspaceOverlap` | 同模型/seed/矩阵在不同电路上的左右子空间重合度和主夹角 |
| `SharedBudget` | 多电路共享 U/V、小型独立 S 的理论参数预算，整个共享组计一次 |
| `SharedErrors` | 共享基底对各电路矩阵/局部层输出的误差 |
| `Checkpoints` | 路径、SHA256、实际配置、总参数/分析覆盖参数、源 commit（如果保存了） |
| `Notes` | 未捕获激活、只有一个匹配电路等未适用情况，不能解读成零误差 |
| `SingularValues_pin.csv` | 全部矩阵的完整奇异值与累计平方能量；不截断成仅前几个值 |
| `report_pin.html` | 离线报告、数值文件链接和谱图；连同 `plots_pin/` 一起下载查看 |
| `probe_pin.json` | 请求设置、实际采样 ID、checkpoint 配置与身份、运行状态和错误信息 |

先看 `Parameters` 找大矩阵，再看 `Matrices.rank_99`，最后同时查看 `IndependentLowRank` 的参数节省和实际层输出误差。不能只根据权重谱曲线就认为截断安全。

## 3. 数学定义

统一按 `y=W x` 分析，`W` 形状 `[m,n]`。权重 SVD 为 `W=U diag(s) Vᵀ`。独立 rank-r 截断使用前 r 个分量，保留原矩阵 dtype 计算数值秩阈值，分解/误差计算使用 float64。请求 rank 超过 `min(m,n)` 时显式记录 `actual_rank`。

- `rank_99`：累计 `s²` 达到总能量 99% 的最小 rank。
- `stable_rank = sum(s²)/max(s²)`。
- `effective_rank_energy = exp(-sum(p log p))`，其中 `p=s²/sum(s²)`；这里明确采用平方能量，不与使用 `s/sum(s)` 的定义混淆。
- `relative_weight_error = ||W-Wr||F / ||W||F`。
- `relative_layer_output_error = ||(W-Wr)X||F / ||WX||F`，X 为真实模型原始前向中截取的训练集输入。
- `optimal_calibration_output_error`：对 `WX` 做 rank-r 截断所得的同子集最优输出误差。可由 `W'=U_y,r U_y,rᵀ W` 达到，但这里只计算诊断值，不替换权重；它是校准子集上的最佳局部结果，不能当成验证集/测试集误差。
- 零矩阵能量秩为 0；零子空间的重合度为空。分母为零且残差非零时相对误差为空。未执行/未捕获的位置也为空，并有 Notes 说明。

每层都使用**原始上游激活**，不会把多个层的截断串起来模拟。因此当前工具不产生“压缩后 R²”。后续需要分别实现实际因式分解替换、训练集微调、验证集选秩和独立 test 评测。

## 4. 共享基底与对齐边界

针对同一个模型、seed、架构和同名同形状矩阵，在多个电路间拟合 `W_c ≈ U S_c Vᵀ`：先将每个非零矩阵按 Frobenius 范数缩放到单位范数，横向拼接求公共左子空间 U，纵向拼接求公共右子空间 V，然后 `S_c=UᵀW_cV`。误差始终在原始权重及该电路自己的激活上计算。架构或矩阵形状不同不会硬拼；Notes 会记录缺少配对。

这是联合行/列 SVD 初始化，不是迭代求解的全局最优 Tucker 分解。C 个电路对应矩阵的预算：原始 `Cmn`，独立低秩 `Cr(m+n)`，共享双侧基底 `r(m+n)+Cr²`，均不含偏置。增加 rank 不一定节省参数，负 saving 会如实保留。

**本版不做隐藏通道/神经元/注意力头重排对齐。** 子空间夹角和共享拟合结果都属于保存的原坐标。相同 seed 也不能保证不同电路训练后坐标语义一致；谱接近不等于方向接近，方向接近也不等于功能相同。输出不能用于声称“跨电路不能共享”或“已经发现共同物理规律”。若原坐标结果差，应先检查归一化/字段约定和模型对齐，再考虑共享压缩。

共享基底拟合使用所有列出的电路，因此这些都是拟合任务，不是 OOD 测试任务。想评估未见电路，需要另留整个电路不参与基底、秩和统计量选择。

## 5. 实现与验证

- `spectral_probe_pin.py`：来源预检、真实 checkpoint 恢复、ID train 抽样、来源校验、分组执行。
- `spectral_analysis_pin.py`：统一矩阵方向、参数账本、有限激活采集、精确 SVD、独立与共享误差。
- `spectral_report_pin.py`：完整 CSV、标准库 Excel 写入器、matplotlib PNG、离线 HTML。
- `single_circuit_pin/inspect_spectrum_pin.sh`：可编辑设置、只读预检、Slurm 提交与日志分配。

可运行测试：

```bash
python -m pytest -q final_exam/tests/test_spectral_probe_pin.py \
  final_exam/single_circuit_pin/test_spectral_launch_pin.py
```

测试覆盖已知谱、低秩共享真值、零矩阵、正交子空间、激活权重导致的重要性变化、padding/预算、专家矩阵方向、真实小型模型训练后的 checkpoint 校准与报表、原文件不变、无数据权重模式、无 torch 的登录预检、输出保护、Slurm 预览/失败路径。CUDA 数值对照只有在有 CUDA 硬件时执行。
