# 同一模型内多个 PE decoder 的共同下基训练

## 实验目的与已有代码的区别

已有 `width_decoder_pin` 给每一个 PE 输出配置独立的性能 decoder，再把各分支的性能特征拼接送入 MoE-MLP。已有 `width_decoder_lowrank_*_pin` 将投影分解为两个小矩阵，但每个 decoder 的上下因子仍全部独立。`inspect_shared_decoder_pin.sh` 则只对保存的权重做 SVD、子空间相似度和临时重建诊断，不会训练共享结构。

本次增加的是**从头训练的共同下基架构**：同一模型内部不同 PE 的对应 decoder 投影共用一份可训练下基，各自保留可训练上基。它没有使用旧 checkpoint 做 SVD 初始化，也没有加入独立残差权重或私有低秩分支。每个电路、每个 seed、每个模型均独立训练；多个电路的权重不放在一起拟合，不跨电路共享参数。

## 模型清单与共享范围

| 模型 ID | 跨 PE decoder 共享下基的位置 | 其他 decoder 投影 |
|---|---|---|
| `width_decoder_pin` | 无，原宽读出对照 | 原始完整矩阵 |
| `width_decoder_shared_kv_pin` | 仅 cross-attention 的 K、V | 原始完整矩阵 |
| `width_decoder_shared_attention_pin` | self-attention、cross-attention 各自的 Q/K/V/O | FFN 保留完整矩阵 |
| `width_decoder_shared_ffn_pin` | FFN 的 `linear1`、`linear2` | attention 保留完整矩阵 |
| `width_decoder_shared_both_pin` | 上述全部 attention 与 FFN 投影 | 无额外投影替换 |

这里的 attention 只指**性能端 decoder 内部的 attention**，不包含编码器的 SE/PE attention。Q、K、V、O 分别拥有自己的下基，self-attention 和 cross-attention 也分别拥有下基。`shared_kv` 不意味着 K 和 V 共用同一个下基。

若 `model.cross_layers=3`、`model.decoder_layers=2`，有三个 PE 读出分支，每个分支的 decoder 含两层。三个分支的第 1 层 cross-K 共用一份 A，第 2 层 cross-K 共用另一份 A；两个内部层号之间不共享，cross-V 又是独立的共享组。对应投影需要相同输入和输出维度。

以下组件沿用原宽读出：

- 器件级 OP predictor 为原始设计编码器加末端器件 MLP，先训练，再冻结。它不使用宽 OP decoder，也不参与本次下基共享。
- 性能端 pin encoder、各 PE 输出的排列、padding、拓扑和命名端口语义保留。
- 各分支的性能 query、LayerNorm、bias 保持独立；最终 MoE 的专家和 router 保留。
- 不添加门控，不引入真实 OP 教师 latent 蒸馏；模型推理仅需拓扑和设计参数，由冻结的 predictor 产生近似 OP。

## 线性映射与实际参数共享

权重按 `[out_features, in_features]` 记。对同一投影组中的第 i 个 decoder：

\[
W_i=B_iA,\qquad y_i=B_i(Ax_i)+b_i.
\]

`A` 的形状为 `[rank, in_features]`，在多个 PE decoder 之间共享；每个 `B_i` 为 `[out_features, rank]`，每个 `b_i` 为 `[out_features]`，均独立。两个线性因子之间不插入激活或 dropout；FFN 原本两层之间的 ReLU 和原有 dropout 位置保留。A 与 B 均从头初始化、均接受性能损失的梯度。A 汇总各分支的梯度，优化器只登记和更新同一个共享参数一次。

训练实际执行两次较小线性映射，不先生成完整 `B_i A` 再执行大矩阵乘法。attention 的头数、softmax、序列长度和掩码规则不变。这不是 token 压缩，也不是保留原矩阵再增加低秩增量的 LoRA。

共享不是“复制初始值”：多个 decoder 引用同一份 A。checkpoint 保存共享结构的配方，并在重建时恢复共享关系；恢复后修改共同下基会同时影响对应的多个 decoder。bias、上基和其他分支参数仍各自保存。参数统计以实际唯一参数计数，不能把共享下基重复乘以分支数。

## 配置与 rank 限制

配置模板：`final_exam/configs/shared_decoder_pin.json`。

```json
"shared_decoder_pin": {
  "attention_rank": 32,
  "ffn_rank": 64
}
```

模型名称控制启用哪些共享组。`attention_rank` 对启用的 attention 投影生效，`ffn_rank` 对启用的两个 FFN 投影生效。未启用的投影保持完整权重；`width_decoder_pin` 始终是未共享的原始对照。

rank 必须是正整数，不超过该投影输入、输出维度的较小值；同时要求整组共享权重严格减少参数。对 m 个相同形状的矩阵，不含 bias：

\[
P_{dense}=m\,out\,in,\qquad
P_{shared}=r\,in+m\,out\,r.
\]

这个限制不同于每个分支独立低秩化时的 `r*(in+out)<in*out`。例如三个 128×128 投影使用共同下基时，rank 64 的参数数为 32,768，小于原组的 49,152，因而可以使用；独立分解的 rank 64 则没有节省参数。参数预算由实际分支数量和投影形状校验，不应套用独立低秩模型的 rank 上限。

当前默认 `dim=128`、`cross_layers=3`、`decoder_layers=1`、`decoder_ffn_dim=2048`。在此形状下，下列是**被列出的投影权重总数**，不含 bias、encoder、predictor、query、归一化和 MoE：

| 投影集合 | 完整独立投影 | 共同下基后的投影 |
|---|---:|---:|
| 三个 decoder 的 cross-K/V，rank 32 | 98,304 | 32,768 |
| 三个 decoder 的全部 attention 投影，rank 32 | 393,216 | 131,072 |
| 三个 decoder 的两层 FFN，rank 64 | 1,572,864 | 557,056 |
| 全部 attention rank 32＋FFN rank 64 | 1,966,080 | 688,128 |

这些是结构公式的参数预算，不是精度、GPU 延迟或显存实测结果。不同 PE 的输入不同，仍分别计算 `A x_i`；减少参数不保证等比例加速。

MoE 仍由 `width_decoder_pin` 配置块控制，例如 `n_experts=4`、`hidden_dim=128`、`dropout=0.1`。修改 decoder 内部层数会影响每个分支的深度，也会增加独立共享组数，应对所有待比较模型使用一致设置。

## 训练阶段与比较口径

同一个作业中，每个 seed 先训练一次原始器件级 OP predictor，默认最多 30 轮；恢复验证集最优权重后冻结。然后按 `MODELS` 逐个训练性能模型，默认每个最多 60 轮。性能阶段一起更新 pin encoder、共享 A、独立 B、query、归一化、MoE 等未冻结参数。选中的模型复用同一 seed 的冻结 predictor 和原有预测 OP 缓存，不重复训练相同依赖。

控制项均在 `op_transfer` 下：

| 配置项 | 默认值 | 含义 |
|---|---:|---|
| `epochs.op_predictor` | 30 | OP 预测器阶段最大轮数 |
| `epochs.readout` | 60 | 每个性能模型最大轮数 |
| `batch_size` | 32 | 训练批大小 |
| `lr` | 0.0001 | 学习率 |
| `patience` | 10 | 验证损失连续不改善的早停等待轮数 |
| `progress_every` | 100 | batch 训练进度打印间隔 |

`patience=0` 关闭早停，仍按验证结果选模。模板保留了公共配置的其他阶段预算，但本组不会因此训练未选择的教师、蒸馏模型或门控。每个模型使用自己的验证集最佳性能权重测试；test 不参与 rank 选择或 checkpoint 选择。

相同最大轮数不代表相同计算量；早停后的实际轮数、阶段时间与参数量应同时比较。与独立低秩模型比较时，同 rank 的共享模型参数更少，因此同 rank 对照和近似同参数量对照是不同实验，不能混写。

## 电路、模型与提交命令

编辑 `final_exam/single_circuit_pin/train_shared_decoder_pin.sh` 顶部：

```bash
CIRCUITS=(two_stage_folded_opamp Opamp1 Peng_ACBC)
MODELS=(
    width_decoder_pin
    width_decoder_shared_kv_pin
    width_decoder_shared_attention_pin
    width_decoder_shared_ffn_pin
    width_decoder_shared_both_pin
)
SEEDS=(17 29 43)
```

模板默认只列二级折叠运放；上例为每个电路分别提交一个单电路作业，不是联合训练或 OOD。可以任意删减模型名单，仅训练和测试所选模型及必要依赖。脚本中同时列出了可取消注释的三个独立低秩模型，用于对比 `W_i=B_i A_i` 与 `W_i=B_i A`。独立低秩模型的 rank 由单独的 `low_rank_decoder_pin` 配置块控制；模板为此也保存 attention 32 / FFN 64，不会因此自动训练未选中的模型。要做其他 rank 的匹配对照，应同时修改两个配置块；同一个 rank 必须满足各自架构的有效性约束。

启动新训练不需要 `SOURCE_RUN` 或旧 checkpoint：

```bash
cd /share/home/202411094934/python_project/Circuit-OP-JEPA
bash final_exam/single_circuit_pin/train_shared_decoder_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_shared_decoder_pin.sh
```

`--dry-run` 只显示命令，不提交、不写实验目录。`--smoke` 提交缩短轮数的训练，但不会自动减少样本；小样本试运行应先在本组配置中设置 `dataset.max_samples_per_circuit`，例如 2000 或 20000。`null` 表示全量；抽样发生在划分之前，`sampling_seed` 固定抽样，所有所选模型使用同一份样本与划分。

原始数据仍放在 `runs/raw/full_data/<电路>/`。默认 Slurm 8 CPU、1 GPU、80G 主机内存、`b_phzhwu` / `ex01A800`、conda `newbase`。训练请求 CUDA，不会因 GPU 缺失静默退回 CPU；数据读取与静态准备仍需要 CPU。脚本会在提交时保存配置快照。

## 输出、日志与结果整理

所有产物都放在当前仓库的 `final_exam/single_circuit_pin/`：

| 内容 | 相对路径 |
|---|---|
| 本次结果根目录 | `results_pin/<电路>_shared_decoder_<时间和唯一编号>_pin/` |
| HPC stdout、stderr、提交配置和记录 | `hpc_logs_pin/<相同run_id>/` |
| 数据导入、训练、评估日志 | `logs_pin/<相同run_id>/` |
| 本次 canonical 数据与配置 | `results_pin/<run_id>/data_pin.json`、`config_pin.json` |
| 自动生成的指标 Excel | `results_pin/<run_id>/training_pin/summary_pin.xlsx` |
| 模型配置、checkpoint、测试报告 | `.../training_pin/seed_<seed>_pin/<model>/` |
| 固定批次级联推理测量 | `.../training_pin/seed_<seed>_pin/<model>/benchmark_pin.json` |
| 冻结的 OP predictor | `.../training_pin/seed_<seed>_pin/shared_pin/device_op_predictor_pin/` |

每个模型的 `config_pin.json`、`checkpoint_pin.pt`、`test_pin.json` 保存适用的架构与共享设置。共享设置是 checkpoint 重建和恢复校验的一部分，不能只改模型名把旧的独立权重当成共享模型。阶段历史、最佳权重、实际训练成本和续训状态按已有阶段目录保存。最终模型保存推理所需的冻结 predictor，部署不需要另找教师文件；本组本身没有性能教师训练阶段。

Excel 保留原有 MAE、RMSE、R² 的均值和跨 seed 样本标准差、逐 seed 指标、配置、训练成本、宽读出及共享下基结构信息。一个 seed 的跨 seed 标准差留空。仅比较样本、划分与 seeds 一致的结果，OP 依赖训练成本按 stage 去重。

新增 `SharedDecoder` 表包括：参数化方式、共享 attention 范围、FFN 是否共享、实际 rank、PE 分支数、共享组数和投影角色、只计一次的共享 A 参数、独立 B 参数、所选投影的完整/因子权重参数预算及节省比例。bias 不计入这部分投影权重预算，但实际模型总参数和可训练参数包含它；模型总参数还包含未改变的 encoder、MoE 和冻结 OP predictor。表中也记录实际 readout 轮数和时间、固定批次推理延迟/延迟标准差、samples/s、CUDA 进程峰值和增量峰值。

推理诊断复用现有宽 decoder 的微基准：完成正常评估后，对已驻留设备的同一 test batch 预热 3 次、测量 10 次完整级联前向，包括执行冻结 OP predictor，不读取预测 OP 缓存，不使用性能标签。数据读取和搬运在计时范围之外，CUDA 计时同步；它不选择 checkpoint、不更新模型。延迟 std 是重复前向的测量标准差，与指标跨 seed std 不同。CUDA 显存是当前进程状态，可能包括驻留数据和缓存，不能视为一个模型独占的部署显存。阶段训练时间另计并包括验证。未测量字段留空，不应写成零。

重新整理结果仍使用原脚本：

```bash
bash final_exam/single_circuit_pin/summarize_pin.sh \
  final_exam/single_circuit_pin/results_pin/<run_id> \
  --output final_exam/single_circuit_pin/results_pin/<run_id>/summary_review_pin.xlsx
```

## 失败后的恢复

```bash
bash final_exam/single_circuit_pin/resume_shared_decoder_pin.sh \
  final_exam/single_circuit_pin/results_pin/<失败run_id> --dry-run

bash final_exam/single_circuit_pin/resume_shared_decoder_pin.sh \
  final_exam/single_circuit_pin/results_pin/<失败run_id>
```

也可把旧结果根目录或其中的 `training_pin` 写入恢复脚本顶部的 `RESUME_FROM`。恢复沿用原电路、样本、划分、模型、seeds 和配置，在新的 `_resume_shared_decoder_..._pin` 目录中继续，原目录保持原状。原 Slurm 作业仍存活时，先停止原作业再提交恢复。

已完成阶段复用保存的结果；未完成阶段从最近保存的完整 epoch 恢复 optimizer、RNG 和早停状态，该 epoch 之后未保存的部分会重做。共享参数关系随结构重建一并恢复。改变 rank、模型范围、PE 数或 decoder 层数属于新结构实验，应重新训练，不能拿不匹配 checkpoint 接续。

源码、配置和说明纳入版本控制。结果、checkpoint、实验日志和 HPC 日志沿用现有 `.gitignore`，不会上传。实际全量数据的预测精度、GPU 加速和显存收益需由集群实验测量；现有权重谱诊断不等于这些新训练模型已经取得收益。
