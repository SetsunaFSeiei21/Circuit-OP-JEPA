# v1 / v2 / v3：蒸馏后联合调整门控与 decoder

三个版本各增加两个模型。它们复用相应版本的门控网络，只改变最后阶段的可训练模块，不改变门控输入、归一化、初始化或作用边界。

| 版本 | 结构层＋融合层 Self-Attention | 仅融合层 Self-Attention |
|---|---|---|
| v1 | `device_op_distill_gate_decoder_pin_v1` | `device_op_distill_gate_decoder_fusion_pin_v1` |
| v2 | `device_op_distill_gate_decoder_pin_v2` | `device_op_distill_gate_decoder_fusion_pin_v2` |
| v3 | `device_op_distill_gate_decoder_pin_v3` | `device_op_distill_gate_decoder_fusion_pin_v3` |

v1 使用 OP＋有序引脚独热编码 MLP；v2 增加样本内相对 OP；v3 使用相同双 OP 输入和 FlowNet 式三层 M-MLP。每个模型仍只有一个跨选定层共享的门控。新模型不把门控插入 Cross-Attention 或 decoder 内部。

## 训练顺序和轮数

| 阶段 | 操作 | 轮数配置 | 模板上限 |
|---|---|---|---|
| 器件 OP 预测器 | 拓扑＋设计参数预测器件 OP；完成后冻结 | `op_transfer.epochs.op_predictor` | 30 |
| 教师 | 拓扑＋设计参数＋真实 OP 监督训练性能预测模型；完成后冻结 | `op_transfer.epochs.teacher` | 60 |
| 学生 encoder 蒸馏 | 用预测 OP 输入，将学生表示对齐教师表示；门控关闭 | `op_transfer.epochs.predicted_op_distill_encoder` | 30 |
| decoder 预训练 | encoder 冻结，只训练 decoder；门控关闭 | `op_transfer.epochs.predicted_op_distill_decoder` | 30 |
| 门控＋decoder 微调 | encoder 与 OP predictor 冻结，同时训练门控和已有 decoder | `op_transfer.epochs.gate` | 30 |

最后阶段不是重置 decoder，而是从蒸馏无门控基座的最优权重继续训练。冻结 encoder 参数不等于断开梯度图：门控仍通过冻结的 attention / encoder 获得梯度。各阶段保留验证选模、早停和实际成本记录，表中的数值是上限。

默认每条路线若完全独立运行，各阶段上限合计 180 轮；其中 predictor 和教师合计 90 轮，学生蒸馏、decoder 预训练、最终微调合计 90 轮。同一实验同一 seed 的 predictor、教师和蒸馏无门控基座在适用模型之间共享，实际总作业成本不能把每条路线的 180 简单相加。每个新增变体需要自己的最后阶段。它与 `distill_gate` 的区别仅为最后阶段额外更新 decoder；原来的 `distill_gate` 仍只更新门控。

## 使用

在对应版本的原脚本中选模型，无需新的提交脚本：

| 版本 | 脚本 | 配置 |
|---|---|---|
| v1 | `final_exam/single_circuit_pin/train_op_gate_pin_v1.sh` | `final_exam/configs/op_gate_pin_v1.json` |
| v2 | `final_exam/single_circuit_pin/train_op_gate_pin_v2.sh` | `final_exam/configs/op_gate_pin_v2.json` |
| v3 | `final_exam/single_circuit_pin/train_op_gate_pin_v3.sh` | `final_exam/configs/op_gate_pin_v3.json` |

三个模板默认均为八个门控模型＋`device_op_cascade_pin`，共九个评测入口。若只跑新增 v3 变体，将 v3 脚本顶部改为：

```bash
MODELS=(
    device_op_distill_gate_decoder_pin_v3
    device_op_distill_gate_decoder_fusion_pin_v3
)
```

按需设置 `CIRCUITS`、`SEEDS`，然后：

```bash
bash final_exam/single_circuit_pin/train_op_gate_pin_v3.sh --dry-run
bash final_exam/single_circuit_pin/train_op_gate_pin_v3.sh
```

最后阶段若要最多 60 轮，在 JSON 的 `op_transfer.epochs` 内将 `"gate": 30` 改为 `"gate": 60`。无需新增 epoch 字段。该预算也控制同一配置下所选的只训练门控变体。

结果仍位于 `final_exam/single_circuit_pin/results_pin/<run_id>/training_pin/`，实验日志和 HPC 日志分别在同级任务目录的 `logs_pin/`、`hpc_logs_pin/`。模型目录使用完整名称；配置、checkpoint、门控审计与前后对比记录最后阶段为 `gate_decoder` / `gate_and_decoder`。Excel 的 `ModelInputs`、门控前后比较和成本表区分该阶段与 `gate_only`。蒸馏教师保留在 `shared_pin/design_op_oracle_pin/`。

## 恢复与兼容

仍使用 `resume_staged_op_gate_pin.sh <失败实验目录>` 恢复同一个实验。模型身份和阶段策略随 checkpoint 保存；只训练门控的旧实验不能仅修改模型名后续训成门控＋decoder。增加新模型应提交新实验，或在一开始就将两者加入同一 `MODELS` 列表以共享依赖。

已有 v0/v1/v2/v3 模型名称及训练策略保持不变。各版本新增模型的门控机制沿用原版本，没有新的门控版本号。

## 验证记录（2026-11-12）

组合回归运行 158 项：155 项通过，3 项 CUDA 专用测试因当前环境没有 GPU 跳过。三个提交脚本的 `bash -n`、模型名称 CLI 和 `git diff --check` 通过。

测试覆盖实际优化后的模块权重摘要：OP predictor/encoder 不变，gate/decoder 更新；三个版本分别验证中断后的优化器、随机数恢复与连续训练权重一致；完成阶段的恢复不重复优化。跨版本混合运行共享同一个蒸馏基座，门控分数检查、Excel 中的阶段标注及成本去重均有验证。测试使用小型合成电路，未进行 A800 实测或新增真实电路精度实验。
