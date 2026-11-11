# 分阶段 OP 门控验证记录

日期：2026-11-10。环境：Python 3.12、PyTorch 2.6.0+cpu、NumPy 2.3.5、XGBoost 2.1.4（CPU 包）。本记录是代码正确性与小数据流程验证，不是电路性能实验结果。

## 已执行检查

| 检查 | 结果 |
|---|---|
| `python -m unittest discover -s final_exam/tests -p 'test_*.py' -q` | 127 项通过；覆盖原器件级、引脚级流程及新增门控方案 |
| `python -m unittest final_exam.single_circuit_pin.test_staged_op_gate_launch_pin final_exam.single_circuit_pin.test_slurm_entrypoints_pin final_exam.single_circuit_pin.test_op_gate_launch_pin final_exam.single_circuit_pin.test_launch_pin -q` | 24 项通过；使用模拟 sbatch 和实际 Bash 脚本 |
| `python -S final_exam/run_pin.py --list-models` | 无第三方依赖也能列出 24 个注册模型，含三个新增选项 |
| `bash -n final_exam/single_circuit_pin/train_staged_op_gate_pin.sh` | 通过 |
| 新入口 `--dry-run` | 正确打印三个模型、三种子、8 CPU/1 GPU、独立配置和原项目内的日志/结果路径 |
| `git diff --check` | 通过 |

## 新方案的关键断言

- 三个模型在小型合成电路上实际完成训练、验证选模、测试、checkpoint 保存/重载、JSON/CSV/XLSX 导出。
- 蒸馏的两个模型共享同一基础模型与初始 gate；改变请求模型顺序不改变权重结果。
- 前置阶段完全不调用门控，门控输出被绕过；测试将未使用的门控参数置为 NaN，并让其 forward 抛错，以检查真实绕过。
- 最后阶段门控得到非零梯度并实际更新；被冻结的 predictor、encoder、decoder 参数摘要保持不变。门控＋decoder 版本中 decoder 也实际更新。
- 冻结模块保持 eval 模式，但门控梯度仍能通过编码器和 decoder 回传。
- 编码器蒸馏不依赖性能标签；推理时污染真实 OP、其掩码及性能标签，不改变三个最终模型的输出。
- 最终 checkpoint 独立包含 predictor 和学生，无需读取教师或基础 checkpoint；阶段、冻结状态和输入契约能够恢复。
- 第三种模型单独运行时不会训练教师。共享阶段成本按 stage ID 去重；三模型的小数据组合包含九个唯一阶段（含一次前向 OP 缓存）。
- 每个模型的 `gate_comparison_pin.json` 与 Excel `GateBeforeAfter` 展示同一测试集的无门控基线和最终模型；这些测试报告不参与训练或选模。
- 非法门控轮数在写实验目录之前报错。提交时保存配置快照，保留 Slurm 失败退出码。

## 数值与验证边界

严格对照门控关闭前后的前向结果时，测试也匹配参考模型的 `requires_grad` 状态。PyTorch 可能依据该状态选择不同 attention 计算路径；冻结状态不一致时观察到约 `7.45e-09` 的输出差异。匹配冻结状态后测试继续要求 `atol=0, rtol=0`，未通过放宽容差掩盖门控行为。

未在此环境执行真实 Slurm 提交、CUDA 运算、60,000 点完整电路训练或多随机种子精度实验。以上通过不证明门控提高精度、改善验证振荡，也不证明完整数据的显存容量。门控＋decoder 版本同时增加 decoder 训练；若要单独归因于门控，需要额外的等预算 decoder-only 对照。
# 2026-11-11：冻结主干、仅门控求梯度的 CUDA SDPA 兼容性

针对 `loss.backward()` 的 `LSE is not correctly aligned (strideH)`，在
`EncoderBlock.forward` 中仅为 Q/K/V 无梯度、浮点 mask 有梯度的调用选择
math SDPA。这个后端使用输入所在设备，未引入 CPU 回退，也没有改变冻结策略。
依据为 PyTorch 上游 issue #148476 和修复 PR #188302。

本次验证环境为 PyTorch **2.6.0+cpu**，没有 CUDA 硬件。

| 检查 | 结果 |
|---|---|
| `tests/test_attention_mask_grad.py` | 4 项 CPU 通过；1 项 CUDA 用例跳过 |
| `final_exam/tests/test_staged_op_gate_models_pin.py` | 7 项 CPU 通过；1 项 CUDA 用例跳过 |
| 门控数据、模型、运行和汇总回归（新增 CUDA 用例前运行） | 39 项通过 |
| 原有 model/shared_structure/baselines/op_only/resident 回归 | 63 项通过；4 项 CUDA 用例跳过 |
| 独立代码审查、`git diff --check` | 通过 |

专项测试将输出和 mask 梯度与显式 bmm/softmax 参考计算比较，检查冻结权重没有
梯度、被排除的边没有梯度，以及后端设置在成功/异常退出后恢复。正常训练的
Q/K/V 梯度路径和 `no_grad` 路径不改变后端选择。CUDA 用例另覆盖 97 token / 8 head
和三个 staged 模型各两步优化，但本环境未执行，不能表述为 A800 验证通过。

此次修复没有增加断点续训。失败目录中的 checkpoint 保留；使用普通启动脚本
重新提交会创建新实验目录并重训前置阶段。
