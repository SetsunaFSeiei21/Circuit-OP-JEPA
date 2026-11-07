# OP 门控 pin 单电路实现：验证记录

日期：2026-11-07。实现以原 pin 第一阶段提交 `a785b4e71ce5ed2d2cc85bba62eb89469640108a` 为基础。

## 执行环境与范围

本地 Python 3.12.14、PyTorch 2.15.0+cpu，CUDA 不可用。以下是 CPU 合成数据机制验证、脚本验证及既有代码回归；没有提交用户集群作业，没有完成实际 60,000 点电路的 CUDA 训练，也没有得到精度提升的实验结论。

## 新增验证：27 项通过

```bash
python -m unittest \
  final_exam.tests.test_op_gate_data_pin \
  final_exam.tests.test_op_gate_models_pin \
  final_exam.tests.test_op_gate_run_pin \
  final_exam.tests.test_op_gate_summary_pin \
  final_exam.single_circuit_pin.test_op_gate_launch_pin -q
```

| 范围 | 项数 | 检查内容 |
|---|---:|---|
| 数据与 shuffle | 7 | 原始器件图/参数一致；严格 D 轴输入；缓存 sample 对齐与权重变化失效；只缓存冻结 eval 模型；训练来源限定；排除自身；batch 顺序、大小和重载不影响替换来源 |
| 模型 | 12 | 真正 D-token OP 编码；按物理器件计 OP 损失；OP embedding 与 encoder dim 一致；逐 head 有向门控；同 net、自环、外部端口、非 MOS 保护；全部结构/融合 self-attention 生效；冻结与梯度；A/B 特征区别；oracle 回退；组内初始化一致；零强度等价；8 模型独立重载 |
| 训练与产物 | 3 | 8 对照端到端训练、共用 OP 预测器及缓存成本、有效配置、独立重载、oracle 覆盖率、XLSX；B 无门控基线无 OP 预训练依赖；无效门控配置在写入前拒绝 |
| 结果汇总 | 3 | 8 个名称和输入契约保留；oracle 标记错误拒绝；XLSX 展示 A/B、特征/门控来源和 shuffle 诊断身份 |
| 启动脚本 | 2 | dry-run 不建输出目录；GPU/CPU、模型、seed 和 smoke 参数正确传递；模拟 sbatch 记录作业和统一运行标识 |

## 既有回归

- `python -m unittest discover -s final_exam/tests -v`：该次发现并执行 92 项，通过；当时新的 12 项核心模型测试文件尚未纳入发现，之后已包含在上述 27 项中单独执行。
- `python -m unittest discover -s final_exam/single_circuit_pin -p 'test*.py' -v`：13 项通过，包含真实 SQLite 小夹具经导入和训练子进程的原有流程测试。
- `python -m unittest discover -s tests -q`：468 项执行成功，其中 6 项跳过；不能把跳过项称为已验证。
- `python -m unittest discover -s experiments/op_transfer -p 'test*.py' -q`：79 项通过。
- `bash -n final_exam/single_circuit_pin/train_op_gate_pin.sh` 及 `git diff --check` 通过。

以上命令存在重复覆盖，项数不应直接相加当作独立测试总数。既有全模型回归现已覆盖 21 个注册 pin 模型、两个 seed 的训练、保存、独立恢复和真实 OP/标签输入边界检查。

## 审查落实的细节

1. 新增真正器件级 OP 编码器，未将旧 `predicted_op_pin` 的 pin 编码＋池化结构重新命名成器件级编码。
2. 第一阶段和下游均保留仓库 ZeroSim-family 框架；共享 OP MLP 输出 `model.dim`，有序端口角色也是该维度，最后映射成各 head 的标量分数。
3. 可选 attention bias 接口不改变旧调用行为；局部和全局 mask、结构层和融合层都接收新偏置，cross-attention 和指标读出保持原接口。
4. A 组节点 OP 一直使用当前样本预测值；oracle/shuffle 仅替换门控来源。B 组不创建 OP 特征 token；B 无门控基线没有 OP 预测器。
5. OP predictor 参数和 eval 状态冻结，保存前后权重摘要核对；门控网络仍有梯度。缓存只存冻结的数值预测，不缓存可训练的偏置或隐表示。
6. Shuffle 来源限定为训练预测库，完整保留器件/字段排列；样本对应关系由 seed 与 sample ID 固定。来源库与模型一并保存，不需要其他测试样本才能恢复输出。
7. 门控不能削弱 G=D、B=S 等真实 net 短接；也不将缺失 OP 的观测模式作为 oracle 独有的额外特征。
8. 依赖 OP 的模型共享一个预训练阶段；前向缓存准备单独记录为零 epoch 阶段，按唯一 stage ID 去重。B 无门控基线只有自身训练成本。
9. 修正审查发现的共享预测器审计相对路径：从模型目录应使用 `../shared_pin/...`；测试检查路径真实存在。
10. 两个无门控基线的来源记录单独描述其结构，避免误写成带门控的模型。

## 实际运行仍需检查

在用户集群上核对原始几何单位、真实 pin 数量及 `model.max_devices`，先用小样本来源运行烟测。`--smoke` 仅减少训练 epoch，不限制原始导入样本数。

完整数据的显存、主机内存、耗时和预测精度尚未在本环境验证。Shuffle 的训练 OP 来源库随 checkpoint 保存，会增加内存与磁盘 I/O；soft attention 偏置本身不提供稀疏算子加速。真实实验完成后应比较相同划分/seed/预算的 MAE、RMSE、R² 与实际阶段成本。
