# 端口级第一阶段验证记录

日期：2026-11-07。交付范围为 `final_exam` 的端口级单电路实验，不包含后续多电路/OOD、消融或迁移学习实验。

## 用户确认的规则

- 图节点采用器件物理引脚＋命名外部端口；MOS D/G/S/B 分开。
- OP predictor 先聚合同器件引脚 latent，预测一组器件 OP，再广播到所属引脚。
- OP 物理监督按器件计数一次；外部端口没有器件 OP。
- 13 个模型、模型源文件、脚本、日志及结果使用 `_pin` 区分，保留器件级版本。
- 其余沿用器件级的数据划分、训练集预处理、验证选模、预算、早停、配置/教师保存、训练成本和 XLSX 汇总标准。

## 实现与检查

本地验证环境为 Python 3.12.14、PyTorch 2.15.0+cpu、XGBoost 3.4.1、NetworkX 3.7，无 CUDA 设备。

| 检查 | 实际结果 |
|---|---|
| `python -m unittest discover -s final_exam/tests -v` | 79 项全部通过，含原器件级 30 项与新增 pin 49 项 |
| `python -m unittest final_exam.single_circuit.test_launch final_exam.single_circuit_pin.test_launch_pin -v` | 21 项全部通过，含原 10 项与新增 pin 11 项 |
| `python -m unittest discover -s tests -v` | 468 项运行成功，其中 6 项跳过 |
| `python -m unittest discover -s experiments/op_transfer -p 'test_*.py' -v` | 79 项全部通过 |
| 新脚本 Bash 语法检查 | 通过 |
| 提交入口 dry-run、Git 产物忽略和差异检查 | 通过 |

端口级新增测试共 **60 项**。其中端到端测试实际训练全部 13 个 pin 模型 × 2 个 seed，导出 `summary_pin.xlsx`，重建全部最终模型与共享依赖。另有小型有效 SQLite 源经真实 importer 子进程、真实 runner 子进程到 XLSX 的完整测试。

这些是机制与链路检查，不是用户真实模拟电路的预测性能结果。未提交 Slurm、未实测 CUDA、未执行完整 60,000 点数据容量或速度验证。

## 关键断言

### 输入与统计

- 标准输入张量的节点维度实际是 pin 数量，encoder memory 长度为 pin 数＋graph token。
- pin 工厂拒绝器件级预处理与无端口映射的 batch，OP oracle 也执行相同粒度检查。
- 展开后的 pin 数值与仓库已有 pin collator 一致；owner 映射保留原始器件身份。
- 参数、OP、标签统计量仅训练集拟合。OP/参数不会因一个器件有更多引脚而在归一化中重复计权。
- 未训练字段在数值变换前屏蔽；测试包含仅验证/测试出现的有限极大值 `1e100`。
- 外部端口、padding、未见端口名字、任意带冒号的器件身份、token 超限均有检查。

### OP 与蒸馏

- 同器件引脚取均值后只产生一组 OP，广播后同器件各引脚数值一致。
- 外部端口无 owner，不参与聚合、OP 输出或 OP 监督。
- 物理 OP loss 的有效计数来自 owner-device 轴，不来自重复广播后的 pin 轴。
- OP support 根据训练集器件 type/model 拟合，不读取测试 OP mask。
- 污染真实 OP、owner OP、OP mask、性能标签及其 mask，不改变部署模型预测。
- 联合学生验证选模只用性能；顺序学生先对齐 encoder，再冻结 encoder 训练 decoder。
- 教师、predictor 始终冻结；顺序学生 encoder 阶段 decoder 不变，decoder 阶段 encoder 不变。
- 只选学生时独立保存必需教师和 predictor，不自动评测无关模型；共享阶段每 seed 去重。
- 独立 checkpoint 重载保持预测一致及冻结状态；pin/device checkpoint 格式不能互换。

### Baseline、结果和目录

- baseline 前向、反向、设计参数梯度、padding、BatchNorm、单样本行为通过。
- GTN 环来自真实 owner/net 拓扑并投影到环端点；不把同网/同器件 clique 或旁支引脚当环。
- GTN 训练/验证和测试按阶段准备 resident 拓扑；训练 batch 不重复 CPU 环发现。
- XGBoost 读取固定顺序的真实 pin/外部端口特征，保存列语义与拓扑，拒绝身份或顺序不匹配；关闭早停仍选验证最佳轮次。
- 所有输出文件均测试 `_pin` 后缀；既有结果不会覆盖，错误输入与 manifest 冲突在写入前拒绝。
- 配置给出的 pin 容量在训练前校验；模型覆盖配置的容量同样检查，并保证测试缓存使用足够容量。
- XLSX 保留 `_pin` 名称，不混合两种粒度；样本 std、单 seed 空 std、真实 OP 分组、完整矩阵、成本去重和树 rounds 均有测试。
- Slurm 与实验日志独立并共享 run ID，提交时配置快照不受后续模板修改影响；失败状态和真实退出码保留。

## 审查后补充的修复

1. **严格 OP pin 输入边界**：最初 OP encoder 底层与器件模型共享算子，形状兼容的器件 batch 可以直接通过。实际 runner 始终使用 pin 数据，但公共接口的检查不足。现加入 pin 映射检查；同时为蒸馏和 latent 尺度拟合加入可覆盖输入过滤接口，避免公共 device 过滤器丢弃 pin 元数据。
2. **OP pooling 的确定性路径**：使用引脚/器件 incidence 与批矩阵乘法完成聚合，避免依赖 GPU 原子 scatter 聚合的执行顺序。CPU 梯度与数值检查通过；没有据此宣称 GPU 已实测。
3. **pin token 容量**：旧小样本测试按器件数量配置的容量不足以容纳全部 pin 和外部端口。保留严格报错，修正测试容量，并将所选模型容量检查前移到结果写入之前。
4. **命名和统计隔离**：树模型原生权重、阶段 history/cost、共享依赖、导出文件和 HPC `%j` 日志均独立加后缀；汇总入口严格拒绝器件级 ID/文件与无 pin 后缀的输出。

公共文件只增加向后兼容的后缀和输入过滤钩子，默认仍使用原有器件级行为。未改写原始采样数据、器件级模型文件或既有实验结果。固定源码来源与 pin 适配细节见 `models/SOURCES_pin.json`。
