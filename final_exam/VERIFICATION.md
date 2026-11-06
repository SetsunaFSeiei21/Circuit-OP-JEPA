# 第一阶段交付验证记录

验证日期：2026-11-06。新增入口为 `final_exam/single_circuit/train.sh`，旧入口不变。

## 本次实现范围

完成第一阶段单电路的 13 个可选模型、器件级统一输入、各架构独立读出、原始数据导入、Slurm 提交脚本、分阶段日志、独立教师与 OP predictor 保存、模型重载、验证选模、真实训练成本及 XLSX 汇总。多电路、消融和迁移学习目录仅建立任务位置与范围说明。

来源固定版本及适配差异见 `models/SOURCES.json`。这组模型是受控器件级适配，不以名字相同冒称逐算子复现。真实 OP oracle 与部署时不使用真实 OP 的模型在结果中区分。

## 已运行的检查

本地环境 Python 3.12、PyTorch 2.15.0+cpu、XGBoost 3.4.1。全部模型训练机制测试使用 CPU；没有可用 CUDA 设备。

| 命令 | 结果 |
|---|---|
| `python -m unittest discover -s final_exam/tests -v` | 30 项通过 |
| `python -m unittest final_exam.single_circuit.test_launch -v` | 10 项通过 |
| `python -m unittest discover -s tests -v` | 468 项运行，通过，含 6 项跳过 |
| `python -m unittest discover -s experiments/op_transfer -p 'test_*.py' -v` | 79 项通过 |
| 三个新 bash 的 `bash -n` | 通过 |
| `train.sh --dry-run --smoke` | 正确打印 sbatch 账户、分区、8 CPU/1 GPU、独立 HPC 日志、13 模型及 3 seeds；无提交、无结果目录写入 |
| `git diff --check`、产物 `git check-ignore` | 通过 |

端到端测试用单拓扑合成电路，完整运行 13 个模型 × 2 个 seed，导出 XLSX、独立恢复全部最终 checkpoint 和共享依赖。另有真实子进程测试：构造有效 SQLite 小样本源，经实际 importer 和 runner CLI，产出完整模型结果和 XLSX。这是导入/训练链路测试，不是实际模拟电路精度评估。

关键断言包括：

- 只选择 MLP 时不训练 OP 依赖；只选择顺序学生时仅训练必需 predictor、教师和两个学生阶段，不额外评测其他模型。
- 设计输入模型与级联模型的预测不受真实 OP、OP mask、性能标签及标签 mask 的污染改变。
- 共享依赖每个 seed 只训练一次；完整教师即使不在评测模型名单中仍独立保存。
- 顺序学生第一阶段 decoder 参数不变，第二阶段 encoder 参数不变；教师和 OP predictor 始终冻结。
- checkpoint 可独立重建，恢复后预测一致；教师冻结标志一并恢复。
- GTN 环来自器件/网络二部图，不把同网 clique 当物理环；训练前缓存环结构，训练 batch 不进行 CPU 环发现；测试集在评测时单独准备。
- 各 baseline 前向、反向、设计参数梯度、padding、BatchNorm 单样本行为正常。
- XGBoost 仅训练/验证拟合、缺失标签处理、native booster 重载、关闭早停仍选验证最佳轮次。
- XLSX 一 seed 的 std 留空、跨 seed 样本 std、不可比结果不高亮、完整结果矩阵校验、神经 epoch 与树 rounds 分开。
- 无写入前检查可拦截非空输出、manifest/配置冲突、非法选择与训练预算；日志打开失败可记录 failed 状态。

## 审查中发现并修复的问题

1. **XGBoost 恢复接口多包了一层 tuple。** 统一直接返回 `(model, metadata)`，最终 13 模型重载测试覆盖。
2. **关闭早停时树模型曾选择最后一轮。** 修改为只要存在该指标验证标签，就按验证 RMSE 最低轮次截取原生 booster；提前停止与最佳选择相互独立。无验证标签的指标按固定预算，并明确记录。
3. **XGBoost 成本没有 epoch，导致 Excel 拒绝导出。** 保留 epoch 为 null，增加 boosting budget 和逐指标实际 rounds，禁止伪装成神经网络 epoch。
4. **GTN 虽有拓扑预计算接口，但 runner 最初没有调用。** 接入训练前 train/val 缓存；测试缓存挪到测试阶段，避免训练前物化测试标签。
5. **独立教师的冻结状态不会随 state_dict 保存。** checkpoint 增加可训练参数名单并在重载时恢复；级联内部仍按模型配方重建冻结关系。
6. **外部 manifest 可能与当前配置声明不同。** 在任何结果写入前同时检查数据身份、划分成员及配置的 split_seed、比例、标签预算，不能用一份配置描述另一份划分。
7. **日志文件打开失败可能留下 running 状态。** 增加覆盖日志获取步骤的异常状态记录；旧日志文件仍保持独占保护。
8. **模型来源与完整训练成本易混淆。** 显式记录 upstream commit、独立实现与适配差异；依赖成本每 seed 去重，阶段总参数含教师/冻结子模块，与实际可训练参数分别呈现。

## 未验证内容

- 没有实际提交用户集群作业或跑 CUDA；Slurm 通过 dry-run 与模拟 sbatch 合约检查，不能据此声称集群资源配置实测成功。
- 没有运行用户完整 60,000 点数据，也没有验证完整数据导入/训练的主机或显存峰值、吞吐与最终预测精度。
- 没有新增多电路/OOD、消融或迁移学习科学实验结论。
- CPU 运行检查不代表不同 CUDA/XGBoost/PyTorch 版本的性能和数值完全一致。
