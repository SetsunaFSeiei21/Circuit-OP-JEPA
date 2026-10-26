# 0.4.1 边界修复记录

日期：2026-10-26。复核基于 `main` 提交 `4396bf48fa948b31de6afbd4a97ff3d6a36c472c`；用户指出的两项问题均在该版本实际复现。

## Smoke 入口在拒绝前修改了已有文件

复现条件：输出目录已有 `data.json` 和旧 `test_report.json`，但没有 `manifest.json`。`smoke` 与 `smoke_real` 都先创建 manifest、改写 data 内容及修改时间，再由训练阶段的目录保护报错。

修复：两个入口在创建目录、读取配置、导入数据及任何输出写入之前检查路径。只接受新路径或已有空目录；已有文件、非空目录和部分实验产物都明确拒绝。直接训练入口原有的阶段衔接规则保持不变。

回归检查同时比较拒绝前后的路径集合、目录/文件修改时间和文件字节，覆盖有无 manifest、已完成实验、子目录及普通文件。新路径和空目录均实际执行两阶段 smoke，确认正常运行；对已完成结果再次运行也必须保持原样。测试见 [test_smoke_output_guard.py](../tests/test_smoke_output_guard.py)。

## 未训练指标在进入选模 mask 前发生转换溢出

复现条件：`gain` 在 labeled_train 中全部缺失，在验证集中取有限值 `1e100`。五模式在该位置为 `None` 时都能训练，改成 `1e100` 后均报 `Non-finite standardized feature gain`。原因是标签先归一化并转 float32，随后才应用选模 mask。

修复：device/pin tensorization 增加可选的 `performance_metric_mask`。训练和验证循环按训练统计 `count>0` 显式传入 mask；不参与目标的字段在读取值、归一化和浮点转换之前跳过，保留零占位和 false 观测 mask。两个无 OP 基线、两个直接 OP 基线及 JEPA decoder 共用此规则。

默认 tensorization、正式 evaluation 和 float64 `evaluation_targets` 保留全部指标的既有语义。参与目标的指标仍执行正常数值检查；验证集中没有任何可支持的观测标签仍会报错。这次没有将异常数值静默裁剪或把所有未训练指标从报告中删掉。

回归对五模式分别比较 `None`、`1e100`、`-1e100`：完整 history、选中 epoch、encoder 与 decoder 摘要必须相同；另外覆盖受支持指标溢出、无受支持验证标签、mask 类型/形状、归一化前跳过以及正式评估保留未训练指标。测试见 [test_validation_label_boundary.py](../tests/test_validation_label_boundary.py)。

## 兼容性和验证

本次为 0.4.1 补丁版本，不改变输入 contract 3、checkpoint/preprocessing 容器版本或 manifest 指纹版本，也不改变 JEPA 共享结构、梯度和冻结协议。0.4.0 的有效新契约实验不需要因本次补丁重新划分；早期不兼容 manifest 的迁移要求仍见 [0.4.0 修复说明](INDEPENDENT_AUDIT_FIXES_ZH.md)。任何新 smoke 仍需新输出目录或空目录。

新增 13 项回归，完整 326 项测试通过（显式启用百万行 reader 测试，零跳过）；7 项独立探针通过。两个实际 CLI smoke 均成功：合成 120 样本、预训练 2 epoch / decoder 3 epoch；原始 Opamp2 32 样本、每阶段 1 epoch。0.4.1 wheel 构建、源码字节比对及隔离安装/CLI 检查通过。

验证命令及最终结果见 [verification/boundary_fixes/results.json](../verification/boundary_fixes/results.json)。短流程验证实现和回归行为，不证明预测效果、多电路泛化或完整 60,000 点导入/训练的内存容量。原始 `tmp_data` 未改写，ZIP 和 SQLite SHA-256 与上一轮一致。
