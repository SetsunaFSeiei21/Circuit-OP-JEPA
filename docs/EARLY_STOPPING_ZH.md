# 验证集选模与早停

性能报告同时给出原始单位下的 MAE、RMSE、决定系数 R²，以及训练标签变换下的归一化误差。R² 越大越好，MAE/RMSE 越小越好。少于两个有效标签或标签方差为零时，R² 为 null。

选模和早停监控同一验证目标：所有选中的监督基线以及 JEPA decoder 使用 val 的归一化性能 MSE，仅包含训练中有观测标签的指标；JEPA 预训练使用 val 的 OP 对齐与正则化总损失，不读取性能标签。test 和 OOD 不参与选模或早停。每个模型、每个 seed 独立维护计数。

## 配置

在训练配置的 `training` 对象中添加：

```json
"early_stopping": {
  "enabled": true,
  "patience": 10,
  "min_delta": 0.0,
  "min_epochs": 1
}
```

- `patience`：连续多少轮没有超过 `min_delta` 的验证损失下降时停止。
- `min_delta`：重置耐心计数所需的绝对下降幅度；默认 0。所有新低值仍保存最佳 checkpoint，即使下降幅度小于这个阈值。
- `min_epochs`：最少执行轮数；此前已累计耐心计数。若阶段最大轮数更少，则执行到阶段上限。
- `enabled: false`：执行完整轮数，仍按验证损失最低值保存 checkpoint。

`configs/default.json` 及安装包默认配置启用早停；两个 smoke 配置关闭。旧自定义配置没有这个对象时，保持原来的固定轮数，需显式添加才能启用。选中的基线、JEPA 预训练和 decoder 均使用这组规则，各阶段的计数互相独立。达到阶段上限时，停止原因为 `max_epochs`。

停止后，test/OOD 评估加载历史验证损失最低的完整 checkpoint，所有性能指标共享它。JEPA 先选预训练 encoder 并冻结，再独立选择 decoder；预训练与 decoder 的最佳轮数和实际执行轮数分别保留。

## 最大预算与实际成本

比较入口仍要求 `pretrain_epochs + decoder_epochs == baseline_epochs`，默认最大预算为 JEPA 30＋50、其余各 80。启用独立早停后，实际总轮数可能不同，这时只能称为最大预算相同。若实验要求实际轮数完全一致，应关闭早停。

checkpoint 和报告 provenance 的 `early_stopping` 包含有效配置、监控目标、最佳轮数、执行轮数和停止原因。JEPA 的 `pretrain_provenance.execution` 从实际预训练 checkpoint 保留；执行结果不进入跨种子固定的协议摘要，早停规则则进入协议摘要，避免把合理的停止轮数差异误判为协议变化。

比较输出 schema 5 中，`pretrain_epochs`、`supervised_epochs`、`total_epochs` 是实际执行轮数；`planned_pretrain_epochs`、`planned_supervised_epochs`、`planned_total_epochs` 是配置上限。`max_total_epochs` 和兼容字段 `total_epoch_budget` 均指相等的总上限。早停比较的 `epoch_budget_policy` 为 `equal_maximum_with_early_stopping`；固定轮数为 `equal_total`。`costs.csv` 继续按真实 run identity 去重，另行记录优化器步数与耗时。

比较器要求各模型的监督早停规则相同，验证执行记录与上限、耐心计数和实际成本一致；缺少执行记录的启用早停报告不能通过。预训练规则与执行经历来自实际阶段一，不能用阶段二配置替换。旧固定轮数报告仍可读取，既有来源完整性规则仍生效。

## 验证

`tests/test_early_stopping.py` 包括真实五模式短训练、独立阶段停止、最佳 checkpoint、冻结 encoder、实际轮数及优化器步数、test/OOD 成本去重、不同种子不同停止轮数、最小轮数、微小改进和错误来源记录检查。有 CUDA 时执行同样的五模式 GPU 测试；无 CUDA 时明确跳过，CPU 通过不代表新增路径已在 GPU 上实测。
