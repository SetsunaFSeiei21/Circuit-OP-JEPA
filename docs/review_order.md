# 五模式代码说明与独立审查顺序

按数据如何流动审查，比先逐行看 Transformer 更容易发现泄漏和错误。

## 1. 原始数据是否被正确理解

读 `device_jepa/adapter.py`，对应 `tests/test_adapter.py`：

- `parse_spice_netlist` / `evaluate_spice_expression` 是否只接受明确支持的 DUT、器件、单位和有界表达式。
- 每条设计向量如何映射到 netlist 参数，W/L 的 um→m 是否只转换一次。
- OP 是否通过 sample_id + 完整器件名精确匹配；15 MOS + 电流源是否都保留。
- 原始 44 字段和 9 指标缺失值是否保留；是否没有 feasibility 筛选。

## 2. 工作检查类别是不是你真正想要的含义

读 `device_jepa/states.py` 与 `tests/test_states.py`：

- `:saturation` 是失败标签，不是饱和区正标签。
- 两个独立 one-hot 组可以同时失败；unchecked 与 checked-but-missing 必须不同。
- 未知 check scope/阈值不得推断 pass；明确的原始失败 reason 只证明对应组被检查。
- `all_mos` 仅是明确标记的派生模式，不覆盖原始 reasons。

## 3. 实验划分和统计量有无泄漏

读 `data.py` 的 `build_manifest`、`verify_manifest`、`fit_preprocessor`，对应 `test_data.py` / `test_normalization.py`：

- 电路间 OOD 先隔离，再在 ID 内按完整设计组做 80/10/10。
- `labeled_train` 只能来自 train；label fraction 不改变验证/测试集合。
- OOD、验证、测试均不能拟合 vocabulary、asinh scale 或 z-score。
- 电路等权不是样本数等权；true 0 与 missing 不混淆。
- 增删样本不自动重划分，旧版本 manifest/checkpoint 不偷偷兼容。

## 4. 张量化是否保持了语义

读 `data.py` 的 `device_category`、`parameter_applicability` 及 `tensors.py`、`baselines.py`：具体 NMOS/PMOS/R/C/源类型和实际命名参数保留，粗分类补充而不替代具体类型。参数 observation/applicability 区分不适用、适用但缺失、真实零，三者实际进入模型的 token/mask 应不同。pin 复制父器件参数、类别及 OP，不跨器件或污染 port。当前 collector 无 passive/source OP，不伪造值。设计参数、OP、状态、性能各自 mask；工作检查类别不做数值归一化，也不进入直接 OP 基线。

## 5. 共享到底共享了什么

读 `model.py` 的 `StructureEncoder`、`DeviceEncoder`、`SharedOPEncoder`、`JEPAModel`，对应 `test_shared_structure.py`：

- 默认前2层结构 prefix 与后4层参数融合边界。
- 两条路径引用同一个模块对象，registered parameters/state 不重复。
- 结构只 forward 一次，两个分支复用同一 dropout realization。
- 两条梯度均能到达共享结构；无 predictor、EMA、stop-gradient。
- 改 OP/状态不能改变设计分支；改设计参数不能改变纯结构/OP 分支。

## 6. 冻结和 checkpoint 是否真的可靠

读 `training.py` 的 `pretrain`、`train_decoder`、`FrozenPerformanceModel`：

- stage1 不用性能标签；stage2 仅 decoder 入优化器。
- encoder 参数、buffers、dropout 行为均固定。
- 后补性能标签只更新标签完整性指纹；设计/OP/身份和划分不变。
- PyTorch >=2.6 才允许反序列化；状态启用时必须验证编码和来源/配置一致。多来源检查记录按 source_id+circuit_id+run_id 隔离。
- v2 checkpoint 保存字段变换、状态配置、manifest、选择依据、时耗和参数量。

## 7. 五个模式的差别是否只有被声明的差别

读 `baselines.py`、`model.py`、`test_baselines.py` 与 `test_op_inputs.py`：pin baseline 有独立 pin/port vocabulary；device baseline 与 JEPA 使用器件节点。原两种监督 baseline 不读 OP/state；新 `zerosim_pin_op` 与 `zerosim_device_op` 训练和推理都读连续 OP/mask，不读 collector 状态类别，端到端更新 encoder+decoder，不做 JEPA 预训练。

独立反例应验证 OP direct 确实消费 OP、缺少输入通道明确失败、masked payload 不污染预测；原 no-OP 和冻结 JEPA 对任意 OP/state 改动不变。共享结构仍不读 OP。五模式共用 labeled_train 和目标变换，OP/no-OP 按相同图粒度配对；标签预算相同不等于同等参数量、总计算或推理信息。

新输入契约为 2；旧 v2 元数据无该字段时按 legacy 1 重建。检查 stage1→stage2、checkpoint load 与报告中的版本一致，不把旧/新输入空间混入公平汇总。

## 8. 结果能不能直接比较

先看 `test_evaluation_precision.py`：物理标签直接读取原始 float64，稳定 asinh 反变换和缩放 RMSE/R² 避免极端尺度丢失。

读 `comparison.py` / 对应测试：检查原数据、OP/性能指纹、划分、训练标签 IDs、目标 scaler 与逐样本目标一致。同时检查参数变换指纹及跨模式架构一致性（pin 的节点上限、端子 bias 是明确例外）。逐指标看原单位 MAE/RMSE/R²，再看归一化误差；不能跨不同物理单位直接平均 raw RMSE。缺失目标不参与分母。`paired_per_seed` 先算同 seed 的 OP-minus-noOP 差值，`paired_summary` 再聚合；MAE/RMSE 负差值、R² 正差值为改善，null 不补零、单 seed 无 SD。OP 采集成本必须保持未测，不得用模型耗时替代；costs 中同 mode/seed 多 split 成本不可重复求和。

## 9. 最后审查入口和运行脚本

读 `cli.py`、`configs/`、`scripts/README.md` 及 `scripts/run/`。先运行全量 unit tests 和小 smoke，再检查正式 run 的电路集合、OOD、label fraction、随机种子、epoch、输出目录与设备。

特别注意：收到的 Opamp2 数据只有一个电路，短流程通过只能证明代码贯通，不能支持任何跨电路 OOD 或方法优劣结论。

完整复审交接、历史与本轮证据边界、P0/P1/P2 验收标准和可直接交给更强模型的提示词见 `docs/REVIEW_HANDOFF_ZH.md`。不要把现有测试数量当作独立审查的结论。
