# 按电路随机抽样后训练

单电路与联合训练/OOD 使用相同的数据量设置：**先按每个电路抽取指定总数，再划分训练、验证和测试集**。设置位于启动脚本 `CONFIG` 指向的 JSON 的 `dataset` 中，不改变 `MODELS` 或 `SEEDS` 的写法。

## 设置位置

| 实验入口 | 默认配置文件 |
|---|---|
| 门控 v1：`final_exam/single_circuit_pin/train_op_gate_pin_v1.sh` | `final_exam/configs/op_gate_pin_v1.json` |
| 端口级普通模型：`final_exam/single_circuit_pin/train_pin.sh` | `final_exam/configs/default_pin.json` |
| 端口级门控 v0：`final_exam/single_circuit_pin/train_op_gate_pin.sh` | `final_exam/configs/op_gate_pin.json` |
| 端口级分阶段门控：`final_exam/single_circuit_pin/train_staged_op_gate_pin.sh` | `final_exam/configs/staged_op_gate_pin.json` |
| 端口级仅融合层分阶段门控：`final_exam/single_circuit_pin/train_staged_op_gate_fusion_pin.sh` | `final_exam/configs/staged_op_gate_fusion_pin.json` |
| 器件级：`final_exam/single_circuit/train.sh` | `final_exam/configs/default.json` |
| OP transfer 单电路/OOD：`experiments/op_transfer/train_single_circuit.sh`、`train_ood.sh` | `experiments/op_transfer/example_config.json` |
| 原有单电路/OOD：`scripts/run/train_single_circuit.sh`、`train_ood.sh` | `configs/default.json` |

例如每个电路最多使用 20,000 条：

```json
"dataset": {
  "train_circuits": ["two_stage_folded_opamp"],
  "ood_circuits": [],
  "split_ratios": [0.8, 0.1, 0.1],
  "split_seed": 17,
  "max_samples_per_circuit": 20000,
  "sampling_seed": 17,
  "finetune_label_fraction": 1.0
}
```

这是现有 JSON 中 `dataset` 对象的示例；保留同一文件里的模型、训练和归一化配置。使用 Slurm 原始数据入口时，电路角色仍由 bash 中的电路名单填入。

修改配置后，照常执行原来的提交命令，例如：

```bash
bash final_exam/single_circuit_pin/train_op_gate_pin_v1.sh
# 或运行原有联合训练/OOD：
bash scripts/run/train_ood.sh
```

两个入口读取各自的配置文件；修改某一份配置不会自动修改其他实验配置。`final_exam` 的多电路任务目录仍是预留位置，此功能没有新增该目录下的 OOD runner。

## 数量与随机种子

- `max_samples_per_circuit: 20000`：在每个电路内无放回随机抽样，最多保留 20,000 条。源数据只有 15,000 条时保留全部 15,000 条；设置大于或等于源样本数时也保留全部。
- `max_samples_per_circuit: null` 或省略：不限制数量，兼容原来的全量实验。仓库模板默认 `null`，不会静默改变已习惯的实验规模。`0`、负数、浮点数和字符串不是合法上限。
- `sampling_seed`：只控制选中哪些样本，默认 17。它与模型训练 `SEEDS`、数据划分 `split_seed` 独立；改变模型训练 seed 不会重抽数据。
- 同一输入、同一抽样上限及 `sampling_seed` 使用同一批样本。每个电路独立抽样，改变命令里的电路排列不会改变其样本选择。

单电路 60,000 条、上限 20,000 时，先得到 20,000 条，再按 0.8/0.1/0.1 划分，通常约为训练 16,000、验证 2,000、测试 2,000。原有相同设计分组规则仍然生效，实际划分数量以 manifest 为准，不拆开同一设计组以强行凑整。

OOD 也按**每个电路**限量，不是所有电路合计 20,000。假设 4 个训练电路和 1 个 OOD 电路各有 60,000 条，上限设为 20,000：训练来源总计 80,000 条，再做 ID train/val/test 划分；另有 20,000 条留出电路样本全部进入 OOD 测试，不加入训练或验证。

抽样不按性能值、可行性或 OP 状态筛选；缺失标签及不可行样本沿用原来的保留规则。归一化和字段支持仍只由抽样并划分后的 ID 训练集拟合。

## 导入、记录和续训

原始 SQLite 路径在昂贵的器件/OP 数据组装前选择样本 ID，再导入选中的样本，减少完整 canonical JSON 和后续训练缓存的规模。不会修改 SQLite、网表或其他原始文件。已有 canonical JSON 的训练入口也应用同一抽样设置，但读取已有大 JSON 本身的 I/O 成本仍然存在。

启用抽样上限时，选中样本的 ID、原始数量、选中数量、抽样上限和随机种子保存于 canonical 数据的 `sampling` 以及 manifest 的 `sampling` 中。同一实验的所有模型和训练 seeds 共用这批样本与划分；恢复分阶段训练时复用保存的数据、配置和 manifest，不重新抽样。需要改样本上限或抽样 seed 时从原始数据开启新实验，不能把新样本集接到旧 checkpoint 的续训状态上；也不能通过改大上限从已抽样的 JSON 恢复被舍弃的样本。

`--smoke` 只改变训练预算，不自行决定抽样数量；需要小数据烟测时同时在配置中设置较小的 `max_samples_per_circuit`。`--max-table-rows` 是原始表行数安全上限，不是训练样本数，不能用来代替本设置。
