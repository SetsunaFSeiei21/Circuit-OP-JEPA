# Circuit-OP-JEPA

器件级电路表示预训练 → 冻结输入编码器 → 性能预测。独立新项目，不改 ZeroSim 或 Analog Agent 仓库。

仓库：https://github.com/SetsunaFSeiei21/Circuit-OP-JEPA （私有）。

## 已实现的约定

1. **输入分支**：拓扑 + 设计参数，参考 ZeroSim 的 Transformer 编码结构，但每个 **器件只有一个 node/token**，不是一个 pin 一个 node。
2. **OP 分支**：所有器件、所有电路共用一个可训练 MLP，输入该器件标准化 OP 向量及 missing mask。
3. **阶段一**：按 device_id 对齐两个分支，优化 OP 表示对齐；不使用性能标签、性能预测头或性能 loss。
4. **阶段二**：加载阶段一输入编码器，固定其全部参数及状态，仅训练 **metric-query Transformer decoder**。每个指标一个 query，包含 query self-attention、读取完整器件序列的 cross-attention，以及共享 scalar regressor。
5. **划分**：指定一个电路整体作为 OOD。剩余四个电路分别按 design/group 做 train/val/test；重复设计不会跨集合。OOD 不参与预训练、scaler/vocabulary 拟合、早停或 checkpoint 选择。

没有 graph-token MLP 性能头，也没有把所有器件平均池化后再给 decoder。保留一个辅助 graph token，但 decoder memory 是 `[graph, device_1, ..., device_N]`。

> 当前仅合成数据验证流程与约束，**不是电路仿真，不代表真实预测精度/OOD 泛化性能**。Analog Agent 的五个真实电路数据尚未接入。

## 快速运行

Python 3.10+。CPU 环境示例，先安装官方 CPU PyTorch，避免意外安装大型 CUDA 依赖：

```bash
git clone https://github.com/SetsunaFSeiei21/Circuit-OP-JEPA.git
cd Circuit-OP-JEPA
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
python -m device_jepa smoke --output runs/smoke
```

`device-jepa smoke` 是等价入口。也可以运行 `bash scripts/run_smoke.sh`。合成 smoke 使用较小模型、2 个预训练 epoch 和 3 个 decoder epoch，仅用于检查数据/梯度/保存加载/评估是否贯通。

默认研究配置在 `configs/default.json`：128 维、6 层 encoder、8 heads、FFN 512、后 4 层参数 cross-attention；2 层 metric-query decoder，decoder FFN 2048。真实实验所需 epoch、batch size、OP feature transforms 和 loss 权重需随后验证。

### 分开执行两个阶段

```bash
python -m device_jepa make_synthetic --output runs/demo/data.json
python -m device_jepa split \
  --data runs/demo/data.json --ood-circuit synthetic_4 \
  --output runs/demo/manifest.json

python -m device_jepa pretrain \
  --data runs/demo/data.json --manifest runs/demo/manifest.json \
  --config configs/default.json --output runs/demo

python -m device_jepa train_decoder \
  --data runs/demo/data.json --pretrained runs/demo/pretrain_best.pt \
  --config configs/default.json --output runs/demo

python -m device_jepa evaluate \
  --data runs/demo/data.json --checkpoint runs/demo/decoder_best.pt \
  --split test --output runs/demo/test_report.json
python -m device_jepa evaluate \
  --data runs/demo/data.json --checkpoint runs/demo/decoder_best.pt \
  --split ood --output runs/demo/ood_report.json
```

阶段一允许所有 performance 都为 {}，随后添加性能标签进入阶段二；输入、OP、记录身份与划分必须不变，性能指标名字需要提前声明。

所有配置均显式保存到 checkpoint。GPU 可传 `--device cuda`，但本版本只在 CPU 上做过验证。严格确定性模式可能使某些 GPU kernel 报错，而不会悄悄改变执行方式。

## 预训练目标：明确的 demo 选择

`h_i = input_encoder(topology, parameters)[i]`，`p_i = predictor(h_i)`，`z_i = shared_OP_MLP(OP_i, observed_i)`。

默认 objective：

```text
25 * MSE(p, z)
+ 25 * variance_penalty(p, z)
+  1 * off_diagonal_covariance_penalty(p, z)
```

- 两个 encoder 都训练，没有偷偷把 OP MLP 固定为随机 teacher。
- 没有 EMA teacher、stop-gradient、masked-context 任务或 OP reconstruction decoder。
- 仅做 MSE 可能产生常数表示，所以加入 VICReg-style 方差/协方差正则。这是**对齐式 demo 的 anti-collapse 设计**，不是声称逐项复现 I-JEPA，也不是已证明不会塌缩或能迁移的结果。
- 所有有至少一个有效 OP 的器件参与对齐；部分缺失 OP 由 mask 明确表示。整个 OP 向量缺失的器件不参与任何对齐/正则项。正则以 minibatch 内有效器件 token 为样本，需要至少 2 个。
- Stage1 checkpoint 根据 ID validation 的同一 OP objective 选择；stage2 根据 ID validation 标准化性能 MSE 选择。没有根据 OOD 调参。

详见 [architecture.md](docs/architecture.md)。

## 数据交接

完整契约与一条样例在 [data_schema.md](docs/data_schema.md) 和 [sample_schema.json](docs/sample_schema.json)。原始 Analog Agent 格式尚未知，**未臆造 converter**。

给到真实数据后主要补充：
- 五个 circuit_id 和明确的 OOD circuit_id；design_id / 重复仿真 group_id。
- 器件 ID、类型、D/G/S/B 或 P/N 的 net 映射，端口信息。
- 设计参数、OP 字段名/单位，失败/部分成功记录的缺失语义。
- 性能指标列表、单位、有效标签 mask，固定的工艺/电源/温度/负载条件。

请把原始 `NULL`、NaN 等转换为 JSON `null`，不要用数值 0 代替缺失。数值 0 是有效测量。性能全部缺失但有 OP 的样本仍可做阶段一预训练。

## 代码导航

- `device_jepa/data.py`：schema、去重分组、manifest、train-only scaler/vocabulary、synthetic fixtures。
- `device_jepa/tensors.py`：标准化、missing/padding masks、器件连接及 terminal-pair features。
- `device_jepa/model.py`：输入 encoder、共享 OP MLP、predictor、对齐 loss、metric-query decoder、冻结包装。
- `device_jepa/training.py`：两个训练阶段、checkpoint integrity、原单位 MAE/RMSE/R² 与逐电路报告。
- `device_jepa/cli.py`：命令行；`configs/` 为可编辑配置。
- `tests/`：数据泄漏、OP/参数缺失、padding、梯度、冻结参数/状态、checkpoint 和端到端回归测试。

## 已知边界

- 独立概念实现，不承诺与 ZeroSim 权重/结果兼容；未复制或分发它的源代码。来源与许可检查见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- 保留 ZeroSim-style absolute learned position embedding，按 device_id 排序；输入 JSON 记录重排不影响输出，但重新命名器件、拓扑同构不保证不变。
- 端子角色进入 attention bias；外部端口进入 terminal-role 指示。端口名字/输入输出功能仅保留在数据中，未做语义 embedding。net 名只用来确定连接关系，不作为数值 token。
- `context` 目前不是模型输入。因此验证器要求所有样本 context 相同；多工艺角/不同电源/温度/负载需要后续显式 context 特征，不能直接混入。
- 图和参数 attention 使用 dense masks，适合小型模拟电路，不是大图高性能实现。超过 `max_devices` 会报错，不截断。
- 精确重复检测保留 net 名，不能自动发现 net 重命名后的同构重复；跨来源样本应提供稳定 group_id。
- scaler 只做按具名字段的 z-score；未自动猜测 OP 符号、对数变换或单位。训练集未观测的设备类型映射到 UNK，不是 padding。
- checkpoint 支持两个阶段衔接及评估，不含训练中断恢复功能。阶段一到阶段二允许只补充/更改性能标签，评估要求同一阶段二数据 manifest；另一个任意新数据集的 predict 入口可在真实数据接入时添加。
- 本 demo 不包含 SPICE/PDK、Analog Agent 数据转换、ACCFormer 独立变体或集群脚本。

## English summary

An independent, device-node adaptation of the ZeroSim-style topology/parameter Transformer. Stage 1 jointly trains a shared per-device OP MLP and the design branch using symmetric alignment plus explicit variance/covariance regularization. Stage 2 freezes the full design encoder, disables its dropout and gradients, and trains only a metric-query Transformer decoder over all device tokens. Four circuit families are ID; the fifth is excluded from every training/scaler/model-selection step. Synthetic tests validate mechanics only, not real circuit accuracy.
