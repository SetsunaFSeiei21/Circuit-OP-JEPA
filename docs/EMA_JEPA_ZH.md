# 独立结构与 EMA 目标分支：`jepa_zerosim_ema`

这是新增的第八个可选模型。原 `jepa_zerosim` 的共享结构、OP MLP 和训练行为保持不变，可以在同一份划分上选择两个模型进行比较。新模型的性能推理仍只需要拓扑与设计参数，不需要 DC OP。

## 架构与更新边界

输入侧仍用拓扑 SE，再在后续融合层通过 own-device 交叉注意力读取设计参数。OP 侧有独立的 SE 和同深度的融合主干，读取该器件的命名电气 OP token、观测 mask，以及配置启用的 collector 检查状态。两边不共享同一个 SE 对象。

| 模块 | 输入侧 | OP 目标侧 |
|---|---|---|
| 拓扑 SE，包括类型、model、位置、端口等结构 embedding | 梯度更新 | 从输入侧 EMA 更新 |
| 后段 local/global attention 与交叉注意力融合主干 | 梯度更新 | 从输入侧 EMA 更新 |
| 输出 normalization/projection、对应的空输入 fallback | 梯度更新 | 从输入侧 EMA 更新 |
| 设计参数字段 embedding 与数值/观测投影 | 独立梯度更新 | 不使用 |
| OP 字段 embedding 与数值/观测投影 | 不使用 | 独立梯度更新 |
| 可选检查状态字段 embedding 与投影 | 不使用 | 独立梯度更新 |

这里“embedding 各自更新”指模态专属的设计参数、OP 和可选状态输入模块。拓扑身份及位置 embedding 属于 SE，因此目标侧这部分随 SE 一起 EMA。

初始化时，只复制两侧形状与语义匹配的结构/融合主干。每次完成预训练 optimizer step 后，目标主干按下式更新一次：

```text
target = ema_decay * target + (1 - ema_decay) * online
```

目标主干参数不进入梯度优化器，验证时不执行 EMA。目标融合计算仍保留对 OP/state 输入的梯度路径，不能对整个目标输出 `detach` 或把整个 OP 分支包进 `no_grad`，否则 OP 字段 embedding 无法学习。目标主干使用 eval 行为，避免目标侧 dropout 随机扰动。

损失沿用原来的两侧 MSE 对齐以及两侧 variance/covariance 防塌缩项，没有引入额外 predictor。因此这是带 EMA 主干和可训练 OP 输入模块的对齐变体，并非标准 I-JEPA 的整条 stop-gradient teacher。是否改善表征或下游性能需要实验验证。

OP 侧不直接消费设计参数。命名电气 OP、真实零、缺失 mask、器件资格和非 MOS 处理沿用已有数据契约；默认 OP 字段继续排除 `w_um/l_um/multiplicity`。阶段一不使用性能标签，OOD 不参与预训练、统计量拟合或模型选择。

## 选择与训练参数

在 `scripts/run/train_single_circuit.sh` 或 `scripts/run/train_ood.sh` 的设置区写：

```bash
MODELS=(jepa_zerosim jepa_zerosim_ema)
# 只训练新模型：MODELS=(jepa_zerosim_ema)
```

其他电路、种子、输出目录和资源设置按原来的脚本管理。启动器原有的模型数组没有自动改变；比较入口的兼容默认仍为原五个模型，只有显式 `--models all` 会选择当前全部八个模型。更新代码后，请用新的实验输出目录。

如果调用底层入口，传入 `--models jepa_zerosim_ema` 即可，例如已有 canonical 数据的两模型比较：

```bash
python scripts/run/run_comparison.py \
  --data runs/my_experiment/data.json \
  --config runs/my_experiment/config.json \
  --output runs/gpu_full/jepa_ema_comparison_new \
  --models jepa_zerosim jepa_zerosim_ema \
  --seeds 17 29 43 --device cuda
```

使用自己的实际数据和配置路径；输出路径必须新建或为空。在集群上可把相同参数传给已有的 `submit_comparison.sh`，或使用上面的便捷 Bash 启动器，在计算节点运行。

新模型额外读取 `training.ema_decay`，省略时为 `0.99`，必须为有限数值且满足 `0 <= ema_decay < 1`，不接受布尔值。需要调整时，在自己使用的完整 JSON 的 `training` 对象中增加该字段；它不是替换整份配置的内容。其余参数与原 JEPA 相同：

- `training.pretrain_epochs`：表示预训练的最大轮数。
- `training.decoder_epochs`：冻结在线输入 encoder 后，decoder 的最大轮数。
- `training.batch_size`、`pretrain_lr`、`decoder_lr`、`early_stopping`：沿用现有训练配置。
- `model.encoder_layers - model.cross_layers`：每侧纯拓扑 SE 层数；`model.cross_layers` 为每侧融合层数。

与监督基线比较时仍要求 `pretrain_epochs + decoder_epochs == baseline_epochs`。验证早停会使实际轮数不同，报告保留实际各阶段成本。相同 epoch 预算不代表相同参数量、运行时间或 FLOPs；新 OP Transformer 的计算成本必须一起比较。

独立阶段入口增加了 `--mode`（别名 `--model`）：

```bash
python -m device_jepa pretrain \
  --mode jepa_zerosim_ema \
  --data runs/my_experiment/data.json \
  --manifest runs/my_experiment/manifest.json \
  --config runs/my_experiment/config.json \
  --output runs/gpu_full/jepa_ema_stages_new --device cuda

python -m device_jepa train_decoder \
  --data runs/my_experiment/data.json \
  --pretrained runs/gpu_full/jepa_ema_stages_new/pretrain_best.pt \
  --config runs/my_experiment/config.json \
  --output runs/gpu_full/jepa_ema_stages_new --device cuda
```

不指定预训练 `--mode` 时仍运行 `jepa_zerosim`。`train_decoder` 和 `evaluate` 从 checkpoint 恢复实际模型类型；阶段二冻结整个在线输入 encoder，只训练 metric-query decoder，OP 目标分支不参与性能预测。

## Checkpoint、结果与验证范围

新 checkpoint 的架构标识为 `separate-structure-ema_device-op-cross-attention_v1`，保存新模型类别、EMA 配置与更新状态。旧共享结构 checkpoint 继续按旧架构加载，不能仅修改模型名称就当作新模型。阶段二保留源预训练的真实来源信息。

比较脚本分别保存两个模型的 `pretrain_best.pt`、`decoder_best.pt`、测试/OOD 报告及训练成本。CSV 和 `comparison/model_comparison.xlsx` 中的新行名为 `jepa_zerosim_ema`；它不进入“直接增加 OP 推理输入”的 OP/no-OP 配对表。

合成数据 smoke 用于检查选择、预训练、冻结、checkpoint 重载和汇总流程：

```bash
python -m device_jepa smoke \
  --models jepa_zerosim jepa_zerosim_ema \
  --output runs/gpu_full/jepa_ema_smoke_new --device cuda
```

smoke 的 1+1 轮仅检查机制，不代表真实预测收益。CPU 回归与小样本检查不能验证集群 CUDA 吞吐、30 万样本驻留容量或 OOD 改善；这些需要在实际 GPU 作业中单独记录。
