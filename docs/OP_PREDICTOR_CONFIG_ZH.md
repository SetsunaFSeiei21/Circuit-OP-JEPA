# OP 预测器独立配置

OP 预测器的结构和预训练超参数现在集中在顶层 `op_predictor`，与后续性能模型的 `model` / `op_transfer` 分开。模型名称、图输入、OP 字段、损失目标、数据划分与结果目录保持原有定义。

## 修改位置

- 单电路端口级：修改启动脚本 `CONFIG` 指向的 JSON，例如 `final_exam/configs/lasd_moe_pin.json`。
- 多电路联合训练：修改 `final_exam/multi_circuit/configs_pin/models/<模型名>.json`；网格在同级 `search_spaces/<模型名>.json`。
- 器件级实验：修改 `final_exam/configs/default.json` 或 `experiments/op_transfer/example_config.json`。
- `--init-model` 为依赖 OP 预测器的新模型生成完整独立配置。MLP、GAT、XGBoost 等不训练 OP 的基线不增加无效配置。

已提供的 OP 配置复制此前的有效数值，默认训练设置不因本次分离而改变。随后修改顶层 `model` 或 `op_transfer.lr`，不会再改变这些已明确配置的 OP 参数。

## 字段和优先级

下面展示完整配置中的 OP 部分；它与顶层 `model`、`op_transfer` 并列：

```json
"op_predictor": {
  "model": {
    "dim": 128,
    "heads": 8,
    "encoder_layers": 6,
    "cross_layers": 3,
    "ffn_dim": 512,
    "decoder_layers": 2,
    "decoder_ffn_dim": 2048,
    "dropout": 0.1,
    "max_devices": 200,
    "terminal_bias": true
  },
  "training": {
    "epochs": 30,
    "lr": 0.0001,
    "batch_size": 32,
    "weight_decay": 0.0001,
    "grad_clip": 1.0,
    "patience": 10,
    "progress_every": 100,
    "torch_threads": 1
  }
}
```

| 参数 | 作用 |
|---|---|
| `op_predictor.model.encoder_layers` | 普通 OP Encoder 的 SE＋PE 总层数 |
| `op_predictor.model.cross_layers` | 普通 OP Encoder 的 PE 层数；SE 数为总层数减 PE 数 |
| `op_predictor.model.dim/heads/ffn_dim/dropout` | OP Encoder 的维度、头数、FFN 宽度和 dropout |
| `op_predictor.training.epochs` | OP 预训练最大轮数；覆盖旧 `op_transfer.epochs.op_predictor` |
| `op_predictor.training.lr/batch_size/weight_decay/grad_clip` | 仅 OP 阶段使用的优化设置 |
| `op_predictor.training.patience` | OP 阶段早停；0 关闭 |
| `op_predictor.training.progress_every/torch_threads` | OP 阶段日志频率与 CPU 线程数 |
| 顶层 `model` | 下游性能模型结构 |
| 顶层 `op_transfer` | 下游、教师及其他阶段训练设置；保留旧 OP 默认值用于兼容 |

例如 OP 使用 `encoder_layers=4, cross_layers=2`，性能模型使用顶层 `encoder_layers=6, cross_layers=3`，就分别得到 OP 的 2SE＋2PE 和性能侧的 3SE＋3PE。普通 Cascade / LASD 的两侧 `dim` 也可不同，因为它们通过物理 OP 字段向量连接。

缺少整个 `op_predictor` 段的旧配置，继续使用以前的共享设置；部分填写时，只覆盖明确给出的字段，其余继承旧有效配置。要完全独立控制，应保留新模板里的完整 OP 段。未知字段会报错。`--smoke` 同样将 OP 阶段缩短到 1 轮，并关闭其早停，不会因为独立轮数而意外进行完整预训练。

训练 seed、样本及划分、OP 字段顺序与归一化仍沿用整个实验，未增加第二套数据协议。教师模型的预算仍是 `op_transfer.epochs.teacher`，不属于 OP 预测器配置。只有依赖 OP 预测器的模型会训练这个阶段。

## 普通 OP、宽 OP 与 DeepGEN

普通器件级 OP 的输出头仍是 `Linear(dim, dim) → GELU → Linear(dim, OP字段数)`，因此普通 OP 不使用 `decoder_layers` / `decoder_ffn_dim`。这两个字段保留在通用模型结构中，只有宽 OP 等实际使用 Decoder 的结构才会读取。

`width_decoder_pin` 仍使用普通 OP 预测器；`width_decoder_op_pin` 采用每个 PE 一个 Decoder 的宽 OP 预测器。后者额外使用 OP 段中的：

```json
"width_decoder_pin": {
  "n_experts": 4,
  "hidden_dim": 128,
  "dropout": 0.1
}
```

它位于 `op_predictor` 内，独立于顶层同名的性能读出配置；OP Decoder 的层数和 FFN 宽度则读取 `op_predictor.model.decoder_layers` / `decoder_ffn_dim`。

`deepgen_op_pretrained_pin` / `deepgen_op_pretrained` 会把预训练 Encoder 复制给性能模型，并不是冻结 OP 后向下游传递物理字段的 Cascade。因此允许独立设置 OP 预训练优化参数，但 Encoder 的有效结构配置 `dim`、`encoder_layers`、`dropout`、`max_devices` 必须与性能模型一致，冲突会明确拒绝。DeepGEN 也没有 SE/PE 的 ZeroSim 分段。它的默认维度网格同时列出 OP 与性能维度，只有两者匹配的组合会运行。

旧 `experiments/op_transfer` 中的 `dual_predicted_op` 直接融合两侧 Encoder 隐藏表示，仍要求 OP 与下游 `dim` 一致；深度和训练参数可以独立设置。这不限制通过物理 OP 字段连接的普通 Cascade / LASD。

## 多电路网格

例如固定性能模型，仅搜索 OP 的 PE 深度和学习率：

```json
{
  "grid": {
    "op_predictor.model.cross_layers": [1, 3],
    "op_predictor.training.lr": [0.0001, 0.0003]
  }
}
```

也可同时搜索两侧：`model.*` / `op_transfer.lr` 控制性能侧，`op_predictor.model.*` / `op_predictor.training.*` 控制 OP 侧。OP 的 `epochs`、`lr`、`batch_size`、`weight_decay`、`grad_clip`、`patience` 支持网格；普通 OP 不接受无效的 Decoder 结构网格，宽 OP 才允许 OP Decoder 与混合读出参数。违反 OP 头数整除关系、SE/PE 深度或 DeepGEN Encoder 迁移兼容性的组合记录为 skipped。

旧模型文件若没有 OP 段，在第一次显式使用 `op_predictor.*` 网格时，会以该文件的原始设置补齐 OP 配置，再展开各 trial，因此性能侧网格不会改变未搜索的 OP 参数。只使用旧网格、且没有 OP 段时仍保持原共享行为。

OP 搜索仍按整个模型的下游验证表现选参，不改为直接按 OP 测试误差挑选。不同 trial / 模型之间不会自动复用 OP checkpoint；单次模型训练内保持已有依赖复用规则。

## 结果与续训

新阶段记录、checkpoint 和模型重载使用实际生效的 OP 结构及训练参数，旧 checkpoint 保留兼容加载。更改 OP 架构或训练设置后，应创建新实验；不要把新超参数用于恢复旧阶段。多电路已有 campaign 继续使用其保存的配置和源码快照，修改当前模板不会改变已提交任务。
