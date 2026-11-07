# 端口级模型适配

本目录中的 `*_pin.py` 是新增端口级模型入口，原器件级模型不变。模型名称也以 `_pin` 结尾；各模型实际输入为**器件引脚及命名外部端口**，不是将器件级结果改名。MOS 的 D/G/S/B 分别产生 token，其他器件按网表实际存在的端子产生 token，不添加不存在的虚拟端子。

同一器件的引脚之间相连，同一电气网络的引脚/外部端口相连。不同命名外部端口即使映射同一网络，也不直接相连。每个引脚保留端子类别、父器件类别和物理 model_name；设计参数由父器件广播，外部端口没有设计参数。参数及 OP 的统计量仍只在原始 ID 训练器件上拟合一次，随后展开到引脚，不按引脚数量重复拟合。

| 名称 | 编码器 | 保留的模型读出 |
|---|---|---|
| `mlp_pin` | 固定引脚顺序展开 MLP | MLP 线性输出 |
| `xgboost_pin` | 固定引脚顺序特征及静态拓扑特征，逐指标树模型 | 原生 booster |
| `gat_pin` | pin graph 上的 GAT | mean pooling + linear |
| `deepgen_supervised_pin` | pin graph 上的 GENConv/DeepGEN | 16×16 query pooling、3 次 XAttn、graph MLP |
| `deepgen_op_pretrained_pin` | 同上，先用完整受支持 OP 预训练 | 与 supervised 变体相同 |
| `gtn_pin` | 邻居及真实电路环双分支 attention | mean pooling + 64/16 MLP |
| `zerosim_supervised_pin` | pin 类型 embedding、结构编码及设计注入 | metric-query Transformer decoder |
| `accformer_pin` | ACCFormer_CR 的 pre-BatchNorm attention | CR 的逐指标 query cross-attention decoder |

GAT、DeepGEN 和 ACCFormer 复用此前已归属来源的数学内核，在独立 `_pin` 类中接入真实 pin embedding。没有将它们改成统一 ZeroSim decoder。所有 baseline 的输入白名单排除真实 OP、OP 缺失模式、仿真状态和性能标签。模型拒绝缺少 pin 映射的器件 batch，防止误用粒度。

GTN 不把同网 pin clique 或同器件 pin clique 误认为电路环。它从 `pin_to_device` 和 `pin_net_index` 复原物理器件/网络二部图，计算 minimum cycle basis，再选取同时位于该环的器件与网络之间的引脚。一个位于环上的 MOS 的旁支引脚不会仅因属于同一器件就自动进入环。外部端口不成为 OP 器件或环成员。拓扑发现预先完成，`gtn_loop_membership_by_pin` 留在 resident cache 中；训练每个 batch 不重复 CPU 环计算。

OP 仍是物理器件属性。预测 OP 时，对同一器件的引脚 memory 取平均后进行一次器件 OP 预测；损失每器件/字段计一次，然后把预测 OP 广播给该器件引脚。不会把四个 MOS 引脚当成四组独立真值。教师及三个级联变体由 `op_transfer_pin.py` 实现，监督、冻结和独立保存要求与器件级阶段一致。

接口为 `models_pin.build_model_pin(name, config, preprocessing)`、`build_encoder_pin`、`build_decoder_pin`；`prepare_model_pin(model, ctx, options, splits=('train','val'))` 为 GTN 准备已用数据划分。测试集应在评测阶段显式使用 `splits=('test',)` 准备。`model.max_devices` 是沿用的配置键，**在端口级中表示引脚＋外部端口 token 上限**，不是物理器件数；超出上限报错，不截断。

来源和固定版本见 `SOURCES_pin.json`。其中 ZeroSim-inspired 部分仍为现有仓库的受控适配，不冒称官方逐参数复现；GAT/DeepGEN 来源、BSD 许可、ACCFormer_CR 固定来源，以及 GTN 原论文独立实现的说明全部保留。GTN 的 pin 特定差异另列于 `gtn_source_pin.json`。当前单电路固定拓扑协议不能用于宣称 MLP/XGBoost 的跨拓扑能力。
