# 器件级 baseline 与来源

所有图网络中，一个 token 对应一个完整器件。MOS 的 D/G/S/B 不拆成四个训练节点，而是保留为 `edge_roles` 中的端口对通道；外部端口名称、器件类型、器件类别和物理 `model_name` 保留在器件特征中。设计参数使用训练集预处理后的值、观测掩码和适用性掩码。baseline 输入不包含真实 OP、OP 缺失模式、仿真状态或性能标签。

第一阶段是单电路比较。MLP 与 XGBoost 使用固定器件顺序展开参数，不能由这里的单电路结果推断其跨拓扑能力。图模型可以处理带 padding 的器件图；所有模型都拒绝超过 `max_devices` 的图，不截断器件。

## 来源与架构边界

优先检查了 ZeroSim 的公开仓库 `xz-group/ZeroSim`（固定版本见 `SOURCES.json`）。该版本仅发布核心 CircuitFormer，并没有其比较实验所用的 GAT、DeepGEN、GTN baseline 源码，因此本目录没有把独立适配冒称为“从 ZeroSim 仓库复制”的实现。

| 模型 | 编码器与监督 | 本模型的读出 |
|---|---|---|
| `mlp` | 固定器件顺序展开；本地 MLP 定义 | 同一 MLP 的线性输出 |
| `xgboost` | 官方 XGBoost；每项指标独立 booster | 原生树预测，见 `../tree.py` |
| `gat` | `circuit-fewshot-code` 的 GAT 架构器件级适配 | mean pool + linear |
| `deepgen_supervised` | 同一原作者仓库的 DeepGEN/GENConv 架构；性能监督 | 该仓库 Node2GraphEmb/XAttn 思路的 query pooling + MLP |
| `deepgen_op_pretrained` | 同一 DeepGEN；先监督完整受支持 OP，再进行性能监督 | 与 `deepgen_supervised` 相同 |
| `gtn` | DAC 2024 原论文的邻居/环双注意力器件级实现 | 论文 mean pool + 64/16 MLP |
| `zerosim_supervised` | 现有项目 ZeroSim-inspired 器件编码器；与级联方法共用架构 | 项目现有 decoder，与级联方法匹配 |
| `accformer` | 用户指定 ACCFormer_CR 的 pre-BatchNorm 编码器 | CR 的逐指标 query cross-attention decoder |

各模型保留不同的读出机制，没有统一替换成同一个 Transformer decoder。DeepGEN 的 dense GENConv、GAT 的 dense attention 在这里是原生 PyTorch 适配，包含端口对信息和 padding 支持，不声称与原始 PyG 模型逐参数数值一致。DeepGEN 保留原生 16×16 query、三次 XAttn pooling 和 256 维图表示；图读出 MLP 的深度使用 `decoder_layers`，query 作为可优化 Parameter 注册。具体差异全部在 `SOURCES.json` 中列明。原仓库 BSD-3-Clause 许可保留于 `LICENSE_circuit_fewshot`。

ACCFormer 来源是 `SetsunaFSeiei21/ACCFormer_CR`，不是名称相近的其他仓库。其 pre-BatchNorm、残差、内部等宽 FFN、逐指标查询及初始化规则保留；为器件图接口增加端口对偏置、按图 mask 和排除 padding 的 BatchNorm。

GTN 没有找到可核验的原作者公开代码，故明确记录为依照论文的独立实现。它先从端口等价关系重建 device/net 二部图并计算最小环，再投影为器件集合。单个网络上三个器件形成的 clique 不会被误认为实际环。环结构仅用于拓扑归纳偏置，不等于显式求解或严格满足 KCL/KVL；多端晶体管的拓扑环也不等同于导通路径。训练默认从第 1 轮启用双路，统一预算下没有执行原论文 450/700 轮邻居分支 warmup。详细机制和差异见 `gtn_source.json`。

## 接口和可复现性

`build_model(name, config['model'], preprocessing)` 构造性能模型，输入为 resident batch，输出 `[batch, metrics]`。所有图模型暴露 `.encoder` 与 `.decoder`；编码器返回 graph token + device memory 和 padding，供 DeepGEN OP 预训练复用。`build_encoder`、`build_decoder` 分别构造两个模块。

`prepare_model(model, ctx, options)` 在 GTN 训练前为指定 split 的 resident 静态拓扑计算环成员（训练前默认仅 train/val，测试前单独准备 test）。计算环的 NetworkX 过程只在拓扑准备时执行；之后 mask gather、注意力和优化在所选 Torch device 上执行。独立小样本推理可以使用 topology cache fallback；大型推理应对 `ResidentDataset` 调用 `encoder.prepare_resident(store)`。

checkpoint 必须同时保存模型名称、完整模型配置、训练集拟合的 preprocessing、state_dict 及阶段依赖。`deepgen_op_pretrained` 的架构工厂本身不会自动执行预训练，runner 必须明确完成 OP 阶段并加载其最佳 encoder；不能仅按名称构造随机模型后报告为已预训练。

这些来源记录证明实现与适配过程，不能代替原论文复现验证。正式结果中应将本组模型标记为器件级受控适配，并一并报告每个模型的实际参数量、训练预算、预训练数据及所选超参数。
