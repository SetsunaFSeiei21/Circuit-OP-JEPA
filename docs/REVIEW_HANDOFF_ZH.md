# Circuit OP JEPA 五模式实现与独立复审交接书

本交接书供未参与实现的研究者或更强模型接手审查。请先核对代码快照，再独立追踪数据、模型依赖、训练状态与统计结果，并主动构造反例。本文提供审查地图，不代替代码证据；测试通过不等于研究假设成立。

本轮需要回答两个不同问题：第一，训练和推理都允许获得 OP 时，直接输入 OP 能否改善性能预测；第二，只有训练时使用 OP 的 JEPA，能否把有用信息迁移到不需要 OP 的推理模型。两者输入权限和部署成本不同，不能混称为同一个零仿真任务。

文档日期：2026 年 10 月 25 日。项目：Circuit-OP-JEPA，软件版本 0.3.0。文档形态：仓库 Markdown、可编辑 Word 及同版 PDF。

已发布并核对的审查代码快照：85ca47392ce5a21ff765008fefe634da1c8378ee。

代码地址：https://github.com/SetsunaFSeiei21/Circuit-OP-JEPA/commit/85ca47392ce5a21ff765008fefe634da1c8378ee

代码树 SHA：edb17d6be1b9c3d92aa976e99f36e73efaceb2fb。准确测试、运行范围和发布状态见第 13 节。本文件随后作为交接文档提交；文档提交号不冒充被审代码号。原始真实数据、逐样本私有预测和训练 checkpoint 不随仓库发布。

阅读建议：先读第 2 至 3 节的用户约定和五模式契约，再沿第 4 至 10 节核对真实数据流；第 13 节区分历史与本轮证据；第 14 至 15 节可直接交给独立审查模型。所有“应当”均为验收要求，不应当自动理解为已经通过。

## 1 研究前提与为什么要做

### 1.1 问题从哪里来

模拟电路性能预测需要把拓扑、器件类型与尺寸等设计变量映射到增益、带宽等性能指标。不同电路结构、不同器件数量及设计分布会使这个映射难以泛化；完整性能仿真与标注可能比获得部分 DC 工作点信息更昂贵。研究问题是：如果训练阶段已有逐器件 operating point 信息，能否利用它塑造一个更有物理关联的设计表示，从而在下游性能标签有限时改善预测与跨电路迁移？

这里的“可能更便宜”是研究动机，不是本仓库已量化的成本结论。OP 也来自仿真或测量，不是免费标签；已有采样历史中 OP 与性能的生成成本、缺失机制和成功率仍需单独测量。不能把预训练样本叫作完全无监督数据，也不能据此宣称节省了多少 SPICE 调用。

### 1.2 本项目要检验的假设

JEPA 的输入分支只看电路结构与设计参数。训练阶段另有一个 OP 分支，它读取共享结构与逐器件电气 OP，构造可学习的目标表示；两个分支直接对齐。随后移除 OP 分支，冻结输入 encoder，仅用少量性能标签训练 metric-query decoder。最终推理只需要结构与设计参数，不需要先运行 OP 仿真。

待检验假设是：OP 监督使输入 encoder 保留更有用的器件物理状态信息，并且这些信息能够迁移到性能预测。新增的两个 OP 直接输入监督基线先检验 OP 在允许推理时获取的条件下是否有预测价值，为解读 JEPA 提供参照；它们不是 JEPA 的替代命名。DC OP 并不足以决定所有 AC、瞬态、噪声或稳定性指标。对齐成功不意味着信息充分，更不意味着 OOD 有效。当前实现不是物理约束求解器、SPICE 替代品或保证准确的电路评估器。

### 1.3 什么不在目标内

本仓库不是 Analog-Agent 的记忆检索系统，不实现 agent 长期记忆、相似案例召回或闭环自动尺寸优化。Analog-Agent 在这里提供原始采样历史格式和脚本组织参考。项目也不提出已证实的新颖性结论，不宣称复现官方 ZeroSim 论文数值，不把当前双分支训练等同于原版 I-JEPA。

### 1.4 理想实验与当前现实

期望的正式实验是明确列出多个 ID 电路和至少一个完全留出的 OOD 电路，例如四个训练电路加一个留出电路，并在 ID 内做分组 train/val/test。当前真实数据只有 Opamp2 一个电路、32 个设计样本。单电路短流程只能检查管线，不能证明跨拓扑泛化。不得复制同一电路、更改 circuit_id 后冒充五个电路。

## 2 用户约定与结论边界

### 2.1 必须保留的研究协议

本轮保留原 pin/port-node 独立复现、device-node 监督与 JEPA 三模式，并增加 zerosim_pin_op 和 zerosim_device_op。五模式共用 canonical 样本、设计分组 split、labeled_train、性能目标变换与可比的监督训练配置。OP 直接输入模式必须分别与同粒度的无 OP 基线成对比较，不只给出一张混合榜单。

器件输入必须保留具体类型，例如 NMOS、PMOS、电阻、电容、电流源与电压源；active、passive、source、unknown 是额外的粗分类，不替代具体类型。设计参数使用实际字段名与单位，不把电阻电容塞进 MOS 的 W/L 槽位。不适用、适用但缺失、真实零必须可区分，不能都变为无含义的零填充。

JEPA 继续使用同一个纯结构前缀、两条梯度回传和直接表示对齐；不增加 EMA、stop-gradient 或独立 predictor。第二阶段冻结整个输入 encoder，只训练性能 decoder。最终 JEPA 数值推理不读 OP；新增 OP 直接输入模型在训练、验证与推理都消费 OP 和观测 mask。

### 2.2 证据不能越过的边界

设计意图是探索 OP 的预测信息及其可迁移性，尚未证明方法收益。代码机制、单元回归、短流程贯通、正式预测效果与科学因果是不同证据层级。历史 211 项测试和真实 32 样本三模式 smoke 不能自动覆盖这次新增模式；本轮证据单独列在第 13 节。

禁止从短 smoke 推出 JEPA 或 OP 输入模式优于其他模式、真实 OOD 有效、可靠节省仿真、三样本测试集足以排序方法，或测试数量证明没有 bug。当前不报告未测量的 OP 采集成本为零，不把缺失 R² 变成零，也不把一个电路复制改名冒充多个拓扑。

## 3 五个模式的输入与训练契约

### 3.1 zerosim_pin

独立 clean-room 实现的 pin/port-node ZeroSim-style 监督参考。显式外部端口节点在前，随后是每个器件的端子节点。器件内部端子形成连接，同一电气 net 上的端子及外部端口形成连接。父器件实际命名参数及相应 mask 复制到其 pin token。训练 encoder 与 metric-query decoder，训练和推理都不读取 OP 或 OP 检查类别。

这不是官方源码导入，也不是逐算子、权重兼容或论文数字复现。它保留 token/位置 embedding、交替 local/global attention、后几层 own-parameter cross-attention、graph token 和逐性能 query decoder 的高层结构。pin 默认关闭器件端子对 bias 扩展。来源与明确差异见 THIRD_PARTY_NOTICES.md。

### 3.2 zerosim_device

每个物理器件一个 token 的监督参考。从随机初始化开始，端到端训练输入 encoder 与 decoder；训练和推理不读取 OP 或状态。它与 JEPA 的设计输入粒度一致，但没有 OP 对齐预训练。非 MOS 器件仍属于完整电路图，不因缺少 MOS OP 而删除。

### 3.3 jepa_zerosim

器件节点设计输入分支与 OP 分支共享纯结构前缀。阶段一用 ID train 的逐器件 OP 对齐、用 ID val 的 OP 对齐损失选择 checkpoint；阶段二固定整个设计输入 encoder，仅用 labeled_train 的性能标签训练 decoder，并用 val 性能选择 checkpoint。

推理只需要结构、具体类型与粗分类、设计参数及其 mask。OP 分支不在性能 forward 路径中。评估工具仍可能要求完整 canonical 数据和历史 OP 指纹来验证实验身份；这是一项审计接口限制，不能误解为 JEPA 的预测数值依赖 OP，也不能把该限制隐藏为已经提供了独立无 OP 部署接口。

### 3.4 zerosim_pin_op

与 zerosim_pin 使用相同 pin/port 图、共享划分和监督预算，区别是把真实逐器件电气 OP 及观测 mask 加入父器件对应的每个 pin 的输入。外部端口和没有可用 collector OP 契约的非 MOS 不获得虚构 OP。命名 OP token 与命名设计参数 token 一起进入后段 own-input cross-attention，纯结构前缀仍不读取 OP。

它从随机初始化端到端训练 encoder 与 decoder，不执行 JEPA 对齐，不冻结输入 encoder。OP 不是训练专用辅助标签：训练、验证、测试、实际推理都需要约定的 OP 输入通道及其可用性信息。单元格缺失可以由 mask 表示，但“没有 OP 输入契约”不等于普通无 OP 模型。

### 3.5 zerosim_device_op

与 zerosim_device 使用相同器件图和监督预算，把每个 OP 目标器件的电气 OP 及观测 mask 加入该器件的命名输入 token。它同样从随机初始化端到端训练，不做 JEPA 预训练，推理仍依赖 OP。直接输入模式默认不消费 collector 的离散检查类别；这些类别仍只在启用时用于 JEPA 目标分支。比较“电气 OP 价值”时不能偷偷多加入 reason 或性能派生状态。

### 3.6 成对比较能说明什么

主比较一是 zerosim_pin_op 对 zerosim_pin，主比较二是 zerosim_device_op 对 zerosim_device。先在相同 seed、split、电路、指标与变换空间内计算 OP 减无 OP 差值，再做跨种子汇总。MAE/RMSE 差值负为改善，R² 差值正为改善。结果仍需逐指标和逐电路查看，不能用混合单位总分掩盖代价。

pin 对 device 同时改变节点粒度、序列长度、参数量及优化难度；device 对 JEPA 同时改变 OP 预训练、总计算量与冻结策略。OP 直接输入对 JEPA 更改变推理时信息权限。新增 OP embedding/projection 还增加参数量，因此这不是严格容量匹配的纯因果实验。相同标签数、监督 epoch 或 seed 不等于等 FLOPs、等信息或初始化逐参数相同，差异不能全部归因于某一个机制。

正式研究需考虑随机 encoder 冻结探针、预训练后解冻、非共享 OP 分支、同拓扑 OP 打乱、只结构训练、移除检查类别、OP 子集和等总计算预算等消融。先固定主协议，再注册消融与指标，避免根据 test/OOD 反复调参。

## 4 从原始文件到模型的实际流程

原始采样目录中的 SQLite 与 DUT SPICE 文本，经 adapter.py 导入 canonical JSON；states.py 根据原 collector 证据生成可选类别；data.py 校验并创建固定 manifest、拟合 preprocessing；tensors.py 或 baselines.py 张量化；model.py 定义表示和 decoder；training.py 或 baselines.py 训练并保存 checkpoint；evaluate 产生逐样本报告；comparison.py 校验可比性并输出表格。

入口是 device_jepa/cli.py 与 device_jepa/__main__.py。批量实验由 scripts/run/run_comparison.py 编排，shell 包装负责环境与 Slurm。默认配置同时存在于 configs/ 与 device_jepa/defaults/，发布包包含内置配置。修改时应检查两份默认配置一致。

### 4.1 原始导入与器件身份

关键文件：device_jepa/adapter.py；重点函数 parse_spice_netlist、evaluate_spice_expression、adapt_analog_agent、combine_datasets；对应 tests/test_adapter.py。

SQLite 只读打开，不执行保存的 SQL、外部 Python 或仿真。网表解析接受明确支持的 MOS、DC 独立源和理想 R/C、参数定义及有界算术表达式。AST 白名单禁止函数调用、属性访问和任意执行；限制表达式长度、深度、节点数、指数和中间数值。未知器件、模型、指令或未解析参数应报错，不静默漏掉器件。include 必须位于指定数据根目录内。

每条样本的 design vector 按 schema 中的参数名解释，并代入网表的 W/L/m 等表达式。输入使用 W_m、L_m、m、dc_current_A 等有明确单位的字段；SPICE 后缀及 .option scale 必须正确处理，M 为 milli、MEG 为 mega。原始 OP 的 w_um、l_um、multiplicity 不被改写。默认严格模式检查输入几何与 OP 中几何是否一致；关闭 strict 仅放宽这一检查，不放宽身份或表达式安全。

OP join 是 sample_id 加完整器件名，器件名按 SPICE 规则忽略大小写，但不做模糊匹配。拒绝孤立样本、未知器件、重复 join、器件类型或模型冲突。canonical sample_id 加 circuit 前缀，保留原 sample_id、design_key、原始参数向量及采样来源。器件参数、OP 与性能不能仅凭数组顺序错配。

### 4.2 当前真实数据的完整性事实

Opamp2：32 样本，17 个独立设计参数，每样本 15 MOS 加 1 电流源，480 条 MOS OP 记录，44 个原始 OP 字段，9 个性能指标。总计 21,120 个 OP 标量位置与 288 个性能位置；23 个性能缺失值保留为 null。它们是数据完整性计数，不是 480 个独立电路，也不是 21,120 个独立训练样本。

9 个性能字段是 CMRR_dB、DC_Gain_dB、PSRR_Minus_dB、PSRR_Plus_dB、Phase_Margin_deg、Power_Quiescent_uW、Slew_Fall_V_us、Slew_Rise_V_us、UGF_Hz。它们横跨 dB、角度、功率、转换速率与频率，更不能直接平均原单位误差。字段名与顺序以 dataset feature_spec 和 checkpoint 为准，本文列举不用于重新排序。

不因 feasible=false、约束失败或部分性能失败而删除整条样本。本次源中 32 条样本均 feasible=false，仍全部保留；另一次只读核对确认 5,725 个负的原始 OP 值保留符号。源 reasons、check scope、阈值来源、状态与采样 provenance 被保留；状态字段并不自动成为输入特征。原始真实文件不在代码仓库，外部审查者若只有 repo，只能复跑合成测试，不能独立重建这些真实数据核对。

### 4.3 缺失值与图语义

真实数值零是观测，不是 missing。null 或缺少声明字段表示 missing；源测量中的非有限值转为 null。参数、OP、性能和状态各自有 mask，padding 与未知类型分开。非 MOS 图节点保留，但当前 collector 只提供 MOS OP，默认 op_target=false；不能伪造电阻、电容或源的电压电流作为“已采集 OP”。全缺失 OP 的 MOS 不贡献 JEPA 对齐损失。

器件图由共享 net 建边，端子角色区分 D→G 和 D→D。输入 encoder 使用类型、稳定排序位置、拓扑、端子与端口以及设计参数。无 OP 基线和 JEPA 设计分支不读取 OP；新增直接输入模式只在后段融合电气 OP。检查类别仍只进入 JEPA 的 OP 目标分支。学习式绝对位置编码依赖器件 ID 排序，不保证器件重命名或图同构不变；这是当前限制，而不是已解决的泛化性质。

### 4.4 张量接口与可检查的不变量

器件模式中 B 为 batch size、N 为补齐后的器件数、P 为声明参数字段数、O 为选中 OP 数、K 为状态字段数、M 为性能指标数、D 为模型维度。parameters/parameter_mask/parameter_applicability 为 [B,N,P]，op/op_mask 为 [B,N,O]，op_state/op_state_mask 为 [B,N,K]，performance/performance_mask 为 [B,M]。结构 token 与设计 memory 为 [B,N+1,D]，首 token 是 graph；OP 分支输出为 [B,N,D]，不包含 graph token。valid 对齐掩码必须同时满足真实器件、op_target 和至少一个被选 OP 有观测。

当前 Opamp2 有 N=16 个真实器件，原始 O=44，默认排除三个几何字段后训练 O=41，默认 K=6、M=9；跨电路合并后的字段并集可能更宽，不能把这些数写死到模型。pin 模式改变 N 的含义与词表，不能把器件数上限直接理解为电路数量。max_devices 超限应显式报错而非截断。mask 对 attention、loss、统计和报告都必须一致，不能只在输入处置零后忘记输出分母。具体类型与 device_categories 为 [B,N]；新输入契约还记录参数适用性，并在 pin 图保留父器件身份。

### 4.5 异构器件参数的三个状态

输入契约版本 2 在具体类型之外加入粗分类 embedding。NMOS 与 PMOS 都属于 active，但仍保留各自具体类型；R/C 属于 passive，独立源属于 source，未知类型保留 unknown；pin 图额外有 port 类，不应把端口视为物理器件。粗分类是固定物理语义映射，不从 test/OOD 拟合词表；具体类型与 pin/port 词表只由 ID train 建立。

实际导入字段为 MOS 的 W_m、L_m、m，电阻 R_ohm，电容 C_F，电流源 dc_current_A 与电压源 dc_voltage_V。合成数据可使用 W/L/M、R/C 等明确声明字段；字段顺序以 feature_spec 为准，不靠数组位置猜单位。未知或自定义类型可显式提供 parameter_applicability；未支持的网表器件仍应在导入时报错，粗分类出现某名称不代表 parser 已支持它。

对每个命名参数分别保存 observation mask 和 applicability mask。适用且观测时为 observed=true、applicable=true；数值等于 0 仍是观测。适用但缺失时为 false、true，保留带字段名的 missing token；不适用时为 false、false，不进入参数注意力。观测但不适用属于矛盾数据，应拒绝。归一化后的数值零也不能被反推成物理零或缺失。

审查时请构造一个含 NMOS、PMOS、R、C、I、V 的最小图，逐个核对实际字段、单位、类别和 mask；尤其测试 R=0 的张量语义、R=null、MOS 对 R 的不适用三种区别。注意理想 R/C 导入器可基于物理约束拒绝非正参数，张量 mask 的零语义不意味着网表必须接受零欧电阻。

## 5 工作检查类别的真实含义

关键文件：device_jepa/states.py；encode_device_checks、apply_op_states；对应 tests/test_states.py。

编码两个独立组：current_missing/current_below/current_pass 与 saturation_missing/saturation_below/saturation_pass。两个组可以同时失败。它们不是 cutoff/triode/saturation 三种互斥 MOS 工作区。

原始 reason 的 :saturation 表示未满足最小饱和裕量要求，不是正向“处于饱和区”标签。没有 reason 不能推断 pass；没有明确检查证据的组全部 mask=false。被检查但测量缺失则是明确 missing one-hot。已知失败 reason 只证明对应组检查过，不能扩展到另一组或全体器件。

默认 collector_checked 保守恢复原检查范围和阈值；all_mos 必须显式选择，表示根据已知阈值派生重算，记录 derived_not_original_collector，不冒充原记录；preserve_only 不生成类别。类别不做 z-score。collector 的 abs(Id) 阈值比较仅用于该电流检查，不等于对所有原始 OP 取绝对值。

已记录真实源的默认编码得到 93 个有证据的饱和检查失败，没有凭空补正常标签。审查者应独立复核阈值、run 绑定、reason 解析和矛盾证据拒绝逻辑，特别避免把“检查缺失”当成器件物理正常。

## 6 数据划分与防止泄漏

关键文件：data.py 的 build_manifest、verify_manifest、_design_groups、fit_preprocessor；对应 test_data.py、test_normalization.py。

先按显式 train_circuits 与 ood_circuits 分离电路，两组必须不重叠且覆盖数据。空 OOD 必须显式写 []。随后在每个 ID 电路内部按 design_key/design_id/group_id 与精确重复设计指纹连接为不可拆分组，再按默认 80/10/10 分到 train/val/test。比例按组分配，小样本时不保证精确样本比例。

finetune_label_fraction 只选择 train 中的完整设计组，不从 val/test 补标签。该比例只限制训练标签，验证集仍用其可用性能标签做 checkpoint 选择；正式比较少标签成本时必须另外计入这部分验证标签，不能只报 labeled_train 数量。所有模式和训练随机种子复用同一 manifest 与 labeled_train；训练 seed 可以改变初始化和批次顺序，但 split_seed 不变。manifest 保存版本、种子、配置、样本成员与指纹。已存在 manifest 只验证，不会因数据或配置变化偷偷重建。

OOD 不参与 OP 预训练、归一化拟合、标签抽样或 checkpoint 选择。阶段一允许没有性能标签；阶段二允许补充或更改性能值，但样本身份、设计、OP、输入 schema、划分必须一致。精确重复检测不求解 net 重命名后的图同构；跨来源重复需要可信 group_id。请重点审查重复、缓存命中和导出副本是否仍可能跨 split。

## 7 物理字段归一化与反变换

默认按字段选择 linear 或 asinh。电压类常用线性 z-score；跨数量级且带符号的电流、电导、电容、比值等先做 asinh(x/s)，再做 z-score。asinh 保留符号与零，不对负值取 log，不统一取 abs，不裁剪极值。具体默认规则是字段名启发式，新增字段须审查单位和变换，不能假设自动正确。

尺度 s 可以固定为正数，也可仅由训练数据拟合；默认取非零绝对值的电路均衡加权中位数。随后计算均值和标准差时真实零仍参与。equal_circuit 对每个字段的有观测电路等权，再在该电路的有效观测槽位内分配权重，不是简单将所有器件展平等权。缺失与 padding 不参与统计，常量或无观测字段有显式计数和有限回退。

参数 vocabulary、输入与 OP 统计只用 ID train；性能统计只用 labeled_train。val/test/OOD 不拟合任何尺度或统计。原始 OP 44 字段保留，但默认训练字段排除 w_um/l_um/multiplicity，防止 OP 几何直接重复输入设计。审查是否还存在从其他派生 OP 间接抄回输入的捷径。

checkpoint 保存字段顺序、规则、拟合尺度、均值/标准差/计数、词表及 fit IDs。阶段二复用输入/OP 变换，只拟合获准标签上的性能变换。反变换必须先逆 z-score，再逆 asinh 为 s*sinh(·)，不能把标准化预测直接当物理单位。manifest/preprocessing/checkpoint 版本与 canonical schema_version=1、input_contract_version 是不同版本轴。新输入契约的类别与适用性字段必须被记录、指纹化并按版本加载。容器保持 v2，旧 v2 元数据未声明 input_contract_version 时按旧投影且无粗分类 embedding 重建；旧容器 v1 仍拒绝。不应把“旧文件能加载”理解为可与新输入契约直接公平汇总。

## 8 模型结构与共享边界

关键文件：model.py 的 StructureEncoder、DeviceEncoder、SharedOPEncoder、JEPAModel；tests/test_shared_structure.py。

默认 dim=128、heads=8、encoder_layers=6、cross_layers=4。前 2 层是首次参数融合之前的完整纯结构前缀；后 4 层是设计专属 self-attention 加 own-parameter cross-attention。共享的不是仅最初 embedding，也不是全部 6 层。smoke 使用更小结构，Opamp2 配置为 2 层总深度与 1 层融合，因此是 1+1，不是验证了默认大模型的训练表现。

StructureEncoder 拥有具体器件类型及粗分类 embedding、绝对位置 embedding、graph token、端口角色投影、可选端子对 attention bias 和前缀 local/global blocks。它只读取结构字段。DeviceEncoder 在同一结构结果上融合命名参数 token；每个器件只能 cross-attend 自己适用的命名参数，graph token 可读取全部适用参数；新契约将观测位一起编码，缺失但适用的字段保留身份，没有适用参数时使用安全 null token，避免全遮蔽 attention 出 NaN。

共享模块仅在输入 encoder 下注册一次，OP encoder 通过同一对象引用它。JEPAModel.forward 只执行一次结构前缀，再把同一非 detach 张量交给两条分支，包含同一次 dropout 实现。两条梯度都回到共享参数。没有 teacher 副本、EMA、stop-gradient 或复制后同步。

OP 分支对每个器件拼接共享结构 token、归一化电气 OP、OP mask、可选类别值及类别 mask，然后使用同一 MLP。直接将设计分支器件表示 h 与 OP 分支表示 z 对齐，不保留独立 predictor。性能 decoder 的 memory 是 graph token 加全部真实器件 token，不是只读 pooled 单 token。

新增 OperatingPointDeviceEncoder 复用上述设计编码器，扩展 input_tokens：以独立 OP 字段名 embedding 和电气值加观测位形成 token，与设计参数并列，经同一 own-node 融合路径进入后段。当前非目标器件和 port 不生成有效 OP token；符合 eligibility 但观测缺失的器件仍保留 OP 名称及 observed=0 的 token；不能把 padding 数值或非 MOS 假值注入。它不是 SharedOPEncoder，也不包含对齐目标分支。

## 9 损失函数与必须警惕的捷径

alignment_loss 为 device-wise MSE 加双方表示的 variance 与 off-diagonal covariance 正则，默认权重 25/25/1，标准差下限 1。每个 batch 至少需要两个有效、有观测 OP 的目标器件；不足时训练循环跳过，某阶段没有可用批次则报错。variance/covariance 以有效器件 token 为样本，而同一电路内器件高度相关，独立样本量不能直接等于 token 数。

这是 VICReg-style anti-collapse 的双分支对齐原型，不是标准 masked-context、EMA teacher 的 I-JEPA。正则可阻止所有表示完全常数，但不能阻止“只学拓扑、位置或器件身份”。共享结构同时进入两边，双方也都可学习，存在共同退化为结构捷径的风险。

至少检查：同一拓扑不同 sizing 的表示距离是否显著；固定拓扑后改变参数是否改变 h；打乱同拓扑 OP 配对后收益是否消失；移除 OP 或类别后是否近似不变；表示方差是否来自器件位置而非设计差异。不能仅用 alignment loss 下降或非零方差宣称学到物理。还应检查大电路在 token-wise loss 中占比过高，训练目标的电路权重与归一化的电路等权不是一回事。variance/covariance 依赖批次组成，不能把不同 batch size 下的正则化 val loss 当作完全相同量；不足两个目标 token 的尾批次跳过也应核对。

## 10 两阶段训练与可恢复性

阶段一 pretrain：不加载性能目标、不拟合性能统计。train 用 OP 对齐优化；val 用 OP 对齐目标选择 pretrain_best.pt。没有可用 train/val OP 会明确失败。日志记录 observed_device_tokens 与跳过的全 OP 缺失记录。OOD 不用于选择。

阶段二 train_decoder：检查输入/OP 指纹、partition、架构、状态配置与归一化契约，然后加载输入 encoder。FrozenPerformanceModel 同时设置 requires_grad=False、eval，并在 no_grad 内执行 encoder；父模型 train() 后仍将 encoder 保持 eval，固定 dropout 行为。优化器只有 decoder 参数。每 epoch 比较所有 encoder 参数和注册 buffers 的 digest，并检查没有 encoder 梯度。

监督 loss 是 observed-target normalized MSE。无标签 batch 跳过；某阶段完全没有目标则报错。val 的性能 normalized MSE 选择 decoder_best.pt。完全没有训练观测的指标会记录 unobserved_training_metrics，其输出不能当作已受监督学习的可靠预测。

checkpoint 支持阶段衔接与评估，保存架构、来源、manifest、预处理、最佳 epoch/选择依据、参数量、优化步数和阶段耗时；不等于完整中断续训状态。标准流程不保存任意中断恢复所需的全部 optimizer/RNG/scheduler 状态，不能把重复运行原输出目录当作正确 resume。

## 11 五模式自动实验与结果解读

scripts/run/run_comparison.py 默认运行全部五模式与指定 seeds，复用一份 manifest，训练后自动 evaluate 和 aggregate。显式 --modes 可运行原三模式或有意选择的子集，不能把缺少新模式的目录标为完整五模式验证。新实验应使用新或空目录，子进程失败应使整次任务失败而不是产出伪完整结果。

输出包含 summary.csv、per_seed.csv、costs.csv、paired_per_seed.csv、paired_summary.csv、comparison.json 和阅读说明；JSON 中的成本集合名为 run_costs，不另输出同名 CSV。paired 文件首先按 seed 配对 OP 与对应无 OP 参考；汇总使用样本标准差 ddof=1，并保留有效种子数。单 seed 的标准差为空，这不是稳定性证明；当前不做显著性检验或置信区间推断。

每指标输出物理单位 MAE/RMSE/R² 和训练目标变换下的 normalized 指标，按 circuit 和 __all__ 分别给出；__all__ 只在同一指标、同一 split 内池化。逐指标 count 排除缺失目标，观测目标上的非有限预测直接失败。少于两个标签或零目标方差时 R² 为 null。配对差值任一侧未定义时，不应填零参与均值。

comparison.py 必须检查数据、性能、输入 OP 身份、partition、labeled IDs、目标 scaler、指标名、评估样本及逐样本真值一致；要求所选模式匹配 seed 集合。监督 epoch、学习率、batch size、weight decay、grad clip 一致，同一模式跨 seed 的架构和训练配置一致。匹配粒度的 OP/no-OP 模式需使用一致基础架构与输入变换；pin 只允许已声明的 max_devices 和 terminal_bias 差异，OP 特征增量另行声明，不应放宽其他公平性检查。

共享参数变换、具体类型词表、粗分类词表、参数名、端子顺序与输入契约有明确指纹；OP 直接输入还须核对 OP 选择和归一化来源。同一模式跨 seed 的 OP 信息协议，包括 JEPA 状态字段、模式和阈值配置，必须一致，不能把不同特征实验作为同方法随机种子汇总。pin 自有词表与 device 空间不同，不要求整个 preprocessing hash 无条件相同。allow-partial 仅供调试，不得绕过身份、标签预算或变换一致性约束。

成本必须区分模型训练与数据获取。JEPA 多出的 OP 预训练时间与步数要单列；OP 直接输入模式没有该阶段，但训练和每次实际推理都承担取得 OP 的前提。OP 的 SPICE/采样耗时、失败率、缓存复用、完整性能标签采集和验证标签成本均未测量，不能用模型计时替代，更不能写成免费。当前 costs.csv 与 JSON run_costs 仍可能按评估 split 重复同次训练成本；使用者必须按一次 mode/seed run 去重，不能因 test/OOD 两份报告重复相加。未来若填入有实测 scope 的 OP 采集成本，则应按对应 scope 核算，不能对所有成本盲目去重。OP coverage 按原始器件统计，不以重复 pin 数夸大观测覆盖。

不要平均不同单位 raw RMSE 成为总分，不混合 test/OOD，不隐藏负 R²，不把 null 变成零，不因 pooled 好看而忽略小电路。短 CPU smoke 的秒数也不能推出正式 GPU 吞吐、显存或端到端加速。

## 12 复现命令与环境

以下命令从项目根目录运行。真实数据路径需要审查者自行获得授权并提供。示例不会运行 ngspice，也不会生成新的真实电路采样。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
python verification/five_mode/independent_probe.py
bash scripts/smoke_test/run_smoke.sh
```

真实单电路五模式短流程：

```bash
python -m device_jepa import_analog_agent --input /path/to/Opamp2 --config configs/opamp2_smoke.json --output runs/opamp2.json
bash scripts/run/run_comparison.sh --data runs/opamp2.json --config configs/opamp2_smoke.json --output runs/opamp2_compare --seeds 17 --smoke
```

单阶段诊断命令，DATA/MANIFEST/CONFIG/RUN 均为需替换的路径：

```bash
python -m device_jepa split --data DATA --config CONFIG --output MANIFEST
python -m device_jepa pretrain --data DATA --manifest MANIFEST --config CONFIG --output RUN
python -m device_jepa train_decoder --data DATA --pretrained RUN/pretrain_best.pt --config CONFIG --output RUN
python -m device_jepa train_baseline --mode zerosim_pin --data DATA --manifest MANIFEST --config CONFIG --output RUN_PIN
python -m device_jepa train_baseline --mode zerosim_device --data DATA --manifest MANIFEST --config CONFIG --output RUN_DEVICE
python -m device_jepa train_baseline --mode zerosim_pin_op --data DATA --manifest MANIFEST --config CONFIG --output RUN_PIN_OP
python -m device_jepa train_baseline --mode zerosim_device_op --data DATA --manifest MANIFEST --config CONFIG --output RUN_DEVICE_OP
python -m device_jepa evaluate --data DATA --checkpoint CHECKPOINT --split test --output report.json
```

仅复跑原三模式时，向 runner 显式传入 --modes zerosim_pin zerosim_device jepa_zerosim；这不是五模式验收。正式比较应保留完整配置、manifest、每模式每 seed 的 checkpoint、history、report 和汇总，不能只保留均值。

真正多电路实验必须先准备并合并真实数据，复制配置明确 train_circuits/ood_circuits、标签比例和预算。configs/default.json 的名字是 synthetic_0 至 synthetic_4，不可直接用于真实数据。示例正式运行：

```bash
bash scripts/run/run_comparison.sh --data runs/real_data.json --config configs/my_experiment.json --output runs/real_compare --seeds 17 29 43 --device cuda
python -m device_jepa.comparison --reports runs/real_compare/*/seed_*/*_report.json --output runs/reprocessed
```

### 12.1 与 Analog Agent 对应的集群脚本

脚本按 scripts/git、scripts/smoke_test、scripts/run 分组。参考来源为 Analog-Agent-NN-Memory 仓库 analog-agent/scripts，已记录实际读取的 submit_sampling.sh、slurm_run.sh、run_sampling.sh 与 git/push.sh 的对应关系，而非声称完全照搬。

资源 preset：账户 b_phzhwu，分区 ex01A800，1 node、1 task、8 CPUs/task、1 GPU、80G 内存、正式时限 10-00:00:00；参考中未见 QoS。Conda 环境为 newbase，激活路径优先 /share/software/anaconda3/etc/profile.d/conda.sh，再回退 conda info --base。账户/分区/邮件均通过环境变量设置，不复制个人邮件，也不包含凭据。

安全差异：本项目默认 DRY_RUN=1，smoke 默认 30 分钟，日志目录防碰撞，sbatch 显式 --chdir。申请 GPU 不会自动切换模型设备，必须 --device cuda。只验证了语法与 mocked dry-run 参数，没有实际向集群提交任务。

```bash
source scripts/run/cluster.env.example
bash scripts/run/submit_comparison.sh --data /absolute/data.json --config /absolute/config.json --output /absolute/new_results --device cuda
# 以下才会实际请求集群资源，需在拥有权限的集群环境执行
DRY_RUN=0 bash scripts/run/submit_comparison.sh --data /absolute/data.json --config /absolute/config.json --output /absolute/new_results --device cuda
```

Git helper 不自动 stage、不 force-push、不 reset/rebase、不保存凭据。先 git add 明确文件并审查 staged diff，默认仅预览；PUSH_CONFIRM=1 才进入提交/推送路径。它是便捷工具，不替代发布审查。

## 13 本轮证据与历史证据必须分开

### 13.1 本轮五模式修订的锁定结果

本轮软件版本 0.3.0 的完整自动测试为 241 项通过，运行环境为 Python 3.12.14、PyTorch 2.14.1+cpu、NumPy 2.5.4。新增 tests/test_op_inputs.py 的 15 项针对性测试覆盖异构器件实际字段、类别、适用性/观测、直接 OP 依赖、无 OP 推理不变性、pin 父器件匹配、held-out 不参与拟合以及旧输入契约衔接。最终标准测试的全部名称和通过摘要保存在 verification/five_mode/test_results.txt，验收范围见同目录 RESULTS.md。

本轮重新完成了五模式短训练。合成数据为 5 个 toy 电路、每电路 24 样本、总计 120；train/val/test/OOD=80/8/8/24，labeled_train=80。真实数据为重新导入的 Opamp2 32 样本；train/val/test/OOD=26/3/3/0，labeled_train=26。两组都只用 seed 17，每个阶段 1 epoch，全部五模式都完成训练、checkpoint 读取、可用 held-out split 评估及严格比较。合成有 10 份报告、140 汇总行、56 成对行；真实有 5 份报告、180 汇总行、72 成对行。行数按模式、指标、电路、变换空间等展开，不是样本数。

训练完成后又修复报告/比较层的协议校验与元数据保留，对全部 15 份评估报告重新运行并重新聚合，同时核对逐样本预测没有因报告修订改变。这里准确区分“本轮重新训练”和“最后报告补丁后重新评估”，不声称报告层末次修改后又重训所有模型。最终各模式的 split、标签集合、目标 scaler、输入变换与连续 OP scaler 身份一致；严格比较 complete=true。

真实源压缩包 SHA256 为 54e16b0e3bb6ffc33e19efc5b01428e83bd672d2faa925e152223e31a6141edc，与此前来源记录一致。本轮没有运行 ngspice、重新采样电路或实际提交集群作业。真实数据与逐样本输出仍只在私有验证目录，仓库只发布必要的聚合证据。

独立复核另外执行了 7 个小型反例检查，脚本与结果分别为 verification/five_mode/independent_probe.py 和 independent_probe_results.txt。它们使用合成异构 NMOS/PMOS/R/C/I/V/unknown fixture，不是真实 R/C 电路采样证据，涵盖 n/a/missing/zero 分离、无 OP 模型不变性、直接 OP 对符号敏感但不读状态、masked NaN/padding 隔离、held-out 极值不污染统计、legacy 权重重建及 unknown 不等于 PAD。还用真正旧代码生成的阶段一 checkpoint 在新代码完成阶段二与评估，核对冻结 digest 和版本一致；这些独立补充不计入 241 项标准测试数量。

shell 语法、Python 编译和 git diff whitespace 检查通过，0.3.0 wheel 构建并安装到干净 target，确认导入、四个监督模式和包内配置可用；该构建产物 SHA256 为 0f1ed556db9feba0fde18d858e6eef1f95bdc7c31b0170ca8841df1d212bd421。wheel hash 只标识本次构建字节，不是方法有效性的证据，也不承诺换环境可得到相同 wheel 字节。

可审计证据集中在 verification/five_mode/RESULTS.md、run_manifest.json、test_results.txt、independent_probe.py、independent_probe_results.txt，以及 synthetic_ 和 opamp2_smoke_ 前缀的 summary、per_seed、costs、paired_summary、paired_per_seed CSV。run_manifest 保存运行范围与实际源码 hashes。请优先读这些原始证据再决定本文是否可信。

这些是 CPU 机制验收：未完成正式多 seed 大预算训练，未验证 GPU/集群执行，未建立真实多电路 OOD，未测 OP 获取或总部署成本。单 seed 和三个真实 test 样本不能给方法排序或显著性结论。

本节绑定文首被审代码快照，不替后续修改背书。复审者应再次核对源码 hashes、日志与聚合结果；测试数量不能独立说明覆盖面，完整绿色测试仍可能遗漏错误假设。

### 13.2 历史三模式证据

以下属于本轮修改之前的 v2 三模式验证，不应误称为新增 OP 直接输入模式的测试结果。verification/V2_RESULTS.md 与 verification/unittest_v2.log 记录 211 tests passed，环境 Python 3.12、PyTorch 2.14.1+cpu；当时 diff whitespace、shell bash -n、Python compilation 和 0.2.0 wheel 构建通过。更早的 163 或 166 计数不是历史 v2 最终候选的计数。

历史合成实验：120 个 toy 样本、5 电路、seed 17/29、三模式、每阶段一轮，分别导出 test/OOD，共 12 报告、84 分组汇总行。历史真实 Opamp2：32 样本，train/val/test=26/3/3，seed 17、三模式、每阶段一轮，9 指标，共 3 报告、108 汇总行。行数含物理/归一化和电路/池化，不是独立样本量。历史审查修复后，真实数据重新导入并重新训练三模式；合成双种子仅用已有 checkpoint 重新评估与聚合，未声称补丁后全部重新训练。

历史真实源核对记录：21,120 OP 单元格及 288 指标位置与 SQLite 一致，23 missing 保留，数据库 hash 未变，无 feasibility 过滤。当前仓库只含验证摘要和汇总 CSV；审查者没有获授权的原文件时，不能将文档记载当作亲自完成的真实核对。

这些历史检查覆盖机制与回归，不覆盖本轮新输入契约，也不覆盖长训练收敛、GPU 数值、速度、显存、跨拓扑真实 OOD、统计显著性或仿真节省。正式运行前仍须锁定代码、数据 hashes、依赖、配置和种子，保留失败日志及逐样本输出。

### 13.3 历史审查已修复的问题

历史 v2 审查不是只阅读文档：在原 166 项测试全部通过后，又构造额外反例，发现现有测试未覆盖的问题。以下问题在历史版已修复并纳入当时最终 211 项回归测试；这不意味着所有未知问题都已排除。

数值与评估真值：旧 float32 反归一化可能把极小的有效数值变为零，或在中间 sinh 溢出时错误地产生无穷值；更严重的是，通过标准化张量反变换来重建真实标签，会丢失大偏置下的小差异。例如均值 1e9、标准差 1 的标签差异可能在 float32 中合并，从而改变方差与 R²。当前使用 float64/log-safe 反变换，并直接用 canonical 原始性能值作为物理单位评估真值，而不是反推标签。RMSE/R² 也使用按量级缩放的计算，覆盖 1e±200 等极端范围反例；真正超过有限表示范围时报错，不静默裁剪。对应 tests/test_evaluation_precision.py 与 tests/test_comparison.py。

Checkpoint 加载依赖：旧依赖下限 torch>=2.2 允许已知存在 weights_only 加载漏洞的版本。官方安全公告 GHSA-53q9-r3pm-6pq6 将修复版本列为 2.6.0；历史修复将项目最低版本提高至 torch>=2.6，并在加载前运行检查，不能因为传了 weights_only=True 就认为旧版本绝对安全。仍只应加载可信来源 checkpoint。公告：https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6 。

状态来源契约：旧训练入口可能接受以 all_mos 派生编码的数据，却按 collector_checked 配置保存 checkpoint provenance。数值能运行不代表语义一致；当前 validate_op_states 会按请求配置重新构造期望编码，对比 dataset 状态元信息、字段、类别和逐器件 provenance，拒绝不匹配或伪造来源，防止把派生标签写成原 collector 证据。

多电路来源隔离：合并数据时不同电路可能恰好使用相同 run_id；旧逻辑若全局匹配这个编号，可能借用另一电路的 checked_devices 或阈值来源，使本来未知的检查被错误编码。当前按 source_id 与 circuit_ids 作用域关联 run，adapter/combine 为来源建立关联，拒绝歧义来源。旧的未限定作用域的 combined 数据不能仅凭唯一 run_id 借用证据。回归覆盖两个源相同 run_id、不同检查范围和缺失 run 的情况，确认互不污染。

这些案例说明单元测试数量不能代替反例构造。审查模型应再次复现边界，并确认修复没有悄悄更改研究协议。另外新增 input_transform_digest 和跨模式架构检查，防止只匹配标签而放过不同输入变换或模型配置。seed_everything 在 CUDA 初始化前设置 CUBLAS_WORKSPACE_CONFIG，拒绝不兼容值；这只是依据文档加上的配置保护与 CPU 测试，不等于已验证 GPU 确定性。

### 13.4 本轮具体修复与审查发现

旧输入契约的阶段衔接曾出现来源元数据矛盾：旧阶段一 checkpoint 实际为 input contract 1，但新训练入口可能把阶段二顶层元数据写成 2。新加载校验会因此拒绝后续文件，或使报告与实际模型不一致。当前从实际 preprocessing 继承输入契约，而不是硬编码新版本；已用真正旧代码产生的 checkpoint 完成阶段二、评估并验证冻结，且旧 contract 1 报告不能与新 contract 2 模型混合公平比较。审查者应再核对 training.py 的阶段二 checkpoint 构造与 load_checkpoint。

比较器曾允许同一 JEPA 模式跨 seed 混合不同 OP 状态字段、模式或阈值配置。这不是随机种子变化，而是不同信息协议；若静默汇总，会把不同实验错误称为同一种方法的重复。当前按模式检查稳定的 OP 信息签名，并保留实际 state_features/state_config、source、eligibility 和 coverage 等元数据；回归测试要求不匹配时拒绝。此问题不影响本轮配置一致的短训练预测，但影响汇总可信性，因此修复后重新评估与聚合。

新 OP 输入与异构参数相关实现还补充了 masked payload 隔离、显式推理 OP 要求、原始器件计数覆盖率和输入契约检查。不能将“新增功能满足约定”全部包装为旧版 bug；旧版并未承诺直接输入 OP 或新适用性投影。仍保留的限制包括结构捷径、missingness 分布漂移、参数量与计算差异、真实数据不足、没有无 OP 独立部署封装及未测成本。

## 14 独立审查顺序与验收标准

### 14.1 P0 先验收数据身份和异构语义

按 adapter.py → states.py → data.py → tensors.py → baselines.py 的顺序，核对 sample/device join、单位、source reason、真实零、null、适用性、重复设计与 OOD 隔离。器件必须同时保留具体类型和粗分类，R/C/源不能退化为无参数空节点，MOS 的尺寸不能被误当作无源参数。

必须构造反例：不同样本同名器件；大小写变化；同名 run_id 来自不同电路；同一设计重复导出；未见器件类型；可观测数值与 applicability 矛盾；一个字段分别为不适用、missing、零；不具备被动器件 OP 的 collector。验收看实际拒绝或正确结果，不看代码注释。任何伪造 OP、静默丢器件、跨样本 join 或 train/test 泄漏均属于阻塞问题。

### 14.2 P0 验收五模式输入权限

对每个模式独立跟踪 forward。无 OP pin/device 以及 JEPA 设计分支不应读取 OP、OP mask 或状态；改变 OP 不能改变其数值推理。对 OP direct 模式，改变一个实际观测的 OP 应有可追踪的依赖/梯度路径；删去必需输入通道应明确失败，mask=false 的填充值应不影响输出。pin OP 必须复制到同一父器件的 pin，而不能跨器件或外部端口传播成伪观测。

同时检查 checkpoint 恢复与 evaluate 分派，不仅检查新建模型。不要用随机一次前向“恰好不同/恰好相同”作为唯一证据；用受控权重、hook、梯度、输入通道追踪或最小反例证明依赖。缺失 mask 本身也是信息，真实部署缺失机制改变时需单列风险。

### 14.3 P0 验收共享与冻结

读 model.py、training.py、test_shared_structure.py 和 test_pipeline.py。默认 2+4 分界必须保持；共享模块是同一对象且只注册一次，forward 只执行一次，两条梯度都返回共享参数。设计参数不进入结构或 JEPA OP 目标分支，OP 不进入结构或设计分支。无 EMA、stop-gradient 或 predictor 是用户指定协议，不应按另一个 JEPA 定义误报为缺陷。

第二阶段需检查参数、buffers、梯度、dropout 行为、optimizer parameter IDs 和输入变换全冻结。调用父模型 train()、保存再加载 checkpoint、补充合法性能标签后，仍需成立。仅 requires_grad=false 或参数 hash 不变不足以单独证明运行行为固定。

### 14.4 P1 验收比较器和数值统计

读 comparison.py、test_comparison.py、test_evaluation_precision.py 和 runner。五模式的 split、标签集合、输入契约、OP 选择、目标 scaler、真实评估样本、seed 集合与监督预算必须匹配。直接 OP 和对应 no-OP 架构需具有明确可解释增量。故意改动其中一项，比较器必须拒绝而不是只给警告。

手算小样本的变换与反变换、masked MSE、MAE/RMSE/R²、跨 seed ddof=1 与先配对后汇总。检查 R²=null、一个 seed、某指标全缺失、极大偏置与极小尺度。对已有汇总重新读取逐样本报告核算，不依赖报告内自报指标。成本必须以一次训练 run 去重，未测的 OP 采集成本保留未测。

### 14.5 P1 评价科学风险而不是伪装成 bug

共享结构可能支持拓扑/位置捷径；同电路 token 高度相关；直接 OP 模型可能利用缺失模式；OP 可能包含几何复述或性能派生信息；JEPA 冻结策略与额外算力仍是混杂。绝对位置编码不是图同构不变，单拓扑小数据无法验证跨拓扑泛化，DC OP 也不能充分解释所有 AC、瞬态和噪声指标。

每项风险给出最小可证伪实验及预期解释。例如同拓扑内打乱 OP 配对、固定拓扑改变 sizing、输入 OP 缺失率干预、随机冻结探针、预训练解冻、等计算预算和逐电路指标。区分“机制上可能发生”“反例已复现”“实际数据已观测到”，不要把担忧直接写成已确认故障。

### 14.6 P2 运行和维护

审查 cli.py、configs、scripts、pyproject.toml 与 checkpoint provenance。检查干净安装、包内默认配置与外部配置一致、输出目录保护、子进程失败传播、GPU 选择和 Slurm dry run。只有提交成功的代码快照才可用于发布声明。没有权限时不得提交真实集群任务、扩大采样、上传原始数据或修改远端仓库。

### 14.7 接受或拒绝的判断

允许启动正式实验的最低条件是：没有未修复的 P0；新增五模式有独立依赖反例和完整小流程证据；比较器拒绝核心不公平配置；数据来源与 OP 权限清楚；正式数据、预算和实验协议经用户确认。P1 若影响关键数值或主结论，应同样阻塞正式结果发布。

“工程验收可通过”不等于“方法有效”。正式实验启动前可以保留明确列出的科学假设，但不能保留会改变样本、标签、推理输入或比较口径的未决实现错误。结论必须注明条件，不能只写全部通过。

## 15 可直接交给更强模型的完整复审提示词

请独立审查 Circuit-OP-JEPA 的指定代码快照 85ca47392ce5a21ff765008fefe634da1c8378ee 和这份交接书。你没有参与本次实现，不应把文档、作者结论、已有测试或测试数量当作正确性的证明。请先校验 commit/tree 与工作区状态，再追踪真实代码、张量依赖、训练状态与统计路径，主动构造能推翻实现声明的最小反例。

研究目标有两层：新增 zerosim_pin_op 与 zerosim_device_op 检验训练和推理都可获取 OP 时的监督预测价值；jepa_zerosim 则只在表示预训练和验证时用 OP，第二阶段冻结设计 encoder、只训练 decoder，数值推理不需要 OP。原 zerosim_pin 是独立 pin/port-node ZeroSim-style 参考，不是官方源码导入或论文数字复现；zerosim_device 是同族器件节点监督参考。五模式共用样本、分组划分、标签预算和 train-only 变换。必须分别比较 pin_op 对 pin、device_op 对 device，不能把 OP 推理成本忽略后混称零仿真方法。

请保留用户已指定的协议：同一个纯结构前缀、两条梯度回传、直接表示对齐、无 EMA、无 stop-gradient、无独立 predictor；第二阶段冻结输入 encoder。如果认为这个协议有科学弱点，请说明风险并设计消融，不要擅自改成标准 I-JEPA 再说原实现错误。具体器件类型不得被 active/passive/source/unknown 粗类替代；实际 W/L/M、R/C 和源参数需有明确命名及单位；适用性与观测 mask 要区分不适用、missing、零。当前 collector 只有 MOS OP，不允许为无源或源伪造 OP。这是采集器覆盖范围，不是宣称无源没有物理 DC 状态。

审查顺序如下。第一，检查网表安全解析、单位、sample/device 精确 join、44 原始 OP 保留和源检查证据。第二，检查重复设计分组、ID/OOD、标签预算、train-only 词表与归一化。第三，检查异构参数及 mask，从 canonical 到 device/pin 张量再到 attention 的每一步。第四，验证五模式的 OP/状态依赖及 checkpoint/evaluate 分派，特别是 OP direct 模式真正消费 OP，而无 OP 模式和 JEPA 推理不消费。第五，验证共享单次 forward、梯度与冻结，以及不存在性能标签泄漏。第六，检查数值反变换、指标计数、null R²、成对种子差值、公平性拒绝与成本去重。最后审查 CLI、runner、打包和 Slurm 防误提交。

先运行全量 unit tests 与小型五模式合成 smoke，再设计针对性反例。只有获得授权的真实 Opamp2 原数据时才做真实源核对；一个电路、32 样本、约三个 test 样本不能证明真实 OOD 或可靠方法排序。不要向第三方上传数据，不跑大型训练，不提交真实集群作业，不改远程仓库。若允许本地修复，请先保留原始发现与最小复现，再提交最小补丁和新增回归测试，不改变原始数据或研究协议。

请按以下格式输出审查报告，每条发现单独成段，不必强行塞入表格。

1. 审查对象与环境：完整 commit/tree、是否脏工作区、依赖与设备、能访问的数据、实际执行的命令。
2. 总体结论：能否启动正式多电路实验，以及清楚的前提；不能仅写测试通过。
3. 逐项发现：编号；严重性 P0/P1/P2；性质为已复现 bug、设计限制、未证实风险或可选改进；文件与函数及该快照行号；触发条件；最小复现；期望行为；实际行为；日志或数值证据；影响模式和数据范围；最小修复；回归测试。没有可验证证据时明确写未证实。
4. 已验证契约：按数据、输入、模型、冻结、比较、运行分别列出验证方式和证据位置，避免仅引用已有测试名。
5. 无法验证的项目：缺失的数据、硬件、权限或预算，以及这如何限制结论。
6. 修复与复测：区分审查前状态、补丁、补丁后命令及结果，保留失败证据，不用新结果覆盖原始问题。
7. 最小下一轮实验：按信息价值排序，给出假设、干预、控制项、预算和可证伪的结果解释；不要预先承诺正向收益。

尤其请挑战本文的声明。已有测试可能验证了错误假设，短 smoke 可能只覆盖路径而未覆盖语义。发现文档与代码不一致时，应以实际代码与可复现行为为据指出差异，同时不要把历史 211 项测试或历史真实三模式结果冒充本轮新功能验证。

## 16 代码地图与来源

数据路径：device_jepa/adapter.py 负责 parse_spice_netlist、evaluate_spice_expression、adapt_analog_agent、combine_datasets；states.py 负责 encode_device_checks、apply_op_states 和来源校验；data.py 负责 validate_dataset、build_manifest、verify_manifest、fit_preprocessor、device_category、parameter_applicability；tensors.py 负责 collate_samples、evaluation_targets 和稳定反变换。

模型路径：model.py 的 StructureEncoder、DeviceEncoder、OperatingPointDeviceEncoder、SharedOPEncoder、JEPAModel 与 FrozenPerformanceModel；training.py 的 pretrain、train_decoder、evaluate、checkpoint/provenance；baselines.py 的 collate_pin_samples、SupervisedPerformanceModel、train_baseline 和基线评估。先确认类的实际定义位置，再给问题行号，别根据本文文件列表猜调用关系。

实验路径：device_jepa/cli.py、device_jepa/comparison.py、scripts/run/run_comparison.py、scripts/run/*.sh；配置在 configs/ 与 device_jepa/defaults/；审查入口为 docs/review_order.md、docs/architecture.md、docs/data_schema.md 和 scripts/README.md。新增测试与本轮日志路径见第 13 节，历史日志保留于 verification/V2_RESULTS.md、AUDIT_FIXES.md、unittest_v2.log 和 comparison/。

高层架构参考：ZeroSim 仓库 https://github.com/xz-group/ZeroSim ，重点为 model/circuitformer.py、model/modules.py、configs/config.yaml、scripts/preprocess_topology.py、utils/dataset_utils.py。来源记录说明实现当时查阅了什么，不等于复制了官方代码、权重或数据，也不替代当前许可核对。

论文背景：Xiaomeng Yang 等的 Zero-Shot Analog Circuit Evaluation with Unified Transformer Embeddings，ICCAD 2025，DOI https://doi.org/10.1109/ICCAD66269.2025.11240890 。anti-collapse 动机参考 VICReg：https://arxiv.org/abs/2105.04906 。脚本组织参考 Analog-Agent-NN-Memory：https://github.com/SetsunaFSeiei21/Analog-Agent-NN-Memory/tree/main/analog-agent/scripts 。这些来源用于解释背景，不构成本项目实验有效性的证据。
