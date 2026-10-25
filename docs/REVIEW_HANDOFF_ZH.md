# Circuit OP JEPA 实现说明与独立审查交接

本文面向接手审查代码、数据语义与实验设计的研究者或更强模型。目标不是证明方法有效，而是说明为什么做、实际实现了什么、证据到哪里、哪些问题必须再次独立核实。请把代码、测试和原始数据视为证据，把本文视为可被质疑的审查地图。

文档日期：2026 年 10 月 25 日。项目：Circuit-OP-JEPA，版本 0.2.0。

已发布并核对的审查代码快照：8fe025e96379e1decb308dfa6e65065df2eec4a2。
https://github.com/SetsunaFSeiei21/Circuit-OP-JEPA/commit/8fe025e96379e1decb308dfa6e65065df2eec4a2

代码树 SHA：eca7c69cb7ce1d1f20e585deec35b25f2acd5b8a，与本次通过审查的本地代码树一致。本文作为后续独立文档提交，不把自身提交号冒充被审代码版本。原始真实数据和训练 checkpoint 不随仓库发布。

## 1 研究前提与为什么要做

### 1.1 问题从哪里来

模拟电路性能预测需要把拓扑、器件类型与尺寸等设计变量映射到增益、带宽等性能指标。不同电路结构、不同器件数量及设计分布会使这个映射难以泛化；完整性能仿真与标注可能比获得部分 DC 工作点信息更昂贵。研究问题是：如果训练阶段已有逐器件 operating point 信息，能否利用它塑造一个更有物理关联的设计表示，从而在下游性能标签有限时改善预测与跨电路迁移？

这里的“可能更便宜”是研究动机，不是本仓库已量化的成本结论。OP 也来自仿真或测量，不是免费标签；已有采样历史中 OP 与性能的生成成本、缺失机制和成功率仍需单独测量。不能把预训练样本叫作完全无监督数据，也不能据此宣称节省了多少 SPICE 调用。

### 1.2 本项目要检验的假设

输入分支只看电路结构与设计参数。训练阶段另有一个 OP 分支，它读取共享结构与逐器件电气 OP，构造可学习的目标表示；两个分支直接对齐。随后移除 OP 分支，冻结输入 encoder，仅用少量性能标签训练 metric-query decoder。最终推理只需要结构与设计参数，不需要先运行 OP 仿真。

待检验假设是：OP 监督使输入 encoder 保留更有用的器件物理状态信息，并且这些信息能够迁移到性能预测。DC OP 并不足以决定所有 AC、瞬态、噪声或稳定性指标。对齐成功不意味着信息充分，更不意味着 OOD 有效。当前实现不是物理约束求解器、SPICE 替代品或保证准确的电路评估器。

### 1.3 什么不在目标内

本仓库不是 Analog-Agent 的记忆检索系统，不实现 agent 长期记忆、相似案例召回或闭环自动尺寸优化。Analog-Agent 在这里提供原始采样历史格式和脚本组织参考。项目也不提出已证实的新颖性结论，不宣称复现官方 ZeroSim 论文数值，不把当前双分支训练等同于原版 I-JEPA。

### 1.4 理想实验与当前现实

期望的正式实验是明确列出多个 ID 电路和至少一个完全留出的 OOD 电路，例如四个训练电路加一个留出电路，并在 ID 内做分组 train/val/test。当前真实数据只有 Opamp2 一个电路、32 个设计样本。单电路短流程只能检查管线，不能证明跨拓扑泛化。不得复制同一电路、更改 circuit_id 后冒充五个电路。

## 2 结论边界与证据层级

设计意图：学习 OP informed 表示，探索少标签性能预测与跨电路迁移。这是研究目标，不是结果。

代码实现：当前提供安全导入、工作检查类别、设计分组划分、训练专属归一化、共享结构双分支对齐、冻结 encoder 的第二阶段、两个监督基线和自动结果比较。下文逐项给出文件与应验证的契约。

已运行的有限证据：最终 v2 验证日志记录 211 项测试通过，合成五电路双种子与真实单电路单种子的三模式短运行贯通；每阶段仅一轮。源数据完整性核对保留了真实数据的全部 OP 单元格和指标位置。正式实验、GPU 数值行为和集群提交未验证。

当前本轮已复现的工程问题均已修复并复测，没有遗留的已确认实现阻塞项；这只支持继续准备受控实验，不代表无需进一步审查。正式多电路数据、实验预算与新增消融仍未就绪。

禁止从现有证据推出：JEPA 优于监督基线；真实 OOD 有效；能可靠节省仿真；三样本测试集足以排序方法；通过单元测试等于科学正确；空 R² 等于零。

## 3 三个模式及比较的含义

### 3.1 zerosim_pin

独立 clean-room 实现的 pin/port-node ZeroSim-style 监督参考。显式外部端口节点在前，随后是每个器件的端子节点。器件内部端子形成连接，同一电气 net 上的端子及外部端口形成连接。器件参数复制到其对应的 pin token。encoder 与 decoder 端到端使用性能标签训练，不读取 OP 或状态张量。

这不是官方源码导入，也不是逐算子或权重兼容的论文复现。实现借鉴高层设计：token/位置 embedding、交替 local/global attention、后几层 own-parameter cross-attention、graph token、逐性能 query decoder。pin 模式默认关闭器件端子对 bias 扩展。差异与来源记录在 THIRD_PARTY_NOTICES.md。

### 3.2 zerosim_device

每个物理器件一个 token 的监督参考。从随机初始化开始，端到端训练输入 encoder 与性能 decoder。它与 JEPA 的输入图粒度一致，但没有 OP 对齐预训练。电流源等非 MOS 仍是图中器件，不因缺少 MOS OP 而删除。

### 3.3 jepa_zerosim

器件节点输入分支与 OP 分支共享纯结构前缀，先以 ID train 的 OP 进行对齐训练，再冻结整个输入 encoder，只训练性能 decoder。两个监督基线与它使用相同 canonical dataset、manifest、labeled_train 和性能目标变换。

### 3.4 这三项比较尚不能完全隔离因果

pin 对 device 可观察节点粒度变化，但参数量、序列长度、端子 bias、优化难度也变化。监督 device 对 JEPA 同时改变预训练信息、总计算量和冻结策略，因此差异不能全部归因于 OP。相同标签数和监督 epoch 不等于相同总计算预算。

正式研究应补充：随机 encoder 冻结探针；预训练 encoder 解冻微调；不共享结构的 OP 分支；仅结构训练；同拓扑内打乱 OP 配对；移除类别特征；不同 OP 字段子集；等总计算预算基线。先固定主比较，再将这些作为独立消融，避免不断调参污染测试集。

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

真实数值零是观测，不是 missing。null 或缺少声明字段表示 missing；源测量中的非有限值转为 null。参数、OP、性能和状态各自有 mask，padding 与未知类型分开。非 MOS 图节点保留，但默认 op_target=false；全缺失 OP 的 MOS 不贡献对齐损失。

器件图由共享 net 建边，端子角色区分 D→G 和 D→D。输入 encoder 使用类型、稳定排序位置、拓扑、端子与端口以及设计参数。OP 和类别仅进入 OP 分支。学习式绝对位置编码依赖器件 ID 排序，不保证器件重命名或图同构不变；这是当前限制，而不是已解决的泛化性质。

### 4.4 张量接口与可检查的不变量

器件模式中 B 为 batch size、N 为补齐后的器件数、P 为声明参数字段数、O 为选中 OP 数、K 为状态字段数、M 为性能指标数、D 为模型维度。parameters/parameter_mask 为 [B,N,P]，op/op_mask 为 [B,N,O]，op_state/op_state_mask 为 [B,N,K]，performance/performance_mask 为 [B,M]。结构 token 与设计 memory 为 [B,N+1,D]，首 token 是 graph；OP 分支输出为 [B,N,D]，不包含 graph token。valid 对齐掩码必须同时满足真实器件、op_target 和至少一个被选 OP 有观测。

当前 Opamp2 有 N=16 个真实器件，原始 O=44，默认排除三个几何字段后训练 O=41，默认 K=6、M=9；跨电路合并后的字段并集可能更宽，不能把这些数写死到模型。pin 模式改变 N 的含义与词表，不能把器件数上限直接理解为电路数量。max_devices 超限应显式报错而非截断。mask 对 attention、loss、统计和报告都必须一致，不能只在输入处置零后忘记输出分母。

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

checkpoint 保存字段顺序、规则、拟合尺度、均值/标准差/计数、词表及 fit IDs。阶段二复用输入/OP 变换，只拟合获准标签上的性能变换。反变换必须先逆 z-score，再逆 asinh 为 s*sinh(·)，不能把标准化预测直接当物理单位。version 2 manifest/preprocessing/checkpoint 与 canonical schema_version=1 是不同版本轴；旧 checkpoint 明确拒绝，不静默兼容。

## 8 模型结构与共享边界

关键文件：model.py 的 StructureEncoder、DeviceEncoder、SharedOPEncoder、JEPAModel；tests/test_shared_structure.py。

默认 dim=128、heads=8、encoder_layers=6、cross_layers=4。前 2 层是首次参数融合之前的完整纯结构前缀；后 4 层是设计专属 self-attention 加 own-parameter cross-attention。共享的不是仅最初 embedding，也不是全部 6 层。smoke 使用更小结构，Opamp2 配置为 2 层总深度与 1 层融合，因此是 1+1，不是验证了默认大模型的训练表现。

StructureEncoder 拥有器件类型 embedding、绝对位置 embedding、graph token、端口角色投影、可选端子对 attention bias 和前缀 local/global blocks。它只读取结构字段。DeviceEncoder 在同一结构结果上融合命名参数 token；每个器件只能 cross-attend 自己的有效参数，graph token 可读取全部有效参数，没有有效参数时使用安全 null token，避免全遮蔽 attention 出 NaN。

共享模块仅在输入 encoder 下注册一次，OP encoder 通过同一对象引用它。JEPAModel.forward 只执行一次结构前缀，再把同一非 detach 张量交给两条分支，包含同一次 dropout 实现。两条梯度都回到共享参数。没有 teacher 副本、EMA、stop-gradient 或复制后同步。

OP 分支对每个器件拼接共享结构 token、归一化电气 OP、OP mask、可选类别值及类别 mask，然后使用同一 MLP。直接将设计分支器件表示 h 与 OP 分支表示 z 对齐，不保留独立 predictor。性能 decoder 的 memory 是 graph token 加全部真实器件 token，不是只读 pooled 单 token。

## 9 损失函数与必须警惕的捷径

alignment_loss 为 device-wise MSE 加双方表示的 variance 与 off-diagonal covariance 正则，默认权重 25/25/1，标准差下限 1。每个 batch 至少需要两个有效、有观测 OP 的目标器件；不足时训练循环跳过，某阶段没有可用批次则报错。variance/covariance 以有效器件 token 为样本，而同一电路内器件高度相关，独立样本量不能直接等于 token 数。

这是 VICReg-style anti-collapse 的双分支对齐原型，不是标准 masked-context、EMA teacher 的 I-JEPA。正则可阻止所有表示完全常数，但不能阻止“只学拓扑、位置或器件身份”。共享结构同时进入两边，双方也都可学习，存在共同退化为结构捷径的风险。

至少检查：同一拓扑不同 sizing 的表示距离是否显著；固定拓扑后改变参数是否改变 h；打乱同拓扑 OP 配对后收益是否消失；移除 OP 或类别后是否近似不变；表示方差是否来自器件位置而非设计差异。不能仅用 alignment loss 下降或非零方差宣称学到物理。还应检查大电路在 token-wise loss 中占比过高，训练目标的电路权重与归一化的电路等权不是一回事。variance/covariance 依赖批次组成，不能把不同 batch size 下的正则化 val loss 当作完全相同量；不足两个目标 token 的尾批次跳过也应核对。

## 10 两阶段训练与可恢复性

阶段一 pretrain：不加载性能目标、不拟合性能统计。train 用 OP 对齐优化；val 用 OP 对齐目标选择 pretrain_best.pt。没有可用 train/val OP 会明确失败。日志记录 observed_device_tokens 与跳过的全 OP 缺失记录。OOD 不用于选择。

阶段二 train_decoder：检查输入/OP 指纹、partition、架构、状态配置与归一化契约，然后加载输入 encoder。FrozenPerformanceModel 同时设置 requires_grad=False、eval，并在 no_grad 内执行 encoder；父模型 train() 后仍将 encoder 保持 eval，固定 dropout 行为。优化器只有 decoder 参数。每 epoch 比较所有 encoder 参数和注册 buffers 的 digest，并检查没有 encoder 梯度。

监督 loss 是 observed-target normalized MSE。无标签 batch 跳过；某阶段完全没有目标则报错。val 的性能 normalized MSE 选择 decoder_best.pt。完全没有训练观测的指标会记录 unobserved_training_metrics，其输出不能当作已受监督学习的可靠预测。

checkpoint 支持阶段衔接与评估，保存架构、来源、manifest、预处理、最佳 epoch/选择依据、参数量、优化步数和阶段耗时；不等于完整中断续训状态。标准流程不保存任意中断恢复所需的全部 optimizer/RNG/scheduler 状态，不能把重复运行原输出目录当作正确 resume。

## 11 自动实验与结果解读

scripts/run/run_comparison.py 对三模式与多个 seed 运行，复用一份 manifest，训练后自动 evaluate 和 aggregate。输出 summary.csv、per_seed.csv、costs.csv、comparison.json 及阅读说明。新实验需要新或空目录，错误会中止而不是把部分失败当作完整成功。

每指标输出物理单位 MAE/RMSE/R² 和训练目标变换下的 normalized 指标，按 circuit 和 __all__ 分别给出；__all__ 只在同一指标、同一 split 内池化。逐指标 count 排除缺失目标，观测目标上的非有限预测直接失败。少于两个标签或零目标方差时 R² 为 null；单 seed 的样本标准差为空。跨 seed 统计用 ddof=1，并报告各分数有定义的 seed 数。

comparison.py 检查数据/性能/输入 OP 身份、partition、labeled IDs、目标 scaler、指标名、评估样本及真实目标一致；各模式须有匹配 seed 集合。监督 epoch、学习率、batch size、weight decay、grad clip 一致。同一模式跨 seed 的架构和训练配置也必须一致。器件监督与 JEPA 架构配置必须完全一致；跨模式公共架构字段一致，pin 仅允许 max_devices 与 terminal_bias 差异。共享输入参数变换、器件词表、参数名及端子顺序有独立 input_transform_digest；pin 自有词表等输入空间不同，所以不强求整个 preprocessing hash 相同。allow-partial 只供调试，仍保留身份检查。

不要平均不同单位的 raw RMSE 形成总分；不要把 test 与 OOD 混在一起；不要把负 R² 隐藏；不要把 null 改成零；不要因 pooled 好看而不看小电路。costs 中同一训练成本可能按评估 split 重复，不能重复求和。JEPA 的额外 OP 预训练与样本获取成本应单列，现有小 CPU 耗时不代表集群正式开销。

## 12 复现命令与环境

以下命令从项目根目录运行。真实数据路径需要审查者自行获得授权并提供。示例不会运行 ngspice，也不会生成新的真实电路采样。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
bash scripts/smoke_test/run_smoke.sh
```

真实单电路三模式短流程：

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
python -m device_jepa evaluate --data DATA --checkpoint CHECKPOINT --split test --output report.json
```

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

## 13 已有验证与不能证明什么

截至所锁定代码快照，verification/V2_RESULTS.md 与 verification/unittest_v2.log 记录 211 tests passed，环境 Python 3.12、PyTorch 2.14.1+cpu；diff whitespace、shell bash -n、Python compilation 和 0.2.0 wheel 构建通过。早期 163 或 166 等计数不作为本版最终证据。

合成实验：120 个 toy 样本、5 电路、seed 17/29、三个模式、每阶段一轮，分别导出 test/OOD，共 12 报告、84 分组汇总行。真实 Opamp2：32 样本，train/val/test=26/3/3，seed 17、三模式、每阶段一轮，9 指标，共 3 报告、108 汇总行。行数含物理/归一化及按电路/池化，不是独立数据量。审查修复后，真实数据重新从源导入，三个模式重新完成一轮训练、评估和聚合；合成双种子使用已有 checkpoint 重新评估全部 12 报告并聚合，没有把它描述为最终补丁后的完整重新训练。

真实源核对记录：21,120 OP 单元格及 288 指标位置与 SQLite 一致，23 missing 保留，源数据库 hash 未变，无 feasibility 过滤。仓库仅保留验证摘要和汇总 CSV，不提交原始真实数据、逐样本私有预测或训练 checkpoint。

这些检查覆盖机制与回归，不覆盖长训练收敛、有效 batch size、GPU 数值、速度、显存、分布漂移、跨拓扑 OOD、统计显著性或仿真节省。正式运行前需锁定代码 commit、数据 hashes、依赖版本、实验配置和种子，保留完整失败日志与逐样本输出。

### 13.1 再次审查发现的具体问题

本轮审查不是只阅读文档：在原 166 项测试全部通过后，又构造额外反例，发现现有测试未覆盖的问题。以下问题已修复并纳入最终 211 项回归测试；这不意味着所有未知问题都已排除。

数值与评估真值：旧 float32 反归一化可能把极小的有效数值变为零，或在中间 sinh 溢出时错误地产生无穷值；更严重的是，通过标准化张量反变换来重建真实标签，会丢失大偏置下的小差异。例如均值 1e9、标准差 1 的标签差异可能在 float32 中合并，从而改变方差与 R²。当前使用 float64/log-safe 反变换，并直接用 canonical 原始性能值作为物理单位评估真值，而不是反推标签。RMSE/R² 也使用按量级缩放的计算，覆盖 1e±200 等极端范围反例；真正超过有限表示范围时报错，不静默裁剪。对应 tests/test_evaluation_precision.py 与 tests/test_comparison.py。

Checkpoint 加载依赖：旧依赖下限 torch>=2.2 允许已知存在 weights_only 加载漏洞的版本。官方安全公告 GHSA-53q9-r3pm-6pq6 将修复版本列为 2.6.0；当前项目已提高最低版本至 torch>=2.6，并在加载前运行检查，不能因为传了 weights_only=True 就认为旧版本绝对安全。仍只应加载可信来源 checkpoint。公告：https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6 。

状态来源契约：旧训练入口可能接受以 all_mos 派生编码的数据，却按 collector_checked 配置保存 checkpoint provenance。数值能运行不代表语义一致；当前 validate_op_states 会按请求配置重新构造期望编码，对比 dataset 状态元信息、字段、类别和逐器件 provenance，拒绝不匹配或伪造来源，防止把派生标签写成原 collector 证据。

多电路来源隔离：合并数据时不同电路可能恰好使用相同 run_id；旧逻辑若全局匹配这个编号，可能借用另一电路的 checked_devices 或阈值来源，使本来未知的检查被错误编码。当前按 source_id 与 circuit_ids 作用域关联 run，adapter/combine 为来源建立关联，拒绝歧义来源。旧的未限定作用域的 combined 数据不能仅凭唯一 run_id 借用证据。回归覆盖两个源相同 run_id、不同检查范围和缺失 run 的情况，确认互不污染。

这些案例说明单元测试数量不能代替反例构造。审查模型应再次复现边界，并确认修复没有悄悄更改研究协议。另外新增 input_transform_digest 和跨模式架构检查，防止只匹配标签而放过不同输入变换或模型配置。seed_everything 在 CUDA 初始化前设置 CUBLAS_WORKSPACE_CONFIG，拒绝不兼容值；这只是依据文档加上的配置保护与 CPU 测试，不等于已验证 GPU 确定性。

## 14 审查优先级与具体验收标准

### 14.1 P0 数据和实验可信性

先读 adapter.py → states.py → data.py → tensors.py 及对应 tests。核对器件身份、单位、sample join、source reason 意义、mask、重复设计和 OOD 隔离。对每项高风险假设写反例：同名器件来自不同 sample；大小写变化；NaN 与零；重复导出；缺少检查范围；不同 context；两个电路重复 run_id；输入字段被修改却沿用 manifest。

通过标准不是“已有测试绿”，而是新增反例被明确拒绝或得到正确、可解释结果。每个问题提供最小复现、实际结果、期望结果、受影响样本/模式和严重性。

### 14.2 P0 架构依赖与冻结

读 model.py → training.py → test_shared_structure.py/test_pipeline.py。独立验证共享对象同一性和单次 forward；两路梯度均到共享参数；OP/状态变化不影响设计分支；设计参数变化不影响纯结构与 OP 分支；结构不读性能。检查参数注册、state_dict 和 load 后仍正确。

第二阶段验证参数、buffers、梯度、dropout 行为、optimizer parameter IDs 与预处理都固定。验证修改 OP 输入不会改变性能推理结果，且后补标签不会接受设计/OP 改动。

### 14.3 P1 比较与数值

读 baselines.py → comparison.py → orchestration/tests。确认三种模式不混淆，pin baseline 真正含 pin/port 图与参数融合而非换名通用 GNN。手算少量变换/反变换、masked MSE、MAE/RMSE/R²、跨 seed 标准差，验证 count 与 null。构造不匹配 manifest、目标 scaler、标签预算或 seed 集合，比较器应拒绝。

### 14.4 P1 科学有效性

即使无工程 blocker，也应评价共享结构 shortcut、token 非独立、OP 派生字段、missingness 信息泄漏、冻结与端到端混杂、单拓扑过拟合、绝对位置编码、性能标签缺失选择偏差，以及 OP 无法充分解释下游指标的问题。提出最小、可证伪实验，不把“可能有风险”伪装为已复现 bug。

### 14.5 P2 运行与维护

读 CLI、configs、scripts、pyproject.toml 与 provenance。检查包内默认配置、干净安装、空输出目录保护、失败传播、GPU 选择、Slurm 环境和 checkpoint 版本。不要在审查时默默提交真实集群任务、跑大训练或改变数据。把用户需要决策的实验预算与代码缺陷分开。

## 15 建议交给审查模型的完整提示词

请独立审查 Circuit-OP-JEPA 的指定代码 commit 和这份交接文档。不要相信 README、文档或测试通过就是正确；逐项追踪真实数据流、张量依赖、训练状态和结果统计。先理解研究前提：训练时利用逐器件 OP 作为额外监督，学习仅凭结构与设计参数可计算的表示；下游冻结输入 encoder，仅训练性能 decoder；推理不需要 OP。方法收益、仿真成本节省、真实 OOD 和新颖性均未得到证明。

请按 P0/P1/P2 报告问题，每项给出文件与函数或行号、影响、最小复现、预期与实际行为、建议修复和回归测试。区分已复现 bug、明确的设计限制、尚未证实的风险与可选改进。不要为了贴近标准 I-JEPA 而擅自加 EMA、stop-gradient 或 predictor；当前约定是同一个纯结构前缀、两路回传、直接对齐。如果认为这个约定科学上不合理，请解释并给消融方案，而不是误报成未按要求实现。

优先检查：1）网表解析、单位、sample/device 精确 join 与原始 44 OP 保留；2）collector 检查失败标签和未知 mask，不把 saturation reason 当正标签；3）按设计组划分、标签预算、OOD 与 train-only preprocessing；4）默认 2 层结构共享加 4 层设计融合的边界、只执行一次共享结构、无 OP/性能泄漏；5）MSE/variance/covariance 的统计与 shortcut；6）冻结 encoder 参数、buffers、dropout、优化器及变换；7）pin/device/JEPA 三模式的公平性与混杂；8）物理单位反变换、缺失指标 count、null R²、seed 汇总与成本去重；9）批量 runner、Slurm dry run、checkpoint/provenance 与安装包。

先运行现有测试与合成 smoke，再设计有针对性的反例测试。有授权的真实 Opamp2 数据才做真实核对；它只有 32 样本和一个电路，不可声称真实 OOD。不要向第三方上传原始数据、运行真实集群任务或修改远程仓库。若可以改本地代码，先报告发现，再提供最小修复和测试结果；保留原始研究约定和原始数据。最终给出：阻塞性问题列表、非阻塞风险、已验证项目、无法验证项目、最小下一轮实验，以及是否足以启动正式多电路实验的有条件判断。

## 16 来源与可核对位置

项目源码与证据：README.md、docs/architecture.md、docs/data_schema.md、docs/review_order.md、scripts/README.md、THIRD_PARTY_NOTICES.md、verification/V2_RESULTS.md、verification/AUDIT_FIXES.md、verification/unittest_v2.log、verification/comparison/RESULTS.md 及 CSV。本文的代码说明以所锁定快照为准，后续修改应重新核对。

高层架构来源：https://github.com/xz-group/ZeroSim ，对应 model/circuitformer.py、model/modules.py、configs/config.yaml、scripts/preprocess_topology.py、utils/dataset_utils.py。项目说明记录其实现时的查阅状态，没有复制官方代码、权重或数据。许可状态请审查者重新核实，不把此前未见 LICENSE 视为永久法律结论。

ZeroSim 论文：Xiaomeng Yang 等，Zero-Shot Analog Circuit Evaluation with Unified Transformer Embeddings，ICCAD 2025，DOI https://doi.org/10.1109/ICCAD66269.2025.11240890 。anti-collapse 动机参考 VICReg：https://arxiv.org/abs/2105.04906 。脚本组织参考：https://github.com/SetsunaFSeiei21/Analog-Agent-NN-Memory/tree/main/analog-agent/scripts 。这些来源说明实现背景，不构成本项目实验有效性的证据。
