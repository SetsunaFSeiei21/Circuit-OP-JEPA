# GPU 常驻数据路径与进度

默认训练和评估入口使用 `device_resident_v1`。在 Slurm 任务中传入 `--device cuda`，缓存与数值运算就位于分配的 GPU。原始 SQLite/JSON 解析、网表字符串解析、完整性摘要、控制逻辑、checkpoint 写入以及最终物理指标报表仍由 CPU 处理。

## 数据准备和计算

- 原始参数、OP 和缺失标志一次性解码为数值数组并上传。训练集统计量在 CUDA float64 下拟合，仍严格限定为 ID train；性能统计仅使用 labeled_train。
- signed asinh、z-score 和物理值逆变换在 GPU 以 float64 计算，训练输入随后转换成 float32。缺失值保持掩码；不参与选模的指标在归一化和 float32 转换前排除。
- 根据器件/模型/命名端口/连接识别拓扑。每种拓扑只缓存一份图；邻接和方向端口关系使用设备上的张量操作构建，避免为每个 sizing point 重复构建和存储稠密图。
- pin 模式先存原始器件的参数与 OP，取 batch 时在 GPU 展开成 pin 输入，避免在整个数据集上重复存四份 MOS OP。
- 每个 epoch 按原来的 seed 与 Python 随机排列顺序选样本，在 GPU 上索引常驻表。训练循环不再调用 CPU 标量 collator，也不再逐 batch 复制输入张量到 GPU。
- JEPA decoder 阶段使用冻结、eval encoder 的特征缓存。encoder 只对训练/验证样本计算一次；之后每轮 decoder 仍正常启用训练 dropout 和梯度。encoder 参数和 buffers 每轮仍做逐字节一致性校验。
- 每个训练阶段只计算一次完整数据身份摘要，后续最佳 checkpoint 复用这一份已经验证的来源元数据。各次独立训练仍有独立 run identity。

旧 collator 保留作输入契约的独立参考。旧 checkpoint 架构仍可重建；新报告记录实际 tensor pipeline，比较器拒绝把旧、新路径混作一次统一实验。

## 进度与资源

`training.progress_every` 默认 100，必须是正整数。进度输出包括第一步、间隔步和最后一步，适合重定向到 Slurm 日志：

```text
zerosim_pin data: preparing 54000 samples on cuda:0 (resident numeric cache)
zerosim_pin data: decoded 5000/54000 samples | 1 unique topologies
zerosim_pin data: ready | 1 topologies | ... MiB resident on cuda:0 | ...s
zerosim_pin epoch 1/80 train: 100/1500 (6.7%) loss=... elapsed=...s ETA=...s
```

这是格式示例。原始导入也会报告已解析样本数；阶段启动会报告校验、读取数据、拟合变换和冻结特征缓存。

缓存预检为训练激活和优化器保留显存余量。CUDA 不可用或容量不足会明确失败，不会静默改到 CPU 或缩减数据。多电路和不同拓扑仍可能需要更多显存，应以目标 GPU 的实际运行记录为准。CPU 资源申请仍是每个任务 8 核，GPU 为 1 张。

## 验证范围

`tests/test_resident.py` 比对新旧张量、图、掩码、命名端口、模型身份、未知字段、缺失/极大标签、物理尺度、拓扑去重、冻结特征复用、摘要计算次数和进度。CUDA 硬件测试仅在有 CUDA 时运行；CPU 通过不能代替 GPU 兼容性与端到端速度验证。统计归约保留确定性设置，兼容旧版 PyTorch 的 GPU 加权中位数采用固定加法树，不关闭确定性保护。

本地 batch 准备基准与真实 Opamp2 三种子五模式短流程的结果位于 `verification/resident_pipeline/results.json`。CPU batch 准备提速不代表 GPU 训练或完整 60,000 点实验的加速倍数。

## 更新和重跑

先停止需要替换的两个旧训练任务，再拉取新版本，使用新的输出目录重新提交。每个进程在启动时加载 Python 代码；正在运行的旧进程不会切换到新实现。统一代码版本也避免同一实验的后续子进程读到不同实现。

```bash
git -c http.version=HTTP/1.1 pull --ff-only origin main
source scripts/run/cluster.env.example
JEPA_RUN_ID=$(date +%Y%m%d_%H%M%S)
for circuit in 5t_ota Opamp0; do
  DRY_RUN=0 bash scripts/run/submit_raw_comparison.sh \
    --input "runs/raw/full_data/$circuit" \
    --config configs/default.json \
    --output "runs/gpu_resident_${JEPA_RUN_ID}/$circuit" \
    --max-table-rows none --seeds 17 29 43 --device cuda
done
```

GPU 数据路径保持配置中的 batch size、模型结构、学习率、划分和最大 epoch 预算。0.4.3 另行加入可配置的验证集早停；启用后，每个模型或 JEPA 阶段可能在上限前停止，并记录实际执行成本，详见 [早停说明](EARLY_STOPPING_ZH.md)。
