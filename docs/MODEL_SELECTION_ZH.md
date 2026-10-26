# 按模型选择训练与评估（0.4.4）

`--models` 指定本次实验唯一的模型名单。原始数据入口、canonical 数据入口、Slurm 提交、smoke 和结果汇总共享模型注册表 `device_jepa/methods.py`。旧参数 `--modes` 是兼容别名。

## 在脚本中选择

```bash
# 只修改这一行即可选择本次模型，也可以只写一个。
MODELS=(jepa_zerosim zerosim_device)

source scripts/run/cluster.env.example
export JEPA_RESULTS_ROOT="$PWD/runs/gpu_full"
export JEPA_LOG_DIR="$PWD/hpc_logs/gpu_full"
DRY_RUN=0 bash scripts/run/submit_raw_comparison.sh \
  --input "$PWD/runs/raw/full_data/two_stage_opamp_otaf" \
  --config "$PWD/configs/default.json" \
  --models "${MODELS[@]}" \
  --max-table-rows none --seeds 17 29 43 --device cuda
```

示例使用默认模型大小与训练参数；使用自己的训练配置时替换 `--config`。
原始数据入口仍只做指定电路的独立实验，自动设置 `ood_circuits=[]`。
上述名单只执行 JEPA 的预训练和 decoder，以及 `zerosim_device` 的监督训练；每个 seed 只评估这两个模型的最佳验证 checkpoint。未选择的模型不会创建训练目录、checkpoint 或评估报告。

结果继续保存到原仓库 `runs/gpu_full/电路名_YYYYMMDD_HHMMSS/`。`config.json` 记录解析后的 `experiment.models`；内部 `results/experiment.json` 记录模型、种子、评估划分和是否完整完成。汇总在 `results/comparison/`。

canonical 数据使用相同选项：

```bash
bash scripts/run/submit_comparison.sh \
  --data /absolute/data.json --config /absolute/config.json \
  --output "$PWD/runs/gpu_full/joint_$(date +%Y%m%d_%H%M%S)" \
  --models jepa_zerosim zerosim_device --device cuda
```

这个入口按配置中的 ID/OOD 集合执行；模型选择不会改变数据划分。Slurm 默认 `DRY_RUN=1`，正式提交时明确设置 `DRY_RUN=0`。

## 模型名与配置

```bash
bash scripts/run/submit_raw_comparison.sh --list-models
# 或安装包入口：
python -m device_jepa --list-models
```

| 模型名 | 训练流程 | 性能推理是否需要 OP |
|---|---|---|
| `zerosim_pin` | pin 节点监督训练 | 否 |
| `zerosim_device` | device 节点监督训练 | 否 |
| `jepa_zerosim` | OP 对齐预训练，再冻结 encoder 训练 decoder | 否 |
| `zerosim_pin_op` | pin 节点监督训练，输入连续 OP | 是 |
| `zerosim_device_op` | device 节点监督训练，输入连续 OP | 是 |

长期使用同一名单时，在训练 JSON 中添加顶层字段（`model` 仍用于设置网络结构参数）：

```json
{
  "experiment": {
    "models": ["jepa_zerosim", "zerosim_device"]
  }
}
```

优先级是 **命令行 `--models` > `experiment.models` > 兼容默认名单**。当前 `configs/default.json` 明确列出原有五个模型；旧配置缺少该字段时，比较脚本保持原有五个模型。新增注册项不会扩展这份默认名单。只有显式使用 `--models all` 才选择当前注册的全部模型。

空名单、重复模型、未知名称、同时重复使用 `--models`/`--modes` 会报错；不会悄悄退回全模型训练。CLI 名称检查在参数解析时进行，配置名单在数据导入和训练前检查。

## smoke 与结果汇总

```bash
# 比较脚本的短流程；可选择一个、多个或 all。
bash scripts/smoke_test/run_smoke.sh runs/selected_smoke \
  --models zerosim_device zerosim_device_op

# 安装包也支持选择；未指定名单且配置未设置时，保留原来的 JEPA-only 默认。
python -m device_jepa smoke --output runs/single_smoke --models zerosim_device
python -m device_jepa smoke_real --input /path/to/raw \
  --config configs/opamp2_smoke.json --output runs/raw_smoke --models jepa_zerosim

# 重新汇总时声明同一份名单，并只传这些模型的报告。
python -m device_jepa.comparison --reports /path/to/selected/reports/*.json \
  --output /path/to/new_summary --models jepa_zerosim zerosim_device
```

smoke 使用 JEPA 1+1、基线 2 个 epoch，并保留全部输入样本；它不会自动抽样大型原始数据。合成 smoke 脚本默认不再先运行整套单元测试；需要开发回归时使用 `scripts/smoke_test/run_smoke.sh --run-tests` 或单独运行 unittest。

汇总只要求选中模型具有一致的种子与评估样本，单模型也可完整成功。OP/no-OP 成对表仅在对应两种模型都选中时生成。`complete` 仍要求严格 provenance；未选择的模型不计为缺失，选择之外的报告也不会混入结果。

最大 epoch、公平性校验、早停、训练集拟合变换和验证集最佳 checkpoint 规则保持不变。选中 JEPA 和基线时仍须满足 `pretrain_epochs + decoder_epochs == baseline_epochs`。

## 后续扩展

新模型的名称、流程、节点类型、OP 需求和配对关系统一登记到 `device_jepa/methods.py`，并接入相应训练、checkpoint 重建和报告身份实现。相应测试应验证“仅选此模型”和“与现有模型组合”。新架构或结构消融的比较协议需要显式定义，不能通过把所有未知名称当作现有基线来运行。

这些改动只影响后续启动的实验。已提交的作业和已有结果不会因修改选择参数而自动变成另一组模型。
