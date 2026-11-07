# 端口级单电路实验

本入口与器件级 `single_circuit/` 并列。每个被观测到的器件引脚是一个节点，命名外部端口也有独立节点；MOS 的 D/G/S/B 不合并成一个学习节点。图上保留同器件关系、同网络连接及端口接入关系。编码、输入权重和教师均使用端口级版本。

真实 OP 仍属于器件。OP predictor 对同一器件的引脚表示聚合，预测一组器件 OP，再广播到所属引脚；OP 监督按器件计算一次，不因 MOS 拆成四个引脚就重复加权四次。外部端口无所属器件，不参与 OP 监督。部署级联模型不读取真实 OP 或其缺失模式。

编辑 `train_pin.sh` 顶部的 `CIRCUITS`、`MODELS`、`SEEDS`、`CONFIG` 与集群设置。默认 Opamp1、13 个 `_pin` 模型、seeds 17/29/43，每个电路提交一个独立作业。作业内依次运行所选模型；仅为级联学生自动训练必需的教师与 OP predictor。

```bash
bash final_exam/single_circuit_pin/train_pin.sh --dry-run
bash final_exam/single_circuit_pin/train_pin.sh --smoke
bash final_exam/single_circuit_pin/train_pin.sh
```

`--dry-run` 不提交、不创建输出目录。`--smoke` 提交短轮次训练，但仍导入完整原始数据；不会把 60,000 条输入自动裁成小样本。原始输入继续使用 `runs/raw/full_data/<电路名>/`，无需复制或重命名 SQLite/网表。输入单位检查和器件级入口相同，不自动猜测几何单位。

模型列表为：

```bash
MODELS=(
  mlp_pin xgboost_pin gat_pin deepgen_supervised_pin deepgen_op_pretrained_pin gtn_pin
  zerosim_supervised_pin accformer_pin
  predicted_op_pin student_predicted_op_pin student_predicted_op_distill_pin
  op_oracle_pin design_op_oracle_pin
)
```

配置模板为 `final_exam/configs/default_pin.json`。`op_transfer` 管理训练预算、batch size、学习率及早停；`final_exam_pin` 管理模型选择、模型覆盖及 XGBoost 设置。虽然为兼容模型接口仍使用 `model.max_devices` 这个字段名，在端口级入口中它表示**引脚节点与独立外部端口节点总数上限，不含额外 graph token**；超限时报错，不截断。不要直接把器件数量当成端口级容量。

默认神经网络下游最大预算保持普通/联合模型 60 轮，顺序蒸馏 encoder 30 + decoder 30；教师 60 轮、OP predictor 30 轮作为额外依赖成本记录。每阶段验证集早停并恢复验证最佳权重。XGBoost 单独按 boosting rounds 记录，不冒充 epoch。

## 日志和结果

所有产物位于当前仓库 `final_exam/single_circuit_pin/`，每次使用以 `_pin` 结尾的唯一 run ID；人工传入的 run ID 也必须以 `_pin` 结尾。

| 内容 | 相对任务目录的路径 |
|---|---|
| Slurm 标准输出、错误 | `hpc_logs_pin/<run_id_pin>/<job_id>_pin.out`、`<job_id>_pin.err` |
| 提交配置快照、记录 | `hpc_logs_pin/<run_id_pin>/submitted_config_pin.json`、`submission_pin.txt` |
| 数据导入和命令日志 | `logs_pin/<run_id_pin>/launch_pin.log` |
| 实验总日志 | `logs_pin/<run_id_pin>/experiment_pin.log` |
| 各 seed/模型训练、评估日志 | `logs_pin/<run_id_pin>/seed_<seed>_pin/<model_pin>/train_pin.log`、`evaluate_pin.log` |
| 完整输入数据及启动配置 | `results_pin/<run_id_pin>/data_pin.json`、`config_pin.json` |
| 作业状态与路径关联 | `results_pin/<run_id_pin>/run_pin.json` |
| 模型结果与训练状态 | `results_pin/<run_id_pin>/training_pin/` |
| 自动生成 Excel | `results_pin/<run_id_pin>/training_pin/summary_pin.xlsx` |

共享教师与 predictor 保存在 `training_pin/seed_<seed>_pin/shared_pin/`；每模型独立保存 `config_pin.json`、`checkpoint_pin.pt`（XGBoost 为原生模型文件及 JSON）、`test_pin.json`。每阶段保存历史、成本和最佳检查点，名称均带 `_pin`。结果和日志不会覆盖器件级实验，也不提交到 Git。

脚本默认账户 `b_phzhwu`、分区 `ex01A800`、环境 `newbase`，每作业 8 CPU/1 GPU/80G 主机内存。正式作业最长 10 天，烟测 30 分钟。CUDA 不可用时报错；原始数据 IO、JSON 解析和静态图准备仍需 CPU，不能解释为整个实验没有 CPU 工作。

配置在提交时保存快照。正在排队或运行的作业不应共享一份不断修改的代码或原始数据；此入口没有复制代码/data 快照。所有选定模型与 seeds 完成且 Excel 成功生成，才标记实验 `complete`。

## 重新整理 Excel

```bash
bash final_exam/single_circuit_pin/summarize_pin.sh \
  final_exam/single_circuit_pin/results_pin/<run_id_pin> \
  --output final_exam/single_circuit_pin/results_pin/<run_id_pin>/summary_review_pin.xlsx
```

输出文件名必须以 `_pin.xlsx` 结尾。默认拒绝覆盖，明确指定 `--overwrite` 才替换。未完成实验默认拒绝导出；`--allow-incomplete` 可生成明确标注的临时表，并禁用最优高亮。

Excel 含 MAE、RMSE、R2、PerSeed、Statistics、TrainingCost、Configuration。跨 seeds 使用样本标准差；只有一个 seed 时留空标准差。真实 OP 参考与无需真实 OP 的模型分组比较；样本数或 seeds 不匹配时不高亮最优。每个模型名称保留 `_pin`，不与器件级结果混合统计。

本入口完成的是第一阶段单电路流程，不代表多电路/OOD、消融或迁移学习已经实现。
