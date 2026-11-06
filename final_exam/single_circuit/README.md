# 单电路实验

编辑 `train.sh` 顶部的 `CIRCUITS`、`MODELS`、`SEEDS`、`CONFIG` 和集群设置。每个电路提交一个独立 Slurm 作业，作业内依次训练所选模型和 seed。

```bash
bash final_exam/single_circuit/train.sh --dry-run
bash final_exam/single_circuit/train.sh
```

使用 `--smoke` 缩短训练预算，可与 `--dry-run` 合用。烟测仍导入完整原始数据，原始数据较大时建议先使用小样本电路。原始输入位于 `runs/raw/full_data/<circuit>/`，目录须包含 `sampling_history.sqlite3` 和对应网表。

所有输出位于当前仓库的 `final_exam/single_circuit`，每次提交生成同一个 UTC 时间与随机后缀组成的 `run_id`。

| 文件 | 路径 |
| --- | --- |
| Slurm 标准输出和错误 | `hpc_logs/<run_id>/<job_id>.out`、`<job_id>.err` |
| 提交记录 | `hpc_logs/<run_id>/submission.txt` |
| 提交时配置快照 | `hpc_logs/<run_id>/submitted_config.json` |
| 导入、命令及启动日志 | `logs/<run_id>/launch.log` |
| 训练总日志及每阶段日志 | `logs/<run_id>/experiment.log` 及该目录的 seed 子目录 |
| 实验关联、状态与路径 | `results/<run_id>/run.json` |
| 实际导入配置及 canonical 数据 | `results/<run_id>/config.json`、`data.json` |
| 模型、逐 seed 指标和汇总 | `results/<run_id>/training/` |
| 自动生成的 Excel | `results/<run_id>/training/summary.xlsx` |

`train.sh` 验证所有输入目录后，为每个作业创建独立的 HPC 日志目录，保存提交时配置快照并提交。作业使用该快照，排队期间编辑配置模板不会改变已提交作业的配置。`run.sh` 在计算节点激活 Conda 环境，`launch.py` 检查 SQLite 电路身份与训练配置后导入数据，再启动模型比较。GPU 不可用、输入冲突或训练失败均返回错误。导入日志实时同时写入 Slurm 输出与 `launch.log`，stderr 保留在 Slurm 错误输出中。日志和结果目录出现冲突时拒绝覆盖。

可使用结果目录中的 `run.json` 查到同一次实验的作业号、节点、全部日志路径和训练状态。配置在开始训练前保存，实验完成需要训练状态为 `complete` 且 Excel 确实生成。

重新整理结果使用任务内入口。

```bash
bash final_exam/single_circuit/summarize.sh \
  final_exam/single_circuit/results/<run_id> \
  --output final_exam/single_circuit/results/<run_id>/summary_review.xlsx
```

默认资源为账户 `b_phzhwu`、分区 `ex01A800`、每作业 1 GPU、8 CPU、80G 主机内存，环境 `newbase`。正式实验时间上限默认 10 天，烟测为 30 分钟。可在 bash 顶部调整。数据、日志和模型权重被 Git 忽略，实验代码和配置模板进入版本管理。
