# 多电路实验

此任务目录为下一阶段保留，用于多电路联合训练及 ID/OOD 测试。该阶段需要明确训练电路、验证划分和整电路留出协议后接入执行入口。

后续 bash 入口位于本目录，Slurm 输出放在 `hpc_logs/<run_id>/`，实验日志放在 `logs/<run_id>/`，配置、模型和 Excel 放在 `results/<run_id>/`。共享模型与数据实现从 `final_exam/models`、`final_exam/common` 导入。
