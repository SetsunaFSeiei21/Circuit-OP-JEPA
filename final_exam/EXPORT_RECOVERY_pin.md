# 端口级实验：最终 Excel 导出失败的恢复

## 触发问题

报告模块 `summarize_pin.py` 需要各版本门控模型的注册信息，包括 `V3_GATE_MODELS_PIN`。即使当前实验只选择 v2，通用报告模块也需要正确导入整套注册信息。

旧入口直到全部模型训练/评估结束后才导入报告模块。若计算节点上的长时间训练进程已经缓存旧注册表，而工作目录在这期间被 `git pull` 或 patch 更新，新报告代码就可能引用旧内存模块中不存在的名称，产生 `ImportError`。磁盘文件合并不完整也可能产生同样错误。仅凭 traceback 不能确定是哪一种；traceback 显示的源码行与实际调用不匹配，是运行期间文件变化的线索。

`code_commit=unavailable (FileNotFoundError)` 是计算节点没有可执行 Git 时的可选元数据警告，本身不阻断训练，与缺失模型注册名称不是同一个错误。

## 代码调整

1. `run_pin.py` 在启动时导入并保留报告函数，报告依赖不再延迟到训练结束才解析。导入不一致会在读取数据/训练前报错，并提示同步代码和重启进程。
2. `summarize_pin.py` 和对应 Bash 包装入口新增 `--recover-export`，仅处理 `training_pin/run_pin.json` 标记为 `failed_export` 的实验。
3. 恢复不绕过检查：必须包含原计划的全部模型/seed/split；指标、模型配置、去重阶段成本、门控前后报告继续经过原来的校验。缺失或不一致时不发布 Excel。
4. Excel 的 MAE/RMSE/R2 页注明 `export recovered; original job failed at export`，Configuration 页保留原始失败状态并说明没有重训。原训练结果、配置、checkpoint、日志、状态 JSON 都保持原样；原 Slurm 作业的失败历史也不会被改为成功。
5. 恢复只依赖 Python 标准库及仓库内的报告代码，不加载 Torch、数据集或 checkpoint。可在登录节点执行，无需重新申请 GPU。

## 使用方式

先确保仓库代码同步完成、没有未解决的合并冲突。对于已结束的失败作业，可以在仓库根目录启动新进程检查报告导入：

```bash
python -c 'from final_exam.summarize_pin import export_results_pin; print("Report import OK")'
```

然后指定原结果根目录（下面应有 `training_pin/`，也支持直接指定这个训练目录）：

```bash
SOURCE_RUN="/absolute/path/to/results_pin/<run_id_pin>"
bash final_exam/single_circuit_pin/summarize_pin.sh "$SOURCE_RUN" --recover-export
```

默认生成 `$SOURCE_RUN/training_pin/summary_pin.xlsx`。若这个 Excel 已存在，默认拒绝覆盖，可以通过 `--output /absolute/path/summary_recovered_pin.xlsx` 另存，或者明确加 `--overwrite` 替换这份派生报表。

`--recover-export` 不能与 `--allow-incomplete` 同时使用。如果状态为 `failed` 或缺少计划中的结果，需先通过训练恢复入口完成缺失部分。`--allow-incomplete` 只生成临时报表，不能把未完成实验声明为完成。

## 边界

提前导入修复了此次延迟导入的风险，但没有创建整个代码仓库的运行快照。仍应避免在排队/运行的作业共用的工作目录中更新 Python 源码。代码更新不会自动替换已运行进程里缓存的模块，也不能通过在旧进程中重载单个模块来保证所有训练代码一致。

导出恢复验证的是保存结果及报告的一致性，不能追溯证明一个运行期间被修改过代码的作业始终使用同一代码版本。正式重复实验应在固定代码版本下重新提交。

## 回归验证

`final_exam/tests/test_export_recovery_pin.py` 覆盖：新进程旧注册表导入失败、报告依赖提前加载、无 site-packages 的 CLI 恢复、缺失模型/seed/配置/成本/门控对照的拒绝、写文件失败的原子性，以及一个真实执行的小型 v2 训练在最终导出失败后不重训地恢复。恢复前后校验所有原始文件的 SHA-256，确保只新增 Excel。
