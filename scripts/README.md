# Three-mode matched experiments

This directory follows the reference repository's `analog-agent/scripts/{git,smoke_test,run}` organization:
https://github.com/SetsunaFSeiei21/Analog-Agent-NN-Memory/tree/main/analog-agent/scripts

It does **not** run ngspice or resample circuits. Import existing simulation histories first. Modes:

- `zerosim_pin`: independently implemented pin-node ZeroSim-style reference, **not official code or an exact original-paper reproduction**.
- `zerosim_device`: supervised device-node reference, trained from scratch.
- `jepa_zerosim`: device/OP alignment pretraining, then frozen structure encoder and supervised decoder.

## Quick CPU mechanics test

From the repository root:

```bash
bash scripts/smoke_test/run_smoke.sh
```

This runs unit tests, creates explicitly synthetic data, runs each phase for one epoch, then exports matched comparisons. The old `scripts/run_smoke.sh` remains a JEPA-only compatibility entrypoint.

## Actual imported data

```bash
python -m device_jepa import_analog_agent --input /path/to/opamp2 --config configs/opamp2_smoke.json --output runs/opamp2.json
bash scripts/run/run_comparison.sh --data runs/opamp2.json --config configs/opamp2_smoke.json --output runs/opamp2_compare --seeds 17 --smoke
```

For a genuine multi-circuit experiment, import all circuit directories together and set `dataset.train_circuits` and `dataset.ood_circuits` explicitly in a copied config. The default config names synthetic circuits, so do not silently use it on real data. Keep `split_seed` fixed while varying training seeds:

```bash
bash scripts/run/run_comparison.sh --data runs/real_data.json --config configs/my_experiment.json --output runs/real_compare --seeds 17 29 43 --device cuda
```

Each new experiment requires a new/empty output directory. One immutable `manifest.json` fixes train/validation/test/OOD membership and `labeled_train` for all methods and seeds. `--smoke` caps pretraining, baseline and decoder phases to one epoch. An existing result is never silently overwritten or treated as a successful resumed run. To resume selectively, use individual `device_jepa` commands into separate output directories, then explicitly compare their reports.

## Slurm

```bash
# Preview only, without sbatch or directory creation:
SLURM_ACCOUNT=b_phzhwu SLURM_PARTITION=ex01A800 bash scripts/run/submit_comparison.sh --data /absolute/data.json --config /absolute/config.json --output /absolute/new_results --device cuda

# Submit only when ready on your cluster:
DRY_RUN=0 SLURM_ACCOUNT=b_phzhwu SLURM_PARTITION=ex01A800 JEPA_CONDA_ENV=newbase bash scripts/run/submit_comparison.sh --data /absolute/data.json --config /absolute/config.json --output /absolute/new_results --device cuda
```

`SLURM_CPUS=8`, `SLURM_GPUS=1`, `SLURM_MEM=80G`, `SLURM_TIME=10-00:00:00` mirror the reference. Override them for your workload. Account, partition and mail are environment parameters, not embedded personal settings. `JEPA_CONDA_HOME` is optional; `JEPA_CONDA_ENV` unset inherits the current environment. `JEPA_PYTHON` selects the interpreter. `scripts/smoke_test/submit_comparison.sh` adds `--smoke` and defaults to 30 minutes. All submission scripts default to `DRY_RUN=1`. GPU allocation does not automatically select CUDA: pass `--device cuda` explicitly. Relative data/config/output paths in submitted jobs resolve from the project root; absolute paths are safest.

## Read and independently regenerate results

```bash
python -m device_jepa.comparison --reports runs/real_compare/*/seed_*/*_report.json --output runs/reprocessed
```

- `comparison/summary.csv`: grouped by method/backend, split, circuit, metric and scale; mean and sample standard deviation (`ddof=1`) across seeds.
- `comparison/per_seed.csv`: counts, MAE, RMSE and R² for each seed; `__all__` pools samples across circuits **only within the same metric and split**. Circuit-specific rows prevent a large circuit from hiding smaller-circuit errors.
- `comparison/costs.csv`: phase timings/parameter counts; the same training cost is repeated for each evaluation split and must not be summed twice.
- `comparison/comparison.json`: complete machine-readable tables.

Strict comparison rejects incompatible dataset/performance identity, OP/input identity, partition, labeled training IDs, target scaler, metric names, evaluated IDs/targets, or incomplete seed sets across modes. Supervised epoch count, learning rate, batch size, weight decay and gradient clipping must match; the full training configuration must remain constant within a mode across seeds (except the seed itself). This is label/epoch-budget parity, not equal compute: JEPA pretraining cost is reported separately. It does not compare entire feature preprocessing hashes because pin/device input spaces differ. `--allow-partial` permits missing modes/seeds for debugging only and retains identity checks.

Physical metrics use original dataset units. Normalized metrics use training-only target transforms and scaling, not division by test-set variance. Missing labels are excluded per metric; nonfinite predictions for observed targets fail loudly. R² is null for fewer than two observed targets or zero target variance; JSON null/blank CSV is not zero. Across-seed SD is blank for a single seed. No single grand score averages incompatible physical units. Validation, test and OOD remain separate; OOD is never a checkpoint-selection set.

A 32-point one-circuit result, including negative or high R², is **only a pipeline check**, not evidence of accuracy or generalization. Use sufficiently large multi-circuit holdouts and multiple seeds for scientific comparisons. This package does not make superiority claims from smoke numbers.

## Safe Git helper

Explicitly stage reviewed files with `git add <paths>`, inspect the staged diff, and run `bash scripts/git/push.sh` for a preview. Set `PUSH_CONFIRM=1` only when you intend to commit/push. Optional arguments supply the commit message, otherwise `scripts/git/commit_message.txt` is used. The helper requires a dedicated project checkout and expected origin; it does not automatically stage unrelated files, reset, rebase, pull, force-push, or store credentials.

## Suggested review order

1. Dataset adapter and manifest validation; verify no group leakage and correct label budget.
2. Train-only normalization and NaN masks.
3. Pin/device model representations, JEPA shared structural path, exact encoder freezing.
4. Training/checkpoint selection and provenance/timing.
5. Evaluation records and `device_jepa/comparison.py` guardrails.
6. `scripts/run/run_comparison.py`, shell dry runs, then synthetic and real-data smoke outputs.

## Verified reference-cluster mapping

The source was actually read from `analog-agent/scripts/run/submit_sampling.sh`, `run/slurm_run.sh`, `run/run_sampling.sh`, `smoke_test/submit_sampling.sh` and `git/push.sh`, rather than inferred from generic Slurm examples.

Observed configuration:

- Account `b_phzhwu`; partition `ex01A800`; **no QoS option** in the inspected scripts.
- One node, one task, one task per node; 8 CPUs per task; `--gres=gpu:1`; `--mem=80G`; `--time=10-00:00:00`.
- `--parsable`, `--export=ALL`, output/error files containing `%j`, mail type `BEGIN,END,FAIL`; the reference embeds a personal mail recipient, which this project deliberately does not copy. Set `SLURM_MAIL_USER` explicitly.
- Conda environment `newbase`; activation script searches `/share/software/anaconda3/etc/profile.d/conda.sh`, then `conda info --base`. The reference top-level `CONDA_HOME` is empty and the wrapper supplies that fallback.
- Submit script resolves project root from `../..`, exports the root, and the job wrapper uses that root or `SLURM_SUBMIT_DIR`, changes directory, and exports `PYTHONPATH`. This project uses the same pattern with `CIRCUIT_JEPA_ROOT` and additionally passes `sbatch --chdir` explicitly.
- Reference formal-run logs: `hpc_logs/run/YYYYMMDD`; smoke logs: `hpc_logs/smoke_test/<timestamp-pid-random>`. This project uses timestamp/pid subdirectories for both to avoid collisions.
- Reference submit scripts default `DRY_RUN=0`; this project intentionally defaults to `DRY_RUN=1` to prevent accidental resource allocation. Reference smoke resource duration is also ten days; our smoke helper uses 30 minutes unless overridden, while the formal default remains ten days.

To load the exact account/partition/resource/Conda preset before previewing a job:

```bash
source scripts/run/cluster.env.example
bash scripts/run/submit_comparison.sh --data /absolute/data.json --config /absolute/config.json --output /absolute/new_results --device cuda
```

This preset contains no credentials. Account/partition identify the scheduler allocation and queue, not login secrets. No cluster submission was executed during implementation/testing; only shell syntax and dry-run command generation were exercised.
