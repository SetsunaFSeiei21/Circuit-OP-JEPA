# Five-mode matched experiments

This directory follows the reference repository's `analog-agent/scripts/{git,smoke_test,run}` organization:
https://github.com/SetsunaFSeiei21/Analog-Agent-NN-Memory/tree/main/analog-agent/scripts

It does **not** run ngspice or resample circuits. Import existing simulation histories first. Modes:

- `zerosim_pin`: independently implemented pin-node ZeroSim-style reference, **not official code or an exact original-paper reproduction**.
- `zerosim_device`: supervised device-node reference, trained from scratch.
- `jepa_zerosim`: device/OP alignment pretraining, then frozen structure encoder and supervised decoder.
- `zerosim_pin_op`: the matched pin-node supervised reference with **continuous DC OP values and missing/applicability masks as additional inputs**.
- `zerosim_device_op`: the matched device-node supervised reference with the same additional continuous DC OP inputs.

The existing design inputs are preserved, not restricted to W/L/M: configured named MOS, R/C and source parameters remain in the representation with their masks. The new models do not consume discrete OP-state categories. `zerosim_pin_op` and `zerosim_device_op` **require DC OP at inference**; they are information-augmented ablations, not zero-simulation deployment alternatives at equal total cost. The original two baselines do not use OP as model input. JEPA uses OP only during pretraining and not performance inference.

## Resident CUDA data and progress

With `--device cuda`, all five methods keep numeric data and reusable graph inputs on the selected GPU. Train-only statistics, signed normalization, graph relations, pin feature expansion, minibatch gathering, forward/backward passes and evaluation inversion run on CUDA. Input file/string parsing, provenance hashes, checkpoint writes and final report statistics remain host operations. GPU unavailability or insufficient resident-cache memory fails explicitly without CPU fallback.

The JEPA decoder uses cached features from its frozen eval encoder; later epochs execute the decoder only. Input/OP hashes are computed once per training stage instead of on every best-checkpoint save. `training.progress_every` (default `100`) emits first/interval/last batch progress, loss, elapsed time and ETA in Slurm-friendly lines. Data preparation reports decoded samples, unique topology count and resident bytes. See [GPU pipeline details](../docs/GPU_PIPELINE_ZH.md).

Stop the old tasks before updating and relaunch into new output directories. New reports record `tensor_pipeline=device_resident_v1`; comparison rejects mixed old/new pipelines. Epoch budgets and label/split contracts are unchanged. CUDA hardware and the full dataset capacity still require validation on the target cluster; CPU mechanics tests do not measure CUDA throughput.

## Quick CPU mechanics test

From the repository root:

```bash
bash scripts/smoke_test/run_smoke.sh
```

This runs unit tests, creates explicitly synthetic data, runs JEPA for one pretraining epoch plus one decoder epoch and each of the four baselines for two epochs, then exports matched comparisons and within-seed OP/no-OP deltas. The old `scripts/run_smoke.sh` remains a JEPA-only compatibility entrypoint.

## Actual imported data

```bash
python -m device_jepa import_analog_agent --input /path/to/opamp2 --config configs/opamp2_smoke.json --output runs/opamp2.json
bash scripts/run/run_comparison.sh --data runs/opamp2.json --config configs/opamp2_smoke.json --output runs/opamp2_compare --seeds 17 --smoke
```

For a genuine multi-circuit experiment, import all circuit directories together and set `dataset.train_circuits` and `dataset.ood_circuits` explicitly in a copied config. The default config names synthetic circuits, so do not silently use it on real data. Keep `split_seed` fixed while varying training seeds:

```bash
bash scripts/run/run_comparison.sh --data runs/real_data.json --config configs/my_experiment.json --output runs/real_compare --seeds 17 29 43 --device cuda
```

Each new experiment requires a new/empty output directory. One immutable `manifest.json` fixes train/validation/test/OOD membership and `labeled_train` for all methods and seeds. `--smoke` sets JEPA pretraining and decoder training to one epoch each and baseline training to two epochs, preserving equal total epochs. Direct `pretrain`, `train_decoder` and `train_baseline` commands also reject duplicate stage artifacts or stale reports; they are not resume APIs. Stage two may use its matching stage-one output directory, including a verified labels-only dataset update. Run replacement stages in new directories, then explicitly compare their reports.

Training budgets are explicit, separate fields under `training` in the JSON config:

```json
{
  "pretrain_epochs": 30,
  "decoder_epochs": 50,
  "baseline_epochs": 80
}
```

`decoder_epochs` controls only the frozen-encoder JEPA decoder. `baseline_epochs` controls all four non-JEPA models and is required for new baseline runs; it does not inherit the JEPA decoder value. Matched runs containing JEPA and baselines require `pretrain_epochs + decoder_epochs == baseline_epochs` before creating output files or starting training. The raw-data runner checks this before import as well. Update copied experiment configs explicitly when changing either JEPA stage. The shipped `smoke.json` uses 2 + 3 versus 5 epochs; `opamp2_smoke.json` uses 1 + 1 versus 2. Passing `--smoke` overrides either config to 1 + 1 versus 2. The default config enables training.early_stopping with patience=10, min_delta=0.0 and min_epochs=1; all five models and both JEPA stages stop independently using validation loss and retain the exact best checkpoint. The supervised stopping policy must match across models. Test/OOD never control stopping. Old custom configs without early_stopping remain fixed-epoch; add the object explicitly to enable it. Actual epochs can differ under equal maxima; see [early stopping](../docs/EARLY_STOPPING_ZH.md). Equal epochs do not imply equal optimizer steps when labeled training is a subset, or equal runtime/FLOPs across architectures.

The standalone `python -m device_jepa smoke` and `smoke_real` entrypoints also require a new/empty output directory. They check this before importing data, creating a manifest or writing `data.json`; a rejected pre-existing directory is left unchanged, including when it has no manifest.

All five modes run by default. Original three-mode experiments remain supported, and explicit pairs can be run for a narrower ablation:

```bash
# Original three-mode experiment (same dataset/config/output requirements):
bash scripts/run/run_comparison.sh --data runs/real_data.json --config configs/my_experiment.json --output runs/legacy_compare --modes zerosim_pin zerosim_device jepa_zerosim
# Device-node OP/no-OP pair only:
bash scripts/run/run_comparison.sh --data runs/real_data.json --config configs/my_experiment.json --output runs/device_pair --modes zerosim_device zerosim_device_op
```

## Slurm

For a large raw Analog-Agent dataset, import and train on the compute node in
one allocation. The single-circuit entrypoint reads the circuit identity from
SQLite, copies the base config, sets that circuit as the only ID circuit with
no OOD, and reuses the existing matched five-mode runner. The default base
config allows at most 30 JEPA pretraining epochs plus 50 JEPA decoder epochs, and 80
epochs for each non-JEPA baseline, with validation early stopping, with full train-label availability. Every
seed uses one fixed design-group partition.

Before submission, a small read-only geometry preflight can identify missing
unit declarations or stale source netlists without constructing canonical JSON:

```bash
python scripts/run/check_raw_geometry.py \
  --input runs/raw/full_data/5t_ota runs/raw/full_data/Opamp0 \
  --max-samples 100
```

It resolves sample design parameters and compares W/L in meters with source
`w_um/l_um` converted from micrometers, and compares multiplicity separately.
Reported counts cover only available geometry; missing values remain missing.
Mismatch examples include both physical values and their ratio. No file is
modified and no scale is guessed. `consistent_observed_geometry` means the
observed fields in the stated scope agree; `unverifiable`, `mismatch` or a source
error produce a nonzero exit code. Omit `--max-samples` to stream all records.
A 100-sample success is a spot check: full import still strictly checks all
observed geometry before any training starts.

```bash
source scripts/run/cluster.env.example
RUN_ID=$(date +%Y%m%d_%H%M%S)
for circuit in 5t_ota Opamp0; do
  DRY_RUN=0 bash scripts/run/submit_raw_comparison.sh \
    --input "runs/raw/full_data/$circuit" \
    --config configs/default.json \
    --output "runs/independent_${RUN_ID}/$circuit" \
    --max-table-rows none \
    --seeds 17 29 43 \
    --device cuda
done
```

This submits two independent jobs (the example cluster preset requests one GPU
and 80 GB host RAM per job). Use `DRY_RUN=1` to preview instead. Each job writes
`config.json` and `data.json` at its output root; trained checkpoints, split and
comparison tables are under `results/`, with tables in `results/comparison/`.
The output must be new/empty, and raw files are read only. Import failures stop
the job before training. Logs are under `hpc_logs/raw_run/`.

This workflow does not subsample the source. `--smoke` uses one JEPA
pretraining epoch plus one decoder epoch and two epochs per baseline while
importing all source samples. Explicitly raising or
disabling the source row limit does not provide out-of-core storage: canonical
objects, state encoding and JSON still require dataset-sized memory. Capacity
for the user's full 60,000-point sources must be measured in their jobs.

Single-circuit results are ID experiments. For later joint training with an
OOD circuit, import the chosen source directories together and use the original
`submit_comparison.sh` with an explicit multi-circuit dataset config. The OOD
circuit must be excluded from pretraining, fitted transforms, supervised
training and checkpoint selection.

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
- `comparison/per_seed.csv`: counts, MAE, RMSE and R² for each seed. `scope=pooled` pools samples across circuits **only within the same metric and split**; `scope=circuit` retains individual circuits, including a circuit literally named `__all__`.
- `comparison/costs.csv`: one training-cost row per verified run identity, with phase timings, optimizer steps and parameter counts; repeated test/OOD reports do not duplicate training cost. Conflicting costs for the same run are rejected.
- `comparison/evaluation_costs.csv`: split-specific OP coverage and information requirements.
- `comparison/op_acquisition_costs.csv`: separately scoped measured OP acquisition costs, when present.
- `comparison/paired_per_seed.csv`: matched seed, split, circuit, metric and scale for `zerosim_pin_op − zerosim_pin` and `zerosim_device_op − zerosim_device`; includes both scores and their differences.
- `comparison/paired_summary.csv`: means and sample SDs of those **within-seed differences**, not subtraction of independently averaged runs. Negative MAE/RMSE deltas or positive R² deltas favor the extra OP input. Undefined R² remains blank/null. Pair files are omitted when no complete pair exists (e.g. legacy three-mode runs).
- `comparison/comparison.json`: complete machine-readable tables.

Summary/per-seed rows expose `requires_op_at_inference` and `op_training_use`; cost tables additionally expose `op_inference_use`, OP source/features/state configuration/eligibility and separate evaluation coverage and acquisition scope. Structured CSV fields are JSON-encoded. OP coverage counts original devices, not repeated pin tokens, and records samples without observed OP; missing legacy coverage stays unknown. Imported OP acquisition/simulation cost is **unknown (`null`/blank, `not_measured`)**, never silently zero. Recorded model phase timings do not include an unmeasured DC solve or imply equal end-to-end cost. Parameter counts also may increase when OP projection inputs are added. Any future measured acquisition cost must include its measurement scope and remain separate from model phase timing.

Strict comparison rejects incompatible dataset/performance identity, OP/input identity, partition, labeled training IDs, target scaler, continuous-OP input scaler, metric names, evaluated IDs/targets, or incomplete seed sets across declared modes. Each OP/no-OP pair must have identical model hyperparameters, in addition to the existing device/JEPA architecture check. OP source, continuous feature names, state-feature channels, state configuration (including available mode/thresholds), and eligibility must remain constant within a mode across seeds/splits. JEPA reports preserve actual source pretraining options and both stage seeds; their pretraining protocol digest includes the actual loss and pretraining budget, independently of stage-two options. Maximum total epoch count must match: JEPA uses the pretraining epoch count from its immutable executed source provenance plus its decoder epoch count; baselines use their independent `baseline_epochs`. The stage-two configuration cannot replace the actual pretraining budget, and the selected best-checkpoint epoch is not the number of epochs executed. Supervised learning rate, batch size, weight decay and gradient clipping must still match; the full training configuration must remain constant within a mode across seeds (except the seed itself). This is label and total-epoch-budget parity, not equal compute or equal available information: JEPA phase costs are reported separately. It does not compare entire feature preprocessing hashes because pin/device input spaces differ. Standalone comparison can aggregate legacy three-mode reports, but missing run identity, JEPA pretraining provenance or explicit baseline epoch provenance sets `strict_provenance=false` and `complete=false`, with explicit gaps even when `cohort_complete=true`. If any OP-input mode is present, all five are expected unless `--modes` declares another cohort. `--allow-partial` permits missing modes/seeds for debugging only and retains identity checks; paired rows use only the actual seed intersection. Comparison output schema is now version 5. Planned and executed epoch counts are separate; independent early-stopping outcomes may differ. Known unequal maximum total epoch budgets are rejected, including in partial comparisons; historical comparisons using equal supervised epochs are not evidence of equal total epochs.

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
