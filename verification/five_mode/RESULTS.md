# Five-mode mechanics verification

**Status: final mechanics checks passed.** The integration owner ran the full smoke script (tests plus synthetic five-mode subprocess workflow), freshly imported the verified real source, and completed its five-mode subprocess workflow. After the final report-only provenance repair, all 15 evaluation reports were regenerated from the same saved checkpoints and every prediction was verified unchanged before strict reaggregation. The final full suite then passed **241 tests**. `run_manifest.json` records the exact final code/test/configuration hashes and runtime; `test_results.txt` retains the condensed full-suite evidence.

This is a **single training seed (17), one epoch per phase** mechanics check, not a formal multi-seed accuracy study. No result here establishes scientific superiority or cross-circuit generalization. The real Opamp2 dataset has only 32 samples, with three held-out test samples and no OOD circuit.

## Completed final runs

| Dataset | Train / validation / test / OOD | Labeled train | Modes | Summary rows | Paired rows |
|---|---|---:|---:|---:|---:|
| Synthetic, 24 samples × 5 circuits | 80 / 8 / 8 / 24 | 80 | 5 | 140 | 56 |
| Imported Opamp2, 32 samples | 26 / 3 / 3 / 0 | 26 | 5 | 180 | 72 |

All five modes completed training, checkpoint reload and every available held-out evaluation. Strict comparison returned `complete: true`. All five shared the same partition, labeled train IDs, target-scaler digest, design-input-transform digest (including input contract/category vocabulary), and continuous-OP-scaler digest. Supervised epoch, optimizer settings, training seed and label budgets matched. Pin/device OP/no-OP pairs used identical model hyperparameters. The OP input projections increase parameter count; these are not parameter-count-matched or equal-information/equal-total-cost comparisons.

The imported source archive was restored and verified against SHA-256 `54e16b0e3bb6ffc33e19efc5b01428e83bd672d2faa925e152223e31a6141edc`. No source dataset, per-sample prediction records, checkpoints or private raw logs are committed here; only aggregate metrics/costs are included.

## Aggregate files

For each `synthetic_` / `opamp2_smoke_` prefix:

- `summary.csv`: physical and train-normalized metric scores by mode/split/circuit/metric.
- `per_seed.csv`: the same aggregate metric scores with explicit training seed.
- `paired_per_seed.csv`: within-seed OP-minus-no-OP contrasts, separately for pin and device nodes.
- `paired_summary.csv`: paired contrasts summarized across seeds; sample SD is blank with one seed.
- `costs.csv`: measured model training phases and parameter counts, with explicit OP information requirements. A training cost repeated across evaluation splits must not be added twice.

Cost rows retain the OP source, selected continuous fields, categorical-state protocol (JEPA only), eligibility and split-specific observed coverage. Structured CSV cells are JSON-encoded. Both OP-input modes have matching original-device coverage: synthetic test 90/90 OP values and OOD 284/288; real Opamp2 test 1845/1845 values on 45 eligible original devices. No evaluated sample in these smoke runs lacked all observed OP. Pin counts are not inflated by repeated terminal tokens; these coverage figures do not imply any measured DC acquisition cost.

Negative MAE/RMSE differences or positive R² differences favor the OP-input model **on that metric and split only**. Undefined R² stays blank, not zero. There is no aggregate score across incompatible units and no significance claim. Both OP-input modes require continuous DC OP at inference; JEPA uses OP only in pretraining; the original baselines do not consume OP. DC acquisition/simulation cost remains `null`/blank and `not_measured`, rather than being manufactured or treated as zero.

## Reproduction commands

Use an installed compatible PyTorch/NumPy environment. Substitute existing source/input paths and use new output directories:

```bash
python -m device_jepa make_synthetic --output /private/synthetic.json --per-circuit 24
bash scripts/run/run_comparison.sh --data /private/synthetic.json --config configs/smoke.json --output /private/synthetic_smoke --seeds 17 --smoke
python -m device_jepa import_analog_agent --input /private/tmp_data/Opamp2 --config configs/opamp2_smoke.json --output /private/opamp2.json
bash scripts/run/run_comparison.sh --data /private/opamp2.json --config configs/opamp2_smoke.json --output /private/opamp2_smoke --seeds 17 --smoke
```

Default modes are `zerosim_pin`, `zerosim_device`, `jepa_zerosim`, `zerosim_pin_op`, `zerosim_device_op`. `--modes zerosim_pin zerosim_device jepa_zerosim` explicitly preserves the original three-mode run; standalone comparison also accepts legacy three-mode reports. Exact comparison/export commands and cost interpretation are documented in `scripts/README.md`.

## Final checks

- Independent counterexamples: **7 additional tests passed**, separately runnable with `python verification/five_mode/independent_probe.py`; results are in `independent_probe_results.txt`. These use analytic mixed-device fixtures, not real R/C sampling data. Review also identified and repaired legacy input-contract provenance during stage-one to decoder continuation; the repository regression and actual historical-code checkpoint probe preserve contract 1 and frozen weights.
- Complete suite: **241 tests passed** after all code and reporting repairs; exact test names and runner summary are retained in `test_results.txt`.
- Comparison: 24 tests passed, including complete five/legacy-three/explicit-pair cohorts, seed/split/data/scaler/label/budget guards, within-seed delta arithmetic, unknown-cost handling, discrete-state rejection for continuous OP baselines, train-only OP scaler membership, OP coverage export, and CSV export/stale-pair cleanup.
- Orchestration: four tests passed with actual in-process train/evaluate commands, including default five-mode and explicit legacy-three-mode completion, overwrite refusal and failure propagation.
- Script checks: three tests passed, including all shell syntax checks and Slurm dry runs that never invoke `sbatch`.
- Packaging: the final version 0.3.0 wheel was rebuilt after reporting changes and installed into a clean target. Import resolved from that target, all four supervised modes and bundled default configuration were present. Wheel SHA-256: `0f1ed556db9feba0fde18d858e6eef1f95bdc7c31b0170ca8841df1d212bd421`. Runtime versions and wheel verification are recorded in `run_manifest.json`.
- Independent review found and fixed one provenance gap: different JEPA state-feature sets or state-mode/threshold configurations could previously be pooled across seeds while continuous OP scaling matched. The comparer now requires a stable per-mode OP protocol; regression tests cover the mismatch. Saved checkpoint state configuration is exposed in reports, and observed OP coverage is preserved in aggregate costs. These changes affect reporting/validation only; all final smoke predictions remained identical after report regeneration.

Pending scientific work: adequately sized real multi-circuit training/validation/test/OOD data, multiple training seeds, realistic training budgets, and separately measured OP acquisition and deployment costs.
