# v2 final candidate verification (2026-10-25)

- Full automated suite: **211 tests passed** on Python 3.12 / PyTorch 2.14.1+cpu. See `unittest_v2.log`.
- `git diff --check`, all shell `bash -n` checks and Python compilation passed.
- Version 0.2.0 wheel built successfully with bundled configurations and all new package modules.
- Actual-source integrity: all 21,120 MOS OP cells and all 288 metric positions match the supplied SQLite source; 23 metric missing values retained. Database hash unchanged; no feasibility filtering.
- Shared structure is a single registered module evaluated once, both branch gradients reach it, and parameter/OP/state dependency boundaries are tested.
- Stage-two encoder parameters, buffers, evaluation behavior and input/OP transforms remain fixed. Old checkpoints and conflicting normalization/state configs fail clearly.
- Post-review fixes: stable float64 physical evaluation with raw source truths; scale-stable RMSE/R²; strict encoded-state/config validation; source_id+circuit_id+run_id scoped run evidence; PyTorch >=2.6 loader gate and pre-CUDA deterministic workspace default. See [AUDIT_FIXES.md](AUDIT_FIXES.md).
- Default OP checks classify only evidenced groups: the supplied source yields 93 known saturation failures and no inferred passes. Unknown/unperformed checks are not converted to “normal.”
- Three-mode integration: synthetic five-family data with two seeds and separate test/OOD reports; real 32-sample Opamp2 with one seed and test-only reports. All training/evaluation/aggregation stages complete. After audit fixes, fresh real import and all three one-epoch training/evaluation/aggregation runs passed again; the existing synthetic two-seed checkpoints were re-evaluated and re-aggregated without extra training. See [comparison/RESULTS.md](comparison/RESULTS.md).
- Reports and comparison guards include input/performance/partition/target-scaler identity, evaluated targets, label IDs, supervised protocol and within-mode and cross-mode model configuration, plus shared parameter-transform identity.

These are bounded CPU mechanics checks. No long training, GPU validation, cluster submission or scientific superiority claim was made. One real circuit with a three-sample test split cannot establish meaningful OOD generalization or reliable method rankings. Actual input data and trained checkpoints are not committed; only aggregate verification tables are retained.

Commands:

```bash
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --no-build-isolation --no-index --wheel-dir /tmp/wheels
bash scripts/smoke_test/run_smoke.sh
```

The latest code/review map is [docs/review_order.md](../docs/review_order.md), with concrete real-data and multi-mode commands in the README and scripts/README.md.
