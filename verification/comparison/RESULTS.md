# Three-mode pipeline verification (2026-10-25)

These are **mechanics checks**, not accuracy or generalization claims. They do not establish any method is superior.

- Full suite: **211 tests passed**, including three-mode orchestration with automatic aggregation, failure propagation, split/label/scaler/config comparison guards, shell syntax and mocked Slurm dry-run tests.
- Synthetic: 120 toy samples (5 circuits), 2 training seeds (17, 29), common split/label budget, one epoch per phase, all three methods, separate test and OOD reports. 12 reports produce 84 grouped summary rows.
- Real Opamp2 fixture: 32 simulation-history samples, train/validation/test = 26/3/3, all three methods, seed 17, one epoch per phase, 9 metrics. The single-command runner completes training, evaluation and aggregation automatically. Three reports produce 108 grouped summary rows (per circuit and pooled, physical and normalized).
- There is **no real OOD holdout** in this one-circuit fixture. Tiny test counts make R² highly unstable; null R² is not zero. Physical units remain per metric; never average them into a grand score.
- The pin baseline is an independent clean-room pin-node implementation, not official ZeroSim code. The device baseline is fully supervised from scratch; JEPA decoder training uses a frozen pretrained encoder.
- CPU PyTorch 2.14.1+cpu. Slurm command generation was tested in dry-run mode only. No cluster jobs, full training, or GPU performance validation were run.

Artifacts here intentionally include only summary/cost CSVs, not input datasets, checkpoints or per-sample predictions. Cost measurements from this small CPU smoke are not representative of production hardware. Seed SD is blank for one seed. Training costs repeat per evaluation split in generic exports; do not count the same run twice.

Reproduce from the project root with installed dependencies:

```bash
bash scripts/smoke_test/run_smoke.sh
python -m device_jepa import_analog_agent --input /path/to/Opamp2 --config configs/opamp2_smoke.json --output runs/opamp2.json
bash scripts/run/run_comparison.sh --data runs/opamp2.json --config configs/opamp2_smoke.json --output runs/opamp2_comparison --seeds 17 --smoke
```

The runner writes `comparison/summary.csv`, `per_seed.csv`, `costs.csv`, `comparison.json` and a reading guide; no separate manual aggregation is needed. Refer to `scripts/README.md` for full multi-seed runs, exact reference cluster mapping, safe submission and review order.

Post-review refresh: real Opamp2 was freshly imported and the complete three-mode one-epoch pipeline rerun on the audited code. Synthetic two-seed checkpoints were reused only for new float64 evaluation and aggregation; their training cost remains the original smoke measurement. Both updated exports enforce shared parameter-transform identity and matched cross-mode architecture.
