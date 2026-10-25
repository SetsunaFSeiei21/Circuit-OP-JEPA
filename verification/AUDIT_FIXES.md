# Final audit fixes and bounded rechecks

The stronger review identified real defects after the initial 166-test candidate. They were fixed before publication; the original candidate is not the final verified code.

1. **Physical evaluation precision.** Float32 inverse transforms could erase small variations around large offsets or mishandle tiny asinh scales. Evaluation now uses the stable float64 scalar inverse and reads physical truths directly from raw records. RMSE and R² use scaled calculations, and nonrepresentable results fail explicitly instead of silently returning zero, infinity or NaN. Three model paths are tested against metrics recomputed from their exported raw records.
2. **State provenance and effective configuration.** Pretraining now re-derives requested states from raw evidence and checks persisted metadata, every state, and device provenance without mutating data. Wrong mode, threshold, feature order, missing fields and contradictory raw evidence fail with re-import guidance. Omitted features in an explicitly enabled mode consistently select the default six; explicit empty features disable the state input.
3. **Multiple-source run isolation.** Combined samples link to a specific provenance source and circuit before matching run_id. Repeated run IDs across circuits cannot lend checks or thresholds to each other. Legacy unscoped combined evidence is not guessed. A source lacking run metadata stays unknown even when another source has a matching run ID. Single-source Opamp2 imports remain unchanged.
4. **Comparison controls.** Reports must match parameter transform/vocabulary/feature order as well as targets, splits, labels and supervised budgets. Device and JEPA architecture must match exactly. Pin mode permits only declared terminal-bias and token-budget differences; other architecture differences are rejected.
5. **Runtime safety and determinism.** Loading requires stable PyTorch >=2.6 before calling `torch.load(weights_only=True)`, addressing [CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6). Deterministic CUDA workspace is set before seeding/initialization; incompatible explicit settings raise. CUDA behavior is covered by preflight tests, not a GPU execution claim.

Verification on 2026-10-25:

- Full suite: 211 tests passed. Log: `unittest_v2.log`.
- Fresh real import: 32 Opamp2 samples, 44 raw OP fields, 9 metrics, no feasibility filter. Fresh three-mode one-epoch train/evaluate/aggregate pipeline passed with the final validation seam.
- Synthetic: existing two-seed checkpoints re-evaluated, all 12 test/OOD reports re-aggregated under the new numerical and fairness guards.
- Wheel build, Python compilation, shell syntax and whitespace checks passed.

Only aggregate smoke tables are versioned. Input datasets, private simulation histories, checkpoints and per-sample predictions are excluded. None of these checks establishes prediction quality, method superiority, full-training convergence, GPU performance or real cross-circuit OOD generalization.
