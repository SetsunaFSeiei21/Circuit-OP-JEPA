# Verification record

Date: 2026-10-21. Tested with Python 3.12 and PyTorch 2.14.1+cpu on CPU.

## Automated suite

```bash
python -m unittest discover -s tests -v
```

**54 tests passed.** Full stdout/stderr is in `unittest.log`.

Covered: strict JSON/schema validation; repeated-design grouping; exact ID/OOD separation; ID-train-only vocabulary/statistics; missing-vs-zero semantics; terminal-pair inputs; padding and ordering; gradients to both pretraining branches; no performance-tensor pretraining; all-missing parameter safety; nine-metric variable output dimensions; metric-query decoder dependence on device tokens; strict encoder parameter/buffer/dropout/gradient freeze; checkpoint round trips; label attachment after OP-only pretraining; rejection of changed inputs/OP/records/partitions; and separate test/OOD metrics.

Additional adversarial checks verified stage-one model tensors are identical when performance labels are replaced with missing or very large values, and stage-two label attachment preserves the frozen encoder and split membership. The default 128-dimensional/six-layer/eight-head configuration also passed a finite forward/backward check.

## Synthetic smoke

```bash
python -m device_jepa smoke --output runs/smoke
```

Passed both optimization stages and separate test/OOD evaluations. The fixture contains 120 toy records across five synthetic families, split 72 train / 12 validation / 12 test / 24 OOD. Logs, selected history and full metric/prediction reports are in this directory. Runtime checkpoints are intentionally not included in the source archive; the command recreates them.

The run uses only 2 pretraining epochs and 3 decoder epochs. Negative R² values in the report are expected to remain visible, not hidden or presented as successful prediction quality. These data are analytic toy formulas, not SPICE simulation or Analog Agent data. **Nothing here establishes real circuit accuracy, usefulness of pretraining, or OOD generalization.**

## Packaging

The project supports editable installation and wheel builds. Its bundled smoke configuration is included as package data, so default smoke does not depend on a source-tree `configs/` directory. An isolated wheel smoke run was also verified end to end. See README for installation and individual CLI commands.

## Remaining integration

Real Analog Agent data conversion, feature/units mapping, fixed simulation conditions, preprocessing transforms, circuit-group identities and the selected OOD circuit must be confirmed against actual files. No remote repository, cluster job, SPICE simulation or public release was created.
