# Canonical data contract (schema_version = 1)

The converter from Analog Agent must be written against real example files, not guessed. This contract is the intermediate format accepted by the demo. [sample_schema.json](sample_schema.json) contains one explicitly synthetic record as a formatting example; it is not a full five-circuit training dataset.

## Dataset fields

- `schema_version`: integer 1.
- `feature_spec.parameter_names`: ordered, unique parameter feature names, e.g. `W`, `L`, `M`, `R`, `C`, `Ib`. No feature count is hardcoded.
- `feature_spec.op_names`: ordered, unique OP fields, e.g. `id`, `gm`, `gds`, `vds`, `vgs`. Do not assume every device supports every field.
- `feature_spec.metric_names`: ordered performance targets. Use actual names/units from your dataset, not the two synthetic placeholders. Declare the metric names before stage one; values may be absent until stage two.
- `feature_spec.terminal_names`: currently exactly `["d", "g", "s", "b", "p", "n"]`, in that order. Map source aliases deliberately.
- `metadata`: optional JSON metadata, including dataset provenance and `synthetic: true` for toy data. Recommended: specify units per named feature and the simulation/tool/PDK versions.
- `samples`: nonempty array of the records below.

## Each sample

- `sample_id`: globally unique, nonempty string for this observation.
- `design_id`: stable sizing/design identity within this circuit. Repeated runs of the same design must retain it.
- `circuit_id`: circuit/topology family. The split command requires exactly five different values and one explicit OOD ID.
- `group_id`: optional circuit-scoped grouping across repeated or related designs; all group members stay together.
- `context`: optional JSON object for fixed simulation conditions. Every sample must have the same context in this demo. Different supply/temperature/corners/loads must first be represented as explicit modeled features in a later extension.
- `ports`: optional map from external port name to net string, e.g. `{ "input": "vin", "output": "vout", "ground": "0" }`. Exact names are retained for adapters/audit; the current model only encodes which terminal roles touch an external port.
- `op_status`: optional `complete`, `partial` or `failed`. This status is provenance; individual OP null masks control learning. A failed record should not carry falsely valid OP numbers.
- `devices`: nonempty array of actual physical devices, not pin nodes. Every record is matched by `device_id` and sorted canonically for batching.
- `performance`: object mapping declared metric names to finite numbers or null. `{}` is valid in stage one. Missing metrics are skipped in stage-two loss and evaluation.

## Each device

- `device_id`: unique within a sample; stable instance name such as `M1`.
- `device_type`: nonempty semantic type string such as `nmos`, `pmos`, `resistor`, `capacitor`, `voltage_source`. `<PAD>` and `<UNK>` are reserved.
- `parameters`: object of declared parameter names and finite numerical values or null. Any omitted declared feature is also missing. Keep units identical for a field across every circuit.
- `op`: same structure for OP observations. A true measured zero stays `0`; unmeasured/failed values are `null` or omitted.
- `terminals`: nonempty map from declared terminal roles to net strings. MOS usually has d/g/s/b; two-terminal devices usually p/n. Connected nets must use consistent names, including ground and global supplies.

Unknown feature names, duplicate JSON keys, duplicate sample/device IDs, invalid port maps, NaN/Infinity and unexpected strings in numerical fields cause clear errors instead of silent coercion. The converter must interpret raw `NULL`, `nan`, unit suffixes and aliases deliberately. This is especially important when a failed run contains partial useful OP data.

## Stages and provenance

Stage one may receive `performance: {}` for every record. It fits parameter/OP statistics and type vocabulary on the ID training partition only, constructs no label tensors, and uses no performance objective. Records with all performance labels missing must not be filtered out of OP pretraining.

Performance labels may be attached or corrected before stage two. The stage-one checkpoint contains an immutable fingerprint excluding **only** performance values, plus an independent partition digest. Stage two accepts changed performance dictionaries but rejects changes to record/design/circuit/device identity, topology, parameters, OP, feature order, context and other metadata. Declare the same metric schema up front. No new records or repartitioning are silently accepted.

Stage two retains the saved train/val/test/OOD memberships, refits only target statistics on ID train, and records both the original pretraining full dataset fingerprint and the newly labeled full fingerprint. Evaluation uses the exact stage-two dataset fingerprint. If you need new feature schemas or data records, create an explicit new experiment rather than editing a manifest.

## Missingness and transformations

The collator constructs separate parameter/OP/performance boolean masks. Values are standardized with train-only statistics and missing entries replaced by zero **only after masking**. Thus a zero-filled slot is not treated as a measured zero. Unknown device types use UNK rather than PAD.

No log transform, signed-log transform, clipping or unit conversion is guessed. Those should be explicitly agreed when the real data arrives. Statistics are global per named field, not a single shared statistic across heterogeneous columns. Constant observed fields use std=1; unobserved fields mean=0/std=1. Consider domain-aware transforms if OP spans many orders of magnitude.

## Duplicate handling

Repeated design_id/group_id records and exact equal design content are grouped. Content fingerprints ignore labels and device/sample names, but retain terminal/net names, parameters, device types, ports and context. They do not solve graph isomorphism. Use a stable group_id when net renaming or near-duplicate exports could otherwise hide the same design. Exact content duplicates appearing under different circuit IDs are rejected, including ID/OOD duplicates.
