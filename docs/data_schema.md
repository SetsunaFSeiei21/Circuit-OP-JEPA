# Canonical dataset and preprocessing contract (v2 pipeline)

The canonical JSON dataset remains schema_version=1; experiment manifests, fitted preprocessing and model checkpoint containers remain version 2. Newly fitted preprocessing/checkpoints declare `input_contract_version: 3`, adding model identity and named external-port associations to the parameter masks and categories of contract 2. New manifests also require `design_fingerprint_version: 2`. Legacy v2 checkpoints without an input-contract field use contract 1 (see compatibility below). The schema example in sample_schema.json is only a small historical synthetic formatting example, not a complete experiment or the real Opamp2 data.

## Dataset fields

- feature_spec.parameter_names, op_names and metric_names: explicit ordered unique field names. Raw op_names preserves all 44 source fields; preprocessing chooses the training subset without altering source records.
- feature_spec.terminal_names: exactly d,g,s,b,p,n in that order.
- metadata: units, synthetic flag and OP-state encoding metadata when used.
- provenance: source file hashes, parser identity, run configuration/constraints and original condition metadata.
- samples: records below. No feasibility filtering is performed by default.

Each sample contains globally unique sample_id, circuit_id, stable design_id/design_key, optional group_id, devices, performance, ports and fixed context. Adapter sample IDs are circuit-prefixed source IDs. Repeated observations/exports of one design retain design/group identity. raw_design_parameters preserves source design-vector semantics; source_sampling and source_observation preserve original run/state/check provenance.

Each device contains device_id, device_type, optional model_name, parameters, op, terminals, op_target and optional op_state/provenance. An optional `parameter_applicability` object may explicitly map declared parameter names to booleans; an observed value cannot be marked inapplicable. MOS terminals are D/G/S/B mapped to lower case; independent sources and ideal R/C use p/n. Current source is a physical graph device even when no measured device OP exists. Non-MOS op_target defaults false; an explicit eligibility override remains supported. Unknown numerical strings, duplicate keys, contradictory device/model joins and nonfinite canonical values are rejected, not silently guessed.

## Actual source adapter

`adapter.py` reads supported SQLite tables in read-only mode, validates tables rather than executing saved SQL, and parses the supplied DUT text. It accepts a bounded set of MOS instances (known models or supported model declarations), DC independent sources, ideal R/C, simple parameter definitions and arithmetic. Unknown DUT devices/directives, unsafe include paths, unsupported expressions or unresolved parameter names are errors.

Duplicate observation fields from samples and DC records must agree after normalization; SQL NULL is absent, while an explicit empty reasons list is meaningful. Conflicting reasons/status fields fail instead of silently overwriting source evidence. Source-table counts are checked within a read-only snapshot before streaming ordered rows. The per-table safety limit defaults to 1,000,000; `import_analog_agent --max-table-rows 1200000` raises it and `--max-table-rows none` removes it explicitly. Streaming avoids keeping full raw and canonical tables simultaneously, but canonical JSON and state encoding still require memory proportional to the complete dataset. A final `.end` terminates the netlist; any later effective input line is rejected.

Expression evaluation uses an AST whitelist for arithmetic, unary signs, parentheses, names and recognized SPICE suffixes. No eval/exec, function calls, arbitrary attributes, imports or simulator execution occur. Expression length, depth, AST count, exponent magnitude and finite intermediate size are bounded. M is milli; MEG is mega. W/L unit semantics and .option scale are applied explicitly. Include/parameter files must remain within the provided dataset root.

Current actual-source parameter fields are W_m, L_m, m and dc_current_A. Optional supported devices add dc_voltage_V, R_ohm or C_F. All physical fields must have consistent declared units across combined circuits. Raw OP field suffixes retain their original units, including w_um/l_um; no raw OP sign or geometry value is overwritten.

The supplied Opamp2 audit contains 32 samples, 17 independent design parameters, 15 MOS + 1 current source per sample, 480 MOS OP records, 44 OP fields and 9 performance metrics. All 21,120 OP scalar positions and all 288 metric positions are preserved; 23 metric missing values remain null. This source has one circuit, not five. These counts are a data-integrity check, not a model-quality result.

## Missingness and state labels

For an applicable field, JSON null or an omitted value means missing. Measured zero is valid. A field outside a device's physical parameter schema is not applicable, rather than an observed zero or a failed measurement. Raw NaN/Infinity from source measurements become null; masks remain explicit in parameter, OP and performance tensors. Padding is separate from unknown device types. Only eligible targets with observed OP contribute to alignment; the current source collector supplies MOS device OP, not passive/source device OP. This is a data-coverage boundary, not a claim that passive devices have no physical DC state.

### Heterogeneous device inputs (all five newly trained modes)

Concrete `device_type` stays intact: NMOS and PMOS remain distinct, as do resistor, capacitor, independent current source and independent voltage source. A second fixed semantic vocabulary supplements these subtypes:

| Coarse category | IDs / meaning |
| --- | --- |
| padding | `0`, never a real node |
| unknown | `1`, unsupported/custom type; not padding |
| active | `2`, NMOS/PMOS |
| passive | `3`, resistor/capacitor (and an explicitly provided inductor type) |
| source | `4`, independent current/voltage sources; not silently forced into active/passive |
| port | `5`, standalone external-port tokens in pin mode |

This fixed `device_category_vocab` is not fitted from held-out samples. Concrete subtype/pin vocabularies remain ID-train-only; unseen subtypes map to their unknown token. The coarse `device_categories` tensor is consumed by an additional learned embedding in the shared structure encoder. The adapter still rejects unsupported DUT primitives; assigning a category does not add parser support.

Parameters use one shared ordered feature space, preserving their own physical fields rather than assigning W/L/M to every device. Known MOS schemas recognize W/L/M (or adapter `W_m/L_m/m`); R and C use resistance and capacitance fields; sources use their actual current/voltage fields. The same applicability helper is used in both device and pin collators. Known primitive schemas distinguish legacy unrelated null padding from a missing applicable parameter; additional observed fields remain supported. For custom types, explicitly supplied keys, including null keys, provide the fallback schema; use `parameter_applicability` when that evidence is insufficient.

The padded tensors distinguish:

| Case | `device_mask` | `parameter_applicability` | `parameter_mask` |
| --- | --- | --- | --- |
| observed value, including true zero | true | true | true |
| applicable but missing value | true | true | false |
| not applicable to this device | true | false | false |
| padded node | false | false | false |

These are actual model inputs, not reporting-only flags. Contract-2 named parameter tokens project `(normalized value, observed bit)`; missing values are replaced by zero before projection. Applicable-but-missing tokens remain visible to own-node feature attention with observed bit 0; inapplicable tokens are excluded by the attention mask. The device type/category stays visible in either case. Pin tokens inherit their parent device's category, parameter values and both masks; standalone ports carry no live parameter tokens.

### Direct DC OP inputs versus JEPA targets

`zerosim_pin_op` and `zerosim_device_op` add selected continuous OP `(value, observed bit)` tokens to the existing own-node feature fusion. They do not add OP-check one-hot states, a predictor, a separate JEPA branch or extra fusion layers. Device-level OP/masks/eligibility are repeated onto that device's own observed pins in pin mode; ports, padding and noneligible devices have no live OP tokens. Current collector non-MOS observations are unavailable, not fabricated. Both new models require the OP tensor/mask/eligibility contract at **training and inference**; per-field missing values remain supported. A training set with no eligible observed selected OP is rejected. Inference reports count observed/possible OP values and samples with no observed OP, counting original devices rather than repeated pins.

The original `zerosim_pin`/`zerosim_device` models and the frozen JEPA input encoder do not read OP at inference. JEPA uses OP on its pretraining target branch only. All five modes share parameter scalers and target-label budgets; all OP-using methods share selected continuous OP fields and ID-train-only transforms. OP acquisition time is unmeasured (`null`, not zero) and reported separately from model training time.

### Input-contract compatibility

The top-level checkpoint and its preprocessing declare matching `input_contract_version`. Contract 3 includes train-only `model_vocab` and `port_role_vocab`, model embeddings and multi-hot terminal-to-named-port features. Model and port names are canonicalized case-insensitively for identity. Model PAD, unknown and absent identities remain distinct; aliases sharing a net retain all their role bits. Contract 2 includes the fixed category vocabulary/embedding and two-channel parameter projection above; old v2-container checkpoints without the input-contract field reconstruct contract 1 (no coarse embedding, one-channel parameter projection, observed-only parameter attention). New weights are never silently loaded into old architectures. Legacy architecture reconstruction remains supported, but manifests without the new fingerprint version are rejected and require a new experiment. Do not mix input contracts in a matched model comparison. The loader rejects inconsistent contract/category/vocabulary metadata and new direct-OP checkpoints that omit their explicit OP training/inference requirement.

`states.py` encodes two independent collector-check groups:

- current_missing, current_below, current_pass;
- saturation_missing, saturation_below, saturation_pass.

Checked-but-missing has an explicit missing one-hot. Unchecked or insufficiently evidenced groups have all null entries, therefore all masks false. A recognized raw reason is positive evidence only for its corresponding group. Thresholds and device-check scope come from unambiguous sample/run provenance or explicit config; no absent reason is interpreted as pass. `:saturation` is a violation of a minimum saturation margin, not a positive saturation-region label. Contradictory reason/measurement/checklist evidence is rejected. These **OP check-state categories** are never z-scored and never enter the shared structure/design-input branch; they are distinct from the coarse device categories described above.

Default mode collector_checked preserves original check scope conservatively. Explicit all_mos recomputes checks under known thresholds and records derived_not_original_collector; it does not overwrite original reasons/status. preserve_only disables categorical encoding. Raw OP vectors and polarity conventions are unchanged in every mode. The collector's abs(Id) check is used only for that specific current threshold, not as a blanket OP preprocessing rule.

## Split configuration

train_circuits and ood_circuits are explicit, disjoint and exhaustive. Empty OOD requires an explicit empty list and is reported as no OOD experiment. There is no hardcoded five-circuit requirement; the intended future four-ID/one-OOD setup is expressed in config.

Within every ID circuit, design_key/design_id/group_id and exact design fingerprints form indivisible groups. Defaults allocate 80/10/10 train/val/test by group count, with at least one group per requested nonempty split. finetune_label_fraction selects complete train groups only. Every mode/seed reuses the same persisted labeled_train and split IDs. Fingerprint version 2 ignores target values but retains parameters, types, canonical model names, terminals, ports and context. Numeric +0.0 and -0.0 have the same design identity; source parameters and signed OP values are not modified. It is not a net-renaming graph-isomorphism detector.

The manifest includes its version, design-fingerprint version, explicit circuits, ratios, seed, memberships, labeled IDs and dataset fingerprint. Existing manifests are verified, never silently rebuilt after additions or config changes. Manifests lacking fingerprint version 2 require a new experiment, because earlier grouping could miss duplicate designs differing only in signed zero or conflate different models. Performance values can be added later through the stage-transition provenance checks; no new records, input changes or repartitioning are silently accepted.

## Fitted physics transforms

Preprocessing version 2 saves raw_feature_spec, selected feature_spec, device/model/port-role vocabularies, transform rules, fitted asinh scales, mean/std/count, weighting and fitted sample IDs. Default selected OP fields exclude w_um/l_um/multiplicity, preserving them only in raw data. Input parameter/vocabulary/OP fitting uses ID train; performance fitting uses only labeled_train. OOD/validation/test never fit anything. A performance field with zero labeled-train observations is excluded from checkpoint-selection loss even when validation labels exist; checkpoints record the supported selection metric names. Validation with no observed supported metric fails rather than selecting on untrained targets.

Default equal_circuit weighting gives each observed circuit equal mass per field, then averages observed values/device slots within each circuit. Signed wide-dynamic OP fields use asinh(x/s), preserving sign and zero; voltage-like fields use linear transform. The default fitted s is the circuit-balanced weighted median of nonzero |x| from training only. Zero observations still participate in the subsequent mean/std. A fixed positive physical scale can be configured. Missing and padding never enter either fit. No sign-erasing abs transform, negative-value log or clipping is applied.

Rules are configurable under normalization with op_fields, weighting and field mappings under parameters/op/performance. Each field rule uses kind linear/asinh and optional scale positive-number/fit. Constant or entirely unobserved fields are explicitly counted and use finite fallback scaling. Transforms and input statistics stay frozen for stage two and evaluation; old scaler versions or conflicting stage-two configs fail clearly.

## Fixed context limitation

All samples currently require identical modeled context. Different process corners, supply, temperature, common-mode or load conditions cannot silently be mixed while omitted from model inputs. Extend the explicit input schema/model before using varied conditions. Unknown extra metadata is preserved for audit, not automatically converted into model features.
