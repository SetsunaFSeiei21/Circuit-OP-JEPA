# Five mode architecture and comparison protocol

## Five declared modes

- `zerosim_pin`: independently implemented pin/port-node ZeroSim-style supervised reference.
- `zerosim_device`: the same local input/decoder family with one token per physical device, supervised end to end.
- `jepa_zerosim`: device-node structure/OP joint alignment, followed by strictly frozen input-encoder probing with a metric-query decoder. Performance inference does not read OP.
- `zerosim_pin_op`: matched pin/port supervised model with continuous device OP and observation masks at training **and inference**.
- `zerosim_device_op`: matched device supervised model with continuous device OP and observation masks at training **and inference**.

The pin reference is a clean-room reimplementation, not imported official source, not an exact original-paper reproduction, and not a generic pin GNN relabeled as ZeroSim. It uses learned token/absolute-position embeddings, alternating local/global Transformer attention, own-parameter cross-attention in the last configured layers, a graph token, and a per-metric query Transformer decoder. Sources and deliberate differences are listed in THIRD_PARTY_NOTICES.md.

All five modes use the same canonical samples, grouped manifest, labeled_train subset and target transforms. Baselines optimize encoder+decoder on labels; JEPA additionally learns from OP on the full ID training partition. Matched experiments require equal total epochs: JEPA pretraining plus decoder epochs equals each baseline's epochs. Equal labeled-data and total-epoch budgets do not imply equal total compute or inference information. Direct OP models are paired with their matching no-OP baseline; their extra projections/embeddings increase parameter count, so this is not a perfectly parameter-matched causal ablation. OP acquisition cost is unmeasured, not zero. Reports record pretraining/supervised time, optimizer counts and parameter counts separately. Multiple seeds must retain the same dataset/split/target preprocessing and a stable within-mode architecture and supervised training protocol.

## Structure sharing boundary

For default encoder_layers=6 and cross_layers=4, the first two blocks precede the first design-parameter fusion. `StructureEncoder` owns this entire pure-structure prefix:

1. Concrete device-type, coarse-category and model-name embeddings, learned absolute-position embedding and graph token.
2. Structural terminal-pair attention bias and named external-port terminal associations.
3. Alternating local/global self-attention plus feed-forward blocks in the prefix.

The remaining four input-side blocks retain self-attention and own-parameter cross-attention. They are parameter-conditioned, design-specific and **not** shared with the OP branch. This boundary keeps the reference's six-layer layout rather than silently adding six new structure layers. It is deliberately not described as “all attention is shared.”

`DeviceEncoder` registers the shared module once. `SharedOPEncoder.structure_encoder` references that exact module object without a second registration. `JEPAModel.forward` calls the structure module exactly once and passes the same non-detached tensor to both branches. Thus both branches see the same structural dropout realization and both gradient paths update the shared parameters. There is no teacher copy, EMA, stop-gradient, or synchronization procedure.

The shared module reads only concrete type, coarse category, model identity, position, topology/terminal and external-port features. It cannot read parameters, OP, state labels or performance. The OP branch receives structure tokens concatenated with normalized selected electrical OP fields, their observation masks, and optional categorical check features/masks. One MLP is reused for every MOS device. No independent predictor remains: the input device representation aligns directly with the OP-branch output.

## Alignment objective and limitations

The objective is symmetric device-wise MSE plus variance and off-diagonal covariance penalties on both branch outputs (VICReg-style anti-collapse regularization). Defaults are 25/25/1 with a standard-deviation floor of 1. At least two eligible observed MOS tokens are required per minibatch. Entirely missing OP vectors and non-target devices do not contribute.

This is an explicit alignment prototype, not a canonical I-JEPA implementation. Sharing structure creates a genuine shortcut risk: the branches could agree on topology while underusing sizing/OP. Regularization addresses constant-vector collapse, not every shortcut. Meaningful research needs within-topology representation checks, OP pairing/shuffle ablations and downstream comparisons, not only decreasing alignment loss. The current smoke verifies mechanics, not scientific benefit.

## Device graph and pin reference

Device mode has one node per DUT device, including current sources and passive elements. Two devices connect when terminals share a net; terminal-pair features distinguish D→G from D→D. Only MOS nodes are default OP targets. The graph token and every real device token remain decoder memory.

Pin mode converts those same samples only at tensorization time: explicit external-port nodes first, then canonical per-device terminal nodes. Pins belonging to a device form an internal clique; shared nets and their external ports create the electrical connectivity. Named parameter values, observation masks and applicability masks are repeated for the corresponding device's pins; parent-device categories are retained, and independent ports use a distinct port category. Pin/port vocabulary is fitted from ID train only, with separate UNK and PAD. Dataset IDs, design grouping and target scalers never change. The pin reference disables the device terminal-pair attention-bias extension by default; its effective config is recorded.

Differences from an exact upstream reproduction include strict missing/padding masks, named parameter embeddings, train-only circuit-balanced preprocessing, safe null-parameter attention, no silent truncation, and independent implementation choices in normalization modules. These safeguards are kept explicit rather than claimed to reproduce published numbers.

## Heterogeneous input contract

Newly fitted preprocessing records `input_contract_version=3`. Concrete NMOS, PMOS, resistor, capacitor and independent-source identity is retained. Fixed coarse categories supplement it: active/passive/source/unknown, plus PAD and pin-mode port. Model names and named external-port roles now have separate train-only vocabularies, alongside the train-only subtype and pin vocabularies. Model names use case-insensitive identity with distinct PAD, UNK and missing entries. Terminal-to-port features are multi-hot, preserving multiple named aliases on the same net; swapping VINP/VOUT wiring therefore changes device inputs. Unseen held-out model/port names use UNK without expanding fitted vocabularies.

Physical design fields remain named and unit-specific: W_m/L_m/m, R_ohm, C_F, dc_current_A and dc_voltage_V for imported data. A field has independent observation and applicability masks. An applicable missing field retains a named token with observed=0, an observed zero has observed=1, and an inapplicable field has no live input token. Contradictory observed-but-inapplicable data are rejected. Unknown/custom types may supply explicit applicability metadata.

`OperatingPointDeviceEncoder` extends the same own-node input fusion, not the pure structure prefix. Named electrical OP tokens encode normalized value and observed bit alongside design tokens, with no additional Transformer layers. Eligible missing observations retain explicit mask information. Pin OP comes only from each pin's own parent device; ports and currently ineligible non-MOS devices have no live OP tokens. Both new modes train the encoder and decoder end to end and do not consume discrete collector state categories. They are not JEPA OP target branches.

The current adapter collects MOS OP only. No passive/source OP is fabricated; this is a collector-scope statement, not a claim that passive devices lack DC states. Explicit OP eligibility is part of the canonical contract and must not be silently reinterpreted.

Checkpoint/preprocessing container version remains 2. Input contract 1 reconstructs the legacy value-only parameter projection and category-less structure; contract 2 adds observed-bit projections and category embeddings; contract 3 additionally includes model identity and named port associations. Older v2 preprocessing without an input-contract field reconstructs contract 1. Legacy architecture loading does not migrate old manifests: grouping now requires `design_fingerprint_version=2`, including model names and canonical signed zero. Rebuild experiments from canonical data when that manifest version is absent. New and old input contracts cannot be mixed in a matched comparison.

## Decoder and frozen behavior

Each metric has a learned query. A Transformer decoder performs query self-attention, cross-attention over `[graph, all real input tokens]`, and feed-forward processing. A shared D→D→1 regressor outputs metric values. The default decoder has two layers and a 2048-wide feed-forward block.

For JEPA stage two, the entire input encoder is set requires_grad=False, is kept in eval even when its wrapper trains, and executes under no_grad. Only decoder parameters enter the optimizer. Every epoch verifies a digest over all encoder parameters and registered buffers and checks absent encoder gradients. All four supervised baselines instead update both input encoder and decoder. The original two receive no OP/state tensors; the new two receive continuous OP and masks but no collector state categories.

## Epoch budgets

Three independent positive integer fields control training: `training.pretrain_epochs` for JEPA representation learning, `training.decoder_epochs` for its frozen-encoder decoder, and `training.baseline_epochs` for each of the four end-to-end baselines. New baseline runs require the explicit baseline field and never fall back to the JEPA decoder count. Defaults preserve JEPA's 30 + 50 epochs and give each baseline 80 epochs. A matched cohort containing JEPA and baselines must satisfy `pretrain_epochs + decoder_epochs == baseline_epochs`; both canonical and raw-data comparison runners check this before writes or training. Smoke comparison uses 1 + 1 versus 2 epochs.

The strict comparison reconstructs JEPA's total using the immutable executed pretraining protocol and the downstream decoder budget, not the stage-two config's proposed pretraining epochs. All configured epochs execute; the selected best checkpoint's epoch identifies model selection, not total training cost. Supervised learning rate, batch size, weight decay and gradient clipping still match across methods. Epoch parity does not imply equal sample exposure when the pretraining and labeled partitions differ, equal FLOPs, or equal elapsed time.

## Provenance and evaluation

The `device_resident_v1` pipeline packs numeric observations once, normalizes them on the selected device in float64, and caches topology separately from per-design values. CUDA runs also fit train-only weighted statistics on CUDA. Pin inputs gather their parent-device values on device. Own-node cross-attention masks use a vectorized construction with the original mask semantics. Frozen decoder training caches encoder memories once and checks exact encoder parameter/buffer bytes each epoch. CPU file/string processing and report generation stay outside the CUDA numerical training path; no implicit CPU fallback is used. Batch progress and phase/total costs remain explicit. Details are in [GPU_PIPELINE_ZH.md](GPU_PIPELINE_ZH.md).

Version-2 checkpoints store architecture, selected field order, signed transform rules and fitted scales, state config, explicit split/label IDs, dataset/partition/input-OP digests, validation selection, phase timing and counts. Stage two preserves the actual source pretraining provenance, including its loss, training options, seed and protocol digest, separately from supervised options and seed. Validation selection uses only metrics observed in labeled training; validation with no supported observed metric fails clearly. Direct training entrypoints reject existing stage artifacts and stale output files; stage two may share a matching stage-one directory. Old v1 checkpoints fail with a clear incompatibility message. Adding downstream performance labels is allowed only while immutable input/OP/identity and split provenance remain unchanged. Changed normalization or OP-state configuration is rejected during stage transition rather than mislabeled as the requested new configuration.

Metrics are computed only on observed targets: original-unit MAE/RMSE/R² plus normalized-space errors under frozen training-target transforms. R² is undefined for fewer than two labels or zero variance and remains null. Target values and both physical/normalized predictions are emitted per sample. Comparison rejects unmatched data, labels, splits, evaluated samples, target scalers, label budgets, total epoch budgets or supervised optimizer settings. Pin and device architectures intentionally differ; comparisons require per-mode consistency across seeds, not identical node counts or parameter counts across methods.

## Paired reporting and cost boundaries

The runner defaults to all five modes; `--modes` can explicitly select a legacy-three cohort or a narrower pair. Strict aggregation checks input/target/dataset identity, grouped split and labels, selected OP transform, total epoch budget, supervised optimizer settings and within-mode architecture. Matching OP/no-OP pairs have identical base model hyperparameters; extra OP projections are declared capacity changes.

`paired_per_seed.csv` computes OP-minus-no-OP differences before aggregation. `paired_summary.csv` summarizes the actual matched differences with sample SD; undefined R2 remains null and one-seed SD remains blank. Negative MAE/RMSE deltas and positive R2 deltas favor OP inputs, but smoke deltas do not establish scientific benefit.

Report provenance exposes `requires_op_at_inference`, `op_information`, train/inference use, train-only OP scaler IDs, and original-device OP coverage (not inflated repeated-pin counts). Acquisition cost is null with `not_measured`. `costs.csv` records training cost once per verified run identity, retaining phase optimizer steps; evaluation coverage is separate from training cost and acquisition records retain their measurement scope. Conflicting costs for the same run are rejected. Comparison output schema 4 records total-epoch parity. Legacy reports without trustworthy run/pretraining identity or explicit baseline epoch provenance remain explicitly incomplete; known unequal totals are rejected. These reports do not measure DC solve, label collection or deployment latency.
