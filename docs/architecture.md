# Architecture and comparison protocol (v2)

## Three declared modes

- `zerosim_pin`: independently implemented pin/port-node ZeroSim-style supervised reference.
- `zerosim_device`: the same local input/decoder family with one token per physical device, supervised end to end.
- `jepa_zerosim`: device-node structure/OP joint alignment, followed by strictly frozen input-encoder probing with a metric-query decoder.

The pin reference is a clean-room reimplementation, not imported official source, not an exact original-paper reproduction, and not a generic pin GNN relabeled as ZeroSim. It uses learned token/absolute-position embeddings, alternating local/global Transformer attention, own-parameter cross-attention in the last configured layers, a graph token, and a per-metric query Transformer decoder. Sources and deliberate differences are listed in THIRD_PARTY_NOTICES.md.

All three modes use the same canonical samples, grouped manifest, labeled_train subset and target transforms. Baselines optimize encoder+decoder on labels; JEPA additionally learns from OP on the full ID training partition. Equal labeled-data budget is not equal total compute. Reports record pretraining/supervised time, optimizer counts and parameter counts separately. Multiple seeds must retain the same dataset/split/target preprocessing and a stable within-mode architecture and supervised training protocol.

## Structure sharing boundary

For default encoder_layers=6 and cross_layers=4, the first two blocks precede the first design-parameter fusion. `StructureEncoder` owns this entire pure-structure prefix:

1. Device-type embedding, learned absolute-position embedding and graph token.
2. Structural terminal-pair attention bias and external-port terminal-role indicators.
3. Alternating local/global self-attention plus feed-forward blocks in the prefix.

The remaining four input-side blocks retain self-attention and own-parameter cross-attention. They are parameter-conditioned, design-specific and **not** shared with the OP branch. This boundary keeps the reference's six-layer layout rather than silently adding six new structure layers. It is deliberately not described as “all attention is shared.”

`DeviceEncoder` registers the shared module once. `SharedOPEncoder.structure_encoder` references that exact module object without a second registration. `JEPAModel.forward` calls the structure module exactly once and passes the same non-detached tensor to both branches. Thus both branches see the same structural dropout realization and both gradient paths update the shared parameters. There is no teacher copy, EMA, stop-gradient, or synchronization procedure.

The shared module reads only type, position, topology/terminal and external-port features. It cannot read parameters, OP, state labels or performance. The OP branch receives structure tokens concatenated with normalized selected electrical OP fields, their observation masks, and optional categorical check features/masks. One MLP is reused for every MOS device. No independent predictor remains: the input device representation aligns directly with the OP-branch output.

## Alignment objective and limitations

The objective is symmetric device-wise MSE plus variance and off-diagonal covariance penalties on both branch outputs (VICReg-style anti-collapse regularization). Defaults are 25/25/1 with a standard-deviation floor of 1. At least two eligible observed MOS tokens are required per minibatch. Entirely missing OP vectors and non-target devices do not contribute.

This is an explicit alignment prototype, not a canonical I-JEPA implementation. Sharing structure creates a genuine shortcut risk: the branches could agree on topology while underusing sizing/OP. Regularization addresses constant-vector collapse, not every shortcut. Meaningful research needs within-topology representation checks, OP pairing/shuffle ablations and downstream comparisons, not only decreasing alignment loss. The current smoke verifies mechanics, not scientific benefit.

## Device graph and pin reference

Device mode has one node per DUT device, including current sources and passive elements. Two devices connect when terminals share a net; terminal-pair features distinguish D→G from D→D. Only MOS nodes are default OP targets. The graph token and every real device token remain decoder memory.

Pin mode converts those same samples only at tensorization time: explicit external-port nodes first, then canonical per-device terminal nodes. Pins belonging to a device form an internal clique; shared nets and their external ports create the electrical connectivity. Named parameter values/masks are repeated for the corresponding device's pins. Pin/port vocabulary is fitted from ID train only, with separate UNK and PAD. Dataset IDs, design grouping and target scalers never change. The pin reference disables the device terminal-pair attention-bias extension by default; its effective config is recorded.

Differences from an exact upstream reproduction include strict missing/padding masks, named parameter embeddings, train-only circuit-balanced preprocessing, safe null-parameter attention, no silent truncation, and independent implementation choices in normalization modules. These safeguards are kept explicit rather than claimed to reproduce published numbers.

## Decoder and frozen behavior

Each metric has a learned query. A Transformer decoder performs query self-attention, cross-attention over `[graph, all real input tokens]`, and feed-forward processing. A shared D→D→1 regressor outputs metric values. The default decoder has two layers and a 2048-wide feed-forward block.

For JEPA stage two, the entire input encoder is set requires_grad=False, is kept in eval even when its wrapper trains, and executes under no_grad. Only decoder parameters enter the optimizer. Every epoch verifies a digest over all encoder parameters and registered buffers and checks absent encoder gradients. Baselines instead update both input encoder and decoder; no OP/state tensors are passed to either baseline model.

## Provenance and evaluation

Version-2 checkpoints store architecture, selected field order, signed transform rules and fitted scales, state config, explicit split/label IDs, dataset/partition/input-OP digests, validation selection, phase timing and counts. Old v1 checkpoints fail with a clear incompatibility message. Adding downstream performance labels is allowed only while immutable input/OP/identity and split provenance remain unchanged. Changed normalization or OP-state configuration is rejected during stage transition rather than mislabeled as the requested new configuration.

Metrics are computed only on observed targets: original-unit MAE/RMSE/R² plus normalized-space errors under frozen training-target transforms. R² is undefined for fewer than two labels or zero variance and remains null. Target values and both physical/normalized predictions are emitted per sample. Comparison rejects unmatched data, labels, splits, evaluated samples, target scalers, label budgets or supervised training protocol. Pin and device architectures intentionally differ; comparisons require per-mode consistency across seeds, not identical node counts or parameter counts across methods.
