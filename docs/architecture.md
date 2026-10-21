# Architecture and invariants

## Input encoder

The reference inspected was the public [ZeroSim repository](https://github.com/xz-group/ZeroSim), especially [circuitformer.py](https://github.com/xz-group/ZeroSim/blob/main/model/circuitformer.py), [modules.py](https://github.com/xz-group/ZeroSim/blob/main/model/modules.py), [config.yaml](https://github.com/xz-group/ZeroSim/blob/main/configs/config.yaml), and [preprocess_topology.py](https://github.com/xz-group/ZeroSim/blob/main/scripts/preprocess_topology.py). This demo is an original conceptual implementation, not a source copy or exact replication.

The default preserves the high-level 128-dimensional, six-layer, eight-head structure, alternating local/global self-attention and own-parameter cross-attention in the final four layers. Each input device has one token. A graph token precedes the device tokens, and all tokens are retained as decoder memory.

1. Canonicalize the supplied device list by stable device_id.
2. Embed the **device type**, not pin type or circuit identity. Add absolute learned position and terminal-role external-port indicators.
3. Build a device graph: two devices are adjacent when any terminals share a net. Preserve a multi-hot terminal-pair vector for every device pair. A learned per-head bias distinguishes D→G from D→D, etc. Self edges are allowed; the graph token attends globally as in the reference design.
4. Embed each observed named parameter as scalar projection + feature-name embedding. Device rows can cross-attend only their own observed parameters. The graph token may attend all observed parameters. Rows with none receive a dedicated null token, preventing all-masked softmax NaNs.
5. Alternate local and global self-attention blocks; insert parameter cross-attention after self-attention in the configured last layers. Use residual post-LayerNorm blocks, feed-forward networks and dropout. Separate norm modules per sublayer are a deliberate robustness difference.
6. Final LayerNorm + linear projection produce `[B, 1+N, D]`. Padding nodes have zero output and remain excluded from attention keys and decoder memory.

Other deliberate adaptations include train-only normalization, UNK distinct from PAD, explicit missing masks, no silent node truncation, named parameter embeddings, terminal-pair bias and safe handling of all-missing parameter rows. They are not claims about the official implementation.

## OP branch and objective

One shared `2O → 2D → D` GELU MLP receives each device's standardized observed OP values concatenated with its O-dimensional observed mask. The input branch's device token passes through a `D → 2D → D` predictor. The predictor is discarded at stage two; the encoder itself is retained.

The loss combines per-device MSE with variance and off-diagonal covariance regularization on **both** learned branch outputs. This follows the anti-collapse motivation in [VICReg](https://arxiv.org/abs/2105.04906), but the exact device-token batching and implementation are demo choices. Variance uses the population centered second moment plus 1e-4 before square root; covariance uses the sample denominator n−1. Each branch's penalty is averaged with the other branch. Tokens in the same circuit are correlated; larger/more varied minibatches and anti-collapse diagnostics should be examined on real data.

There is no performance prediction objective, OP value reconstruction, EMA teacher or stop-gradient in stage one. It is a joint embedding alignment prototype rather than a canonical I-JEPA reproduction. The graph token is not directly aligned to an OP vector, but receives gradients through its interactions with device tokens. Unit tests confirm nonzero graph-token gradients.

## Frozen-encoder downstream stage

Each metric has one learned query. A standard Transformer decoder applies query self-attention, cross-attention to `[graph, every device]`, and a feed-forward sublayer. A shared `D → D → 1` GELU regressor predicts one scalar per query. The decoder feed-forward default is 2048, matching PyTorch's default used by the inspected reference.

`FrozenPerformanceModel` enforces all three:

- `encoder.requires_grad_(False)` and optimizer containing decoder parameters only.
- `encoder.eval()` even after the wrapper is switched to train mode, disabling encoder dropout.
- Encoder forward runs inside `torch.no_grad()`.

Every epoch compares a SHA-256 digest over the encoder's entire state_dict (parameters AND registered buffers) against the stage-one checkpoint and verifies absence of encoder gradients. Stage two never constructs OP tensors. Performance labels are masked and standardized using ID train statistics only. Their MSE is averaged over observed scalar targets.

## Split and checkpoint provenance

Only circuit IDs in the manifest's train/val/test partitions appear in any fitting step. Dataset validation inspects and integrity-hashes every record, including OOD, but this does not fit statistics or feed OOD to the model. Feature names are an explicit caller-provided schema, not learned by inspecting the fifth circuit.

Splits unite records sharing a circuit-scoped design_id, group_id or exact design fingerprint. The fingerprint excludes OP/performance labels and sample/device identity strings. It retains device types, named parameters, terminal/net mapping, ports and context. Exact duplicates across different circuits raise an error rather than silently contaminating ID/OOD. The detector is not a graph-isomorphism solver.

Within each ID circuit, seed-based hashes order independent groups; validation and test each receive max(1, floor(group_count × fraction)) groups, with the remainder training. All four ID circuits must have at least three groups. No sample appears in multiple splits.

Checkpoints store architecture/options, feature order, train-only preprocessing/vocabulary, explicit sample split IDs, full dataset fingerprint, immutable input/OP fingerprint, independent partition digest, source encoder hash, selection objective/epoch, Python/Torch versions, and a synthetic-data flag. Labels affect the integrity hash only, never pretraining gradients, preprocessing or selection. A separate adversarial review verified identical stage-one tensors under performance-label poisoning.

Stage two explicitly permits later performance-label attachment. It compares the input/OP fingerprint against stage one, verifies the original partition digest and scaler fitted IDs, then refreshes the full dataset fingerprint for the labeled data without changing any membership. The original full pretraining fingerprint is retained in the stage-two checkpoint. Changes to inputs, OP, records, schema or partitions are rejected.
