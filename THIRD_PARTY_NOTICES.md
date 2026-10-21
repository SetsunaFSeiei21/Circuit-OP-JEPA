# Sources and implementation status

Inspected on 2026-10-21:

- [ZeroSim official repository](https://github.com/xz-group/ZeroSim).
- [Input encoder and decoder](https://github.com/xz-group/ZeroSim/blob/main/model/circuitformer.py).
- [Transformer layers](https://github.com/xz-group/ZeroSim/blob/main/model/modules.py).
- [Default configuration](https://github.com/xz-group/ZeroSim/blob/main/configs/config.yaml).
- [Pin-graph preprocessing](https://github.com/xz-group/ZeroSim/blob/main/scripts/preprocess_topology.py).
- [Dataset representation](https://github.com/xz-group/ZeroSim/blob/main/utils/dataset_utils.py).
- [VICReg, Bardes et al.](https://arxiv.org/abs/2105.04906), for variance/covariance anti-collapse motivation.
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) and [TransformerDecoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html).

The inspected ZeroSim repository root did not show a license file, and its main-branch `LICENSE` endpoint returned 404. This is not a claim about the authors' ultimate licensing intentions. Accordingly no ZeroSim source files, weights, datasets or textual source excerpts have been copied or bundled. This standalone project implements the described high-level architecture independently and clearly identifies changes. It is not the official implementation or a reproduction claim.

ZeroSim citation: Xiaomeng Yang, Jian Gao, Yanzhi Wang and Xuan Zhang, “ZeroSim: Zero-Shot Analog Circuit Evaluation with Unified Transformer Embeddings,” ICCAD 2025, DOI [10.1109/ICCAD66269.2025.11240890](https://doi.org/10.1109/ICCAD66269.2025.11240890).

PyTorch and NumPy are installed dependencies and retain their own licenses. No dependency binaries are bundled in the source archive. This demo is hosted in the private Circuit-OP-JEPA repository. Choose an appropriate project license before public distribution.
