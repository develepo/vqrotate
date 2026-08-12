# vqrotate

**One-line Rotation Trick for VQ-VAEs.**

Drop-in wrapper that attaches the ICLR 2025 Rotation Trick to any VQ layer.
[![PyPI version](https://badge.fury.io/py/vqrotate.svg)](https://badge.fury.io/py/vqrotate)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
# Why VQRotate
Vector quantization introduces a non-differentiable nearest-neighbour operation. The standard solution is the Straight Through Estimator, which approximates the backward pass as an identity transformation.

The Rotation Trick provides another way to propagate gradients by taking the geometry of the encoder output and quantized vector into account.

Combining adaptive scaling experimentally with rotation trick it tells when to amplify and dampen.

vqrotate packages these approaches behind a simple API so that they can be easily applied and compared.

## Features

- **One-line API:** `attach_rotation(quantizer)`
- **Drop-in compatible:** Works with any VQ layer that has a `forward` method.
- **Multiple strategies:** `ste`, `rotation` (default), `adaptive` (experimental), `reflection`.
- **Multi-head support:** Handles lucidrains-style multi-head codebooks.
- **Mixed precision ready:** AMP compatible.
- **No user codebook required:** The rotation trick is applied automatically.

---





## Installation

```bash
pip install vqrotate
```

# Quick Start
```python
from vqrotate import attach_rotation
from vector_quantize_pytorch import VectorQuantize

# Create a quantizer
vq = VectorQuantize(dim=256, codebook_size=1024)

# Attach the Rotation Trick (one line)
attach_rotation(vq, strategy="rotation")

# Use it normally
quantized, indices, commit_loss = vq(x)
```
To change the strategy, simply change the strategy argument:

```python
attach_rotation(vq, strategy="adaptive")   # experimental
attach_rotation(vq, strategy="ste")        # baseline
```
# Benchmark

Refer to Benchmark-repo and experiments-docs

# Contributing

Please, it might be a great help :(

# Citation
If you use this library in your research, please cite the original paper:

```bibtex
@inproceedings{fifty2025rotation,
  title={Restructuring Vector Quantization with the Rotation Trick},
  author={Fifty, Christopher and Junkins, Ronald G. and Duan, Dennis and Iyengar, Aniketh and Liu, Jerry W. and Amid, Ehsan and Thrun, Sebastian and Ré, Christopher},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

# Acknowledgements
This project is based on the ICLR 2025 paper by Christopher Fifty, Ronald G. Junkins, et al. The Householder reflection implementation is inspired by the lucidrains/vector-quantize-pytorch library.

# Contact
Reach out to:
Discord : @develepo
Or just ping me or any maintainer(if any?) on github!
