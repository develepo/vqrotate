# vqrotate

**One-line Rotation Trick for VQ-VAEs.**

Drop-in wrapper that attaches the ICLR 2025 Rotation Trick to any VQ layer.

## Features
- Single line of code: `attach_rotation(quantizer)`
- Multi-head support (lucidrains compatible)
- Mixed precision (AMP) ready
- Adaptive gradient scaling (89% codebook usage on Speech Commands)

## Install
```bash
pip install git+https://github.com/develepo/vqrotate.git