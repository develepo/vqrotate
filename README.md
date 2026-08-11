# vqrotate

**One-line Rotation Trick for VQ-VAEs.**

Drop-in wrapper that attaches the ICLR 2025 Rotation Trick to any VQ layer.

## Features
- Single line of code: `attach_rotator(quantizer)`
- Multi-head support (lucidrains compatible)
- Mixed precision (AMP) ready

## Install
```bash
pip install git+https://github.com/develepo/vqrotate.git