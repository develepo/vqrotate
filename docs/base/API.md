# API Reference

## One-Line API

The library handles the gradient propagation internally.

### `attach_rotator`

`attach_rotator` attaches a gradient propagation strategy to an existing quantizer.

#### Parameters

| Parameter   | Description                                                      |
|-------------|------------------------------------------------------------------|
| `quantizer` | The vector quantizer to which the strategy will be attached.     |
| `strategy`  | The gradient propagation strategy to use. See available strategies below. |

---

### Available Strategies

| Strategy   | Description                                               |
|------------|-----------------------------------------------------------|
| `ste`      | Uses the Straight Through Estimator.                      |
| `rotation` | Uses the Rotation Trick.                                  |
| `adaptive` | Uses Adaptive Scaling. **Experimental:** intended for experimentation and research. |

---

## Basic Usage

Suppose you already have a quantizer in your VQ-VAE:

```python
quantizer = MyQuantizer()
```
You can attach the Rotation Trick with one line:
```python
attach_rotator(quantizer, strategy="rotation")
```

The quantizer can then be used normally:
```python
z_q, loss, indices = quantizer(z)
```
our lib handles the selected gradient propagation strategy internally.

***If you wanna use any other strategy, just change it like "ste" for using Straight through estimator and "adaptive" for adaptive scaling***

## Complete Example
```python
from vqrotate import attach_rotator
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(dim=256, codebook_size=1024)

attach_rotator(vq, strategy="rotation")

quantized, indices, commit_loss = vq(x)
```
To change the strategy, only the strategy argument needs to be changed:

```python
attach_rotator(vq, strategy="adaptive")
```
attach_rotation:
attach_rotation is an alias for attach_rotator.
It provides the same functionality as attach_rotator.

Version
Current version: 0.3.0

