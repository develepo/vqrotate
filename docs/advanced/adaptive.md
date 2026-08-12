# Gradient Norm Adaptive Scaling

## Problem Statement

The Rotation Trick applies a fixed scaling:

$$
\lambda_{\text{base}} = \frac{\|q\|}{\|e\|}
$$

This treats all points equally. Points with large gradients and points with small gradients are scaled the same way.

We want a data-dependent scaling that amplifies useful gradients and dampens less useful ones.

## Assumption
Gradient norm is a proxy for "usefulness":

- Large gradients indicate the point is in a high-curvature region (near a boundary). These points should be pushed harder.
- Small gradients indicate the point is in a flat region (already settled). These points should be left alone.

## Step 1: Compute Gradient Norms

Let `g = ∇qL`. Compute the norm for each point:

$$
n_i = \|g_i\|
$$

## Step 2: Batch Statistics

Compute the mean and standard deviation of the gradient norms in the batch:

$$
\mu = \frac{1}{B} \sum_{i=1}^B n_i
$$

$$
\sigma = \sqrt{\frac{1}{B} \sum_{i=1}^B (n_i - \mu)^2}
$$

## Step 3: Z-Score

Compute the z-score for each point:

$$
z_i = \frac{n_i - \mu}{\sigma + \epsilon}
$$

where `ε` is a small constant to prevent division by zero.

## Step 4: Adaptive Factor

We want a function `α(z)` such that:

- `α(0) = 1` (neutral)
- `α(z) > 1` for `z > 0` (amplify)
- `α(z) < 1` for `z < 0` (dampen)
- `α(z)` is bounded to prevent instability

The hyperbolic tangent satisfies all these conditions:

$$
\alpha(z) = 1 + \tanh(z)
$$

## Step 5: Full Scaling

The final scaling is:

$$
\lambda_{\text{final}} = \lambda_{\text{base}} \cdot \alpha(z)
$$

## Step 6: Clamping

To prevent extreme scaling, we clamp:

$$
\lambda_{\text{final}} = \text{clamp}(\lambda_{\text{final}}, 0.1, 2.0)
$$

# Properties

## 1. Boundedness

Since `tanh(z) ∈ (-1, 1)`, we have:

$$
\alpha \in (0, 2)
$$

## 2. Centered at 1

When `z = 0` (gradient is average), `α = 1`. No change.

## 3. Monotonicity

`α(z)` is strictly increasing. Larger gradients get larger amplification.

## 4. Smoothness

The function is smooth and differentiable, preserving gradient flow.


## Limitations

1. **Batch size**: `μ` and `σ` are only reliable with `B ≥ 16`.
2. **Outliers**: A few very large gradients can skew `μ` and `σ`.
3. **Direction blind**: Only uses norm, not direction.


## In the Code

```python
def _adaptive_scale(e, q, grad_e, dim):
    eps = 1e-8
    
    # Base scaling
    e_norm = _safe_norm(e, dim=-1, eps=eps)
    q_norm = _safe_norm(q, dim=-1, eps=eps)
    base_scale = q_norm / e_norm

    # Gradient norms
    grad_norm = grad_e.norm(dim=-1, keepdim=True)
    grad_flat = grad_norm.view(-1, 1)
    
    # Batch statistics
    mu = grad_flat.mean(dim=0, keepdim=True).detach()
    sigma = grad_flat.std(dim=0, keepdim=True).detach()
    
    # Z-score
    z_score = (grad_norm - mu) / (sigma + eps)
    
    # Adaptive factor
    alpha = 1.0 + th.tanh(z_score)
    
    # Final scaling
    final_scale = base_scale * alpha
    final_scale = th.clamp(final_scale, 0.1, 2.0)
    
    return grad_e * final_scale
```
