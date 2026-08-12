# Construction of the Rotation Matrix
## Goal

Construct a rotation matrix `R` such that:

$$
\mathbf{R} \hat{e} = \hat{q}
$$

## Step 1: Two Reflections Make a Rotation

We know that two reflections make a rotation
so,

Let `R` be the product of:

1. Reflection across `r^⊥` (the hyperplane orthogonal to the halfway vector)
2. Reflection across `q_hat^⊥` (the hyperplane orthogonal to `q_hat`)

$$
\mathbf{R} = \mathbf{H}_{\hat{q}} \mathbf{H}_r
$$

## Step 2: Apply to e_hat

First reflection:

$$
\mathbf{H}_r \hat{e} = -\hat{q}
$$

Second reflection:

$$
\mathbf{H}_{\hat{q}} (-\hat{q}) = \hat{q}
$$

Therefore:

$$
\mathbf{R} \hat{e} = \mathbf{H}_{\hat{q}} (\mathbf{H}_r \hat{e}) = \mathbf{H}_{\hat{q}} (-\hat{q}) = \hat{q}
$$

## Step 3: Expand the Product

$$
\mathbf{R} = (\mathbf{I} - 2\hat{q}\hat{q}^T)(\mathbf{I} - 2rr^T)
$$

Expanding:

$$
\mathbf{R} = \mathbf{I} - 2rr^T - 2\hat{q}\hat{q}^T + 4\hat{q}\hat{q}^T r r^T
$$

## Step 4: Simplify Using `q_hat^T r`

We need to compute `q_hat^T r`.

Let `s = ||e_hat + q_hat||`. Then:

$$
r = \frac{\hat{e} + \hat{q}}{s}
$$

So:

$$
\hat{q}^T r = \frac{\hat{q} \cdot (\hat{e} + \hat{q})}{s} = \frac{\hat{q} \cdot \hat{e} + 1}{s} = \frac{s^2/2}{s} = \frac{s}{2}
$$

## Step 5: Substitute into the Product

$$
4\hat{q}\hat{q}^T r r^T = 4\hat{q}(\hat{q}^T r) r^T = 4\hat{q}\left(\frac{s}{2}\right) r^T = 2s \hat{q} r^T
$$

Since `2s r^T = 2(e_hat^T + q_hat^T)`:

$$
4\hat{q}\hat{q}^T r r^T = 2\hat{q}(\hat{e}^T + \hat{q}^T) = 2\hat{q}\hat{e}^T + 2\hat{q}\hat{q}^T
$$

## Step 6: Final Simplification

Substitute back into the expanded product:

$$
\mathbf{R} = \mathbf{I} - 2rr^T - 2\hat{q}\hat{q}^T + 2\hat{q}\hat{e}^T + 2\hat{q}\hat{q}^T
$$

The `-2q_hat q_hat^T + 2q_hat q_hat^T` cancel:

$$
\boxed{\mathbf{R} = \mathbf{I} - 2rr^T + 2\hat{q}\hat{e}^T}
$$

## Matrix-Free Application

For any vector `v`:

$$
\mathbf{R}v = v - 2r(r \cdot v) + 2\hat{q}(\hat{e} \cdot v)
$$

This is O(d) memory and O(d) compute.
