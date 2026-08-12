# Householder Reflection
The Householder reflection is the mathematical foundation of the Rotation Trick. It is a linear transformation that reflects a vector across a hyperplane.

## Definition
For a unit vector `v` (||v|| = 1), the Householder matrix is:

$$
\mathbf{H}_v = \mathbf{I} - 2vv^T
$$

## Derivation

We want a linear transformation that reflects any vector `x` across the hyperplane orthogonal to `v`.

**Step 1:** Decompose `x` into components parallel and perpendicular to `v`:

$$
x = x_\parallel + x_\perp
$$

where:

$$
x_\parallel = (x \cdot v)v
$$

and:

$$
x_\perp = x - x_\parallel
$$

**Step 2:** A reflection should vanish the parallel component and leave the perpendicular component unchanged:
$$
\mathbf{H}_v x = x_\perp - x_\parallel
$$

**Step 3:** Substitute `x_perp = x - x_parallel`:

$$
\mathbf{H}_v x = (x - x_\parallel) - x_\parallel = x - 2x_\parallel
$$

**Step 4:** Substitute `x_parallel = (x · v)v`:

$$
\mathbf{H}_v x = x - 2(x \cdot v)v
$$

In matrix form:

$$
\mathbf{H}_v = \mathbf{I} - 2vv^T
$$

## Proof of Properties
### 1. Orthogonality
$$
\mathbf{H}_v^T \mathbf{H}_v = (\mathbf{I} - 2vv^T)^T(\mathbf{I} - 2vv^T)
$$

Since `vv^T` is symmetric and `v^T v = 1`:

$$
\mathbf{H}_v^T \mathbf{H}_v = (\mathbf{I} - 2vv^T)(\mathbf{I} - 2vv^T) = \mathbf{I} - 4vv^T + 4v(v^T v)v^T = \mathbf{I} - 4vv^T + 4vv^T = \mathbf{I}
$$

So `H_v` is orthogonal.
 `H_v` is orthogonal.

### 2. Involution (Self-Inverse)

$$
\mathbf{H}_v \mathbf{H}_v = \mathbf{I}
$$

This follows directly from orthogonality.

### 3. Determinant

$$
\det(\mathbf{H}_v) = -1
$$

Reflections flip orientation. that's why two reflections make a rotation.

## Matrix-Free Application
For any vector `x`:

$$
\mathbf{H}_v x = x - 2v(v \cdot x)
$$

This is O(d) memory and O(d) compute. No matrix is ever stored.
### How?
A householder is defined as:

$$
\mathbf{H}_v = \mathbf{I} - 2vv^T
$$

Matrix multiplication distributes over addition:

$$
\mathbf{H}_v x = \mathbf{I}x - 2(vv^T)x
$$

- `I x = x` (identity matrix leaves a vector unchanged).
- For `(vv^T)x`, use the associativity of matrix multiplication:

$$
(vv^T)x = v(v^T x)
$$

Since `v^T x` is a **scalar** (the dot product of `v` and `x`), we write:

$$
(vv^T)x = v (v \cdot x)
$$

Substitute back:

$$
\boxed{\mathbf{H}_v x = x - 2v(v \cdot x)}
$$

## In the Code

The Householder reflection is used to compute the halfway vector `r` and to apply the rotation:

```python
# Compute r
r = sum_hat / sum_hat_norm
```
# Apply reflection matrix-free
grad_e = grad - 2 * r * (r · grad)