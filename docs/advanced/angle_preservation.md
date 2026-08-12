# Angle Preservation
## Theorem
The Rotation Trick preserves the angle between the codebook vector `q` and its gradient `∇qL`:

$$
\angle(e, \nabla_e \mathcal{L}) = \angle(q, \nabla_q \mathcal{L})
$$

# Given

The Rotation Trick sets:

$$
\nabla_e \mathcal{L} = \lambda \mathbf{R} \nabla_q \mathcal{L}
$$

where:

- `λ = ||q|| / ||e||`
- `R` is the rotation matrix such that `R e = q / λ`

## Proof

### Step 1: Dot Product

Compute the dot product at `e`:

$$
e \cdot \nabla_e \mathcal{L} = e \cdot (\lambda \mathbf{R} \nabla_q \mathcal{L}) = \lambda (\mathbf{R}^T e) \cdot \nabla_q \mathcal{L}
$$

Since `R` is orthogonal, `R^T e = λ^{-1} q`:

$$
e \cdot \nabla_e \mathcal{L} = \lambda (\lambda^{-1} q) \cdot \nabla_q \mathcal{L} = q \cdot \nabla_q \mathcal{L}
$$

### Step 2: Norms

Compute the norm of `∇eL`:

$$
\|\nabla_e \mathcal{L}\| = \|\lambda \mathbf{R} \nabla_q \mathcal{L}\| = \lambda \|\nabla_q \mathcal{L}\|
$$

because `R` is orthogonal (preserves norm).

Compute the norm of `e`:

$$
\|e\| = \|\lambda^{-1} q\| = \lambda^{-1} \|q\|
$$

### Step 3: Cosine Similarity

The cosine of the angle at `e` is:

$$
\cos(\angle e, \nabla_e) = \frac{e \cdot \nabla_e}{\|e\| \|\nabla_e\|}
$$

Substitute the preserved quantities:

$$
\cos(\angle e, \nabla_e) = \frac{q \cdot \nabla_q}{(\lambda^{-1} \|q\|)(\lambda \|\nabla_q\|)} = \frac{q \cdot \nabla_q}{\|q\| \|\nabla_q\|}
$$

The right-hand side is exactly the cosine of the angle at `q`.

### Step 4: Conclusion

Since the cosine is equal and both angles are in `[0, π]`:

$$
\boxed{\angle(e, \nabla_e \mathcal{L}) = \angle(q, \nabla_q \mathcal{L})}
$$

Point to be noted that the angle preservation is independent of the scaling factor `λ` and the specific choice of `R`.

## Also, an usefull insight
The Straight-Through Estimator sets:

$$
\nabla_e \mathcal{L} = \nabla_q \mathcal{L}
$$

In this case:

$$
\cos(\angle e, \nabla_e) = \frac{e \cdot \nabla_q}{\|e\| \|\nabla_q\|}
$$

This is not equal to `cos(angle q, ∇q)` in general. The angle information is lost.

## Geometric Interpretation

The angle preservation means that the gradient direction relative to the codebook vector is transmitted through the quantization layer. This is what allows the Rotation Trick to push boundary points differently from center points.
