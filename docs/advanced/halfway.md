# The Halfway Vector
## Definition

Let `e_hat = e / ||e||` and `q_hat = q / ||q||`. The halfway vector is:

$$
r = \frac{\hat{e} + \hat{q}}{\|\hat{e} + \hat{q}\|}
$$

## Derivation of the Reflection Property

We prove that reflecting `e_hat` across the hyperplane orthogonal to `r` maps it to `-q_hat`.

**Step 1:** The reflection of `e_hat` across `r^⊥` is:

$$
\mathbf{H}_r \hat{e} = \hat{e} - 2r(r \cdot \hat{e})
$$

**Step 2:** Let `s = ||e_hat + q_hat||`. Compute `r · e_hat`:

$$
r \cdot \hat{e} = \frac{(\hat{e} + \hat{q}) \cdot \hat{e}}{\|\hat{e} + \hat{q}\|} = \frac{\hat{e} \cdot \hat{e} + \hat{q} \cdot \hat{e}}{s}
$$

Since `e_hat · e_hat = 1`:

$$
r \cdot \hat{e} = \frac{1 + \hat{q} \cdot \hat{e}}{s}
$$

**Step 3:** Note that:

$$
s^2 = \|\hat{e} + \hat{q}\|^2 = 2 + 2(\hat{e} \cdot \hat{q}) = 2(1 + \hat{e} \cdot \hat{q})
$$

Therefore:

$$
1 + \hat{e} \cdot \hat{q} = \frac{s^2}{2}
$$

**Step 4:** Substitute:

$$
r \cdot \hat{e} = \frac{s^2/2}{s} = \frac{s}{2}
$$

**Step 5:** Substitute back into the reflection:

$$
\mathbf{H}_r \hat{e} = \hat{e} - 2r\left(\frac{s}{2}\right) = \hat{e} - s r
$$

Since `s r = e_hat + q_hat`:

$$
\mathbf{H}_r \hat{e} = \hat{e} - (\hat{e} + \hat{q}) = -\hat{q}
$$

## Conclusion

A single reflection across `r^⊥` maps `e_hat` to `-q_hat`. The second reflection (across `q_hat^⊥`) maps `-q_hat` to `q_hat`, completing the rotation.

## Edge Case

When `e_hat = -q_hat`, the vector `e_hat + q_hat = 0`. The halfway vector is undefined. This case is handled by the fallback to STE.

## Visual Proof

The halfway vector is the angle bisector of `e_hat` and `q_hat`. Reflecting across the bisector flips `e_hat` to the opposite side, giving `-q_hat`.
