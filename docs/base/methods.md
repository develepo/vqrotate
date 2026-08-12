# Methods

We have used 3 main methods or strategy as we like to call it:

## 1. Straight Through Estimator (STE)
This is the standard to propagate gradients through the VQ operation.

After the quantization during forward pass:
$$
z_q = q_k
$$

where $q_k$ is the nearest codebook vector to $z_e$.

Since, our nearest neighbour is not differantiable becuase of it being piecewise constant because of argmin.
STE simply pass this backward by treating the quantization operation as an identity function $I$

Formulatingly :
$$
\frac{\partial z_q}{\partial z_e} \approx I
$$

Because of this : 

$$
\frac{\partial L}{\partial z_e}
\approx
\frac{\partial L}{\partial z_q}
$$

This results in loss of information, that's the problem, rotation and adaption methods are intended to solve.

## 2. Rotation Trick
As proposed in ICLR 2025 paper by Fifty et al. , this trick preserves the information by replacing the identity approximation used by STE with a differantiable transformation 

Formulatingly:

$$
g_e = R g_q
$$

where $g_q$ is the gradient arriving from the decoder and $R$ represents the transformation determined by the encoder and quantized vectors derived using Householder reflection

The resulting gradient can be written as:

$$
g_e = \lambda R g_q
$$

where:

- $R$ is the rotation/reflection transformation between $z_e$ and $z_q$.
- $\lambda$ is a scaling factor determined by the magnitudes of the two
  vectors.

The scaling factor is:

$$
\lambda = \frac{\|z_e\|}{\|z_q\|}
$$

Thus, the Rotation Trick does two things:

1. **Rotates the gradient** according to the relationship between $z_e$ and
   $z_q$.
2. **Scales the gradient** according to the relative magnitudes of the
   encoder output and quantized vector.

This allows the backward pass to retain geometric information that is lost when the quantization operation is simply treated as the identity.

For more brief dive, you can check the original paper or the \advanced modules.

## 3. Adaptive Scaling

Adaptive Scaling builds on top of the Rotation Trick.

### Why? What is the need?
The Rotation Trick tells us how the gradient should be transformed based on the geometry of the encoder output and its quantized representation.

But the morm of the resulting gradient may not necessarily be equally useful for every point in a batch. I mean by this that some points may produce relatively large gradients, while others may produce relatively small gradients. Treating all of these gradients in the same way
may not always be desirable.

##

The gradient after the Rotation Trick is:

$$
g_e = \lambda R g_q
$$

Adaptive Scaling add $\alpha$:

$$
g_e' = \alpha \lambda R g_q
$$

where $\alpha$ controls the question should the gradient is amplified or dampened.

The adaptive factor is:

$$
\alpha = 1 + \tanh(z)
$$

where $z$ is the batch-normalized gradient norm:

$$
z = \frac{\|g\|-\mu}{\sigma+\epsilon}
$$

Here:

- $g$ is the rotated gradient.
- $\|g\|$ is the gradient norm for each point.
- $\mu$ is the mean gradient norm in the current batch.
- $\sigma$ is the standard deviation of the gradient norms.
- $\epsilon$ is a small constant ($10^{-8}$) used to prevent division by zero.

The resulting $\alpha$ is bounded between $0$ and $2$:

$$
0 < \alpha < 2
$$

So:

- **Large gradient relative to the batch** → $z>0$ → $\alpha>1$ → gradient is amplified.
- **Average gradient** → $z\approx0$ → $\alpha\approx1$ → little to no modulation.
- **Small gradient relative to the batch** → $z<0$ → $\alpha<1$ → gradient is dampened.

Hence, Adaptive Scaling does not replace the Rotation Trick. It adds a batch-dependent modulation of its gradient magnitude.

> **⚠️ Experimental:** Adaptive Scaling is an experimental feature.
> It is provided for experimentation and research and should not be
> considered a validated improvement over the other gradient propagation
> strategies.
