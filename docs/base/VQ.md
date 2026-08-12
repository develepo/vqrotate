# Vector Quantization
Suppose we have an encoder which gives an output $z_e$ and we have a codebook
$\mathcal{C} = \{q_1, q_2, \ldots, q_K\}$.

The quantized vector $z_q$ is obtained by finding the nearest codebook
vector to $z_e$:

$$
z_q = q_k
\quad \text{where} \quad
k = \arg\min_j \|z_e - q_j\|_2
$$

This looking for nearest-neighbor is the vector quantization operation.

for example,
An encoder gives the output:

$$
z_e = [2.1,\ 1.8]
$$

And our codebook contains these vectors:

$$
q_1 = [2,\ 2]
$$

$$
q_2 = [5,\ 4]
$$

$$
q_3 = [0,\ 1]
$$

To quantize $z_e$, we calculate its Euclidean distance from every
codebook vector.

For $q_1$:

$$
\begin{aligned}
\|z_e-q_1\|_2
&= \sqrt{(2.1-2)^2 + (1.8-2)^2} \\
&= \sqrt{0.1^2 + (-0.2)^2} \\
&= \sqrt{0.01 + 0.04} \\
&= \sqrt{0.05} \\
&\approx 0.224
\end{aligned}
$$

For $q_2$:

$$
\begin{aligned}
\|z_e-q_2\|_2
&= \sqrt{(2.1-5)^2 + (1.8-4)^2} \\
&= \sqrt{(-2.9)^2 + (-2.2)^2} \\
&= \sqrt{8.41 + 4.84} \\
&= \sqrt{13.25} \\
&\approx 3.640
\end{aligned}
$$

For $q_3$:

$$
\begin{aligned}
\|z_e-q_3\|_2
&= \sqrt{(2.1-0)^2 + (1.8-1)^2} \\
&= \sqrt{2.1^2 + 0.8^2} \\
&= \sqrt{4.41 + 0.64} \\
&= \sqrt{5.05} \\
&\approx 2.247
\end{aligned}
$$

Comparing the distances:

$$
0.224 < 2.247 < 3.640
$$

Therefore, $q_1$ is the nearest codebook vector.

So the quantized output is:

$$
z_q = q_1 = [2,\ 2]
$$

This nearest-neighbor selection is the vector quantization operation.

In Python using PyTorch, this can be done in one line:

```python
z_q = codebook[torch.argmin(torch.cdist(z_e, codebook), dim=1)] 
```

> **Note:** The quantization function is piecewise constant. Within a Voronoi cell, all encoder outputs are mapped to the same codebook vector. The output only changes when the encoder output crosses a Voronoi boundary or the hyperplane.