# Why a Custom Autograd Function?
To override the gradient of the VQ operation. 
# Why Matrix-Free Householder?
Building a full d x d rotation matrix is O(d²) memory. The matrix-free approach is O(d) memory.
# Why FP32 Casting in the Backward?
Because the Householder reflection involves division.
# Why the Fallback for e_hat + q_hat ≈ 0?
YK what happens when you divide by a 0
# Why is the adaptive strategy experimental?
Uhm, because I am still trying to make it work plus I lack resources to test it broadly. Maybe someday.
Advice : Use it for experimentation, but rely on `"rotation"` for production or stable benchmarks.
if it works for u, congratulations.
# What is the difference between `attach_rotator` and `attach_rotation`?
Nothing. They are **aliases**., I was just being cool.
# Does vqrotate support multi‑head codebooks?
Yes ig, ping me if it doesn't.
# Can I use vqrotate with JIT compilation?
no unless you are willing to rewrite backward pass
# What is the difference between STE, Rotation, and Adaptive?
Better explained in base docs
# What is α in the adaptive strategy?
Adaptive Factor, refer to GNAS.md in \advanced
# Why does the benchmark use 10 epochs?
Lack of resources and impatience
# What are the hardware requirements to run benchmark?
Cuda GPU(minimum 4gb)
will work on cpu too but training is slower
ram 8gb
4gb space(mostly dataset)

# Why do I get different results across runs?
Sensitivity to random initilization, my advice is to run 3-5 seeds and report the average. why I didn't do this? Laziness.
# Where can I report issues?
Open an issue



