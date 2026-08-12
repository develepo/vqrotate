# VQRotate

Vqrotate implements ICLR 2025 Rotation Trick Paper (Fifty et al.), in the paper, they present a way to propagate gradients through the vector quantization layer of VQ-VAEs without losing the information (preserving direction...) from VQ operator.

This library is focused on implementing it efficiently and provide a one liner PyPI package to apply ways of propagating grads such as **STE (Straight Through Estimator)**, **Rotation Trick** and **Adaptive Scaling (experimental)**.

Rotation Trick provides a way to propagate gradients through the VQ layer using a rotation-based transformation, whereas STE just estimates the Jacobian is approximated as the identity matrix.

This library has been tested by benchmark using **SPEECH COMMANDS** dataset, where it shows behaviours affected by the constraints. This is further explained well.