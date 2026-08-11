"""
vqrotate.core -- Rotation Trick Applied for Vector Quantization.

Implements the Rotation Trick (Fifty et al., ICLR 2025), which overrides the
backward pass of the vector quantization operation. Instead of the standard
Straight-Through Estimator (STE) that copies gradients from q to e, this module
rotates the gradient using a matrix-free Householder reflection.

This implementation includes:
    - Multi-head codebook support (lucidrains compatible)
    - Mixed precision (AMP) handling via FP32 casting
    - Adaptive gradient scaling with dimension-normalized thresholds
    - Graceful fallbacks for numerical edge cases
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Callable, Literal, Any
import warnings

strategy = Literal["ste", "adaptive", "rotation", "reflection"]
shape_info = Tuple[Tuple[int, ...], int]  

def _handle_multihead(e: th.Tensor) -> Tuple[th.Tensor, shape_info]:
    orginal_shape = e.shape
    ndim = e.ndim
    if ndim ==2:
        return e, (orginal_shape, 0)
    elif ndim == 3:
        mode =1 
    elif ndim == 4:
        mode = 2
    else:
        mode = 3

    flattened = e.view(-1, e.shape[-1])
    return flattened, (orginal_shape, mode)    

def _restore_multihead(flattened: th.Tensor, shape_info: shape_info) -> th.Tensor:
    original_shape, mode = shape_info
    if mode == 0:
        return flattened

    return flattened.view(*original_shape)

def _safe_norm(t : th.Tensor, dim : int = -1 , eps : float = 1e-8) -> th.Tensor:
    norm = t.norm(dim=dim, keepdim=True)
    return norm.clamp(min=eps)

def _householder_rotation(e: th.Tensor, q: th.Tensor, grad_output: th.Tensor, eps: float = 1e-8) -> th.Tensor:
    original_dtype=e.dtype
    e_f32 = e.float()
    q_f32 = q.float()
    grad_output_f32 = grad_output.float()
    e_norm = _safe_norm(e_f32, dim=-1, eps=eps)
    q_norm = _safe_norm(q_f32, dim=-1, eps=eps)
    e_hat = e_f32 / e_norm
    q_hat = q_f32 / q_norm
    sum_hat = e_hat + q_hat
    sum_hat_norm = _safe_norm(sum_hat, dim=-1, eps=eps)
    r = sum_hat / sum_hat_norm
    is_dead = (sum_hat_norm <= eps).squeeze(-1)

    r_dot_v = th.sum(r * grad_output_f32, dim=-1, keepdim=True)
    e_hat_dot_v = th.sum(e_hat * grad_output_f32, dim=-1, keepdim=True)
    grad_e_f32 = grad_output_f32 - 2 * r * r_dot_v + 2 * q_hat * e_hat_dot_v
    if is_dead.any():
        grad_e_f32[is_dead] = grad_output_f32[is_dead]

    return grad_e_f32.to(original_dtype)

def _adaptive_scale(e : th.Tensor, q: th.Tensor, grad_e : th.Tensor, dim: int, codebook: Optional[th.Tensor]=None) -> th.Tensor:
    eps = 1e-8
    e_norm = _safe_norm(e, dim=-1,  eps=eps)
    q_norm = _safe_norm(q, dim=-1, eps=eps)
    base_scale = q_norm / e_norm
    lamba_here = grad_e * base_scale
    if codebook is None :
        return lamba_here
    #computing q2 coz we need it for geometric adaption
    e_flat = e.view(-1, dim)
    distances = th.cdist(e_flat, codebook)
    indices = th.argsort(distances, dim=1)
    q2_indices = indices[:, 1]
    q2 = codebook[q2_indices]
    q2 = q2.view(e.shape)
    #computing the distance ratio
    d1 = th.norm(e-q, dim=-1, keepdim=True)
    d2 = th.norm(e-q2, dim=-1, keepdim=True)
    gamma = d1/(d1+d2+eps)
    #alignment
    direction = q2-q
    direction_hat = direction/(direction.norm(dim=-1, keepdim=True) + eps)
    grad_hat = grad_e / (grad_e.norm(dim=-1, keepdim=True) + eps)
    alignment = (grad_hat * direction_hat).sum(dim=-1, keepdim=True)
    usefulness = F.relu(alignment)
    #assembling
    geo_adapt = 1.0 + gamma*usefulness
    final_adapt = base_scale*geo_adapt
    final_scale = th.clamp(final_adapt, 0.1,2.0)
    return grad_e* final_scale

class RotationQuantization(th.autograd.Function):
    @staticmethod
    def forward(ctx, e,q, strategy, multihead, eps, codebook):
        ctx.save_for_backward(e, q)
        ctx.strategy = strategy
        ctx.multihead = multihead
        ctx.eps = eps
        ctx.embedding_dim = e.shape[-1]
        ctx.original_dtype = e.dtype
        ctx.shape_info = None
        ctx.codebook = codebook

        if multihead:
            e_flat, shape_info = _handle_multihead(e)
            q_flat, _ = _handle_multihead(q)
            ctx.e_flat = e_flat
            ctx.q_flat = q_flat
            ctx.shape_info = shape_info
        else:
            ctx.e_flat = e
            ctx.q_flat = q
        return q
    @staticmethod
    def backward(ctx, grad_output):
        e,q = ctx.saved_tensors
        strategy = ctx.strategy
        multihead = ctx.multihead
        eps = ctx.eps
        dim = ctx.embedding_dim
        codebook = ctx.codebook
        if multihead:
            e_flat = ctx.e_flat
            q_flat = ctx.q_flat
            grad_flat, _= _handle_multihead(grad_output)
            shape_info = ctx.shape_info
        else:
            e_flat = e
            q_flat = q
            grad_flat = grad_output
        grad_e_flat = _householder_rotation(e_flat, q_flat, grad_flat, eps)
        if strategy == "ste":
            grad_e_flat = grad_flat
        elif strategy == "rotation":
            e_norm = _safe_norm(e_flat, dim=-1, eps=eps)
            q_norm = _safe_norm(q_flat, dim=-1, eps=eps)
            grad_e_flat = grad_e_flat * (q_norm / e_norm)
        elif strategy == "adaptive":
            grad_e_flat = _adaptive_scale(e_flat, q_flat, grad_e_flat, dim, codebook)
        elif strategy == "reflection":
            grad_e_flat = grad_flat
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        if multihead:
            grad_e = _restore_multihead(grad_e_flat, shape_info)
        else:  
            grad_e = grad_e_flat
        return grad_e, None, None, None, None, None, None

class Rotator():
    @staticmethod
    def wrap(quantize_fn, strategy: Literal["ste", "rotation", "adaptive", "reflection"] = "rotation", multihead=True, eps:float=1e-8):
        def wrapped(z):
            output = quantize_fn(z)
            z_q = output[0] if isinstance(output, (tuple, list)) else output
            if strategy == "adaptive":
                try:
                    codebook = quantize_fn.__self__.embedding.weight
                except AttributeError:
                    codebook=None
            else:
                codebook=None        
            z_q_rotated = RotationQuantization.apply(z, z_q, strategy, multihead, eps, codebook)
            if isinstance(output, (tuple, list)):
                return (z_q_rotated, *output[1:])
            return z_q_rotated
        return wrapped

def attach_rotator(quantizer, strategy: Literal["ste", "rotation", "adaptive", "reflection"] = "rotation", multihead=True, eps:float=1e-8):
    if not hasattr(quantizer, 'forward'):
        raise ValueError("The quantizer must have a 'forward' method.")
    original_forward = quantizer.forward
    wrapped_forward = Rotator.wrap(original_forward, strategy=strategy, multihead=multihead, eps=eps)
    quantizer.forward = wrapped_forward
    return quantizer