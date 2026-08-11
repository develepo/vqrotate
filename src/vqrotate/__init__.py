'''vqrotator: A library for vector quantization with rotation and adaptive scaling.
one line API:
from vqrotator import attach_rotator
attach_rotator(quantizer, strategy="adaptive")'''

from .core import attach_rotator, RotationQuantization, Rotator

attach_rotation = attach_rotator

__all__ = [
    "attach_rotation",
    "attach_rotator",
    "RotationQuantization",
    "Rotator",
]
__version__ = "0.2.1"