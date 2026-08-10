from core import _handle_multihead, strategy, shape_info
import torch as th
A =  [2,3]

_handle_multihead(th.rand(2,3,4))