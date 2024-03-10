# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/07_convolutions.ipynb.

# %% auto 0
__all__ = ['def_device', 'conv', 'to_device', 'collate_device']

# %% ../nbs/07_convolutions.ipynb 3
import torch
from torch import nn

from torch.utils.data import default_collate
from typing import Mapping

from .training import *
from .datasets import *

# %% ../nbs/07_convolutions.ipynb 49
def conv(ni, nf, ks=3, stride=2, act=True):
    result = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2)
    if act:
        result = nn.Sequential(result, nn.ReLU())
    return result

# %% ../nbs/07_convolutions.ipynb 54
def_device = "cuda" if torch.cuda.is_available() else "cpu"

# %% ../nbs/07_convolutions.ipynb 55
def to_device(x, device=def_device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device)
    if isinstance(x, Mapping):
        return {k: to_device(v, device) for k, v in x.items()}
    return type(x)(to_device(i, device) for i in x)

# %% ../nbs/07_convolutions.ipynb 56
def collate_device(b):
    return to_device(default_collate(b))
