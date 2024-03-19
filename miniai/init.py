# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/11_initializing.ipynb.

# %% auto 0
__all__ = ['clean_ipython_hist', 'clean_tb', 'clean_mem', 'BatchTransformCB', 'GeneralReLU', 'plot_func', 'init_weights',
           'lsuv_init', 'LsuvCB', 'conv', 'get_model']

# %% ../nbs/11_initializing.ipynb 3
import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl,numpy as np,matplotlib.pyplot as plt
import sys,gc,traceback
import fastcore.all as fc
from collections.abc import Mapping
from pathlib import Path
from operator import attrgetter,itemgetter
from functools import partial
from copy import copy
from contextlib import contextmanager

import torchvision.transforms.functional as TF,torch.nn.functional as F
from torch import tensor,nn,optim
from torch.utils.data import DataLoader,default_collate
from torch.nn import init
from torcheval.metrics import MulticlassAccuracy
from datasets import load_dataset,load_dataset_builder

from .datasets import *
from .conv import *
from .learner import *
from .activations import *

# %% ../nbs/11_initializing.ipynb 15
def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals(): return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc): user_ns.pop('_i'+repr(n),None)
    user_ns.update(dict(_i='',_ii='',_iii=''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 =  ''

# %% ../nbs/11_initializing.ipynb 16
def clean_tb():
    # h/t Piotr Czapla
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'): delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'): delattr(sys, 'last_value')

# %% ../nbs/11_initializing.ipynb 17
def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()

# %% ../nbs/11_initializing.ipynb 37
class BatchTransformCB(Callback):
    def __init__(self, transform, on_train=True, on_valid=True): fc.store_attr()

    def before_batch(self, learn):
        if (learn.training and self.on_train) or (not learn.training and self.on_valid):
            learn.batch = self.transform(learn.batch)

# %% ../nbs/11_initializing.ipynb 50
class GeneralReLU(nn.Module):
    def __init__(self, leak=0, sub=0, maxv=0):
        assert(leak >= 0)
        assert(sub >= 0)
        assert(maxv >= 0)
        super().__init__()
        fc.store_attr()

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak else F.relu(x)
        if self.sub:
            x -= self.sub
        if self.maxv:
            x = torch.clamp_max_(x, self.maxv)
        return x

# %% ../nbs/11_initializing.ipynb 51
def plot_func(f, start=-10, end=10, steps=100):
    xs = torch.linspace(start, end, steps)
    plt.plot(xs, f(xs))
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.grid()

# %% ../nbs/11_initializing.ipynb 56
def init_weights(m, leak):
    if isinstance(m, nn.Conv2d):
        nonlin = "leaky_relu" if leak else "relu"
        init.kaiming_normal_(m.weight, nonlinearity=nonlin, a=leak)

# %% ../nbs/11_initializing.ipynb 62
def _lsuv_stats(hook, mod, inp, outp):
    hook.mean = to_cpu(outp.mean())
    hook.std = to_cpu(outp.std())

# %% ../nbs/11_initializing.ipynb 63
def lsuv_init(model, m, m_in, xb):
    hook = Hook(m_in, _lsuv_stats)
    with torch.no_grad():
        while model(xb) is not None and hook.mean.abs() > 1e-3 and (hook.std - 1).abs() > 1e-3:
            m.weight /= hook.std
            m.bias -= hook.mean
    hook.remove()
            

# %% ../nbs/11_initializing.ipynb 70
class LsuvCB(Callback):
    def __init__(self):
        super().__init__()
        self.done = False

    def reset(self, learn):
        learn.lsuvcb_done = False

    def before_fit(self, learn):
        if not hasattr(learn, "lsuvcb_done"):
            learn.lsuvcb_done = False
        if not learn.training or learn.lsuvcb_done:
            return

        prev = None
        weights = []
        acts = []
        for cur in learn.model.modules():
            if prev and hasattr(prev, "weight"):
                weights.append(prev)
                if hasattr(cur, "weight"):
                    acts.append(prev)
                else:
                    acts.append(cur)
            prev = cur

        to_init = list(zip(weights, acts))
        xb = next(iter(learn.dls.train))[0].to(device=next(learn.model.parameters()).device)
        for m, m_in in to_init:
            lsuv_init(model, m, m_in, xb)
        learn.lsuvcb_done = True

# %% ../nbs/11_initializing.ipynb 81
def conv(ni, nf, ks=3, stride=2, act=nn.ReLU, norm=None, bias=None):
    if bias is None:
        bias = not isinstance(norm, nn.BatchNorm2d)
            
    layers = [nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)]
    if act:
        layers.append(act())
    if norm:
        layers.append(norm(nf))
    result = nn.Sequential(*layers)
    return result

# %% ../nbs/11_initializing.ipynb 83
def get_model(act=nn.ReLU, nfs=[1,8,16,32,64], norm=None, bias=None):
    layers = [conv(nfs[i], nfs[i+1], act=act, norm=norm, bias=bias) for i in range(len(nfs)-1)]
    model = nn.Sequential(*layers, conv(64, 10, act=False), nn.Flatten())
    return model
