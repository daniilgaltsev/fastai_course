# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/14_augment.ipynb.

# %% auto 0
__all__ = ['MEGA', 'summary', 'CapturePreds', 'rand_erase', 'RandErase', 'rand_copy', 'RandCopy', 'get_rand_rect_start',
           'rand_swap', 'RandSwap']

# %% ../nbs/14_augment.ipynb 2
import torch,random
import fastcore.all as fc

from torch import nn
from torch.nn import init

from .datasets import *
from .conv import *
from .learner import *
from .activations import *
from .init import *
from .sgd import *
from .resnet import *

# %% ../nbs/14_augment.ipynb 15
def _flops(x, h, w):
    d = x.dim()
    if d > 4:
        raise NotImplementedError()
    if d == 4:
        return h * w * x.numel()
    return x.numel()

# %% ../nbs/14_augment.ipynb 16
MEGA = 1000*1000

@fc.patch
def summary(self:Learner):
    result = "|Module|Input|Output|Num params|~MFLOPS|\n|-|-|-|-|-|\n"
    total = 0
    total_flops = 0
    def _summary(hook, mod, inp, outp):
        nonlocal total, result, total_flops
        num = sum(i.numel() for i in mod.parameters())
        h, w = outp.shape[-2], outp.shape[-1]
        flops = sum(_flops(p, h, w) for p in mod.parameters())
        result += f"|{type(mod).__name__}|{str(inp[0].shape)[11:-1]}|{str(outp.shape)[11:-1]}|{num}|{flops/MEGA:.1f}|\n"
        total += num
        total_flops += flops
    with Hooks(self.model, _summary) as hooks:
        self.fit(1, train=False, cbs=[SingleBatchCB()])
    print(f"TOTAL: params={total}, ~MFLOPs={total_flops/MEGA:.1f}")
    if fc.IN_NOTEBOOK:
        from IPython.display import Markdown
        result = Markdown(result)
    return result

# %% ../nbs/14_augment.ipynb 58
class CapturePreds(Callback):
    def before_fit(self, learn):
        self.captured_inps = []
        self.captured_preds = []
        self.captured_targets = []

    def after_batch(self, learn):
        self.captured_preds.append(to_cpu(learn.preds))
        self.captured_inps.append(to_cpu(learn.batch[0]))
        self.captured_targets.append(to_cpu(learn.batch[1]))

# %% ../nbs/14_augment.ipynb 76
def _rand_erase1(x, pct, xm, xs, mn, mx):
    start_x = int(random.random() * (1 - pct) * x.shape[-2])
    start_y = int(random.random() * (1 - pct) * x.shape[-1])
    size_x = int(pct * x.shape[-2])
    size_y = int(pct * x.shape[-1])
    init.normal_(x[:, :, start_x:start_x+size_x, start_y:start_y+size_y], mean=xm, std=xs)
    torch.clamp_(x, mn, mx)

# %% ../nbs/14_augment.ipynb 78
def rand_erase(x, pct, max_num):
    xm,xs,mn,mx = x.mean(), x.std(), x.min(), x.max()
    num = random.randint(0, max_num)
    for i in range(num): _rand_erase1(x, pct, xm, xs, mn, mx)
    return x

# %% ../nbs/14_augment.ipynb 80
class RandErase(nn.Module):
    def __init__(self, pct, max_num): super().__init__(); fc.store_attr()

    def forward(self, x):
        return rand_erase(x, self.pct, self.max_num)

# %% ../nbs/14_augment.ipynb 85
def _rand_copy1(x, pct):
    start_x1 = int(torch.rand(1) * (1 - pct) * x.shape[-2])
    start_y1 = int(torch.rand(1) * (1 - pct) * x.shape[-1])
    start_x2 = int(torch.rand(1) * (1 - pct) * x.shape[-2])
    start_y2 = int(torch.rand(1) * (1 - pct) * x.shape[-1])
    size_x = int(pct * x.shape[-2])
    size_y = int(pct * x.shape[-1])
    x[:, :, start_x1:start_x1+size_x, start_y1:start_y1+size_y] = x[:, :, start_x2:start_x2+size_x, start_y2:start_y2+size_y]

# %% ../nbs/14_augment.ipynb 86
def rand_copy(x, pct, max_num):
    num = random.randint(0, max_num)
    for i in range(num): _rand_copy1(x, pct)
    return x

# %% ../nbs/14_augment.ipynb 88
class RandCopy(nn.Module):
    def __init__(self, pct, max_num): super().__init__(); fc.store_attr()

    def forward(self, x):
        return rand_copy(x, self.pct, self.max_num)

# %% ../nbs/14_augment.ipynb 94
def get_rand_rect_start(x, pct):
    return (
        int(torch.rand(1) * (1 - pct) * x.shape[-2]),
        int(torch.rand(1) * (1 - pct) * x.shape[-1])
    )

# %% ../nbs/14_augment.ipynb 96
def _rand_swap1(x, pct):
    start_x1, start_y1 = get_rand_rect_start(x, pct)
    start_x2, start_y2 = get_rand_rect_start(x, pct)
    size_x = int(pct * x.shape[-2])
    size_y = int(pct * x.shape[-1])
    p1 = x[:, :, start_x1:start_x1+size_x, start_y1:start_y1+size_y].clone()
    p2 = x[:, :, start_x2:start_x2+size_x, start_y2:start_y2+size_y].clone()
    x[:, :, start_x1:start_x1+size_x, start_y1:start_y1+size_y] = p2
    x[:, :, start_x2:start_x2+size_x, start_y2:start_y2+size_y] = p1

# %% ../nbs/14_augment.ipynb 97
def rand_swap(x, pct, max_num):
    num = random.randint(0, max_num)
    for i in range(num): _rand_swap1(x, pct)
    return x

# %% ../nbs/14_augment.ipynb 99
class RandSwap(nn.Module):
    def __init__(self, pct, max_num): super().__init__(); fc.store_attr()

    def forward(self, x):
        return rand_swap(x, self.pct, self.max_num)
