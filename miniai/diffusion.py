# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/28_diffusion-attn-cond.ipynb.

# %% auto 0
__all__ = ['abar', 'inv_abar', 'add_noise', 'collate_ddpm', 'dl_ddpm', 'timestamp_embedding', 'pre_conv', 'upsample',
           'heads_to_batch', 'batch_to_heads', 'SelfAttention', 'SelfAttention2D', 'lin', 'EmbResBlock', 'SaveModule',
           'SaveEmbResBlock', 'SaveConv', 'DownBlock', 'UpBlock', 'EmbUNetModel', 'ddim_step', 'sample', 'cond_sample']

# %% ../nbs/28_diffusion-attn-cond.ipynb 4
def abar(t):
    return (t * (math.pi / 2)).cos() ** 2

# %% ../nbs/28_diffusion-attn-cond.ipynb 5
def inv_abar(x):
    return x.sqrt().acos() * (2 / math.pi)

# %% ../nbs/28_diffusion-attn-cond.ipynb 6
def add_noise(x):
    device = x.device
    bs = x.shape[0]

    t = torch.rand((bs,), device=device)
    alpha_bar_t = abar(t).reshape((bs,) + (1,) * (len(x.shape) - 1))
    
    original_part = alpha_bar_t.sqrt() * x
    epsilon = torch.randn(x.shape, device=device)
    noise_part = (1 - alpha_bar_t).sqrt() * epsilon

    xt = original_part + noise_part
    return (xt, t.to(device)), epsilon

# %% ../nbs/28_diffusion-attn-cond.ipynb 7
def collate_ddpm(b, fm_x="image"):
    return add_noise(default_collate(b)[fm_x])

# %% ../nbs/28_diffusion-attn-cond.ipynb 8
def dl_ddpm(ds, bs):
    return DataLoader(ds, batch_size=bs, collate_fn=collate_ddpm, num_workers=4)

# %% ../nbs/28_diffusion-attn-cond.ipynb 12
def timestamp_embedding(tsteps, emb_dim, max_period=1000):
    mult = 1 / max_period ** torch.linspace(0, 1, emb_dim // 2, device=tsteps.device)
    emb_t = tsteps[:, None] * mult[None]
    emb = torch.cat((torch.sin(emb_t), torch.cos(emb_t)), dim=1)
    return emb

# %% ../nbs/28_diffusion-attn-cond.ipynb 14
def pre_conv(ni, nf, ks=3, stride=1, act=nn.SiLU, norm=None, bias=True):
    layers = []

    if act:
        layers.append(act())
    if norm:
        layers.append(norm(ni))
    layers.append(nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, bias=bias, padding=ks//2))

    return nn.Sequential(*layers)

# %% ../nbs/28_diffusion-attn-cond.ipynb 16
def upsample(nf):
    return nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(nf, nf, kernel_size=3, padding=1)
    )

# %% ../nbs/28_diffusion-attn-cond.ipynb 18
def heads_to_batch(x, heads):
    bs, c, d = x.shape
    x = x.reshape(bs, c, heads, -1)  # (bs, c, heads, dh)
    x = x.transpose(1, 2)  # (bs, heads, c, dh)
    return x.reshape(bs * heads, c, -1)

def batch_to_heads(x, heads):
    n, c, dh = x.shape
    x = x.reshape(-1, heads, c, dh)  # (bs, heads, c, dh)
    x = x.transpose(1, 2)  # (bs, c, heads, dh)
    bs = n // heads
    return x.reshape(bs, c, heads * dh)

class SelfAttention(nn.Module):
    def __init__(self, n_dim, attn_channels):
        super().__init__()

        self.nheads = nheads = n_dim // attn_channels
        self.scale = 1 / math.sqrt(n_dim / nheads)
        self.qkv = nn.Linear(n_dim, 3 * n_dim)
        self.norm = nn.LayerNorm(n_dim)
        self.lin = nn.Linear(n_dim, n_dim)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        init_x = x
        x = self.qkv(x)
        x = heads_to_batch(x, self.nheads)
        n_dim = x.shape[-1] // 3
        q, k, v = x[..., :n_dim], x[..., n_dim: 2 * n_dim], x[..., 2 * n_dim: 3 * n_dim]
        x = (q @ k.transpose(1, 2) * self.scale).softmax(dim=-1) @ v
        x = batch_to_heads(x, self.nheads)
        x = self.lin(x)
        return self.norm(init_x + x).transpose(1, 2)

# %% ../nbs/28_diffusion-attn-cond.ipynb 20
class SelfAttention2D(SelfAttention):
    def forward(self, x):
        bs, c, h, w = x.shape
        return super().forward(x.reshape(bs, c, -1)).reshape(bs, c, h, w)

# %% ../nbs/28_diffusion-attn-cond.ipynb 22
def lin(ni, nf, act=nn.SiLU, norm=None, bias=True):
    layers = []

    if act:
        layers.append(act())
    if norm:
        layers.append(norm(ni))
    layers.append(nn.Linear(ni, nf, bias=bias))
    return nn.Sequential(*layers)

# %% ../nbs/28_diffusion-attn-cond.ipynb 23
class EmbResBlock(nn.Module):
    def __init__(self, n_emb, ni, nf=None, ks=3, act=nn.SiLU, norm=nn.BatchNorm2d, attn_channels=0):
        super().__init__()
        if nf is None:
            nf = ni

        self.emb_layer = lin(n_emb, nf * 2)
        self.nf = nf
        self.convs = nn.Sequential(
            pre_conv(ni, nf, ks=ks, act=act, norm=norm),
            pre_conv(nf, nf, ks=ks, act=act, norm=norm)
        )
        if ni == nf:
            self.id_conv = fc.noop
        else:
            self.id_conv = pre_conv(ni, nf, ks=1, act=None)

        self.attention = fc.noop
        if attn_channels:
            self.attention = SelfAttention2D(nf, attn_channels)

    def forward(self, x):
        x, t = x
        init_x = x
        x = self.convs[0](x)
        emb = self.emb_layer(t)[..., None, None]
        x = x * (1 + emb[:, :self.nf])  + emb[:, self.nf:]
        x = self.convs[1](x)
        x = x + self.id_conv(init_x)
        return x + self.attention(x)

# %% ../nbs/28_diffusion-attn-cond.ipynb 25
class SaveModule:
    def forward(self, x, *args, **kwargs):
        self.output = super().forward(x, *args, **kwargs)
        return self.output

class SaveEmbResBlock(SaveModule, EmbResBlock):
    ...

class SaveConv(SaveModule, nn.Conv2d):
    ...

# %% ../nbs/28_diffusion-attn-cond.ipynb 27
class DownBlock(nn.Module):
    def __init__(self, n_emb, ni, nf, add_down=True, num_layers=1, attn_channels=0):
        super().__init__()

        layers = [SaveEmbResBlock(n_emb, ni, nf)]
        for i in range(1, num_layers):
            layers.append(SaveEmbResBlock(n_emb, nf, nf, attn_channels=attn_channels))
        self.layers = nn.Sequential(*layers)

        self.add_down = add_down
        if add_down:
            self.down = SaveConv(nf, nf, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x, t = x
        self.output = []
        for l in self.layers:
            x = l((x, t))
            self.output.append(l.output)
        if self.add_down:
            x = self.down(x)
            self.output.append(self.down.output)
        return x

# %% ../nbs/28_diffusion-attn-cond.ipynb 29
class UpBlock(nn.Module):
    def __init__(self, n_emb, ni, prev_nf, nf, add_up=True, num_layers=2, attn_channels=0):
        super().__init__()
        blocks = [EmbResBlock(n_emb, prev_nf + nf, nf, attn_channels=attn_channels)]
        for i in range(1, num_layers - 1):
            blocks.append(EmbResBlock(n_emb, nf + nf, nf, attn_channels=attn_channels))
        blocks.append(EmbResBlock(n_emb, nf + ni, nf, attn_channels=attn_channels))
        self.blocks = nn.Sequential(*blocks)
        
        if add_up:
            self.up = upsample(nf)
        else:
            self.up = fc.noop

    def forward(self, x, ups):
        x, t = x
        for i, block in enumerate(self.blocks):
            x = block((torch.cat((x, ups[i]), dim=1), t))
        return self.up(x)

# %% ../nbs/28_diffusion-attn-cond.ipynb 31
class EmbUNetModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nfs=(224, 448, 672, 896), num_layers=1, attn_channels=8, attn_start=1):
        super().__init__()

        self.n_emb_t = n_emb_t = nfs[0]
        self.n_emb = n_emb = nfs[0]

        self.pre_down = nn.Conv2d(in_channels, nfs[0], kernel_size=3, padding=1)
        self.pre_down_emb = nn.Sequential(
            lin(n_emb_t, n_emb, norm=nn.BatchNorm1d),
            lin(n_emb, n_emb, norm=nn.BatchNorm1d)
        )
        self.down = nn.Sequential(*[
            DownBlock(
                n_emb,
                nfs[max(0, i - 1)], nfs[i], 
                add_down=(i != len(nfs) - 1), num_layers=num_layers,
                attn_channels=(attn_channels if i > attn_start else 0)
            ) for i in range(len(nfs))
        ])

        self.pre_up = EmbResBlock(n_emb, nfs[-1])
        self.up = nn.Sequential(*[
            UpBlock(
                n_emb,
                nfs[max(0, i - 1)], nfs[min(len(nfs) - 1, i + 1)], nfs[i],
                add_up=(i != 0), num_layers=num_layers+1,
                attn_channels=attn_channels
            ) for i in range(len(nfs) - 1, -1, -1)
        ])
        self.post_up = pre_conv(nfs[0], out_channels, norm=nn.BatchNorm2d)

    def forward(self, x):
        x, t = x
        emb_t = timestamp_embedding(t, self.n_emb_t, 1000)
        
        x = self.pre_down(x)
        emb = self.pre_down_emb(emb_t)
        down_act = []
        new = [x]

        for b in self.down:
            x = b((x, emb))
        for b in self.down:
            new = [new[0], *b.output[:-1]]
            down_act.append(new)
            new = [b.output[-1]]
        down_act[-1].append(new[0])

        x = self.pre_up((x, emb))
        for i, b in enumerate(self.up):
            x = b((x, emb), down_act[-i - 1][::-1])
        x = self.post_up(x)

        return x

# %% ../nbs/28_diffusion-attn-cond.ipynb 40
def ddim_step(x_t, noise, alpha_bar_t, alpha_bar_t_minus_1, beta_bar_t, beta_bar_t_minus_1, eta, clipv=2):
    # Equation (12)
    predicted_coef = alpha_bar_t_minus_1.sqrt()
    predicted_x0 = ((x_t - beta_bar_t.sqrt() * noise) * (1 / alpha_bar_t.sqrt())).clip(-clipv, clipv)
    sigma_t = (beta_bar_t_minus_1 / beta_bar_t).sqrt() * (1 - alpha_bar_t / alpha_bar_t_minus_1).sqrt() * eta
    if sigma_t.isnan().item():
        sigma_t = 0
    noise_coef = beta_bar_t_minus_1 - sigma_t ** 2
    if noise_coef < 1e-5:
        noise_coef = beta_bar_t_minus_1
    direction_to_x_t = (beta_bar_t_minus_1 - sigma_t ** 2).sqrt() * noise
    random_noise = sigma_t * torch.randn(x_t.shape, device=x_t.device)
    x_t_minus_1 = predicted_coef * predicted_x0 + direction_to_x_t + random_noise
    return x_t_minus_1

# %% ../nbs/28_diffusion-attn-cond.ipynb 41
def sample(f, model, sz, steps, eta=1., return_process=False):
    ts = torch.linspace(0.99, 0, steps)
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.randn(sz, device=device)
        bs = x.shape[0]
        if return_process: process = []

        for idx, t in enumerate(progress_bar(ts)):
            t_batch = torch.full((bs,), t, dtype=torch.float, device=device)
            noise_pred = model((x, t_batch))

            alpha_bar_t = abar(t)
            alpha_bar_t_minus_1 = abar(t - 1 / steps) if t >= 1 / steps else tensor(1)
            beta_bar_t = 1 - alpha_bar_t
            beta_bar_t_minus_1 = 1 - alpha_bar_t_minus_1
            x = f(x, noise_pred, alpha_bar_t, alpha_bar_t_minus_1, beta_bar_t, beta_bar_t_minus_1, eta)

            if return_process: process.append(to_cpu(x))
    if return_process: return process
    return to_cpu(x)

# %% ../nbs/28_diffusion-attn-cond.ipynb 57
def cond_sample(c, f, model, sz, steps, eta=1., return_process=False):
    ts = torch.linspace(0.99, 0, steps)
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.randn(sz, device=device)
        bs = x.shape[0]
        if return_process: process = []

        c_batch = torch.full((bs,), c, dtype=torch.long, device=device)
        for idx, t in enumerate(progress_bar(ts)):
            t_batch = torch.full((bs,), t, dtype=torch.float, device=device)
            noise_pred = model((x, t_batch, c_batch))

            alpha_bar_t = abar(t)
            alpha_bar_t_minus_1 = abar(t - 1 / steps) if t >= 1 / steps else tensor(1)
            beta_bar_t = 1 - alpha_bar_t
            beta_bar_t_minus_1 = 1 - alpha_bar_t_minus_1
            x = f(x, noise_pred, alpha_bar_t, alpha_bar_t_minus_1, beta_bar_t, beta_bar_t_minus_1, eta)

            if return_process: process.append(to_cpu(x))
    if return_process: return process
    return to_cpu(x)
