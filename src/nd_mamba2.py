from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

Device = torch.device


@dataclass
class Mamba2Config:
    d_model: int  
    n_layer: int = 24  
    d_state: int = 128  
    d_conv: int = 4  
    expand: int = 2  
    headdim: int = 64  
    chunk_size: int = 64  
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class InferenceCache(NamedTuple):
    conv_state: Tensor  
    ssm_state: Tensor  

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )


class Mamba2(nn.Module):
    def __init__(self, d_model: int,  
                 n_layer: int = 24,  
                 d_state: int = 128,  
                 d_conv: int = 4,  
                 expand: int = 2,  
                 headdim: int = 64,  
                 chunk_size: int = 64,  
                 vocab_size: int = 50277,
                 pad_vocab_size_multiple: int = 16, ):
        super().__init__()
        args = Mamba2Config(d_model, n_layer, d_state, d_conv, expand,
                            headdim, chunk_size, vocab_size, pad_vocab_size_multiple)
        self.args = args
        
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False)

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
        )

        self.dt_bias = nn.Parameter(torch.empty(args.nheads, ))
        self.A_log = nn.Parameter(torch.empty(args.nheads, ))
        self.D = nn.Parameter(torch.empty(args.nheads, ))
        self.norm = RMSNorm(args.d_inner, )
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, )

    def forward(self, u: Tensor, h=None):
        
        if h:
            return self.step(u, h)

        A = -torch.exp(self.A_log)  
        zxbcdt = self.in_proj(u)  
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  

        
        conv_state = F.pad(
            rearrange(
                xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )

        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(
                1, 2)[:, : u.shape[1], :]
        )  
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state,
                  self.args.d_state], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=x.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)
        return y, h

    def step(self, u: Tensor, h: InferenceCache):
        
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))  
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )

        
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state,
                  self.args.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)  

        
        dt = F.softplus(dt + self.dt_bias)  
        dA = torch.exp(dt * A)  
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h


def segsum(x: Tensor, device: Device = None) -> Tensor:
    
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool,
                      device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(
        T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    
    assert x.shape[1] % chunk_size == 0

    
    
    
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    
    
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    
    
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(
        segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    
    
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp",
                         C, states, state_decay_out)

    
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    
    return x * F.sigmoid(x)


class BaseNdMamba2(nn.Module):
    def __init__(self, cin,  cout, mamba_dim, **mamba2_args):
        super().__init__()
        assert mamba_dim % 64 == 0, "cmid 必须是64的倍数"
        self.fc_in = nn.Linear(cin, mamba_dim, bias=False)  
        self.mamba2_for = Mamba2(mamba_dim, **mamba2_args)  
        self.mamba2_back = Mamba2(mamba_dim, **mamba2_args)  
        self.fc_out = nn.Linear(mamba_dim, cout, bias=False)  


class NdMamba2_1d(BaseNdMamba2):
    def __init__(self, cin, cout, cmid,  **mamba2_args):
        super().__init__(cin, cout, cmid, **mamba2_args)

    def forward(self, x):
        l = x.shape[2]
        
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))
        x = rearrange(x, 'b c l-> b l c')  
        x = self.fc_in(x)  
        x1, h1 = self.mamba2_for(x)
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        x = x1 + x2
        x = self.fc_out(x)  
        x = rearrange(x, 'b l c -> b c l')  
        x = x[:, :, :l]  
        return x


class NdMamba2_2d(BaseNdMamba2):
    def __init__(self, cin, mamba_dim, cout, **mamba2_args):
        super().__init__(cin, mamba_dim, cout, **mamba2_args)

    def forward(self, x):
        h, w = x.shape[2:]
        x = F.pad(x, (0, (8 - x.shape[3] % 8) % 8,
                      0, (8 - x.shape[2] % 8) % 8)
                  )  
        h8, w8 = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')  
        x = self.fc_in(x)  
        x1, h1 = self.mamba2_for(x)
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        x = x1 + x2
        x = self.fc_out(x)  
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=h8)
        x = x[:, :, :h, :w]  
        return x


class NdMamba2_3d(BaseNdMamba2):
    def __init__(self, cin,  mamba_dim, cout, **mamba2_args):
        super().__init__(cin, mamba_dim, cout, **mamba2_args)

    def forward(self, x):
        d, h, w = x.shape[2:]
        x = F.pad(x, (0, (4 - x.shape[4] % 4) % 4,
                      0, (4 - x.shape[3] % 4) % 4,
                      0, (4 - x.shape[2] % 4) % 4)
                  )  
        d4, h4, w4 = x.shape[2:]
        
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        x = self.fc_in(x)  
        x1, h1 = self.mamba2_for(x)
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        x = x1 + x2
        x = self.fc_out(x)  
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=d4,
                      h=h4, w=w4)  
        x = x[:, :, :d, :h, :w]  
        return x


class NdMamba2(BaseNdMamba2):
    def __init__(self, cin,  cout, mamba_dim, **mamba2_args):
        super().__init__(cin, cout, mamba_dim, **mamba2_args)

    def forward(self, x):
        size = x.shape[2:]
        x = torch.flatten(x, 2)  
        l = x.shape[2]
        
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))
        x = rearrange(x, 'b c l-> b l c')  
        x = self.fc_in(x)  
        x1, h1 = self.mamba2_for(x)
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        x = x1 + x2
        x = self.fc_out(x)  
        x = rearrange(x, 'b l c -> b c l')  
        x = x[:, :, :l]  
        x = torch.unflatten(x, 2, size)
        return x


if __name__ == '__main__':
    
    net_n = NdMamba2(64, 64, 128, ).cuda()

    
    net1 = NdMamba2_1d(64,  64,128,).cuda()
    net2 = NdMamba2_2d(64,  64,128,).cuda()
    net3 = NdMamba2_3d(64,  64,128,).cuda()

    
    x1 = torch.randn(1, 64, 32).cuda()  
    x2 = torch.randn(1, 64, 32, 77).cuda()  
    x3 = torch.randn(1, 64, 32, 77, 25).cuda()  
    x4 = torch.randn(1, 64, 12, 17, 15, 15).cuda()  

    
    y1 = net_n(x1)
    print(y1.shape)
    y2 = net_n(x2)
    print(y2.shape)
    y3 = net_n(x3)
    print(y3.shape)
    y4 = net_n(x4)
    print(y4.shape)

    y1 = net1(x1)
    print(y1.shape)
    y2 = net2(x2)
    print(y2.shape)
    y3 = net3(x3)
    print(y3.shape)
