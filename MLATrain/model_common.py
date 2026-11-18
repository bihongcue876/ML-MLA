"""MLATrain 共用组件：RMSNorm、RoPE、SwiGLU、残差块与通用 Transformer 骨架。"""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(head_dim: int, max_len: int, theta: float, device: torch.device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """rotate_half 形式：前 d/2 维与后 d/2 维配对，与 Qwen3 / Llama 一致。

    官方 minimind-3 的 config 为 model_type=qwen3，其 RoPE 用 rotate_half 而非
    相邻维配对；两者频率相同但维度配对方式不同，写错则加载官方权重后数值全错。
    """
    t = x.size(-2)
    c = cos[start_pos:start_pos + t].unsqueeze(0).unsqueeze(0)
    s = sin[start_pos:start_pos + t].unsqueeze(0).unsqueeze(0)
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, attn: nn.Module, d_model: int, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps)
        self.attn = attn
        self.norm2 = RMSNorm(d_model, eps)
        self.mlp = SwiGLU(d_model, hidden)
        self.grad_checkpoint = False

    def forward(self, x: torch.Tensor, past=None, start_pos: int = 0):
        if self.grad_checkpoint and self.training and x.requires_grad:
            def run(xx):
                h, _ = self.attn(self.norm1(xx), None, start_pos)
                xx = xx + h
                return xx + self.mlp(self.norm2(xx))

            return torch.utils.checkpoint.checkpoint(run, x, use_reentrant=False), None
        h, new_past = self.attn(self.norm1(x), past, start_pos)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_past


class Transformer(nn.Module):
    """通用解码器骨架，注意力层由 attn_factory 提供。"""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, hidden: int,
                 max_len: int, attn_factory, tie: bool = True, eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(attn_factory(), d_model, hidden, eps) for _ in range(n_layers)]
        )
        self.norm_f = RMSNorm(d_model, eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_grad_checkpoint(self, enabled: bool = True):
        for blk in self.blocks:
            blk.grad_checkpoint = enabled

    def forward(self, idx: torch.Tensor, past=None, start_pos: int = 0):
        x = self.embed(idx)
        new_past = []
        for i, blk in enumerate(self.blocks):
            p = past[i] if past is not None else None
            x, np_ = blk(x, p, start_pos)
            new_past.append(np_)
        x = self.norm_f(x)
        return self.lm_head(x), new_past
