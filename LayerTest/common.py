"""LayerTest 共用组件：RMSNorm 与 RoPE。

三层实现共享这两件工具，保证基准的唯一变量是注意力结构本身。
"""

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
    """返回 cos/sin，形状均为 [max_len, head_dim // 2]。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """x: [B, H, T, D]，D 为偶数；cos/sin: [max_len, D // 2]。

    rotate_half 形式：前 D/2 维与后 D/2 维配对，与 Qwen3 / Llama 官方实现一致。
    """
    t = x.size(-2)
    c = cos[start_pos:start_pos + t].unsqueeze(0).unsqueeze(0)
    s = sin[start_pos:start_pos + t].unsqueeze(0).unsqueeze(0)
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
         mode: str = "causal", scale=None):
    """统一处理三种注意力形状的 mask。

    mode="causal"：past 为空、一次喂完整段，交给 SDPA 的因果 mask；
    mode="full"  ：decode 单 token，能看到全部历史，不需要 mask；
    mode="chunk" ：带 past 又一次喂多个 token（分块 prefill）。此时 SDPA 的
                   is_causal 会把新块当成从位置 0 开始，因果性丢失，必须显式
                   构造 [t, t_all] 下三角，对角线偏移取 t_all - t。
    """
    if mode == "chunk":
        t = q.size(-2)
        total = k.size(-2)
        mask = torch.ones(t, total, dtype=torch.bool, device=q.device)
        mask = mask.tril(diagonal=total - t)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=(mode == "causal"), scale=scale)


def attn_mode(t: int, past) -> str:
    """按输入块长度与是否有历史决定 mask 形态。"""
    if t == 1:
        return "full"
    return "causal" if past is None else "chunk"


def cache_bytes(past) -> int:
    """递归统计 cache 占用字节数，支持单层 dict、多层 list。"""
    if past is None:
        return 0
    if isinstance(past, torch.Tensor):
        return past.numel() * past.element_size()
    if isinstance(past, dict):
        return sum(cache_bytes(v) for v in past.values())
    if isinstance(past, (list, tuple)):
        return sum(cache_bytes(v) for v in past)
    return 0
