"""朴素 MLA 层：cache 只存压缩潜变量与解耦 RoPE 的 key，但每步都显式展开 K/V。

cache 口径：每层每 token 保存 d_c 个潜变量元素与 d_rope 个共享 RoPE key 元素。
"""

import torch
from torch import nn
from torch.nn import functional as F

import common


class NaiveMLALayer(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, head_dim: int = 64,
                 d_c: int = 256, d_rope: int = 32, max_len: int = 2048,
                 theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_c = d_c
        self.d_rope = d_rope
        self.w_dkv = nn.Linear(d_model, d_c, bias=False)
        self.w_uk = nn.Linear(d_c, n_heads * head_dim, bias=False)
        self.w_uv = nn.Linear(d_c, n_heads * head_dim, bias=False)
        self.w_kr = nn.Linear(d_model, d_rope, bias=False)
        self.w_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_qr = nn.Linear(d_model, n_heads * d_rope, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        cos, sin = common.rope_tables(d_rope, max_len, theta, torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def cache_elements_per_token(self) -> int:
        return self.d_c + self.d_rope

    def forward(self, x: torch.Tensor, past=None, start_pos: int = 0):
        b, t, _ = x.shape
        c = self.w_dkv(x)
        k_rope = self.w_kr(x).view(b, t, 1, self.d_rope).transpose(1, 2)
        q_nope = self.w_q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q_rope = self.w_qr(x).view(b, t, self.n_heads, self.d_rope).transpose(1, 2)
        q_rope = common.apply_rope(q_rope, self.cos, self.sin, start_pos)
        k_rope = common.apply_rope(k_rope, self.cos, self.sin, start_pos)
        if past is not None:
            c_all = torch.cat([past["c"], c], dim=1)
            k_rope_all = torch.cat([past["k_rope"], k_rope], dim=2)
        else:
            c_all, k_rope_all = c, k_rope
        t_all = c_all.size(1)
        k_nope = self.w_uk(c_all).view(b, t_all, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_uv(c_all).view(b, t_all, self.n_heads, self.head_dim).transpose(1, 2)
        k = torch.cat([k_nope, k_rope_all.expand(-1, self.n_heads, -1, -1)], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)
        out = common.sdpa(q, k, v, common.attn_mode(t, past))
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.w_o(out), {"c": c_all, "k_rope": k_rope_all}
