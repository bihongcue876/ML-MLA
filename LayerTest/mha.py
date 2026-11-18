"""标准多头注意力层，三层基准中的对照项。

cache 口径：每层每 token 保存完整的 K 与 V，共 2 * n_heads * head_dim 个元素。
"""

import torch
from torch import nn
from torch.nn import functional as F

import common


class MHALayer(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, head_dim: int = 64,
                 max_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        cos, sin = common.rope_tables(head_dim, max_len, theta, torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def cache_elements_per_token(self) -> int:
        return 2 * self.n_heads * self.head_dim

    def forward(self, x: torch.Tensor, past=None, start_pos: int = 0):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q = common.apply_rope(q, self.cos, self.sin, start_pos)
        k = common.apply_rope(k, self.cos, self.sin, start_pos)
        if past is not None:
            k = torch.cat([past["k"], k], dim=2)
            v = torch.cat([past["v"], v], dim=2)
        out = common.sdpa(q, k, v, common.attn_mode(t, past))
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out), {"k": k, "v": v}
