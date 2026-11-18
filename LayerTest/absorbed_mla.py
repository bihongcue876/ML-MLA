"""吸收式 MLA 层：把 W_UK 并入查询投影、W_UV 并入输出投影，推理不展开 K/V。

与 NaiveMLALayer 参数同名同形，可直接 load_state_dict 对照，验证数值等价。
cache 口径与朴素版一致：d_c 个潜变量元素加 d_rope 个共享 RoPE key 元素。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

import common


class AbsorbedMLALayer(nn.Module):
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
        q_nope = self.w_q(x).view(b, t, self.n_heads, self.head_dim)
        q_rope = self.w_qr(x).view(b, t, self.n_heads, self.d_rope).transpose(1, 2)
        q_rope = common.apply_rope(q_rope, self.cos, self.sin, start_pos)
        k_rope = common.apply_rope(k_rope, self.cos, self.sin, start_pos)
        if past is not None:
            c_all = torch.cat([past["c"], c], dim=1)
            k_rope_all = torch.cat([past["k_rope"], k_rope], dim=2)
        else:
            c_all, k_rope_all = c, k_rope
        t_all = c_all.size(1)

        # W_UK 并入查询：q_nope @ W_UK^T，得到 d_c 维查询。
        # q_abs 为 [b, t, h, d_c]，q_rope 为 [b, h, t, d_rope]，
        # 必须先统一到 [b, t, h, *] 再拼最后一维，否则位置与头维错位。
        w_uk = self.w_uk.weight.view(self.n_heads, self.head_dim, self.d_c)
        q_abs = torch.einsum("bthd,hde->bthe", q_nope, w_uk)
        q = torch.cat([q_abs, q_rope.transpose(1, 2)], dim=-1).transpose(1, 2)
        c_exp = c_all.unsqueeze(2).expand(b, t_all, self.n_heads, -1)
        k = torch.cat(
            [c_exp, k_rope_all.transpose(1, 2).expand(-1, -1, self.n_heads, -1)], dim=-1
        ).transpose(1, 2)
        v = c_exp.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim + self.d_rope)
        out = common.sdpa(q, k, v, common.attn_mode(t, past), scale=scale)
        out = out.transpose(1, 2)

        # W_UV 并入输出投影：先还原每个头的 V，再与 W_O 的对应切片合并。
        w_uv = self.w_uv.weight.view(self.n_heads, self.head_dim, self.d_c)
        tmp = torch.einsum("bthe,hde->bthd", out, w_uv)
        # tmp 最后一维是 head_dim，w_o 首维是 d_model，两者不能共用同一字母。
        w_o = self.w_o.weight.view(self.d_model, self.n_heads, self.head_dim)
        y = torch.einsum("bthx,phx->btp", tmp, w_o)
        return y, {"c": c_all, "k_rope": k_rope_all}
