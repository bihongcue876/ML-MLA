"""MLA 模型：只替换注意力层，块外结构沿用官方 MiniMind-3。

层内按 DeepSeek MLA 设计：d_c = 384（= 4 * head_dim）、d_h^R = 48（= head_dim / 2）。
use_qk_norm 默认为 False；与 MHA 对照时置 True，此时 q_nope 与 k_nope 各做逐头
RMSNorm，位置与 MHA 的 q_norm / k_norm 一致，但吸收模式随之不可用（k_norm 的分母
依赖 c，折不进 q），推理只能走 naive。
两种模式 cache 都只保存潜变量 c 与共享 RoPE key，每层每 token 为 384 + 48 = 432 个元素。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from model_common import RMSNorm, Transformer, apply_rope, rope_tables


class MLAAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 8, head_dim: int = 96,
                 d_c: int = 384, d_rope: int = 48, max_len: int = 512,
                 theta: float = 1e6, mode: str = "naive", use_qk_norm: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        assert mode in ("naive", "absorbed")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_c = d_c
        self.d_rope = d_rope
        self.mode = mode
        self.use_qk_norm = use_qk_norm
        self.w_dkv = nn.Linear(d_model, d_c, bias=False)
        self.w_uk = nn.Linear(d_c, n_heads * head_dim, bias=False)
        self.w_uv = nn.Linear(d_c, n_heads * head_dim, bias=False)
        self.w_kr = nn.Linear(d_model, d_rope, bias=False)
        self.w_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.w_qr = nn.Linear(d_model, n_heads * d_rope, bias=False)
        self.w_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(head_dim, eps) if use_qk_norm else None
        self.k_norm = RMSNorm(head_dim, eps) if use_qk_norm else None
        cos, sin = rope_tables(d_rope, max_len, theta, torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def cache_elements_per_token(self) -> int:
        return self.d_c + self.d_rope

    def set_mode(self, mode: str):
        assert mode in ("naive", "absorbed")
        if mode == "absorbed" and self.use_qk_norm:
            raise ValueError("启用 QK-Norm 时无法使用吸收模式：k 的归一化不可并入 W_UK")
        self.mode = mode

    def forward(self, x: torch.Tensor, past=None, start_pos: int = 0):
        b, t, _ = x.shape
        c = self.w_dkv(x)
        k_rope = self.w_kr(x).view(b, t, 1, self.d_rope).transpose(1, 2)
        q_nope = self.w_q(x).view(b, t, self.n_heads, self.head_dim)
        q_rope = self.w_qr(x).view(b, t, self.n_heads, self.d_rope).transpose(1, 2)
        if self.use_qk_norm:
            q_nope = self.q_norm(q_nope)
        q_rope = apply_rope(q_rope, self.cos, self.sin, start_pos)
        k_rope = apply_rope(k_rope, self.cos, self.sin, start_pos)
        if past is not None:
            c_all = torch.cat([past["c"], c], dim=1)
            k_rope_all = torch.cat([past["k_rope"], k_rope], dim=2)
        else:
            c_all, k_rope_all = c, k_rope
        t_all = c_all.size(1)
        # 分块 prefill 时 past 非空且 t > 1，此时 is_causal 会把掩码关掉、看到未来 token，
        # 必须显式给出因果掩码。
        attn_mask = None
        if past is not None and t > 1:
            attn_mask = torch.ones(t, t_all, dtype=torch.bool, device=x.device).tril(
                diagonal=t_all - t)

        if self.mode == "naive":
            k_nope = self.w_uk(c_all).view(b, t_all, self.n_heads, self.head_dim)
            if self.use_qk_norm:
                k_nope = self.k_norm(k_nope)
            k_nope = k_nope.transpose(1, 2)
            v = self.w_uv(c_all).view(b, t_all, self.n_heads, self.head_dim).transpose(1, 2)
            k = torch.cat([k_nope, k_rope_all.expand(-1, self.n_heads, -1, -1)], dim=-1)
            q = torch.cat([q_nope.transpose(1, 2), q_rope], dim=-1)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                is_causal=(attn_mask is None and t > 1))
            out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
            y = self.w_o(out)
        else:
            w_uk = self.w_uk.weight.view(self.n_heads, self.head_dim, self.d_c)
            # q_abs 为 [b, t, h, d_c]，q_rope 为 [b, h, t, d_rope]，
            # 必须先统一到 [b, t, h, *] 再拼最后一维，否则位置与头维错位。
            q_abs = torch.einsum("bthd,hde->bthe", q_nope, w_uk)
            q = torch.cat([q_abs, q_rope.transpose(1, 2)], dim=-1).transpose(1, 2)
            # k/v 只保留 1 个头，由 enable_gqa 在 kernel 内广播给 n_heads。
            # 旧写法把 c_all expand 成 [b, h, t_all, d_c] 再 cat 出 [b, h, t_all, 432]，
            # prefill 时会物化 3.34M 元素（MHA 仅 786K），这才是首 token 慢的原因。
            k = torch.cat([c_all.unsqueeze(1), k_rope_all], dim=-1)
            v = c_all.unsqueeze(1)
            scale = 1.0 / math.sqrt(self.head_dim + self.d_rope)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=scale, enable_gqa=True,
                is_causal=(attn_mask is None and t > 1)
            ).transpose(1, 2)
            w_uv = self.w_uv.weight.view(self.n_heads, self.head_dim, self.d_c)
            tmp = torch.einsum("bthe,hde->bthd", out, w_uv)
            # tmp 最后一维是 head_dim，w_o 首维是 d_model，两者不能共用同一字母。
            w_o = self.w_o.weight.view(self.d_model, self.n_heads, self.head_dim)
            y = torch.einsum("bthx,phx->btp", tmp, w_o)
        return y, {"c": c_all, "k_rope": k_rope_all}


class MLAModel(Transformer):
    def __init__(self, vocab_size: int = 6400, d_model: int = 768, n_layers: int = 8,
                 n_heads: int = 8, head_dim: int = 96, d_c: int = 384, d_rope: int = 48,
                 hidden: int = 2432, max_len: int = 512, theta: float = 1e6,
                 tie: bool = True, mode: str = "naive", use_qk_norm: bool = False,
                 eps: float = 1e-6):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            hidden=hidden,
            max_len=max_len,
            attn_factory=lambda: MLAAttention(
                d_model=d_model, n_heads=n_heads, head_dim=head_dim, d_c=d_c,
                d_rope=d_rope, max_len=max_len, theta=theta, mode=mode,
                use_qk_norm=use_qk_norm, eps=eps,
            ),
            tie=tie,
            eps=eps,
        )
        self.arch_config = {
            "arch": "mla", "vocab_size": vocab_size, "d_model": d_model,
            "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
            "d_c": d_c, "d_rope": d_rope, "hidden": hidden, "max_len": max_len,
            "theta": theta, "tie": tie, "use_qk_norm": use_qk_norm, "eps": eps,
        }

    def set_mode(self, mode: str):
        for blk in self.blocks:
            blk.attn.set_mode(mode)
