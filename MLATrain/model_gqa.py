"""GQA 系注意力（MHA / GQA-4 / MQA 同一实现）：严格对齐 MiniMind-3 官方结构。

官方权重实测的每层组件：input_layernorm、post_attention_layernorm、
self_attn.{q_proj, k_proj, v_proj, o_proj, q_norm, k_norm}、mlp.{gate_proj, up_proj, down_proj}。
其中 q_norm / k_norm 为每个头的 RMSNorm，Qwen3 结构特有，本项目保留。
GQA-4 下 cache 只保存 4 组 KV，每层每 token 为 2 * 4 * 96 = 768 个元素。

MHA、GQA、MQA 是同一族的三个取值，统称广义 GQA。查询头固定 8 个、每头 96 维，
由 KV 头数 n_kv_heads 决定共享程度：等于 8 是 MHA（每头独立 KV，每层 1536 元素），
等于 4 是 GQA-4（768 元素），等于 1 是 MQA（全部查询头共享一组 KV，192 元素）。
三者只差这一个整数，投影形状与缓存量随之线性变化，q_norm、RoPE、SDPA、掩码
完全共用，因此互相对照是干净的消融，不需要各写一份实现。

入口：
    python MLATrain/train.py --arch mha  --seed 42    # n_kv_heads = 8
    python MLATrain/train.py --arch gqa4 --seed 42    # n_kv_heads = 4
    # MQA 未列入 choices：在 build_model 里取 n_kv_heads = 1 即可，无需新文件。

MLA 不属于这个族。它每头保留独立查询，把 KV 压成低秩潜在向量，见 model_mla.py。
"""

import torch
from torch import nn
from torch.nn import functional as F

from model_common import RMSNorm, Transformer, apply_rope, rope_tables


class GQAAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 8, n_kv_heads: int = 4,
                 head_dim: int = 96, max_len: int = 512, theta: float = 1e6,
                 eps: float = 1e-6):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.group = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)
        cos, sin = rope_tables(head_dim, max_len, theta, torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def cache_elements_per_token(self) -> int:
        return 2 * self.n_kv_heads * self.head_dim

    def forward(self, x: torch.Tensor, past=None, start_pos: int = 0):
        b, t, _ = x.shape
        # 官方顺序：投影 -> 每头 RMSNorm -> RoPE -> 注意力。
        q = self.q_norm(self.q_proj(x).view(b, t, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.cos, self.sin, start_pos)
        k = apply_rope(k, self.cos, self.sin, start_pos)
        if past is not None:
            k = torch.cat([past["k"], k], dim=2)
            v = torch.cat([past["v"], v], dim=2)
        k_cache, v_cache = k, v
        k_used = k.repeat_interleave(self.group, dim=1)
        v_used = v.repeat_interleave(self.group, dim=1)
        # 分块 prefill 时 past 非空且 t > 1，此时 is_causal 会把掩码关掉、看到未来 token，
        # 必须显式给出因果掩码。
        attn_mask = None
        if past is not None and t > 1:
            total = k_used.size(2)
            attn_mask = torch.ones(t, total, dtype=torch.bool, device=x.device).tril(
                diagonal=total - t)
        out = F.scaled_dot_product_attention(
            q, k_used, v_used, attn_mask=attn_mask,
            is_causal=(attn_mask is None and t > 1))
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out), {"k": k_cache, "v": v_cache}


class GQAModel(Transformer):
    def __init__(self, vocab_size: int = 6400, d_model: int = 768, n_layers: int = 8,
                 n_heads: int = 8, n_kv_heads: int = 4, head_dim: int = 96,
                 hidden: int = 2432, max_len: int = 512, theta: float = 1e6,
                 tie: bool = True, eps: float = 1e-6, arch_name: str = "gqa4"):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            hidden=hidden,
            max_len=max_len,
            attn_factory=lambda: GQAAttention(
                d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
                head_dim=head_dim, max_len=max_len, theta=theta, eps=eps,
            ),
            tie=tie,
            eps=eps,
        )
        self.arch_config = {
            "arch": arch_name, "vocab_size": vocab_size, "d_model": d_model,
            "n_layers": n_layers, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
            "head_dim": head_dim, "hidden": hidden, "max_len": max_len,
            "theta": theta, "tie": tie, "eps": eps,
        }

    def key_mapping(self) -> dict:
        """官方 safetensors 键 -> 本实现 state_dict 键。"""
        mapping = {
            "model.embed_tokens.weight": "embed.weight",
            "model.norm.weight": "norm_f.weight",
        }
        for i in range(len(self.blocks)):
            src = f"model.layers.{i}."
            dst = f"blocks.{i}."
            mapping.update({
                src + "input_layernorm.weight": dst + "norm1.weight",
                src + "post_attention_layernorm.weight": dst + "norm2.weight",
                src + "mlp.gate_proj.weight": dst + "mlp.w1.weight",
                src + "mlp.up_proj.weight": dst + "mlp.w3.weight",
                src + "mlp.down_proj.weight": dst + "mlp.w2.weight",
            })
            for name in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]:
                mapping[src + f"self_attn.{name}.weight"] = dst + f"attn.{name}.weight"
        return mapping

    def load_official(self, sd: dict):
        """把官方 safetensors 的键映射到本实现的键，返回 (已加载, 缺失) 两个列表。"""
        mapping = self.key_mapping()
        own = dict(self.named_parameters())
        loaded, missing = [], []
        with torch.no_grad():
            for src_name, dst_name in mapping.items():
                if src_name in sd and dst_name in own:
                    own[dst_name].copy_(sd[src_name].to(own[dst_name].dtype))
                    loaded.append(src_name)
                else:
                    missing.append(src_name)
        return loaded, missing
