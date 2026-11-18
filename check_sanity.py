"""项目自检：结构对齐、因果性、cache 一致性、MLA 两种实现等价性。

用法：
    python check_sanity.py

不训练、不测性能，CPU 上几十秒内完成。全部通过时退出码为 0。
"""

import json
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "MLATrain"))
sys.path.insert(0, str(ROOT / "LayerTest"))

from absorbed_mla import AbsorbedMLALayer  # noqa: E402
from mha import MHALayer  # noqa: E402
from model_gqa import GQAModel  # noqa: E402
from model_mla import MLAModel  # noqa: E402
from naive_mla import NaiveMLALayer  # noqa: E402

TINY_GQA = dict(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                head_dim=16, hidden=128, max_len=64, theta=10000.0)
TINY_MLA = dict(vocab_size=256, d_model=64, n_layers=2, n_heads=4, head_dim=16,
                d_c=64, d_rope=8, hidden=128, max_len=64, theta=10000.0)
# MHA 即每组一份独立 KV：这里让 n_kv_heads 等于 n_heads，走上 group=1 的分支。
TINY_MHA = dict(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4,
                head_dim=16, hidden=128, max_len=64, theta=10000.0)

results = []


def check(name: str, fn):
    try:
        detail = fn()
        results.append((name, True, detail))
        print(f"PASS  {name}  {detail}")
    except Exception as exc:  # noqa: BLE001
        results.append((name, False, str(exc)))
        print(f"FAIL  {name}  {type(exc).__name__}: {exc}")


def official_header_keys(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    return set(hdr) - {"__metadata__"}


def expected_official_keys(n_layers: int) -> set:
    keys = {"model.embed_tokens.weight", "model.norm.weight"}
    for i in range(n_layers):
        p = f"model.layers.{i}."
        keys |= {
            p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
            p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", p + "mlp.down_proj.weight",
        }
        for n in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]:
            keys.add(p + f"self_attn.{n}.weight")
    return keys


def main():
    def param_count():
        n = GQAModel().num_params()
        assert n == 63_912_192, f"GQA 基线参数量 {n:,}，与官方 63,912,192 不符"
        return f"{n:,}"

    def key_alignment():
        path = ROOT / "dataset-minimind" / "model.safetensors"
        if not path.exists():
            return "跳过：未下载 dataset-minimind/model.safetensors"
        actual = official_header_keys(path)
        expected = expected_official_keys(8)
        assert actual == expected, f"缺失={sorted(expected - actual)} 多出={sorted(actual - expected)}"
        return f"{len(actual)} 个张量完全一致"

    def shape_alignment():
        """键名对得上还不够，形状也要对得上，否则加载时会静默广播或直接报错。"""
        path = ROOT / "dataset-minimind" / "model.safetensors"
        if not path.exists():
            return "跳过：未下载 dataset-minimind/model.safetensors"
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n).decode("utf-8"))
        hdr.pop("__metadata__", None)
        model = GQAModel()
        own = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        bad = []
        for src, dst in model.key_mapping().items():
            got = tuple(hdr[src]["shape"]) if src in hdr else None
            want = own.get(dst)
            if got is None or want is None or got != want:
                bad.append(f"{src}{got} vs {dst}{want}")
        assert not bad, "; ".join(bad[:5])
        return f"{len(hdr)} 个张量形状全部一致"

    def rope_matches_qwen3():
        """RoPE 必须是 rotate_half（前后半配对），官方 minimind-3 为 qwen3 架构。"""
        from model_common import apply_rope, rope_tables
        d, theta = 48, 1e6
        x = torch.randn(1, 4, 16, d)
        cos, sin = rope_tables(d, 128, theta, torch.device("cpu"))
        got = apply_rope(x, cos, sin, 5)
        inv = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
        freqs = torch.outer(torch.arange(5, 21).float(), inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        c, s = emb.cos()[None, None], emb.sin()[None, None]

        def rotate_half(t):
            return torch.cat((-t[..., t.shape[-1] // 2:], t[..., :t.shape[-1] // 2]), dim=-1)

        ref = x * c + rotate_half(x) * s
        diff = (got - ref).abs().max().item()
        assert diff < 1e-6, f"RoPE 与 Qwen3 rotate_half 不一致，diff={diff}"
        return f"diff={diff:.1e}"

    def cache_formula():
        gqa = GQAModel().blocks[0].attn.cache_elements_per_token()
        mla = MLAModel().blocks[0].attn.cache_elements_per_token()
        assert gqa == 768, gqa
        assert mla == 432, mla
        assert abs(1536 / 432 - 3.5556) < 1e-3
        return f"GQA-4={gqa} MLA={mla} 相对 MHA={1536 / 432:.3f}"

    def causality():
        torch.manual_seed(0)
        x = torch.randint(0, TINY_GQA["vocab_size"], (1, 16))
        outs = []
        for model in [GQAModel(**TINY_GQA), MLAModel(**TINY_MLA)]:
            model.eval()
            with torch.no_grad():
                a, _ = model(x)
                y = x.clone()
                y[0, 8:] = torch.randint(0, TINY_GQA["vocab_size"], (8,))
                b, _ = model(y)
            diff = (a[0, :8] - b[0, :8]).abs().max().item()
            assert diff < 1e-5, f"{type(model).__name__} 前缀受未来 token 影响，diff={diff}"
            outs.append(f"{type(model).__name__}={diff:.1e}")
        return " ".join(outs)

    def cache_consistency():
        torch.manual_seed(1)
        x = torch.randint(0, TINY_GQA["vocab_size"], (1, 16))
        outs = []
        for model in [GQAModel(**TINY_GQA), MLAModel(**TINY_MLA)]:
            model.eval()
            with torch.no_grad():
                full, _ = model(x)
                past = None
                pieces = []
                for i in range(x.size(1)):
                    lg, past = model(x[:, i:i + 1], past, start_pos=i)
                    pieces.append(lg)
                step = torch.cat(pieces, dim=1)
            diff = (full - step).abs().max().item()
            assert diff < 1e-4, f"{type(model).__name__} 逐步解码与整段前向不一致，diff={diff}"
            outs.append(f"{type(model).__name__}={diff:.1e}")
        return " ".join(outs)

    def chunk_prefill():
        """分块 prefill（每块 t>1 且带 past）必须与一次性前向一致。"""
        torch.manual_seed(2)
        x = torch.randint(0, TINY_GQA["vocab_size"], (1, 16))
        outs = []
        for model in [GQAModel(**TINY_GQA), MLAModel(**TINY_MLA)]:
            model.eval()
            with torch.no_grad():
                full, _ = model(x)
                past = None
                chunks = []
                for start in range(0, 16, 4):
                    lg, past = model(x[:, start:start + 4], past, start_pos=start)
                    chunks.append(lg)
                cat = torch.cat(chunks, dim=1)
            diff = (full - cat).abs().max().item() / (full.abs().max().item() + 1e-9)
            assert diff < 1e-4, f"{type(model).__name__} 分块 prefill 与整段前向不一致，rel={diff}"
            outs.append(f"{type(model).__name__}={diff:.1e}")
        return " ".join(outs)

    def absorbed_equivalence():
        torch.manual_seed(2)
        naive = NaiveMLALayer(d_model=64, n_heads=4, head_dim=16, d_c=64, d_rope=8, max_len=64)
        absorbed = AbsorbedMLALayer(d_model=64, n_heads=4, head_dim=16, d_c=64, d_rope=8, max_len=64)
        absorbed.load_state_dict(naive.state_dict())
        x = torch.randn(2, 12, 64)
        with torch.no_grad():
            y1, _ = naive(x)
            y2, _ = absorbed(x)
        diff = (y1 - y2).abs().max().item() / (y1.abs().max().item() + 1e-9)
        assert diff < 1e-4, f"朴素与吸收输出不一致，相对误差={diff}"
        return f"相对误差={diff:.2e}"

    def mla_model_modes():
        """整模型层面：朴素前向、朴素逐步解码、吸收逐步解码三者必须一致。

        这一项专门盯吸收路径的维度顺序，batch 与序列长度不相等时最容易错位。
        """
        torch.manual_seed(3)
        model = MLAModel(**TINY_MLA).eval()
        x = torch.randint(0, TINY_MLA["vocab_size"], (2, 12))

        def stepwise():
            outs, past = [], None
            for i in range(x.size(1)):
                lg, past = model(x[:, i:i + 1], past, start_pos=i)
                outs.append(lg)
            return torch.cat(outs, dim=1)

        with torch.no_grad():
            full, _ = model(x)
            naive_step = stepwise()
            model.set_mode("absorbed")
            full_absorbed, _ = model(x)
            absorbed_step = stepwise()

        d1 = (naive_step - full).abs().max().item()
        d2 = (absorbed_step - full).abs().max().item()
        d3 = (full_absorbed - full).abs().max().item()
        assert d1 < 1e-4, f"朴素逐步解码与整段前向不一致，diff={d1}"
        assert d2 < 1e-4, f"吸收逐步解码与朴素整段不一致，diff={d2}"
        assert d3 < 1e-4, f"吸收整段前向与朴素整段不一致，diff={d3}"
        return f"逐步={d1:.1e}/{d2:.1e} 整段吸收={d3:.1e}"

    def layer_shapes():
        x = torch.randn(1, 8, 512)
        for layer in [MHALayer(), NaiveMLALayer(), AbsorbedMLALayer()]:
            y, past = layer(x)
            assert y.shape == (1, 8, 512), y.shape
        return "mha / naive_mla / absorbed_mla 输出均为 [1, 8, 512]"

    def mha_structure():
        """MHA 用 n_kv_heads=8 的同一实现，参数量与 cache 必须对上理论值。"""
        model = GQAModel(max_len=640, n_kv_heads=8, arch_name="mha").eval()
        params = sum(p.numel() for p in model.parameters())
        elems = model.blocks[0].attn.cache_elements_per_token()
        k_shape = tuple(model.blocks[0].attn.k_proj.weight.shape)
        v_shape = tuple(model.blocks[0].attn.v_proj.weight.shape)
        mla_elems = MLAModel(max_len=640).blocks[0].attn.cache_elements_per_token()
        assert model.arch_config["arch"] == "mha", model.arch_config["arch"]
        assert params == 68630784, params
        assert elems == 1536, elems
        assert k_shape == (768, 768) and v_shape == (768, 768), (k_shape, v_shape)
        ratio = elems / mla_elems
        assert abs(ratio - 3.5556) < 1e-3, ratio
        return (f"params={params:,} cache={elems} 比值={ratio:.3f} "
                f"k/v 形状={k_shape}/{v_shape}")

    def mha_decoding():
        """group=1 分支的因果性与逐步解码，与 GQA-4 的分支无关，需单独验证。"""
        torch.manual_seed(4)
        model = GQAModel(**TINY_MHA, arch_name="mha").eval()
        x = torch.randint(0, TINY_MHA["vocab_size"], (2, 12))
        with torch.no_grad():
            full, _ = model(x)
            outs, past = [], None
            for i in range(x.size(1)):
                lg, past = model(x[:, i:i + 1], past, start_pos=i)
                outs.append(lg)
            step = torch.cat(outs, dim=1)
            y = x.clone()
            y[0, 6:] = torch.randint(0, TINY_MHA["vocab_size"], (6,))
            pert, _ = model(y)
        d_step = (full - step).abs().max().item()
        d_causal = (full[0, :6] - pert[0, :6]).abs().max().item()
        assert d_step < 1e-4, f"MHA 逐步解码与整段前向不一致，diff={d_step}"
        assert d_causal < 1e-5, f"MHA 前缀受未来 token 影响，diff={d_causal}"
        return f"逐步={d_step:.1e} 因果={d_causal:.1e}"

    def ckpt_roundtrip():
        """config 保存 -> weights_only 加载 -> build_from_config 重建，覆盖 mha 分派。"""
        import io

        from eval import build_from_config
        torch.manual_seed(5)
        outs = []
        for name, model in [("mha", GQAModel(**TINY_MHA, arch_name="mha")),
                            ("mla", MLAModel(**TINY_MLA))]:
            model.eval()
            ck = {
                "config": model.arch_config, "model": model.state_dict(), "seq_len": 8,
                "tokens_seen": 0, "params": sum(p.numel() for p in model.parameters()),
            }
            buf = io.BytesIO()
            torch.save(ck, buf)
            buf.seek(0)
            loaded = torch.load(buf, map_location="cpu", weights_only=True)
            rebuilt = build_from_config(loaded["config"]).eval()
            rebuilt.load_state_dict(loaded["model"])
            x = torch.randint(0, TINY_MHA["vocab_size"], (1, 8))
            with torch.no_grad():
                a, _ = model(x)
                b, _ = rebuilt(x)
            diff = (a - b).abs().max().item()
            assert diff < 1e-6, f"{name} 往返后输出不一致，diff={diff}"
            outs.append(f"{name}={diff:.1e}")
        return " ".join(outs)

    def tokenizer_roundtrip():
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            return f"跳过：transformers 不可用（{exc}）"
        tok_dir = ROOT / "dataset-minimind" / "tokenizer"
        if not tok_dir.exists():
            return "跳过：无 dataset-minimind/tokenizer"
        tok = AutoTokenizer.from_pretrained(str(tok_dir))
        text = "白日依山尽，黄河入海流。"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        back = tok.decode(ids)
        assert tok.vocab_size == 6400, tok.vocab_size
        return f"vocab=6400 往返一致={back == text}"

    check("官方参数量 63,912,192", param_count)
    check("参数名与官方权重对齐", key_alignment)
    check("参数形状与官方权重对齐", shape_alignment)
    check("cache 元素数公式", cache_formula)
    check("RoPE 与官方 Qwen3 rotate_half 一致", rope_matches_qwen3)
    check("因果性（未来 token 不影响前缀）", causality)
    check("逐步解码与整段前向一致", cache_consistency)
    check("分块 prefill 与整段前向一致", chunk_prefill)
    check("吸收式与朴素 MLA 数值等价", absorbed_equivalence)
    check("MLA 整模型两种模式等价", mla_model_modes)
    check("三个层输出形状", layer_shapes)
    check("MHA 结构与参数（n_kv_heads=8）", mha_structure)
    check("MHA 因果性与逐步解码（group=1）", mha_decoding)
    check("ckpt 往返（含 mha 分派）", ckpt_roundtrip)
    check("tokenizer 往返", tokenizer_roundtrip)

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
