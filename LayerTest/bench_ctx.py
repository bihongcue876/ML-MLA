"""长上下文层测：分块 prefill 到 16K / 32K，在锚点处测 decode 单步耗时。

与 bench.py 的三点差别：
1. 分块 prefill。单次喂 32768 个 token 会让 SDPA 物化 32768^2 的注意力矩阵
   （8 头 × 2 字节约 17 GB），6 GB 卡必然 OOM，只能按块推进。
2. 一次 prefill 走到最长上下文，中途在若干锚点停下载测 decode，而不是每个
   长度重建一次 cache。
3. 默认单层。测的是层内结构差异，多层堆叠会把差异和层数混在一起。

分块 prefill 会撞上一个真实的坑：SDPA 的 is_causal 只对"从位置 0 开始的整段"
成立，带 past 再喂多 token 时它会当成新段从 0 开始，因果性丢失。所以这里给
common.sdpa 加了显式下三角 mask 分支，并由 --check-only 验证分块与一次性的
输出完全一致。

用法：
    python LayerTest/bench_ctx.py --check-only
    python LayerTest/bench_ctx.py --ctx-lens 32768 --rounds 10
    python LayerTest/bench_ctx.py --ctx-lens 16384 32768 --rounds 10
输出：data/runs/layerctx_<时间戳>/ 下的 config.json 与 bench.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
from absorbed_mla import AbsorbedMLALayer  # noqa: E402
from mha import MHALayer  # noqa: E402
from naive_mla import NaiveMLALayer  # noqa: E402

ROOT = HERE.parent

BASE = dict(d_model=512, n_heads=8, head_dim=64, d_c=256, d_rope=32, max_len=2048)


def build_layer(impl: str, max_len: int):
    kw = {"d_model": BASE["d_model"], "n_heads": BASE["n_heads"],
          "head_dim": BASE["head_dim"], "max_len": max_len}
    if impl == "mha":
        return MHALayer(**kw)
    kw["d_c"] = BASE["d_c"]
    kw["d_rope"] = BASE["d_rope"]
    if impl == "naive_mla":
        return NaiveMLALayer(**kw)
    if impl == "absorbed_mla":
        return AbsorbedMLALayer(**kw)
    raise ValueError(impl)


class Stack(nn.Module):
    """同一种层堆叠 n 次；n=1 时就是单层直测。"""

    def __init__(self, impl: str, n_layers: int, max_len: int):
        super().__init__()
        self.layers = nn.ModuleList([build_layer(impl, max_len) for _ in range(n_layers)])
        self.n_layers = n_layers

    def cache_elements_per_token(self) -> int:
        return self.layers[0].cache_elements_per_token() * self.n_layers

    def forward(self, x: torch.Tensor, past=None, start_pos: int = 0):
        new_past = []
        for i, layer in enumerate(self.layers):
            x, lp = layer(x, past[i] if past is not None else None, start_pos)
            new_past.append(lp)
        return x, new_past


def past_len(past) -> int:
    if isinstance(past, (list, tuple)):
        past = past[0]
    return past["c"].size(1) if "c" in past else past["k"].size(2)


def build(impl: str, n_layers: int, max_len: int, device, dtype):
    return Stack(impl, n_layers, max_len).to(device=device, dtype=dtype).eval()


# --------------------------------------------------------------------------- #
# 自检：分块 prefill 与一次性 prefill 必须给出同一份 cache
# --------------------------------------------------------------------------- #

def check_chunk(impl: str, device, chunk: int = 64) -> dict:
    """用 fp32 比较一次性喂入与分块喂入的输出，mask 写错时误差会在 0.1 以上。"""
    torch.manual_seed(0)
    total = 256
    model = build(impl, 1, 1024, device, torch.float32)
    x = torch.randn(1, total, BASE["d_model"], device=device, dtype=torch.float32)
    with torch.no_grad():
        full, past_full = model(x)
        outs, past, pos = [], None, 0
        while pos < total:
            n = min(chunk, total - pos)
            y, past = model(x[:, pos:pos + n], past, start_pos=pos)
            outs.append(y)
            pos += n
        chunked = torch.cat(outs, dim=1)
    scale = full.abs().max().item() or 1.0
    rel = (full - chunked).abs().max().item() / scale

    # cache 张量也要逐一对齐，否则时间对了但状态错了。
    cache_rel = 0.0
    for key in past_full[0]:
        a, b = past_full[0][key], past[0][key]
        if a.shape != b.shape:
            cache_rel = float("inf")
            break
        d = (a - b).abs().max().item() / ((a.abs().max().item()) or 1.0)
        cache_rel = max(cache_rel, d)
    return {"impl": impl, "out_rel": rel, "cache_rel": cache_rel,
            "chunk": chunk, "total": total}


def check_equiv(device) -> float:
    """朴素与吸收在同一份权重下的输出相对误差。"""
    torch.manual_seed(0)
    naive = build_layer("naive_mla", 1024).to(device).float().eval()
    absorbed = build_layer("absorbed_mla", 1024).to(device).float().eval()
    absorbed.load_state_dict(naive.state_dict())
    x = torch.randn(2, 16, BASE["d_model"], device=device)
    with torch.no_grad():
        y1, _ = naive(x)
        y2, _ = absorbed(x)
    return (y1 - y2).abs().max().item() / ((y1.abs().max().item()) or 1.0)


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #

def push(model, past, pos: int, target: int, chunk: int, d_model: int, device, dtype):
    """把 cache 从 pos 推进到 target，返回 (past, 实际长度, 本段耗时列表)。"""
    times = []
    while pos < target:
        n = min(chunk, target - pos)
        x = torch.randn(1, n, d_model, device=device, dtype=dtype)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            _, past = model(x, past, start_pos=pos)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
        pos += n
    return past, pos, times


def timed_decode(model, past, steps: int, d_model: int, device, dtype):
    """逐步 decode，返回 (中位单步毫秒, 全部样本, 推进后的 past)。"""
    samples = []
    for _ in range(steps):
        p = past_len(past)
        x = torch.randn(1, 1, d_model, device=device, dtype=dtype)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            _, past = model(x, past, start_pos=p)
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    ordered = sorted(samples)
    return ordered[len(ordered) // 2], samples, past


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    mean = sum(xs) / n
    med = xs[n // 2]
    var = sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)
    sd = var ** 0.5
    half = 1.96 * sd / max(n ** 0.5, 1)
    return {"n": n, "mean": round(mean, 4), "median": round(med, 4),
            "std": round(sd, 4), "ci95": [round(mean - half, 4), round(mean + half, 4)]}


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-lens", type=int, nargs="+", default=[16384, 32768])
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--n-heads", type=int, default=None,
                    help="覆盖 BASE 的头数；d_c/head_dim/d_rope 比例保持不变")
    ap.add_argument("--d-c", type=int, default=None,
                    help="覆盖 KV 潜变量维度；吸收能否占优由它决定")
    ap.add_argument("--impls", nargs="+",
                    default=["mha", "naive_mla", "absorbed_mla"])
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA 设备")
    if args.n_heads:
        BASE["n_heads"] = args.n_heads
    if args.d_c:
        BASE["d_c"] = args.d_c
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    print("=== 自检 ===", flush=True)
    equiv = check_equiv(device)
    print(f"naive vs absorbed 输出相对误差 = {equiv:.3e}")
    checks = []
    for impl in args.impls:
        rec = check_chunk(impl, device)
        checks.append(rec)
        print(f"[{impl}] 分块 {rec['chunk']} 对一次性 {rec['total']}: "
              f"输出误差 {rec['out_rel']:.3e}  cache 误差 {rec['cache_rel']:.3e}")
        if rec["out_rel"] > 3e-3 or rec["cache_rel"] > 3e-3:
            print(f"  !! {impl} 分块与一次性不一致，mask 有问题", flush=True)
    if args.check_only:
        return
    if any(r["out_rel"] > 3e-3 for r in checks):
        raise SystemExit("分块自检未通过，停止计速")

    ctx = sorted(args.ctx_lens)
    max_len = max(ctx) + args.decode_steps + 16
    run_id = "layerctx_" + time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else ROOT / "data" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {"run_id": run_id, "kind": "layer_ctx", "base": BASE,
           "n_layers": args.n_layers, "dtype": str(dtype), "chunk": args.chunk,
           "ctx_lens": ctx, "decode_steps": args.decode_steps, "rounds": args.rounds,
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "gpu_mem_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
           "equiv_rel": equiv, "checks": checks, "argv": " ".join(sys.argv)}
    (out_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总上下文 {max(ctx)}，块 {args.chunk}，每锚点 {args.decode_steps} 步，"
          f"{args.rounds} 轮 -> {out_dir}\n", flush=True)

    log_path = out_dir / "bench.jsonl"
    with open(log_path, "w", encoding="utf-8") as log:
        raw = {}
        for r in range(args.rounds):
            order = args.impls if r % 2 == 0 else args.impls[::-1]
            for impl in order:
                model = build(impl, args.n_layers, max_len, device, dtype)
                elems = model.cache_elements_per_token()
                d_model = BASE["d_model"]
                past, pos = None, 0
                torch.cuda.reset_peak_memory_stats()
                for target in ctx:
                    past, pos, pre_times = push(
                        model, past, pos, target, args.chunk, d_model, device, dtype)
                    pre_ms = sum(pre_times)
                    med, samples, past = timed_decode(
                        model, past, args.decode_steps, d_model, device, dtype)
                    pos = past_len(past)
                    rec = {
                        "kind": "bench", "round": r, "impl": impl, "ctx": target,
                        "actual_len": pos, "prefill_blocks": len(pre_times),
                        "prefill_ms": round(pre_ms, 2),
                        "prefill_ms_per_block": round(pre_ms / max(len(pre_times), 1), 3),
                        "decode_med_ms": round(med, 4),
                        "decode_samples": [round(s, 4) for s in samples],
                        "cache_measured": common.cache_bytes(past),
                        "cache_theoretical": elems * pos * 2,
                        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
                    }
                    log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    raw.setdefault((impl, target), []).append(med)
                print(f"  round {r + 1}/{args.rounds} {impl:12s} "
                      f"最长 {pos} cache {common.cache_bytes(past) / 1024 ** 2:.1f} MiB "
                      f"峰值 {torch.cuda.max_memory_allocated() / 1024 ** 2:.0f} MB", flush=True)
                del model
                torch.cuda.empty_cache()

        print("\n=== 汇总：decode 单步中位耗时 (ms) ===")
        print("impl          " + "".join(f"{c:>12}" for c in ctx))
        for impl in args.impls:
            line = f"{impl:14s}"
            for c in ctx:
                line += f"{stats(raw[(impl, c)])['median']:>12.3f}"
            print(line)
        summary = {f"{impl}@{c}": stats(raw[(impl, c)])
                   for impl in args.impls for c in ctx}
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        if "naive_mla" in args.impls and "absorbed_mla" in args.impls:
            print("\n=== 吸收 / 朴素 比值 (<1 才算吸收占优) ===")
            for c in ctx:
                a = stats(raw[("absorbed_mla", c)])["median"]
                b = stats(raw[("naive_mla", c)])["median"]
                print(f"  {c:>6} tok  absorbed/naive = {a / b:.3f}")
    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
