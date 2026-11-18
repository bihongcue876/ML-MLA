"""LayerTest 基准：三层同超参堆叠 4 层，测量 cache、prefill 延迟、decode 延迟与峰值显存。

用法：
    python LayerTest/bench.py
    python LayerTest/bench.py --seq-lens 128 512 2048 --decode-steps 64

输出：data/runs/layer_<时间戳>/ 下的 config.json 与 bench.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

import common
from absorbed_mla import AbsorbedMLALayer
from mha import MHALayer
from naive_mla import NaiveMLALayer

ROOT = Path(__file__).resolve().parents[1]

BASE = dict(d_model=512, n_heads=8, head_dim=64, d_c=256, d_rope=32,
            max_len=2048, n_layers=4)


def single_layer(impl: str):
    if impl == "mha":
        return MHALayer(d_model=BASE["d_model"], n_heads=BASE["n_heads"],
                        head_dim=BASE["head_dim"], max_len=BASE["max_len"])
    if impl == "naive_mla":
        return NaiveMLALayer(**{k: BASE[k] for k in
                                ["d_model", "n_heads", "head_dim", "d_c", "d_rope", "max_len"]})
    if impl == "absorbed_mla":
        return AbsorbedMLALayer(**{k: BASE[k] for k in
                                   ["d_model", "n_heads", "head_dim", "d_c", "d_rope", "max_len"]})
    raise ValueError(impl)


class Stack(nn.Module):
    """把同一种层堆叠 n_layers 次，模拟真实模型的逐层 cache 增长。"""

    def __init__(self, impl: str, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([single_layer(impl) for _ in range(n_layers)])
        self.n_layers = n_layers

    def cache_elements_per_token(self) -> int:
        return self.layers[0].cache_elements_per_token() * self.n_layers

    def forward(self, x: torch.Tensor, past=None, start_pos: int = 0):
        new_past = []
        for i, layer in enumerate(self.layers):
            x, layer_past = layer(x, past[i] if past is not None else None, start_pos)
            new_past.append(layer_past)
        return x, new_past


def past_len(past) -> int:
    first = past[0]
    return first["c"].size(1) if "c" in first else first["k"].size(2)


def verify() -> float:
    """在 float32 下把朴素版权重复制给吸收版，返回相对误差。"""
    torch.manual_seed(0)
    kw = {k: BASE[k] for k in ["d_model", "n_heads", "head_dim", "d_c", "d_rope", "max_len"]}
    naive = NaiveMLALayer(**kw).float()
    absorbed = AbsorbedMLALayer(**kw).float()
    absorbed.load_state_dict(naive.state_dict())
    x = torch.randn(2, 16, BASE["d_model"])
    with torch.no_grad():
        y1, _ = naive(x)
        y2, _ = absorbed(x)
    denom = y1.abs().max().item() or 1.0
    return (y1 - y2).abs().max().item() / denom


def timeit(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    n = len(samples)
    # 同时给中位数与均值：中位数抗抖动，均值反映总开销。
    return samples[n // 2], sum(samples) / n


def bench_prefill(model, seq_len: int, device, dtype, warmup: int, iters: int):
    x = torch.randn(1, seq_len, model.layers[0].d_model, device=device, dtype=dtype)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        latency, latency_mean = timeit(lambda: model(x), warmup, iters)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    return latency, latency_mean, peak


def bench_decode(model, prefill_len: int, steps: int, device, dtype):
    d_model = model.layers[0].d_model
    x = torch.randn(1, prefill_len, d_model, device=device, dtype=dtype)
    with torch.no_grad():
        _, past = model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    with torch.no_grad():
        for _ in range(steps):
            step_x = torch.randn(1, 1, d_model, device=device, dtype=dtype)
            pos = past_len(past)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _, past = model(step_x, past, start_pos=pos)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
    samples.sort()
    n = len(samples)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    cache_measured = common.cache_bytes(past)
    # 单步中位数、单步均值、全部步数总和，三个口径一起落盘。
    return samples[n // 2], sum(samples) / n, sum(samples), peak, cache_measured, past_len(past)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # RoPE 表要覆盖 prefill 长度加 decode 步数，否则逐步生成时位置越界。
    BASE["max_len"] = max(args.seq_lens) + args.decode_steps + 16

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA 设备；LayerTest 的计时在 CPU 上没有参考意义")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("LayerTest 的计时依赖 cuda.Event，只支持 CUDA 设备")
    dtype = torch.bfloat16
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    run_id = "layer_" + time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else ROOT / "data" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "kind": "layer_bench",
        "base": BASE,
        "dtype": str(dtype),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_mem_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
        "seq_lens": args.seq_lens,
        "decode_steps": args.decode_steps,
        "warmup": args.warmup,
        "iters": args.iters,
        "argv": " ".join(__import__("sys").argv),
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    log_path = out_dir / "bench.jsonl"
    with open(log_path, "w", encoding="utf-8") as log:
        rel_err = verify()
        log.write(json.dumps({"kind": "verify", "relative_error": rel_err}, ensure_ascii=False) + "\n")
        print(f"verify: naive vs absorbed relative error = {rel_err:.3e}")

        for impl in ["mha", "naive_mla", "absorbed_mla"]:
            model = Stack(impl, BASE["n_layers"]).to(device=device, dtype=dtype).eval()
            params = sum(p.numel() for p in model.parameters())
            elems = model.cache_elements_per_token()
            print(f"[{impl}] layers={BASE['n_layers']} params={params:,} "
                  f"cache={elems} elem/token", flush=True)
            for seq_len in args.seq_lens:
                latency, latency_mean, peak = bench_prefill(
                    model, seq_len, device, dtype, args.warmup, args.iters
                )
                rec = {
                    "kind": "bench", "impl": impl, "phase": "prefill", "seq_len": seq_len,
                    "batch": 1, "latency_ms": round(latency, 4),
                    "mean_ms": round(latency_mean, 4),
                    "tokens_per_s": round(seq_len / latency * 1000, 1),
                    "peak_mem_mb": round(peak, 2),
                    "cache_theoretical": elems * seq_len * 2,
                    "cache_measured": None, "params": params,
                }
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(rec)
            latency, latency_mean, latency_total, peak, cache_measured, total_len = bench_decode(
                model, max(args.seq_lens), args.decode_steps, device, dtype
            )
            rec = {
                "kind": "bench", "impl": impl, "phase": "decode", "seq_len": total_len,
                "batch": 1, "latency_ms": round(latency, 4),
                "step_mean_ms": round(latency_mean, 4),
                "steps": args.decode_steps,
                "total_ms": round(latency_total, 4),
                "steps_per_s": round(args.decode_steps / latency_total * 1000, 2),
                "peak_mem_mb": round(peak, 2),
                "cache_theoretical": elems * total_len * 2,
                "cache_measured": cache_measured, "params": params,
            }
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(rec)
            del model
            torch.cuda.empty_cache()

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
