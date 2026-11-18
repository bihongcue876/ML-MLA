"""KV cache 实测与推理延迟的重复测量。

两件事分开做，因为它们的统计性质不同：

1. cache 字节数。直接读 forward 返回的 past 张量，按 numel * element_size() 累加，
   不用公式推。这个量没有随机性，同一配置每次都是同一个整数，所以每档测一次就够。
2. 推理延迟。同一进程内多配置交替、每轮前固定预热，重复 runs 轮。
   单次测量的漂移远大于架构差距，必须靠重复次数把标准误压下来，故默认 30 轮。

MLA 的两种推理路径（naive / absorbed）用同一个检查点，只切换模式，
权重、层数、cache 完全一致，差异只来自计算路径本身，是严格配对对照。

用法：
    python scripts/model/bench_cache.py --runs 30
    python scripts/model/bench_cache.py --runs 2 --seq-lens 512
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MLATrain"))

from model_gqa import GQAModel  # noqa: E402
from model_mla import MLAModel  # noqa: E402

MHA_CKPT = ROOT / "models/mha/mha_42_20261117_121454/model.pt"
GQA4_CKPT = ROOT / "models/gqa4/gqa4_42_20261117_005456/model.pt"
MLA_CKPT = ROOT / "models/mla_qknorm/mla_42_20261117_202030/model.pt"

# (标签, 检查点, 强制模式, 说明)
# 带 QK-Norm 的 MLA 无法把 k 的归一化并入 W_UK，只能走朴素路径。
CONFIGS = [
    ("MHA", MHA_CKPT, None, "基准，每层每 token 1536 元素"),
    ("GQA-4", GQA4_CKPT, None, "中间档，每层每 token 768 元素"),
    ("MLA", MLA_CKPT, "naive", "每层每 token 432 元素，只能走朴素路径"),
]


def build(cfg: dict, max_len: int):
    base = dict(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], head_dim=cfg["head_dim"], hidden=cfg["hidden"],
        max_len=max_len, theta=cfg["theta"], tie=cfg["tie"], eps=cfg.get("eps", 1e-6),
    )
    if cfg["arch"] in ("gqa4", "mha"):
        return GQAModel(**base, n_kv_heads=cfg["n_kv_heads"], arch_name=cfg["arch"])
    return MLAModel(**base, d_c=cfg["d_c"], d_rope=cfg["d_rope"],
                    use_qk_norm=cfg.get("use_qk_norm", False))


def load(path: Path, max_len: int, forced_mode, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    # 参数留在 FP32，前向统一走 autocast bfloat16，与训练和部署的实际执行方式一致；
    # 这样 cache 落盘也是 BF16，延迟也不会被「裸 FP32」或「裸 BF16 参数」扭曲。
    model = build(cfg, max_len).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    if cfg["arch"] == "mla":
        if forced_mode == "absorbed" and cfg.get("use_qk_norm", False):
            raise SystemExit(f"{path} 启用了 QK-Norm，吸收模式在数学上不成立")
        model.set_mode(forced_mode or ("naive" if cfg.get("use_qk_norm") else "absorbed"))
    return model, cfg


def past_len(past) -> int:
    return past[0]["c"].size(1) if "c" in past[0] else past[0]["k"].size(2)


def cache_bytes(past) -> int:
    """实际驻留的 cache 字节数，逐张量量出来的。"""
    return sum(v.numel() * v.element_size() for layer in past for v in layer.values())


def cache_elements(past) -> int:
    return sum(v.numel() for layer in past for v in layer.values())


def cast_rope_buffers(model):
    """把 cache 从 FP32 拉回 BF16 所必需的两处修正。

    autocast 只覆盖 matmul / conv，不覆盖逐元素乘与归约。于是有两处漏网：
      1. RoPE 表 cos / sin 是 FP32，`x1 * cos` 直接把 k 拉成 FP32；
      2. RMSNorm 内部 `x.pow(2).mean(-1)` 的归约在 autocast 下输出 FP32，
         再把整个 k 拉成 FP32——这一条更隐蔽，光改表没用。
    k 一旦按 FP32 写进 cache，每元素 4 字节而不是 2 字节：MHA 的 cache 从
    12 MiB 涨到 18 MiB（+50%），账面压缩比会虚高到 5.33 倍。
    这里只在本测量进程内覆盖，不改库代码，也不影响已训练的权重。
    """
    from types import MethodType

    from model_common import RMSNorm

    def rms_forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return y.to(x.dtype)  # 归一化不该改变输入精度

    n_rope = 0
    for blk in model.blocks:
        for name in ("cos", "sin"):
            buf = getattr(blk.attn, name, None)
            if isinstance(buf, torch.Tensor) and buf.dtype != torch.bfloat16:
                setattr(blk.attn, name, buf.to(torch.bfloat16))
                n_rope += 1
    n_norm = 0
    for mod in model.modules():
        if isinstance(mod, RMSNorm):
            mod.forward = MethodType(rms_forward, mod)
            n_norm += 1
    return {"rope_buffers": n_rope, "rmsnorm_patched": n_norm}


def gpu_state() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def stats(values):
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(float(np.median(arr)), 4),
        "ci95": [round(mean - half, 4), round(mean + half, 4)],
    }


@torch.no_grad()
def measure_cache(model, seq_lens, device):
    """先量未处理 RoPE 表的原始 dtype，再把表降到 BF16 量一遍。"""
    raw = _measure_cache(model, seq_lens, device)
    casted = cast_rope_buffers(model)
    fixed = _measure_cache(model, seq_lens, device)
    return {"dtype_cast": casted, "raw": raw, "fixed": fixed}


@torch.no_grad()
def _measure_cache(model, seq_lens, device):
    """每档序列长度各跑一次前向，读真实 cache。返回逐档字节数。"""
    out = []
    for L in seq_lens:
        x = torch.randint(0, 1000, (1, L), device=device)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        e0.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, past = model(x)
        e1.record()
        torch.cuda.synchronize()
        n_elem = cache_elements(past)
        n_byte = cache_bytes(past)
        out.append({
            "seq_len": L,
            "prefill_ms": round(e0.elapsed_time(e1), 3),
            "cache_elements": n_elem,
            "cache_bytes": n_byte,
            "cache_mib": round(n_byte / 1024 ** 2, 3),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2),
            "bits_per_element": n_byte * 8 // max(1, n_elem),
        })
        del past, logits
        torch.cuda.empty_cache()
        print(f"    seq {L:>5}: cache {n_byte / 1024 ** 2:8.3f} MiB, "
              f"prefill {out[-1]['prefill_ms']:8.3f} ms", flush=True)
    return out


@torch.no_grad()
def timed_once(model, seq_len, samples, decode_steps, device, data):
    prefill_ms, decode_ms = [], []
    step = max(1, (len(data) - seq_len - decode_steps - 2) // max(1, samples))
    for i in range(samples):
        start = i * step
        x = torch.from_numpy(
            np.asarray(data[start:start + seq_len]).astype(np.int64)[None]
        ).to(device)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, past = model(x)
        e1.record()
        torch.cuda.synchronize()
        prefill_ms.append(e0.elapsed_time(e1))
        nxt = logits[:, -1:].argmax(-1)
        # 整段 decode 只用一对事件包围再除以步数。逐步单独 event + synchronize 会把
        # 每步的同步与采样开销算进延迟，量级可达真实每 token 耗时的两倍，
        # 三组骨架会被同一个固定下限压平而失去可比性（与 eval.py 的 per_token 对不上）。
        d0 = torch.cuda.Event(enable_timing=True)
        d1 = torch.cuda.Event(enable_timing=True)
        d0.record()
        for _ in range(decode_steps):
            pos = past_len(past)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, past = model(nxt, past, start_pos=pos)
            nxt = logits[:, -1:].argmax(-1)
        d1.record()
        torch.cuda.synchronize()
        decode_ms.append(d0.elapsed_time(d1) / decode_steps)
    return float(np.median(prefill_ms)), float(np.median(decode_ms))


@torch.no_grad()
def warmup(model, seq_len, device, steps=3):
    x = torch.randint(0, 1000, (1, seq_len), device=device)
    for _ in range(steps):
        _, past = model(x)
        nxt = torch.zeros((1, 1), dtype=torch.long, device=device)
        for _ in range(4):
            _, past = model(nxt, past, start_pos=past_len(past))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/tokens/test.bin"))
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 2048, 4096])
    ap.add_argument("--bench-seq-len", type=int, default=512, help="延迟测量用的 prefill 长度")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "data/results/model/tables/cache_bench.json"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA 设备")

    device = torch.device("cuda")
    # 不开 cudnn.deterministic：它会强制确定性内核，本机实测 512 长度 prefill 慢约 40%、
    # 逐步 decode 慢约一倍，导致本脚本的数字与 eval.py 的 per_token 对不上。
    # 这里只要与 eval.py 同口径、能横向比较架构，不要求 bit-exact 复现。
    torch.backends.cudnn.benchmark = False
    max_len = max(max(args.seq_lens), args.bench_seq_len + args.decode_steps + 8)

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    print(f"[env] gpu: {tot_mem(device)} MiB total, {gpu_state()}", flush=True)
    print(f"[env] max_len 覆盖为 {max_len}（cache 与延迟不受 RoPE 表限制影响）", flush=True)

    models = {}
    for tag, path, mode, _note in CONFIGS:
        if not path.exists():
            print(f"[skip] {tag}: 缺少 {path}", flush=True)
            continue
        model, cfg = load(path, max_len, mode, device)
        models[tag] = model
        warmup(model, args.bench_seq_len, device)
        print(f"[load] {tag}: {sum(p.numel() for p in model.parameters()):,} 参数", flush=True)

    result = {
        "env": {
            "gpu": tot_mem(device),
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(device),
            "max_len_override": max_len,
            "cudnn_deterministic": False,
        },
        "cache": {},
        "latency": {"settings": {
            "seq_len": args.bench_seq_len, "samples": args.samples,
            "decode_steps": args.decode_steps, "runs": args.runs,
        }, "configs": {}, "rounds": []},
    }

    # 1) cache 实测：无随机性，每档一次
    print("\n[1] cache 实测", flush=True)
    for tag, model in models.items():
        print(f"  {tag}", flush=True)
        result["cache"][tag] = measure_cache(model, args.seq_lens, device)

    # 2) 延迟重复测量：每轮按顺序轮换，抵消先后效应
    print(f"\n[2] 延迟重复 {args.runs} 轮", flush=True)
    tags = list(models)
    prefill_all = {t: [] for t in tags}
    decode_all = {t: [] for t in tags}
    t_start = time.time()
    for r in range(args.runs):
        order = tags if r % 2 == 0 else tags[::-1]
        row = {"round": r + 1, "gpu": gpu_state()}
        for tag in order:
            warmup(models[tag], args.bench_seq_len, device, steps=1)
            pre, dec = timed_once(models[tag], args.bench_seq_len, args.samples,
                                  args.decode_steps, device, data)
            prefill_all[tag].append(pre)
            decode_all[tag].append(dec)
            row[tag] = {"prefill_med": round(pre, 3), "decode_med": round(dec, 3)}
        result["latency"]["rounds"].append(row)
        if (r + 1) % 5 == 0 or r == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (r + 1) * (args.runs - r - 1)
            print(f"  round {r + 1}/{args.runs}  已用 {elapsed / 60:.1f} min  "
                  f"剩余约 {eta / 60:.1f} min", flush=True)

    for tag in tags:
        result["latency"]["configs"][tag] = {
            "prefill_ms": stats(prefill_all[tag]),
            "decode_ms": stats(decode_all[tag]),
        }

    # 配对差值：以 MHA 为基准
    if "MHA" in tags:
        base_dec = np.asarray(decode_all["MHA"])
        base_pre = np.asarray(prefill_all["MHA"])
        for tag in tags:
            if tag == "MHA":
                continue
            dd = np.asarray(decode_all[tag]) - base_dec
            dp = np.asarray(prefill_all[tag]) - base_pre
            result["latency"].setdefault("delta_vs_mha", {})[tag] = {
                "decode_ms": stats(dd.tolist()),
                "prefill_ms": stats(dp.tolist()),
                "decode_pct": round(float(dd.mean() / base_dec.mean() * 100), 2),
                "prefill_pct": round(float(dp.mean() / base_pre.mean() * 100), 2),
            }

    out = Path(args.out)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 写入 {out}", flush=True)

    print("\n=== 摘要 ===", flush=True)
    for tag, block in result["cache"].items():
        for kind in ("raw", "fixed"):
            for row in block[kind]:
                print(f"{tag:<14} {kind:<5} seq {row['seq_len']:>5}  "
                      f"cache {row['cache_mib']:9.3f} MiB  "
                      f"{row['bits_per_element']:>2} bit/元素", flush=True)
    for tag, c in result["latency"]["configs"].items():
        print(f"{tag:<14} prefill {c['prefill_ms']['mean']:8.3f} ± {c['prefill_ms']['std']:.3f}  "
              f"decode {c['decode_ms']['mean']:8.3f} ± {c['decode_ms']['std']:.3f} ms", flush=True)


def tot_mem(device) -> int:
    return int(torch.cuda.get_device_properties(device).total_memory / 1024 ** 2)


if __name__ == "__main__":
    main()
