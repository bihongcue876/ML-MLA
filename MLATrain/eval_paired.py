"""配对交替评测：在同一进程、同一时钟状态下对比两个检查点的推理开销。

用法：
    uv run python MLATrain/eval_paired.py \
        --a models/mha/mha_42_*/model.pt \
        --b models/mla_qknorm/mla_42_*/model.pt \
        --out data/results/model/paired/mha42_vs_mla42.json

为什么需要这个脚本：`eval.py` 单独测两个模型时，两次运行落在不同的 GPU 时钟状态上。
实测同一个 mla_42 检查点先后测出的 per_token 为 10.384 与 9.783，漂移 5.8%，
而架构之间的差距只有 0.6%，噪声把结论淹没。

本脚本的做法：
- 两个模型在同一进程内常驻，按轮次交替测量，先用配对差值 Δ = B - A 抵消时钟漂移；
- 每轮前做固定预热，让 boost 频率先进入稳态；
- 单步延迟取 decode_steps 步的中位数，同时给出 p10 / p90，暴露长尾；
- 每轮记录 SM 频率、温度与功耗，若中途降频可在数据里直接看到。

test PPL 仍是确定性指标，两模型各自独立计算，与 `eval.py` 口径一致。
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model_gqa import GQAModel
from model_mla import MLAModel

ROOT = Path(__file__).resolve().parents[1]


def build_from_config(cfg: dict):
    base = dict(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], head_dim=cfg["head_dim"], hidden=cfg["hidden"],
        max_len=cfg["max_len"], theta=cfg["theta"], tie=cfg["tie"],
        eps=cfg.get("eps", 1e-6),
    )
    if cfg["arch"] in ("gqa4", "mha"):
        return GQAModel(**base, n_kv_heads=cfg["n_kv_heads"], arch_name=cfg["arch"])
    return MLAModel(**base, d_c=cfg["d_c"], d_rope=cfg["d_rope"],
                    use_qk_norm=cfg.get("use_qk_norm", False))


def past_len(past) -> int:
    return past[0]["c"].size(1) if "c" in past[0] else past[0]["k"].size(2)


def load_model(path: Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    model = build_from_config(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    mode = "naive" if cfg["arch"] == "mla" else "plain"
    if cfg["arch"] == "mla" and not cfg.get("use_qk_norm", False):
        model.set_mode("absorbed")
        mode = "absorbed"
    return model, cfg, mode


@torch.no_grad()
def perplexity(model, data, seq_len: int, device, iters: int = 20) -> float:
    losses = []
    stride = max(1, (len(data) - seq_len - 2) // iters)
    for i in range(iters):
        start = i * stride
        chunk = np.asarray(data[start:start + seq_len + 1]).astype(np.int64)
        x = torch.from_numpy(chunk[None, :-1]).to(device)
        y = torch.from_numpy(chunk[None, 1:]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item())
    return float(np.exp(sum(losses) / len(losses)))


def chunk_at(data, index: int, seq_len: int):
    start = index * max(1, (len(data) - seq_len - 1) // 8)
    chunk = np.asarray(data[start:start + seq_len]).astype(np.int64)
    return torch.from_numpy(chunk[None])


@torch.no_grad()
def timed_decode(model, data, seq_len: int, samples: int, decode_steps: int, device):
    prefill_ms, decode_ms = [], []
    for i in range(samples):
        x = chunk_at(data, i, seq_len).to(device)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        logits, past = model(x)
        e1.record()
        torch.cuda.synchronize()
        prefill_ms.append(e0.elapsed_time(e1))
        nxt = logits[:, -1:].argmax(-1)
        for _ in range(decode_steps):
            pos = past_len(past)
            d0 = torch.cuda.Event(enable_timing=True)
            d1 = torch.cuda.Event(enable_timing=True)
            d0.record()
            logits, past = model(nxt, past, start_pos=pos)
            d1.record()
            torch.cuda.synchronize()
            decode_ms.append(d0.elapsed_time(d1))
            nxt = logits[:, -1:].argmax(-1)
    return prefill_ms, decode_ms


@torch.no_grad()
def warmup(model, data, seq_len: int, device, steps: int = 3):
    for i in range(steps):
        x = chunk_at(data, i, seq_len).to(device)
        _, past = model(x)
        nxt = torch.zeros((1, 1), dtype=torch.long, device=device)
        for _ in range(4):
            _, past = model(nxt, past, start_pos=past_len(past))


def gpu_state() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception as exc:  # 取不到状态不影响测量本身
        return f"unavailable: {exc}"


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


def percentile(values, q):
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="基准检查点")
    ap.add_argument("--b", required=True, help="对照检查点")
    ap.add_argument("--data", default=str(ROOT / "data/tokens"))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA 设备")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    a_path, b_path = Path(args.a), Path(args.b)
    model_a, cfg_a, mode_a = load_model(a_path, device)
    model_b, cfg_b, mode_b = load_model(b_path, device)
    seq_len = 512
    for cfg in (cfg_a, cfg_b):
        if seq_len + args.decode_steps > cfg["max_len"]:
            raise SystemExit(
                f"prefill {seq_len} 加 decode {args.decode_steps} 超过 RoPE 表长度 {cfg['max_len']}"
            )

    test_data = np.memmap(Path(args.data) / "test.bin", dtype=np.uint16, mode="r")
    ppl_a = perplexity(model_a, test_data, seq_len, device)
    ppl_b = perplexity(model_b, test_data, seq_len, device)

    warmup(model_a, test_data, seq_len, device)
    warmup(model_b, test_data, seq_len, device)

    rounds = []
    for r in range(args.rounds):
        # 奇数轮交换顺序，抵消「先测者占优」的顺序效应。
        order = [("A", model_a), ("B", model_b)] if r % 2 == 0 else [("B", model_b), ("A", model_a)]
        seen = {}
        for tag, model in order:
            state = gpu_state()
            prefill, decode = timed_decode(
                model, test_data, seq_len, args.samples, args.decode_steps, device
            )
            seen[tag] = {
                "prefill_med": round(median(prefill), 3),
                "decode_med": round(median(decode), 3),
                "decode_p10": round(percentile(decode, 0.10), 3),
                "decode_p90": round(percentile(decode, 0.90), 3),
                "samples": len(decode),
                "gpu": state,
            }
        seen["round"] = r + 1
        seen["delta_decode_med"] = round(seen["B"]["decode_med"] - seen["A"]["decode_med"], 3)
        seen["delta_prefill_med"] = round(seen["B"]["prefill_med"] - seen["A"]["prefill_med"], 3)
        rounds.append(seen)
        print(f"round {r + 1}: A decode {seen['A']['decode_med']} ms, "
              f"B decode {seen['B']['decode_med']} ms, delta {seen['delta_decode_med']} ms",
              flush=True)

    def summarize(key, tag):
        values = [r[tag][key] for r in rounds]
        return {
            "median": round(median(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "values": values,
        }

    deltas = [r["delta_decode_med"] for r in rounds]
    deltas_pre = [r["delta_prefill_med"] for r in rounds]
    result = {
        "a": {"ckpt": str(a_path), "arch": cfg_a["arch"], "mode": mode_a, "test_ppl": round(ppl_a, 4),
              "prefill": summarize("prefill_med", "A"), "decode": summarize("decode_med", "A")},
        "b": {"ckpt": str(b_path), "arch": cfg_b["arch"], "mode": mode_b, "test_ppl": round(ppl_b, 4),
              "prefill": summarize("prefill_med", "B"), "decode": summarize("decode_med", "B")},
        "settings": {"seq_len": seq_len, "samples": args.samples,
                     "decode_steps": args.decode_steps, "rounds": args.rounds},
        "paired_delta": {
            "decode_med": {"mean": round(sum(deltas) / len(deltas), 3),
                           "min": round(min(deltas), 3), "max": round(max(deltas), 3),
                           "same_sign": all(d > 0 for d in deltas) or all(d < 0 for d in deltas)},
            "prefill_med": {"mean": round(sum(deltas_pre) / len(deltas_pre), 3),
                            "min": round(min(deltas_pre), 3), "max": round(max(deltas_pre), 3),
                            "same_sign": all(d > 0 for d in deltas_pre) or all(d < 0 for d in deltas_pre)},
        },
        "rounds": rounds,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[paired] 结果写入 {out}")


if __name__ == "__main__":
    main()
