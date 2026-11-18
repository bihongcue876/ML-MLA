"""评测已训练模型：测试集 PPL 与推理指标。

用法：
    python MLATrain/eval.py --ckpt models/mla_qknorm/mla_42_*/model.pt
    python MLATrain/eval.py --ckpt models/gqa4/gqa4_42_*/model.pt

带 QK-Norm 的 MLA 只能走朴素路径，与 GQA-4 在同一输入、同一生成长度下计时。
结果写入 checkpoint 同目录的 infer.jsonl。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model_gqa import GQAModel
from model_mla import MLAModel

ROOT = Path(__file__).resolve().parents[1]

# 架构名取目录名：`--arch mla` 的两种配置共用 config 里的 arch="mla"。
ARCH_LABEL = {"mha": "MHA", "gqa4": "GQA-4", "mla_qknorm": "MLA", "mla_plain": "MLA plain"}


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
    mean = sum(losses) / len(losses)
    return float(np.exp(mean))


def past_len(past) -> int:
    return past[0]["c"].size(1) if "c" in past[0] else past[0]["k"].size(2)


@torch.no_grad()
def measure(model, data, seq_len: int, samples: int, decode_steps: int, device) -> dict:
    prefill_ms, decode_ms = [], []
    torch.cuda.reset_peak_memory_stats()
    for i in range(samples):
        start = i * max(1, (len(data) - seq_len - 1) // samples)
        chunk = np.asarray(data[start:start + seq_len]).astype(np.int64)
        x = torch.from_numpy(chunk[None]).to(device)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        logits, past = model(x)
        e.record()
        torch.cuda.synchronize()
        prefill_ms.append(s.elapsed_time(e))
        next_id = logits[:, -1:].argmax(-1)

        for _ in range(decode_steps):
            pos = past_len(past)
            ds = torch.cuda.Event(enable_timing=True)
            de = torch.cuda.Event(enable_timing=True)
            ds.record()
            logits, past = model(next_id, past, start_pos=pos)
            de.record()
            torch.cuda.synchronize()
            decode_ms.append(ds.elapsed_time(de))
            next_id = logits[:, -1:].argmax(-1)

    prefill_ms.sort()
    decode_ms.sort()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    prefill_med = prefill_ms[len(prefill_ms) // 2]
    decode_med = decode_ms[len(decode_ms) // 2]
    total_s = (sum(prefill_ms) + sum(decode_ms)) / 1000
    tokens = samples * (seq_len + decode_steps)
    return {
        "samples": samples, "decode_steps": decode_steps,
        "first_token_ms": round(prefill_med, 3),
        "per_token_ms": round(decode_med, 3),
        "peak_mem_mb": round(peak, 2),
        "throughput_tok_s": round(tokens / max(total_s, 1e-9), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/tokens"))
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA 设备")

    device = torch.device("cuda")
    ckpt_path = Path(args.ckpt)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ck["config"]

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    model = build_from_config(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # mha/gqa4 没有 naive/absorbed 之分，单记 plain，避免被读成 naive MLA。
    mode = "naive" if cfg["arch"] == "mla" else "plain"
    if cfg["arch"] == "mla" and not cfg.get("use_qk_norm", False):
        model.set_mode("absorbed")
        mode = "absorbed"

    test_data = np.memmap(Path(args.data) / "test.bin", dtype=np.uint16, mode="r")
    seq_len = int(ck.get("seq_len") or 512)
    if seq_len + args.decode_steps > cfg["max_len"]:
        raise SystemExit(
            f"prefill {seq_len} 加 decode {args.decode_steps} 超过 RoPE 表长度 {cfg['max_len']}"
        )
    test_ppl = perplexity(model, test_data, seq_len, device)
    metrics = measure(model, test_data, seq_len, args.samples, args.decode_steps, device)

    record = {
        "arch": ARCH_LABEL.get(ckpt_path.parent.parent.name, cfg["arch"]),
        "mode": mode, "seq_len": seq_len,
        "test_ppl": round(test_ppl, 4), **metrics,
        "params": ck.get("params"), "tokens_seen": ck.get("tokens_seen"),
        "device": torch.cuda.get_device_name(0),
    }
    out = Path(args.out) if args.out else ckpt_path.parent
    # --out 可能指向尚不存在的目录，先建出来再追加写。
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "infer.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
