"""从零预训练 GQA-4 与 MLA 两个同骨架模型。

用法：
    python MLATrain/train.py --arch gqa4 --seed 42
    python MLATrain/train.py --arch mla  --seed 42

日志写入 models/<架构>/<run_id>/ 下的 train.jsonl 与 val.jsonl，
训练结束后保存 model.pt（权重 + 结构配置）与 last.pt（含优化器，可续训）。
架构目录：gqa4 / mha 同名；mla 默认带 QK-Norm 并写入 mla_qknorm，
加 --no-use-qk-norm 才写入 mla_plain（该配置已作废，仅留作对照）。
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model_gqa import GQAModel
from model_mla import MLAModel

ROOT = Path(__file__).resolve().parents[1]
ROPE_HEADROOM = 128


def build_model(arch: str, max_len: int, use_qk_norm: bool = False):
    if arch == "gqa4":
        return GQAModel(max_len=max_len)
    if arch == "mha":
        # MHA 即每头独立的 KV，KV 头数取满 8 组，与 GQA/MLA 只差注意力结构。
        return GQAModel(max_len=max_len, n_kv_heads=8, arch_name="mha")
    if arch == "mla":
        return MLAModel(max_len=max_len, use_qk_norm=use_qk_norm)
    raise ValueError(arch)


def lr_at(step: int, total: int, base: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def get_batch(data, batch: int, seq_len: int, device, rng):
    ix = rng.integers(0, len(data) - seq_len - 2, size=batch)
    idx = ix[:, None] + np.arange(seq_len + 1)[None, :]
    chunk = np.asarray(data[idx])
    x = torch.from_numpy(chunk[:, :-1].astype(np.int64)).to(device)
    y = torch.from_numpy(chunk[:, 1:].astype(np.int64)).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, data, seq_len: int, device, iters: int = 20):
    model.eval()
    losses = []
    stride = max(1, (len(data) - seq_len - 2) // iters)
    for i in range(iters):
        start = i * stride
        chunk = np.asarray(data[start:start + seq_len + 1]).astype(np.int64)
        x = torch.from_numpy(chunk[None, :-1]).to(device)
        y = torch.from_numpy(chunk[None, 1:]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    mean = sum(losses) / len(losses)
    return mean, math.exp(mean)


def save_ckpt(path: Path, model, opt, step: int, tokens_seen: int, rng, config: dict):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optim": opt.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "rng": rng.bit_generator.state,
        "config": config,
    }, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["gqa4", "mha", "mla"], required=True,
                    help="gqa4=KV 头 4 组；mha=同一模型取 8 组（无独立实现）；mla=启用 QK-Norm 的低秩 KV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokens", type=int, default=150_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=5_000_000)
    ap.add_argument("--ckpt-every", type=int, default=20_000_000)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--use-qk-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="MLA 默认带 QK-Norm（DeepSeek-V2 结构）；关闭后可用吸收模式")
    ap.add_argument("--data", default=str(ROOT / "data/tokens"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA 设备")

    device = torch.device("cuda")
    data_dir = Path(args.data)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    torch.manual_seed(args.seed)
    model = build_model(args.arch, args.seq_len + ROPE_HEADROOM, args.use_qk_norm).to(device)
    model.set_grad_checkpoint(args.grad_checkpoint)

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    tokens_per_step = args.batch * args.grad_accum * args.seq_len
    available = len(train_data) - args.seq_len - 2
    total_steps = max(1, min(args.tokens, available) // tokens_per_step)
    total_tokens = total_steps * tokens_per_step

    run_id = f"{args.arch}_{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.arch == "mla":
        arch_dir = "mla_qknorm" if args.use_qk_norm else "mla_plain"
    else:
        arch_dir = args.arch
    out = Path(args.out) if args.out else ROOT / "models" / arch_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "arch": args.arch,
        "seed": args.seed,
        "model": model.arch_config,
        "params": model.num_params(),
        "train": {
            "tokens": total_tokens, "steps": total_steps, "seq_len": args.seq_len,
            "batch": args.batch, "grad_accum": args.grad_accum, "lr": args.lr,
            "warmup": args.warmup, "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip, "grad_checkpoint": args.grad_checkpoint,
            "tokens_per_step": tokens_per_step,
        },
        "data": {"dir": str(data_dir), "train_tokens": int(len(train_data)),
                 "val_tokens": int(len(val_data)), "vocab_size": meta.get("vocab_size")},
        "env": {
            "device": torch.cuda.get_device_name(0),
            "gpu_mem_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "dtype": "bfloat16",
        },
        "argv": " ".join(__import__("sys").argv),
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    step = 0
    tokens_seen = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        step = int(ck["step"])
        tokens_seen = int(ck["tokens_seen"])
        rng.bit_generator.state = ck["rng"]
        print(f"resumed from {args.resume} at step={step} tokens={tokens_seen}", flush=True)

    train_log = open(out / "train.jsonl", "a", encoding="utf-8")
    val_log = open(out / "val.jsonl", "a", encoding="utf-8")

    print(f"run={out} arch={args.arch} params={model.num_params():,} "
          f"steps={total_steps} tokens={total_tokens:,}", flush=True)

    model.train()
    wall_start = time.time()
    while step < total_steps:
        lr = lr_at(step, total_steps, args.lr, args.warmup)
        for group in opt.param_groups:
            group["lr"] = lr
        step_start = time.time()
        loss_value = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch, args.seq_len, device, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            (loss / args.grad_accum).backward()
            loss_value += loss.item() / args.grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += tokens_per_step
        dt = time.time() - step_start

        record = {
            "step": step, "tokens_seen": tokens_seen, "loss": round(loss_value, 6),
            "lr": lr, "grad_norm": round(float(grad_norm), 4),
            "tokens_per_s": round(tokens_per_step / max(dt, 1e-6), 1),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
            "mem_now_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
            "elapsed_ms": round(dt * 1000, 2),
        }
        train_log.write(json.dumps(record, ensure_ascii=False) + "\n")
        train_log.flush()

        if step % 20 == 0:
            elapsed = time.time() - wall_start
            eta = elapsed / step * (total_steps - step)
            print(f"step {step}/{total_steps} loss={loss_value:.4f} lr={lr:.2e} "
                  f"grad={float(grad_norm):.2f} {record['tokens_per_s']:.0f} tok/s "
                  f"mem={record['peak_mem_mb']:.0f}MB eta={eta / 60:.1f}min", flush=True)

        if tokens_seen // args.val_every > (tokens_seen - tokens_per_step) // args.val_every:
            val_loss, val_ppl = evaluate(model, val_data, args.seq_len, device)
            vrec = {"step": step, "tokens_seen": tokens_seen,
                    "val_loss": round(val_loss, 6), "val_ppl": round(val_ppl, 4)}
            val_log.write(json.dumps(vrec, ensure_ascii=False) + "\n")
            val_log.flush()
            print(f"  val loss={val_loss:.4f} ppl={val_ppl:.2f}", flush=True)

        if tokens_seen // args.ckpt_every > (tokens_seen - tokens_per_step) // args.ckpt_every:
            save_ckpt(out / "last.pt", model, opt, step, tokens_seen, rng, config)
            print(f"  checkpoint saved at {tokens_seen:,} tokens", flush=True)

    val_loss, val_ppl = evaluate(model, val_data, args.seq_len, device)
    val_log.write(json.dumps(
        {"step": step, "tokens_seen": tokens_seen,
         "val_loss": round(val_loss, 6), "val_ppl": round(val_ppl, 4)},
        ensure_ascii=False) + "\n")
    val_log.flush()
    print(f"  final val loss={val_loss:.4f} ppl={val_ppl:.2f}", flush=True)

    save_ckpt(out / "last.pt", model, opt, step, tokens_seen, rng, config)
    torch.save({
        "model": model.state_dict(),
        "config": model.arch_config,
        "tokens_seen": tokens_seen,
        "step": step,
        "params": model.num_params(),
        "seq_len": args.seq_len,
    }, out / "model.pt")
    train_log.close()
    val_log.close()
    print(f"done in {(time.time() - wall_start) / 3600:.2f}h -> {out / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
