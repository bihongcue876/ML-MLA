"""把 JSONL 语料切成训练、验证、测试三个 uint16 二进制文件。

用法：
    python MLATrain/prepare_data.py

默认从 dataset-minimind/corpus/pretrain_t2t_mini.jsonl 取 1.5 亿训练 token，
其后各取 100 万 token 作验证集与测试集，输出到 data/tokens/。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 200_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "dataset-minimind/corpus/pretrain_t2t_mini.jsonl"))
    ap.add_argument("--tok", default=str(ROOT / "dataset-minimind/tokenizer"))
    ap.add_argument("--out", default=str(ROOT / "data/tokens"))
    ap.add_argument("--train-tokens", type=int, default=150_000_000)
    ap.add_argument("--eval-tokens", type=int, default=1_000_000)
    ap.add_argument("--batch-lines", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tok)
    eos = tok.eos_token_id

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targets = [
        ("train", args.train_tokens),
        ("val", args.eval_tokens),
        ("test", args.eval_tokens),
    ]
    files = {name: open(out / f"{name}.bin", "wb") for name, _ in targets}
    counts = {name: 0 for name, _ in targets}
    cur = {"i": 0}

    def drain(buf: list) -> None:
        while buf and cur["i"] < len(targets):
            name, limit = targets[cur["i"]]
            room = limit - counts[name]
            take = min(room, len(buf))
            files[name].write(np.asarray(buf[:take], dtype=np.uint16).tobytes())
            counts[name] += take
            del buf[:take]
            if counts[name] >= limit:
                files[name].close()
                cur["i"] += 1

    buf: list = []
    last_report = 0
    last_t = time.time()
    started = last_t
    line_no = 0
    print(f"src   = {args.src}", flush=True)
    print(f"tok   = {args.tok}", flush=True)
    print(f"out   = {out}", flush=True)
    print(f"目标  = " + " ".join(f"{k}:{v:,}" for k, v in targets), flush=True)
    with open(args.src, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            line_no += 1
            lines.append(line)
            if len(lines) < args.batch_lines:
                continue
            texts = [json.loads(x)["text"] for x in lines]
            lines = []
            for ids in tok(texts, add_special_tokens=False)["input_ids"]:
                buf.extend(ids)
                buf.append(eos)
            now = time.time()
            # 20 秒没有输出就打印一次，避免长时间黑屏；token 满 1000 万也打印。
            if len(buf) >= CHUNK or now - last_t >= 20:
                drain(buf)
                total = sum(counts.values())
                if total - last_report >= 10_000_000 or now - last_t >= 20:
                    last_report = total
                    last_t = now
                    elapsed = now - started
                    print(f"lines={line_no:,} tokens={total:,} "
                          f"({total / max(elapsed, 1e-6):.0f} tok/s, {elapsed / 60:.1f} min) "
                          + " ".join(f"{k}={v:,}" for k, v in counts.items()), flush=True)
            if cur["i"] >= len(targets):
                break
        if lines and cur["i"] < len(targets):
            texts = [json.loads(x)["text"] for x in lines]
            for ids in tok(texts, add_special_tokens=False)["input_ids"]:
                buf.extend(ids)
                buf.append(eos)
        drain(buf)

    for name in files:
        try:
            files[name].close()
        except Exception:
            pass

    meta = {
        "src": args.src,
        "tokenizer": args.tok,
        "vocab_size": tok.vocab_size,
        "eos_token_id": eos,
        "counts": counts,
        "targets": {k: v for k, v in targets},
        "short": {k: counts[k] < v for k, v in targets},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
