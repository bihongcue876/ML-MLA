"""把层级基准的原始 jsonl 汇总成一张宽表，不手写任何数字。

用法：
    python scripts/layer/summarize_layer.py

输入：data/runs/layer_*/config.json 与 bench.jsonl
      config.kind 为 layer_bench 的是短上下文基准（字段 phase/seq_len/latency_ms），
      config.kind 为 layer_ctx 的是长上下文基准（字段 ctx/prefill_ms/decode_med_ms）。
      两套字段合并进同一张宽表，缺的留空，避免任何一方的指标被丢弃。
输出：data/results/layer/tables/layer_bench.csv
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "runs"
OUT = ROOT / "data" / "results" / "layer" / "tables"

FIELDS = [
    "run_id", "kind", "impl", "round", "n_heads", "d_c", "d_rope",
    "ctx", "seq_len", "actual_len", "batch", "phase", "latency_ms",
    "prefill_blocks", "prefill_ms", "prefill_ms_per_block", "decode_med_ms",
    "peak_mem_mb", "cache_theoretical", "cache_measured", "params",
]


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # 基准被中断时最后一行可能写了一半，跳过而不是整体失败。
                continue
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for run in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        cfg_path = run / "config.json"
        bench_path = run / "bench.jsonl"
        if not cfg_path.exists() or not bench_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        base = cfg.get("base", {})
        kind = cfg.get("kind")

        for row in read_jsonl(bench_path):
            if row.get("kind") != "bench":
                # verify 行只记录数值等价性，不是性能测量。
                continue
            rows.append({
                "run_id": run.name,
                "kind": kind,
                "impl": row.get("impl"),
                "round": row.get("round"),
                "n_heads": base.get("n_heads"),
                "d_c": base.get("d_c"),
                "d_rope": base.get("d_rope"),
                "ctx": row.get("ctx"),
                "seq_len": row.get("seq_len"),
                "actual_len": row.get("actual_len"),
                "batch": row.get("batch"),
                "phase": row.get("phase"),
                "latency_ms": row.get("latency_ms"),
                "prefill_blocks": row.get("prefill_blocks"),
                "prefill_ms": row.get("prefill_ms"),
                "prefill_ms_per_block": row.get("prefill_ms_per_block"),
                "decode_med_ms": row.get("decode_med_ms"),
                "peak_mem_mb": row.get("peak_mem_mb"),
                "cache_theoretical": row.get("cache_theoretical"),
                "cache_measured": row.get("cache_measured"),
                "params": row.get("params"),
            })

    if not rows:
        print("skip layer_bench.csv（无数据）")
        return
    path = OUT / "layer_bench.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} 行)")


if __name__ == "__main__":
    main()
