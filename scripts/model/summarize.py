"""把模型级训练与评测的原始 jsonl 汇总成 CSV，不手写任何数字。

用法：
    python scripts/model/summarize.py

输入：models/<架构>/<run_id>/ 下的 config.json、train.jsonl、val.jsonl、infer.jsonl
输出：data/results/model/tables/{train_runs,train_curve,infer,absorbed_reference}.csv

架构名取目录名而不是 config 里的 arch 字段：`--arch mla` 的两种配置共用 arch="mla"，
只有目录名（mla_qknorm）能区分，否则两组会合并成同一条曲线。
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUT = ROOT / "data" / "results" / "model" / "tables"
ARCHIVED = ROOT / "data" / "results" / "model" / "archived"

ARCH_LABEL = {"mha": "MHA", "gqa4": "GQA-4", "mla_qknorm": "MLA", "mla_plain": "MLA plain"}


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
                # 训练被中断时最后一行可能写了一半，跳过而不是整体失败。
                continue
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_runs, train_curve, infer = [], [], []

    for run in sorted(p for p in MODELS.glob("*/*") if p.is_dir()):
        cfg_path = run / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        train = read_jsonl(run / "train.jsonl")
        if not train:
            continue
        val = read_jsonl(run / "val.jsonl")
        steps = train[-1].get("step") or cfg.get("train", {}).get("steps") or len(train)
        wall_hours = sum(r.get("elapsed_ms", 0) for r in train) / 1000 / 3600
        final_ppl = val[-1]["val_ppl"] if val else None
        train_runs.append({
            "run_id": run.name, "arch": ARCH_LABEL.get(run.parent.name, run.parent.name),
            "seed": cfg.get("seed"),
            "params": cfg.get("params"), "tokens_seen": train[-1]["tokens_seen"],
            "steps": steps, "final_val_ppl": final_ppl,
            "avg_tokens_per_s": round(sum(r["tokens_per_s"] for r in train) / len(train), 1),
            "peak_mem_mb": max(r["peak_mem_mb"] for r in train),
            "wall_hours": round(wall_hours, 3),
        })
        for row in val:
            train_curve.append({
                "arch": ARCH_LABEL.get(run.parent.name, run.parent.name),
                "seed": cfg.get("seed"),
                "tokens_seen": row["tokens_seen"], "val_ppl": row["val_ppl"],
            })
        label = ARCH_LABEL.get(run.parent.name, run.parent.name)
        for row in read_jsonl(run / "infer.jsonl"):
            infer.append({**row, "run_id": run.name, "arch": label})

    def write(name: str, rows: list, fields: list):
        if not rows:
            print(f"skip {name}（无数据）")
            return
        path = OUT / name
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} 行)")

    write("train_runs.csv", train_runs,
          ["run_id", "arch", "seed", "params", "tokens_seen", "steps",
           "final_val_ppl", "avg_tokens_per_s", "peak_mem_mb", "wall_hours"])
    write("train_curve.csv", train_curve, ["arch", "seed", "tokens_seen", "val_ppl"])
    write("infer.csv", infer,
          ["run_id", "arch", "mode", "seq_len", "test_ppl", "samples", "decode_steps",
           "first_token_ms", "per_token_ms", "peak_mem_mb", "throughput_tok_s",
           "params", "tokens_seen", "device"])

    # 作废权重只剩记录：吸收路径的模型级时延与当时的验证困惑度。
    archived = []
    for run in sorted(p for p in ARCHIVED.glob("*") if p.is_dir()):
        cfg_path = run / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        val = read_jsonl(run / "val.jsonl")
        for row in read_jsonl(run / "infer.jsonl"):
            archived.append({
                "run_id": run.name, "arch": "MLA plain", "mode": row.get("mode"),
                "seed": cfg.get("seed"), "seq_len": row.get("seq_len"),
                "test_ppl": row.get("test_ppl"),
                "first_token_ms": row.get("first_token_ms"),
                "per_token_ms": row.get("per_token_ms"),
                "peak_mem_mb": row.get("peak_mem_mb"),
                "throughput_tok_s": row.get("throughput_tok_s"),
                "final_val_ppl": val[-1]["val_ppl"] if val else None,
                "note": "weights deleted 2026-11-18",
            })
    write("absorbed_reference.csv", archived,
          ["run_id", "arch", "mode", "seed", "seq_len", "test_ppl", "first_token_ms",
           "per_token_ms", "peak_mem_mb", "throughput_tok_s", "final_val_ppl", "note"])


if __name__ == "__main__":
    main()
