"""按相同 token 位置对比两种注意力结构的验证 PPL，并给出各 run 的训练进度。

配对方式完全由 models/<架构>/<run_id>/config.json 的 arch 与 seed 推导，不硬编码任何 run 名。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def collect():
    """返回 {arch: {seed: {"val": {tokens: ppl}, "last": ..., "cfg": ...}}}"""
    data = {}
    if not MODELS.exists():
        return data
    for run in sorted(p for p in MODELS.glob("*/*") if p.is_dir()):
        cfg_path = run / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "train" not in cfg:
            continue
        val = {r["tokens_seen"]: r["val_ppl"] for r in read_jsonl(run / "val.jsonl")}
        train = read_jsonl(run / "train.jsonl")
        data.setdefault(run.parent.name, {})[str(cfg["seed"])] = {
            "val": val, "last": train[-1] if train else None, "cfg": cfg, "dir": run.name,
        }
    return data


def main():
    data = collect()
    archs = sorted(data)
    seeds = sorted({s for a in data.values() for s in a})

    for seed in seeds:
        present = [a for a in archs if seed in data[a]]
        if len(present) < 2:
            continue
        base = "mha" if "mha" in present else present[0]
        print(f"seed {seed}: 以 {base} 为基准，验证困惑度按 token 对齐")
        for other in present:
            if other == base:
                continue
            va, vo = data[base][seed]["val"], data[other][seed]["val"]
            common = sorted(set(va) & set(vo))
            if not common:
                print(f"  {other}: 尚无对齐的验证位置")
                continue
            shown = common[::5]
            if common[-1] not in shown:
                shown.append(common[-1])
            print(f"  token(M) | {base} | {other} | 差")
            for tok in shown:
                print(f"  {tok / 1e6:8.1f} | {va[tok]:8.3f} | {vo[tok]:8.3f} | {vo[tok] - va[tok]:+7.3f}")
        print()

    print("训练进度")
    for arch in archs:
        for seed, info in sorted(data[arch].items()):
            last, cfg = info["last"], info["cfg"]
            if not last:
                continue
            total = cfg["train"]["steps"]
            pct = last["step"] / total * 100
            vv = info["val"]
            ppl = f"{vv[max(vv)]:.3f}" if vv else "-"
            print(f"  {info['dir']:34} step {last['step']:>5}/{total} ({pct:5.1f}%) "
                  f"loss={last['loss']:.3f} ppl={ppl} mem={last['peak_mem_mb']:.0f}MB")


if __name__ == "__main__":
    main()
