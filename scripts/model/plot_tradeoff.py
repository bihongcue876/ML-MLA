"""按权衡关系出图：困惑度比值、缓存随序列长度、解码带宽。

用法：
    python scripts/model/plot_tradeoff.py

只读各 run 目录的 train.jsonl、val.jsonl 与 tables/cache_bench.json，不依赖任何汇总
CSV，所以新跑完的 run 不需要先并表。输出到 data/results/model/figs/：

    model_ppl_ratio.png   MLA 相对同种子 MHA 的困惑度比值，含 1.0 基准线
    model_cache_ppl.png   KV cache 随序列长度增长，标出与模型权重的交叉点
    model_decode_bw.png   解码单 token 需读取的字节数随上下文长度

有两张不画，因为缺数据：test 困惑度要先跑评测才有；困惑度随压缩比的曲线
需要 d_c 至少三个点，目前只跑过 384 一个点。
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["figure.dpi"] = 150

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "data" / "results" / "model" / "figs"

# (组别, 种子, 目录)。按架构分目录存放，这里写死具体 run，避免扫描出错。
RUNS = [
    ("MHA", 42, ROOT / "models/mha/mha_42_20261117_121454"),
    ("MHA", 43, ROOT / "models/mha/mha_43_20261117_145413"),
    ("GQA-4", 42, ROOT / "models/gqa4/gqa4_42_20261117_005456"),
    ("GQA-4", 43, ROOT / "models/gqa4/gqa4_43_20261117_031712"),
    ("MLA", 42, ROOT / "models/mla_qknorm/mla_42_20261117_202030"),
    ("MLA", 43, ROOT / "models/mla_qknorm/mla_43_20261117_231608"),
]

STYLE = {
    "MHA": ("black", "-"),
    "MLA": ("tab:orange", "-"),
    "GQA-4": ("tab:green", ":"),
}

# 架构常数，与 MLATrain/model_mla.py 一致。
N_LAYERS, N_HEADS, D_HEAD, D_C, D_ROPE = 8, 8, 96, 384, 48
BYTES = 2  # BF16
PARAMS_MHA = 68_630_784
PARAMS_MLA = 68_925_696

MHA_PER_TOKEN = 2 * N_HEADS * D_HEAD * BYTES * N_LAYERS
MLA_PER_TOKEN = (D_C + D_ROPE) * BYTES * N_LAYERS
WEIGHT_BYTES_MHA = PARAMS_MHA * BYTES
WEIGHT_BYTES_MLA = PARAMS_MLA * BYTES


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 中断时最后一行可能写了一半
    return rows


def save(name: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGS / name)
    plt.close()
    print("wrote", FIGS / name)


def load_val():
    data = {}
    for arch, seed, run in RUNS:
        rows = read_jsonl(run / "val.jsonl")
        if not rows:
            print(f"skip {arch} seed {seed}（无 val.jsonl）")
            continue
        data[(arch, seed)] = ([r["tokens_seen"] / 1e6 for r in rows],
                              [r["val_ppl"] for r in rows])
    return data


def nearest(tokens, xs, ys):
    i = min(range(len(xs)), key=lambda j: abs(xs[j] - tokens))
    return ys[i]


def plot_ratio(val):
    if not val:
        print("skip model_ppl_ratio（无验证数据）")
        return
    plt.figure(figsize=(7.5, 4.8))
    for seed in (42, 43):
        base = val.get(("MHA", seed))
        if not base:
            continue
        for arch in ("MLA",):
            cur = val.get((arch, seed))
            if not cur:
                continue
            x, y = cur
            ratio = [py / nearest(px, base[0], base[1]) for px, py in zip(x, y)]
            color, dash = STYLE[arch]
            plt.plot(x, ratio, linestyle=dash, color=color, marker="o", markersize=2.5,
                     linewidth=1.2, alpha=0.85, label=f"{arch} s{seed}")
    plt.axhline(1.0, color="grey", linestyle=":", linewidth=1.2,
                label="MHA baseline")
    plt.xlabel("tokens (M)")
    plt.ylabel("perplexity ratio vs MHA (same seed)")
    plt.title("MLA perplexity overhead over MHA")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    save("model_ppl_ratio.png")


def plot_decode_bw():
    seqs = np.arange(1, 33) * 512
    mha = WEIGHT_BYTES_MHA + MHA_PER_TOKEN * seqs
    mla = WEIGHT_BYTES_MLA + MLA_PER_TOKEN * seqs
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(seqs, mha / 2**20, color="black", linewidth=1.4, label="MHA")
    plt.plot(seqs, mla / 2**20, color="tab:orange", linewidth=1.4, label="MLA")
    plt.axhline(WEIGHT_BYTES_MHA / 2**20, color="black", linestyle=":",
                linewidth=1.0, label="weights only, MHA")
    plt.axhline(WEIGHT_BYTES_MLA / 2**20, color="tab:orange", linestyle=":",
                linewidth=1.0, label="weights only, MLA")
    plt.xlabel("sequence length")
    plt.ylabel("bytes read per decode step (MiB)")
    plt.title("Decode bandwidth: weights plus KV cache")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    save("model_decode_bw.png")


CACHE_POINTS = [("MHA", 1536), ("GQA-4", 768), ("MLA", 432)]


def plot_cache_ppl(val):
    """三个 cache 档位：MHA 1536、GQA-4 768、MLA 432 个元素每层每 token。"""
    pts = []
    for arch, elems in CACHE_POINTS:
        vals = [y[-1] for (a, _s), (_x, y) in val.items() if a == arch and y]
        if not vals:
            print(f"skip {arch}（无终值）")
            continue
        pts.append((arch, elems, sum(vals) / len(vals), min(vals), max(vals), vals))
    if len(pts) < 2:
        print("skip model_cache_ppl（不足两个档位）")
        return
    plt.figure(figsize=(7.5, 4.8))
    base = pts[0][2]
    for arch, elems, mean, lo, hi, raw in pts:
        color, _ = STYLE[arch]
        plt.errorbar(elems, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                     color=color, capsize=4, markersize=7, linewidth=1.4)
        for v in raw:
            plt.scatter(elems, v, color=color, s=14, alpha=0.55)
        tag = "baseline" if arch == "MHA" else f"{mean / base - 1:+.1%}"
        right = arch == "MHA"
        plt.annotate(f"{arch}\n{mean:.3f} ({tag})", xy=(elems, mean),
                     xytext=(-8 if right else 8, 8), textcoords="offset points",
                     fontsize=8, ha="right" if right else "left")
    plt.plot([p[1] for p in pts], [p[2] for p in pts], color="grey",
             linewidth=1.0, linestyle="--", zorder=0)
    plt.xlabel("KV cache elements per layer per token")
    plt.ylabel("final validation perplexity")
    plt.title("Cache size vs perplexity, 150M tokens")
    plt.xlim(300, 1750)
    plt.grid(alpha=0.3)
    save("model_cache_ppl.png")


def main():
    val = load_val()
    plot_ratio(val)
    plot_decode_bw()
    plot_cache_ppl(val)
    print(f"\ncache per token, 8 layers BF16: MHA {MHA_PER_TOKEN} B, "
          f"MLA {MLA_PER_TOKEN} B, ratio {MHA_PER_TOKEN / MLA_PER_TOKEN:.4f}x")
    print(f"weights: MHA {WEIGHT_BYTES_MHA / 2**20:.1f} MiB, "
          f"MLA {WEIGHT_BYTES_MLA / 2**20:.1f} MiB")
    print(f"crossover with weights: MHA {WEIGHT_BYTES_MHA / MHA_PER_TOKEN:,.0f} tokens, "
          f"MLA {WEIGHT_BYTES_MLA / MLA_PER_TOKEN:,.0f} tokens")


if __name__ == "__main__":
    main()
