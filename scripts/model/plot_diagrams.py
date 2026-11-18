"""生成 readme 用的两张示意图：注意力族架构图与实验流程。

与 data/results 下的结果图分开：这里是示意，不含测量值，标签用英文
（matplotlib 默认字体无中文字形）。运行后写入 assets/。
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "assets"
RES = {"d" + "p" + "i": 160}


def box(ax, x, y, w, h, text, fc, fs=8, ec="#333333"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.004,rounding_size=0.015",
            linewidth=0.8, edgecolor=ec, facecolor=fc,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                        linewidth=lw, color=color, linestyle=ls,
                        shrinkA=0, shrinkB=0)
    )


def gqa_panel(ax, title, sub, n_kv, tally, note):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"{title}\n{sub}", fontsize=9.5, pad=8, color="#222222")

    n_h = 8
    qw = 0.86 / n_h
    for i in range(n_h):
        box(ax, 0.07 + i * qw, 0.74, qw * 0.82, 0.13, "Q", "#cfe3f7", 6.5)

    kw = 0.86 / n_kv
    for j in range(n_kv):
        x = 0.07 + j * kw
        box(ax, x, 0.40, kw * (0.82 if n_kv > 1 else 0.5), 0.14,
            "K V", "#c4e6c4", 6.5)
        if n_kv > 1:
            for i in range(n_h):
                if i * n_kv // n_h == j:
                    qx = 0.07 + i * qw + qw * 0.41
                    arrow(ax, qx, 0.73, x + kw * 0.41, 0.545)

    ax.text(0.5, 0.30, note, ha="center", fontsize=6.5, color="#444444")
    ax.text(0.5, 0.17, tally, ha="center", fontsize=8.5, fontweight="bold")


def mla_panel(ax, title, sub, tally, note):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"{title}\n{sub}", fontsize=9.5, pad=8, color="#b03a2e")

    n_h = 8
    qw = 0.86 / n_h
    for i in range(n_h):
        box(ax, 0.07 + i * qw, 0.74, qw * 0.82, 0.13, "Q", "#cfe3f7", 6.5)
    ax.text(0.5, 0.66, "8 query heads, each with its own K and V", ha="center",
            fontsize=6.5, color="#444444")
    ax.text(0.5, 0.58, "no sharing: KV is compressed, not grouped", ha="center",
            fontsize=6.5, color="#444444", style="italic")

    box(ax, 0.07, 0.40, 0.52, 0.14, "c  (d_c = 384)", "#f7d9c4", 7.5)
    box(ax, 0.63, 0.40, 0.30, 0.14, "k_rope  (48)", "#e8e4c0", 7.5)
    ax.text(0.33, 0.36, "one latent per layer", ha="center", fontsize=6.2,
            color="#444444")
    ax.text(0.78, 0.36, "decoupled RoPE", ha="center", fontsize=6.2, color="#444444")

    ax.text(0.5, 0.26, note, ha="center", fontsize=6.2, color="#444444")
    ax.text(0.5, 0.17, tally, ha="center", fontsize=8.5, fontweight="bold")


def fig_arch():
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.0))
    fig.subplots_adjust(left=0.02, right=0.985, top=0.80, bottom=0.10, wspace=0.10)

    gqa_panel(
        axes[0], "MHA", "n_kv_heads = 8", 8,
        "1,536 elements / layer / token",
        "one K,V pair per query head",
    )
    gqa_panel(
        axes[1], "GQA-4", "n_kv_heads = 4", 4,
        "768 elements / layer / token",
        "4 query heads share one K,V pair",
    )
    gqa_panel(
        axes[2], "MQA", "n_kv_heads = 1", 1,
        "192 elements / layer / token",
        "all 8 query heads share one K,V pair",
    )
    mla_panel(
        axes[3], "MLA", "low-rank latent + decoupled RoPE",
        "432 elements / layer / token",
        "decode: expand per head (naive)\nor fold W_UK, W_UV into query and output (absorbed)",
    )

    fig.suptitle(
        "One family, one integer: MHA -> GQA -> MQA only change n_kv_heads.  "
        "MLA compresses KV per head instead, so it is a separate class.",
        fontsize=10.5, y=0.955,
    )
    fig.text(0.5, 0.015,
             "d_model 768, n_layers 8, n_heads 8, head_dim 96, BF16.  "
             "Element counts are per layer per token for the whole model.",
             ha="center", fontsize=7.5, color="#444444")

    OUT.mkdir(exist_ok=True)
    path = OUT / "arch_family.png"
    fig.savefig(path, **RES)
    plt.close(fig)
    print(f"wrote {path}")


def fig_flow():
    fig, ax = plt.subplots(figsize=(12.6, 4.6))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    w, h = 2.45, 0.98
    steps = [
        (0.25, 3.30, "prepare_data.py", "tokenize 1.5e8 tokens\n-> data/tokens/", "#dbe7f3"),
        (3.30, 3.30, "train.py", "--arch mha | gqa4 | mla\n--seed 42 | 43", "#dbe7f3"),
        (6.35, 3.30, "eval.py / eval_paired.py", "PPL, latency, throughput\n-> models/*/infer.jsonl", "#dbe7f3"),
        (9.40, 3.30, "summarize.py", "jsonl -> tables/*.csv", "#e5f0dc"),
        (9.40, 1.55, "plot_*.py", "tables -> figs/*.png", "#e5f0dc"),
        (6.35, 1.55, "share/", "conclusion, reference results,\nexperiment plan", "#f6e0d6"),
        (3.30, 1.55, "readme.md / guide.md", "entry points, conventions,\nhow to rerun", "#f6e0d6"),
    ]
    for x, y, t, s, c in steps:
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                           linewidth=0.9, edgecolor="#333333", facecolor=c)
        )
        ax.text(x + w / 2, y + h - 0.24, t, ha="center", va="center",
                fontsize=8.6, fontweight="bold")
        ax.text(x + w / 2, y + 0.30, s, ha="center", va="center", fontsize=6.8,
                color="#333333")

    arrow(ax, 2.70, 3.79, 3.30, 3.79)
    arrow(ax, 5.75, 3.79, 6.35, 3.79)
    arrow(ax, 8.80, 3.79, 9.40, 3.79)
    arrow(ax, 10.62, 3.30, 10.62, 2.53)
    arrow(ax, 9.40, 2.04, 8.80, 2.04)
    arrow(ax, 6.35, 2.04, 5.75, 2.04)

    ax.add_patch(
        FancyBboxPatch((0.25, 1.55), 2.45, 1.45, boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=0.9, edgecolor="#333333", facecolor="#f2f2f2")
    )
    ax.text(1.47, 2.78, "LayerTest/", ha="center", fontsize=8.6, fontweight="bold")
    ax.text(1.47, 2.10, "bench.py, bench_ctx.py\nlayer-level latency\n-> data/runs/layer_*\nsummary by summarize_layer.py",
            ha="center", va="center", fontsize=6.8)

    ax.text(6.3, 0.55,
            "LayerTest writes data/runs/layer_*; summarize_layer.py and plot_layer.py fold it into data/results/layer/.",
            ha="center", fontsize=7.5, color="#444444")

    path = OUT / "flow.png"
    fig.savefig(path, bbox_inches="tight", **RES)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    fig_arch()
    fig_flow()
