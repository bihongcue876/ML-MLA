"""把 `docs/实验设计.md` 里写下的两条预期优势，逐条对到实测数据上。

两条预期（原文）：
  1. KV Cache 显存压缩，是 MHA 方案的 1/7 到 1/4；
  2. 吞吐量提升（可能被其他因素掩盖此优势）。

对应的图：
  model_claim_cache.png   预期 1。cache 随序列长度增长，含 1/7~1/4 参考带，
                          右 panel 展示不做 dtype 收敛时账面压缩比如何虚高。
  model_claim_latency.png 预期 2。三种配置的 prefill / decode 延迟，30 轮重复，
                          误差棒是 95% 置信区间。

第三条预期（吸收优于朴素）只在层级基准上检验，见 scripts/layer/plot_layer.py：
模型级已无同权重的两条路径对照，吸收版权重在 2026-11-18 作废并删除。

数据来自 data/results/model/tables/cache_bench.json（scripts/model/bench_cache.py 生成）。
用法：
    python scripts/model/plot_claims.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "data" / "results" / "model" / "figs"
SRC = ROOT / "data" / "results" / "model" / "tables" / "cache_bench.json"

# 与训练配置一致的架构常数。
N_LAYERS, N_HEADS, N_KV_HEADS, D_HEAD, D_C, D_ROPE = 8, 8, 4, 96, 384, 48
BYTES = 2

PER_TOKEN = {
    "MHA": 2 * N_HEADS * D_HEAD * N_LAYERS,          # 12288
    "GQA-4": 2 * N_KV_HEADS * D_HEAD * N_LAYERS,     # 6144
    "MLA": (D_C + D_ROPE) * N_LAYERS,                # 3456
}
COLOR = {"MHA": "black", "GQA-4": "tab:green", "MLA": "tab:orange"}
LABEL = {"MHA": "MHA (baseline)", "GQA-4": "GQA-4", "MLA": "MLA with QK-Norm"}
EXTRAP = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768])


def load():
    if not SRC.exists():
        raise SystemExit(f"缺少 {SRC}，先跑 scripts/model/bench_cache.py")
    return json.loads(SRC.read_text(encoding="utf-8"))


def mib(elements: int) -> float:
    return elements * BYTES / 1024 ** 2


def fig_cache(data):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    # 左：cache 随序列长度。实测点只在 512/2048/4096，之外是解析外推。
    curves = {}
    for tag, per_tok in PER_TOKEN.items():
        xs = EXTRAP
        ys = np.array([mib(per_tok * x) for x in xs])
        curves[tag] = ys
        ax.plot(xs, ys, ls="--", lw=1.1, color=COLOR[tag], alpha=0.55)
        obs = data["cache"].get(tag, {}).get("fixed", [])
        ox = [r["seq_len"] for r in obs]
        oy = [r["cache_bytes"] / 1024 ** 2 for r in obs]
        ax.plot(ox, oy, "o", ms=6.5, color=COLOR[tag], mec="white", mew=0.8,
                label=f"{LABEL[tag]}  {mib(per_tok * 512):.2f} MiB / 512 tokens", zorder=5)

    # 1/7 ~ 1/4 参考带：以 MHA 为基准，预期论 MLA 应落在这条带里。
    ax.fill_between(EXTRAP, curves["MHA"] / 7, curves["MHA"] / 4, color="tab:gray",
                    alpha=0.22, lw=0,
                    label="expected band: MHA / 7 ... MHA / 4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([512, 2048, 8192, 32768])
    ax.set_xticklabels(["512", "2k", "8k", "32k"])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("KV cache per sequence (MiB, BF16)")
    ax.set_title("KV cache grows linearly with context")
    ax.grid(alpha=0.25, which="both", ls=":")
    ax.legend(loc="lower right", fontsize=7.2, framealpha=0.92)
    ax.annotate("MLA sits above the band,\nso compression is short of 1/4",
                xy=(9000, mib(PER_TOKEN["MLA"] * 9000)),
                xytext=(0, 46), textcoords="offset points",
                fontsize=7.4, color="tab:orange", ha="center",
                arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "tab:orange",
                            "shrinkA": 2, "shrinkB": 2})

    # 右：dtype 收敛前后，512 token 的 cache 占用。
    tags = ["MHA", "GQA-4", "MLA"]
    raw, fixed, bits = [], [], []
    for tag in tags:
        block = data["cache"][tag]
        r = next(x for x in block["raw"] if x["seq_len"] == 512)
        f = next(x for x in block["fixed"] if x["seq_len"] == 512)
        raw.append(r["cache_mib"])
        fixed.append(f["cache_mib"])
        bits.append((r["bits_per_element"], f["bits_per_element"]))

    x = np.arange(len(tags))
    w = 0.36
    b1 = ax2.bar(x - w / 2, raw, w, color="tab:gray", alpha=0.55,
                 label="before fix (24 bit)")
    b2 = ax2.bar(x + w / 2, fixed, w, color="tab:blue", alpha=0.85,
                 label="after fix (16 bit)")
    for rects, vals, bs in ((b1, raw, [b[0] for b in bits]), (b2, fixed, [b[1] for b in bits])):
        for rect, v, b in zip(rects, vals, bs):
            ax2.text(rect.get_x() + rect.get_width() / 2, v + 0.35, f"{v:.2f}\n{b} bit",
                     ha="center", va="bottom", fontsize=7.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABEL[t] for t in tags])
    ax2.set_ylabel("KV cache at 512 tokens (MiB)")
    ax2.set_ylim(0, max(raw) * 1.32)
    ax2.set_title("Cache at 512 tokens before and after the dtype fix")
    ax2.grid(axis="y", alpha=0.25, ls=":")
    ax2.legend(loc="upper right", fontsize=7.2)

    ratio_fixed = fixed[0] / fixed[2]
    ratio_raw = raw[0] / raw[2]
    fig.suptitle(
        f"claim 1: KV cache compression  —  measured {ratio_fixed:.2f}x "
        f"(before the dtype fix: {ratio_raw:.2f}x)", fontsize=9.5)
    fig.tight_layout()
    save("model_claim_cache.png")


def fig_latency(data):
    lat = data["latency"]["configs"]
    order = ["MHA", "GQA-4", "MLA"]
    order = [t for t in order if t in lat]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    for ax, key, title in ((axes[0], "prefill_ms", "prefill 512 tokens"),
                           (axes[1], "decode_ms", "decode, one token at a time")):
        ys = np.arange(len(order))[::-1]
        for y, tag in zip(ys, order):
            s = lat[tag][key]
            lo = s["mean"] - s["ci95"][0]
            hi = s["ci95"][1] - s["mean"]
            ax.barh(y, s["mean"], height=0.55, color=COLOR.get(tag, "tab:gray"),
                    alpha=0.85, xerr=[[lo], [hi]], capsize=3,
                    error_kw={"lw": 0.9})
            ax.text(s["mean"] + hi + 0.25, y, f"{s['mean']:.2f} ms",
                    va="center", fontsize=7.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([LABEL.get(t, t) for t in order])
        ax.set_title(title)
        ax.set_xlabel("latency (ms), mean and 95% CI over "
                      f"{lat[order[0]][key]['n']} rounds")
        ax.grid(axis="x", alpha=0.25, ls=":")
        ax.set_xlim(0, max(lat[t][key]["ci95"][1] for t in order) * 1.22)

    fig.suptitle("claim 2: latency — the smaller cache does not buy latency "
                 "at 512 tokens", fontsize=9.5)
    fig.tight_layout()
    save("model_claim_latency.png")



def save(name: str):
    out = FIGS / name
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[fig] {out}")


def main():
    data = load()
    fig_cache(data)
    fig_latency(data)


if __name__ == "__main__":
    main()
