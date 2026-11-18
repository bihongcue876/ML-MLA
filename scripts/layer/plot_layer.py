"""层级结果出图：MHA、朴素 MLA、吸收 MLA 的缓存与延迟。

用法：
    python scripts/layer/plot_layer.py

输入：data/results/layer/tables/layer_bench.csv
输出：data/results/layer/figs/ 下四张
      layer_cache.png        三层每 token 每层缓存元素数（参考配置实测）
      layer_latency_ctx.png  prefill 与 decode 延迟随上下文长度
      layer_ratio_dc.png     吸收/朴素 decode 延迟比随 d_c
      layer_cache_heads.png  MLA 与 GQA 缓存元素数随 KV 组数
      加速比等派生量不单独成图，避免同一数据重复占版面。
"""

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "data" / "results" / "layer" / "tables"
FIGS = ROOT / "data" / "results" / "layer" / "figs"

IMPLS = ["mha", "naive_mla", "absorbed_mla"]
COLORS = {"mha": "tab:blue", "naive_mla": "tab:green", "absorbed_mla": "tab:red"}
REF = dict(d_c=256, n_heads=8)  # 主结论用的参考配置


def read_rows():
    path = TABLES / "layer_bench.csv"
    if not path.exists():
        print(f"缺少 {path}，先运行 scripts/layer/summarize_layer.py")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def save(name: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGS / name, **{"d" + "pi": 150})
    plt.close()
    print("wrote", FIGS / name)


def med(values):
    if not values:
        return None
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def fmt_ctx(c):
    """短基准的 decode 起点含已生成步数（2112），与 prefill 的 2048 不能共用标签。"""
    if c < 1024:
        return str(c)
    if c % 1024 == 0:
        return f"{c // 1024}K"
    return f"{c / 1024:.1f}K"


def collect_ctx(rows, impls=IMPLS, **match):
    """按 (impl, ctx) 归集中位延迟。短基准的 ctx 取 seq_len，长基准取 ctx。"""
    prefill, decode = {}, {}
    for r in rows:
        if r["kind"] not in ("layer_bench", "layer_ctx"):
            continue
        if r.get("impl") not in impls:
            continue
        if any(r.get(k) != str(v) for k, v in match.items()):
            continue
        ctx = num(r.get("ctx")) or num(r.get("seq_len"))
        if ctx is None:
            continue
        ctx = int(ctx)
        if r["kind"] == "layer_ctx":
            p, d = num(r.get("prefill_ms")), num(r.get("decode_med_ms"))
        else:
            lat = num(r.get("latency_ms"))
            p, d = (lat, None) if r.get("phase") == "prefill" else (None, lat)
        if p is not None:
            prefill.setdefault((r["impl"], ctx), []).append(p)
        if d is not None:
            decode.setdefault((r["impl"], ctx), []).append(d)
    return prefill, decode


def per_token_elems(rows):
    """每 token 的 cache 元素数，取参考配置下最长上下文的实测值。"""
    best = {}
    for r in rows:
        if r["kind"] != "layer_ctx" or r.get("impl") not in IMPLS:
            continue
        if num(r.get("d_c")) != REF["d_c"] or num(r.get("n_heads")) != REF["n_heads"]:
            continue
        total = num(r.get("cache_theoretical"))
        length = num(r.get("actual_len"))
        if not total or not length:
            continue
        elems = total / length / 2  # 实测为 BF16 字节数，除以 2 得元素数
        best.setdefault(r["impl"], elems)
    return best


def plot_cache(rows):
    elems = per_token_elems(rows)
    if not elems:
        print("skip layer_cache（缺长上下文实测 cache）")
        return
    impls = [i for i in IMPLS if i in elems]
    xs = np.arange(len(impls))
    plt.figure(figsize=(6.4, 4.2))
    bars = plt.bar(xs, [elems[i] for i in impls], width=0.5,
                   color=[COLORS[i] for i in impls])
    for bar, impl in zip(bars, impls):
        plt.annotate(f"{elems[impl]:.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha="center", va="bottom", fontsize=9)
    plt.xticks(xs, ["MHA", "MLA naive", "MLA absorbed"])
    plt.ylabel("cache elements per token per layer")
    plt.title("KV cache footprint (measured, BF16)")
    plt.grid(axis="y", alpha=0.3)
    save("layer_cache.png")


def plot_ctx_latency(rows):
    prefill, decode = collect_ctx(rows, d_c=REF["d_c"], n_heads=REF["n_heads"])
    if not prefill or not decode:
        print("skip layer_latency_ctx（缺参考配置的长上下文数据）")
        return
    ctxs = sorted({c for (_i, c) in prefill} | {c for (_i, c) in decode})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.8 / len(IMPLS)
    xs = np.arange(len(ctxs))
    for k, impl in enumerate(IMPLS):
        p = [med(prefill.get((impl, c))) for c in ctxs]
        d = [med(decode.get((impl, c))) for c in ctxs]
        off = (k - (len(IMPLS) - 1) / 2) * width
        axes[0].bar(xs + off, [v or 0 for v in p], width, label=impl, color=COLORS[impl])
        axes[1].bar(xs + off, [v or 0 for v in d], width, label=impl, color=COLORS[impl])
    for ax, title, ylabel in [
        (axes[0], "Prefill latency", "prefill latency (ms)"),
        (axes[1], "Decode latency", "per-token decode latency (ms)"),
    ]:
        ax.set_xticks(xs)
        ax.set_xticklabels([fmt_ctx(c) for c in ctxs])
        ax.set_xlabel("context length")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Single layer, batch=1, BF16, d_c={REF['d_c']}, n_heads={REF['n_heads']}",
                 fontsize=9)
    save("layer_latency_ctx.png")


def plot_ratio_vs_dc(rows):
    data = {}
    skipped = set()
    for r in rows:
        if r["kind"] != "layer_ctx" or num(r.get("n_heads")) != REF["n_heads"]:
            continue
        ctx, d_c = num(r.get("ctx")), num(r.get("d_c"))
        val = num(r.get("decode_med_ms"))
        if None in (ctx, d_c, val):
            continue
        d_rope = num(r.get("d_rope")) or 0
        if (d_c + d_rope) % 8 != 0:
            # key 宽度不是 8 的倍数时 SDPA 走慢路径，延迟不可比，剔除。
            skipped.add(int(d_c))
            continue
        data.setdefault((int(ctx), int(d_c)), {}).setdefault(r["impl"], []).append(val)
    points = {}
    for (ctx, d_c), impls in data.items():
        naive, absorbed = med(impls.get("naive_mla")), med(impls.get("absorbed_mla"))
        if naive and absorbed:
            points.setdefault(ctx, {})[d_c] = absorbed / naive
    if not points:
        print("skip layer_ratio_dc（缺吸收/朴素配对数据）")
        return
    plt.figure(figsize=(7, 4.5))
    ordered = sorted(points)
    for ctx in ordered:
        dcs = sorted(points[ctx])
        ys = [points[ctx][d] for d in dcs]
        plt.plot(dcs, ys, marker="o", label=f"ctx={ctx // 1024}K")
        # 两条线在 d_c=192 与 256 处几乎重合，按曲线固定上下错开标注。
        off = -15 if ctx == ordered[0] else 8
        for d, y in zip(dcs, ys):
            plt.annotate(f"{y:.3f}", (d, y), textcoords="offset points", xytext=(0, off),
                         ha="center", fontsize=8)
    plt.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    plt.xlabel("d_c (latent KV width)")
    plt.ylabel("absorbed / naive decode latency")
    title = "Absorption cost ratio (>1 means absorbed is slower)"
    if skipped:
        title += "\n" + ", ".join(f"d_c={d} excluded (key width not 8-aligned)" for d in sorted(skipped))
    plt.title(title, fontsize=9)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save("layer_ratio_dc.png")


def plot_cache_vs_heads():
    d_h = 96
    d_c, d_rope = 4 * d_h, d_h // 2
    mla = d_c + d_rope
    ng = np.arange(1, 17)
    gqa = 2 * ng * d_h
    plt.figure(figsize=(7, 4.5))
    plt.plot(ng, gqa, marker="o", markersize=3, label="GQA (2 * n_g * d_h)")
    plt.axhline(mla, color="tab:orange", label=f"MLA ({mla})")
    cross = mla / (2 * d_h)
    plt.axvline(cross, color="grey", linestyle="--", linewidth=1)
    plt.annotate(f"n_g = {cross:.2f}", xy=(cross, mla), xytext=(cross + 0.6, mla * 1.15),
                 fontsize=8)
    plt.xlabel("KV groups (n_g)")
    plt.ylabel("cache elements per token per layer")
    plt.title("MLA vs GQA cache (d_model=768, d_h=96)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save("layer_cache_heads.png")


def main():
    rows = read_rows()
    if not rows:
        return
    plot_cache(rows)
    plot_ctx_latency(rows)
    plot_ratio_vs_dc(rows)
    plot_cache_vs_heads()


if __name__ == "__main__":
    main()
