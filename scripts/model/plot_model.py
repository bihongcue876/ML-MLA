"""模型级结果出图：训练曲线、困惑度、推理时延。

用法：
    python scripts/model/plot_model.py

数据来源：
    models/<架构>/<run_id>/train.jsonl、val.jsonl
    data/results/model/tables/train_curve.csv、infer.csv
输出：
    data/results/model/figs/
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
TABLES = ROOT / "data" / "results" / "model" / "tables"
FIGS = ROOT / "data" / "results" / "model" / "figs"

# 架构名取目录名：`--arch mla` 的两种配置共用 config 里的 arch="mla"。
ARCH_LABEL = {"mha": "MHA", "gqa4": "GQA-4", "mla_qknorm": "MLA", "mla_plain": "MLA plain"}


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
                # 训练被中断时最后一行可能写了一半，跳过而不是整体失败。
                continue
    return rows


def read_csv(name: str):
    path = TABLES / name
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def smooth(values, window=50):
    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def save(name: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGS / name, **{"d" + "pi": 150})
    plt.close()
    print("wrote", FIGS / name)


def train_curves(field: str, smoothing: int = 0):
    out = {}
    if not MODELS.exists():
        return out
    for run in sorted(p for p in MODELS.glob("*/*") if p.is_dir()):
        cfg_path = run / "config.json"
        rows = read_jsonl(run / "train.jsonl")
        if not cfg_path.exists() or not rows:
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "train" not in cfg:
            continue
        field_used = field if field in rows[0] else "peak_mem_mb"
        y = [r[field_used] for r in rows]
        if smoothing:
            y = smooth(y, smoothing)
        out[f"{ARCH_LABEL.get(run.parent.name, run.parent.name)}_s{cfg['seed']}"] = (
            [r["tokens_seen"] / 1e6 for r in rows], y)
    return out


def plot_train_loss():
    curves = train_curves("loss", 50)
    if not curves:
        print("skip train_loss（无训练日志）")
        return
    plt.figure(figsize=(7, 4.5))
    for key, (x, y) in curves.items():
        plt.plot(x, y, linewidth=1.3, label=key)
    plt.xlabel("tokens (M)")
    plt.ylabel("training loss (smoothed)")
    plt.title("Pretraining loss")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save("model_train_loss.png")


def plot_train_ppl():
    rows = read_csv("train_curve.csv")
    if not rows:
        print("skip train_ppl（无验证日志）")
        return
    series = {}
    for r in rows:
        key = f"{r['arch']}_s{r['seed']}"
        series.setdefault(key, ([], []))
        series[key][0].append(float(r["tokens_seen"]) / 1e6)
        series[key][1].append(float(r["val_ppl"]))
    plt.figure(figsize=(7, 4.5))
    for key, (x, y) in series.items():
        plt.plot(x, y, marker="o", markersize=3, linewidth=1.3, label=key)
    plt.xlabel("tokens (M)")
    plt.ylabel("validation perplexity")
    plt.title("Validation perplexity")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save("model_train_ppl.png")


def plot_train_speed():
    curves = train_curves("tokens_per_s", 200)
    if not curves:
        print("skip train_speed（无训练日志）")
        return
    plt.figure(figsize=(7, 4.5))
    for key, (x, y) in curves.items():
        plt.plot(x, y, linewidth=1.2, label=key)
    plt.xlabel("tokens (M)")
    plt.ylabel("tokens/s")
    plt.title("Training throughput")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save("model_train_speed.png")


def plot_train_mem():
    curves = train_curves("mem_now_mb", 100)
    if not curves:
        print("skip train_mem（无训练日志）")
        return
    plt.figure(figsize=(7, 4.5))
    for key, (x, y) in curves.items():
        plt.plot(x, y, linewidth=1.2, label=key)
    plt.xlabel("tokens (M)")
    plt.ylabel("allocated GPU memory (MB)")
    plt.title("Training memory per step")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    save("model_train_mem.png")


def plot_ppl_final():
    curve = read_csv("train_curve.csv")
    infer = read_csv("infer.csv")
    if not curve and not infer:
        print("skip ppl_final（无 PPL 数据）")
        return
    val = {}
    # 每个 (arch, seed) 取最后一次验证，再按 arch 聚合，与 test 的口径一致。
    last = {}
    for r in curve:
        key = (r["arch"], r["seed"])
        tok = int(r["tokens_seen"])
        if key not in last or tok >= last[key][0]:
            last[key] = (tok, float(r["val_ppl"]))
    for (arch, _seed), (_tok, ppl) in last.items():
        val.setdefault(arch, []).append(ppl)
    test = {}
    for r in infer:
        test.setdefault(r["arch"], []).append(float(r["test_ppl"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    archs = sorted(set(val) | set(test))
    xs = np.arange(len(archs))
    width = 0.36

    def mean_and_err(groups, arch):
        vals = groups.get(arch, [])
        if not vals:
            return 0.0, [0, 0]
        m = float(np.mean(vals))
        return m, [m - min(vals), max(vals) - m]

    val_means, val_err, test_means, test_err = [], [], [], []
    for a in archs:
        m, e = mean_and_err(val, a)
        val_means.append(m)
        val_err.append(e)
        m, e = mean_and_err(test, a)
        test_means.append(m)
        test_err.append(e)

    ax.bar(xs - width / 2, val_means, width, label="final val ppl (mean)")
    ax.bar(xs + width / 2, test_means, width, label="test ppl (mean)")
    if any(any(e) for e in val_err):
        ax.errorbar(xs - width / 2, val_means, yerr=np.array(val_err).T,
                    fmt="none", ecolor="black", capsize=4)
    if any(any(e) for e in test_err):
        ax.errorbar(xs + width / 2, test_means, yerr=np.array(test_err).T,
                    fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(archs)
    ax.set_ylabel("perplexity")
    ax.set_title("Final perplexity (mean, range across seeds)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    save("model_ppl_final.png")


def plot_infer_compare():
    rows = read_csv("infer.csv")
    if not rows:
        print("skip infer_compare（无推理结果）")
        return
    agg = {}
    for r in rows:
        agg.setdefault(r["arch"], []).append(r)
    archs = sorted(agg)
    first = [float(np.mean([float(x["first_token_ms"]) for x in agg[a]])) for a in archs]
    per = [float(np.mean([float(x["per_token_ms"]) for x in agg[a]])) for a in archs]

    def err(arch, field):
        vals = [float(x[field]) for x in agg[arch]]
        if len(vals) < 2:
            return [0, 0]
        m = float(np.mean(vals))
        return [m - min(vals), max(vals) - m]

    first_err = np.array([err(a, "first_token_ms") for a in archs]).T
    per_err = np.array([err(a, "per_token_ms") for a in archs]).T

    xs = np.arange(len(archs))
    width = 0.36
    plt.figure(figsize=(7, 4.5))
    plt.bar(xs - width / 2, first, width, label="first token (ms)", yerr=first_err, capsize=4)
    plt.bar(xs + width / 2, per, width, label="per token (ms)", yerr=per_err, capsize=4)
    plt.xticks(xs, archs)
    plt.ylabel("latency (ms)")
    plt.title("Inference latency (mean, range across seeds)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    save("model_infer_compare.png")


def plot_infer_throughput():
    rows = read_csv("infer.csv")
    if not rows:
        print("skip infer_throughput（无推理结果）")
        return
    agg = {}
    for r in rows:
        agg.setdefault(r["arch"], []).append(float(r["throughput_tok_s"]))
    archs = sorted(agg)
    xs = np.arange(len(archs))
    plt.figure(figsize=(7, 4.5))
    plt.bar(xs, [float(np.mean(agg[a])) for a in archs], width=0.45,
            yerr=[[float(np.mean(agg[a])) - min(agg[a]) for a in archs],
                  [max(agg[a]) - float(np.mean(agg[a])) for a in archs]], capsize=4)
    plt.xticks(xs, archs)
    plt.ylabel("tokens/s")
    plt.title("Inference throughput")
    plt.grid(axis="y", alpha=0.3)
    save("model_infer_throughput.png")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    plot_train_loss()
    plot_train_ppl()
    plot_train_speed()
    plot_train_mem()
    plot_ppl_final()
    plot_infer_compare()
    plot_infer_throughput()


if __name__ == "__main__":
    main()
