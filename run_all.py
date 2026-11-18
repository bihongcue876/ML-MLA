"""一键流水线：环境检查 -> 数据切分 -> 训练 -> 评测 -> 汇总 -> 出图。

用法：
    python run_all.py                          # 全流程
    python run_all.py --stages train,eval      # 只跑训练与评测
    python run_all.py --archs gqa4 --seeds 42  # 只训一个模型一个 seed
    python run_all.py --dry-run                # 只打印将执行的命令

产物：
    logs/run-logs/<时间戳>/master.log   全部子进程输出
    logs/run-logs/<时间戳>/errors.log   失败命令与堆栈
    logs/run-logs/<时间戳>/status.json  各阶段起止时间与返回码
    data/logs/live/live_training.png              训练期间每 30 秒刷新的监控图
    data/results/model/figs/*.png                 训练结束后的模型级终图
    data/results/layer/figs/*.png                 层级基准终图
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PY = sys.executable
MODELS = ROOT / "models"
BENCH_RUNS = ROOT / "data" / "runs"
FIGS = ROOT / "data" / "results" / "model" / "figs"
LIVE_DIR = ROOT / "data" / "logs" / "live"
LOGS = ROOT / "data" / "logs" / "run-logs"

# 训练产物按架构分目录。mla 即带 QK-Norm 的配置，写入 mla_qknorm。
ARCH_DIR = {"gqa4": "gqa4", "mha": "mha", "mla": "mla_qknorm"}

# train.py 的 --arch 取值与 run_id 前缀。
TRAIN_ARCH = {"mla": "mla"}


def arch_dir(arch: str) -> str:
    return ARCH_DIR.get(arch, arch)


def smooth(values, window=50):
    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


class LiveMonitor:
    """后台线程周期读取 train.jsonl，刷新训练监控图，不阻塞子进程输出。"""

    def __init__(self, interval: int = 30):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.last_error = None
        self.fig = None
        LIVE_DIR.mkdir(parents=True, exist_ok=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 5)
        try:
            self.render()
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.render()
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"

    def _collect(self):
        runs = []
        if not MODELS.exists():
            return runs
        for path in sorted(p for p in MODELS.glob("*/*") if p.is_dir()):
            cfg_path = path / "config.json"
            train_path = path / "train.jsonl"
            if not cfg_path.exists() or not train_path.exists():
                continue
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if "train" not in cfg:
                continue
            rows = []
            with open(train_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 正在写入的最后一行可能不完整，下一轮再读。
                        continue
            if not rows:
                continue
            val = []
            val_path = path / "val.jsonl"
            if val_path.exists():
                with open(val_path, "r", encoding="utf-8") as f:
                    for x in f:
                        if not x.strip():
                            continue
                        try:
                            val.append(json.loads(x))
                        except json.JSONDecodeError:
                            continue
            runs.append((f"{cfg.get('arch')}_s{cfg.get('seed')}", cfg, rows, val))
        return runs

    def render(self):
        runs = self._collect()
        if not runs:
            return
        if self.fig is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 7))
        axes = self.axes
        for ax in axes.ravel():
            ax.clear()

        for label, cfg, rows, _ in runs:
            x = [r["tokens_seen"] / 1e6 for r in rows]
            axes[0, 0].plot(x, smooth([r["loss"] for r in rows]), linewidth=1.2, label=label)
            axes[0, 1].plot(x[-2000:], [r["tokens_per_s"] for r in rows][-2000:],
                            linewidth=0.9, label=label)
        axes[0, 0].set_title("training loss (smoothed)")
        axes[0, 0].set_xlabel("tokens (M)")
        axes[0, 0].legend(fontsize=7)
        axes[0, 1].set_title("throughput")
        axes[0, 1].set_xlabel("tokens (M)")
        axes[0, 1].set_ylabel("tokens/s")
        axes[0, 1].legend(fontsize=7)

        for label, cfg, _, val in runs:
            if val:
                axes[1, 0].plot([v["tokens_seen"] / 1e6 for v in val],
                                [v["val_ppl"] for v in val], marker="o", markersize=3,
                                linewidth=1.2, label=label)
        axes[1, 0].set_title("validation perplexity")
        axes[1, 0].set_xlabel("tokens (M)")
        axes[1, 0].legend(fontsize=7)

        lines = []
        for label, cfg, rows, _ in runs:
            total = cfg.get("train", {}).get("tokens") or 0
            done = rows[-1]["tokens_seen"]
            pct = done / total * 100 if total else 0
            lines.append(f"{label}: {done / 1e6:.1f}/{total / 1e6:.1f}M ({pct:.1f}%) "
                         f"loss={rows[-1]['loss']:.3f} mem={rows[-1]['peak_mem_mb']:.0f}MB")
        axes[1, 1].axis("off")
        axes[1, 1].text(0.02, 0.95, "\n".join(lines) if lines else "waiting...",
                        va="top", fontsize=8, family="monospace")

        for ax in axes.ravel():
            ax.grid(alpha=0.3)
        self.fig.tight_layout()
        self.fig.savefig(LIVE_DIR / "live_training.png", dpi=110)


class Pipeline:
    def __init__(self, args):
        self.args = args
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = LOGS / stamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.master = open(self.log_dir / "master.log", "a", encoding="utf-8")
        self.errors = open(self.log_dir / "errors.log", "a", encoding="utf-8")
        self.status = {"started": stamp, "stages": {}}
        self.failed = False
        self.monitor = LiveMonitor(args.live_interval)

    def log(self, text: str, to_file: bool = True):
        print(text, flush=True)
        if to_file:
            self.master.write(text.rstrip("\n") + "\n")
            self.master.flush()

    def _record(self, stage: str, **kw):
        self.status["stages"].setdefault(stage, {}).update(kw)
        (self.log_dir / "status.json").write_text(
            json.dumps(self.status, ensure_ascii=False, indent=2), encoding="utf-8")

    def run_cmd(self, stage: str, cmd: list) -> int:
        self.log(f"\n=== [{stage}] {' '.join(str(c) for c in cmd)}")
        if self.args.dry_run:
            self._record(stage, cmd=[str(c) for c in cmd], dry_run=True, code=0)
            return 0
        started = time.time()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        # 子进程输出接到管道时 Python 默认块缓冲，日志会整段延迟出现；
        # 强制行缓冲，控制台与 master.log 才能按行实时看到进度。
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            bufsize=1, env=env,
        )
        tail = collections.deque(maxlen=80)
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                self.master.write(line)
                self.master.flush()
                tail.append(line)
            code = proc.wait()
        except BaseException:
            # 中断或异常时先收掉子进程，否则它会继续占着 GPU。
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            self.errors.write(f"\n[interrupt] stage={stage} 已终止子进程\n")
            self.errors.flush()
            raise
        elapsed = time.time() - started
        self._record(stage, cmd=[str(c) for c in cmd], code=code,
                     seconds=round(elapsed, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        if code != 0:
            self.failed = True
            self.errors.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] stage={stage} code={code}\n"
                f"cmd={' '.join(str(c) for c in cmd)}\n"
                f"--- last output ---\n{''.join(tail)}\n"
            )
            self.errors.flush()
            self.log(f"!!! [{stage}] 失败，返回码 {code}，详情见 {self.log_dir / 'errors.log'}")
        else:
            self.log(f"--- [{stage}] 完成，用时 {elapsed:.1f}s")
        return code

    def existing_model(self, arch: str, seed: int):
        prefix = TRAIN_ARCH.get(arch, arch)
        for path in sorted((MODELS / arch_dir(arch)).glob(f"{prefix}_{seed}_*")):
            if (path / "model.pt").exists():
                return path / "model.pt"
        return None

    def stage_check(self):
        return self.run_cmd("check", [PY, "check_sanity.py"])

    def stage_prepare(self):
        tokens_dir = ROOT / "data" / "tokens"
        # 上次被中断会留下 0 字节的 val.bin / test.bin，只判断存在会误跳过。
        needed = {"train.bin": 2, "val.bin": 2, "test.bin": 2, "meta.json": 1}
        if self.args.skip_existing and all(
            (tokens_dir / n).exists() and (tokens_dir / n).stat().st_size >= min_size
            for n, min_size in needed.items()
        ):
            self.log("[prepare] token 文件齐全，跳过")
            return 0
        return self.run_cmd("prepare", [
            PY, "MLATrain/prepare_data.py",
            "--train-tokens", str(self.args.tokens),
        ])

    def stage_train(self):
        codes = []
        for arch in self.args.archs:
            for seed in self.args.seeds:
                existing = self.existing_model(arch, seed) if self.args.skip_existing else None
                if existing:
                    self.log(f"[train] {arch} seed={seed} 已有 {existing}，跳过")
                    continue
                cmd = [
                    PY, "MLATrain/train.py", "--arch", TRAIN_ARCH.get(arch, arch),
                    "--seed", str(seed),
                    "--tokens", str(self.args.tokens), "--seq-len", str(self.args.seq_len),
                    "--batch", str(self.args.batch),
                    "--grad-accum", str(self.args.grad_accum),
                ]
                if self.args.grad_checkpoint:
                    cmd.append("--grad-checkpoint")
                code = self.run_cmd(f"train:{arch}:{seed}", cmd)
                codes.append(code)
                if code != 0 and not self.args.continue_on_error:
                    return code
        return max(codes) if codes else 0

    def stage_eval(self):
        codes = []
        for arch in self.args.archs:
            for seed in self.args.seeds:
                model = self.existing_model(arch, seed)
                if not model:
                    self.log(f"[eval] 未找到 {arch} seed={seed} 的 model.pt，跳过")
                    continue
                code = self.run_cmd(f"eval:{arch}:{seed}", [PY, "MLATrain/eval.py", "--ckpt", model])
                codes.append(code)
                if code != 0 and not self.args.continue_on_error:
                    return code
        return max(codes) if codes else 0

    def stage_bench(self):
        if self.args.skip_existing and any(BENCH_RUNS.glob("layer_*/bench.jsonl")):
            self.log("[bench] 已存在层基准记录，跳过")
            return 0
        return self.run_cmd("bench", [
            PY, "LayerTest/bench.py",
            "--decode-steps", str(self.args.decode_steps),
        ])

    def stage_summary(self):
        code = self.run_cmd("summarize", [PY, "scripts/model/summarize.py"])
        if code != 0 and not self.args.continue_on_error:
            return code
        code = self.run_cmd("plot-model", [PY, "scripts/model/plot_model.py"])
        if code != 0 and not self.args.continue_on_error:
            return code
        code = self.run_cmd("summarize-layer", [PY, "scripts/layer/summarize_layer.py"])
        if code != 0 and not self.args.continue_on_error:
            return code
        code = self.run_cmd("plot-layer", [PY, "scripts/layer/plot_layer.py"])
        if code != 0 and not self.args.continue_on_error:
            return code
        code = self.run_cmd("plot-tradeoff", [PY, "scripts/model/plot_tradeoff.py"])
        if code != 0 and not self.args.continue_on_error:
            return code
        cache_json = ROOT / "data" / "results" / "model" / "tables" / "cache_bench.json"
        if not cache_json.exists():
            self.log("[plot-claims] 缺 cache_bench.json，跳过；先跑 scripts/model/bench_cache.py")
            return 0
        return self.run_cmd("plot-claims", [PY, "scripts/model/plot_claims.py"])

    def run(self):
        stages = [s.strip() for s in self.args.stages.split(",") if s.strip()]
        self.log(f"pipeline start, logs -> {self.log_dir}")
        self.log(f"archs={self.args.archs} seeds={self.args.seeds} tokens={self.args.tokens}")
        self.monitor.start()
        try:
            if "check" in stages:
                self.stage_check()
            if "prepare" in stages and not self.failed:
                self.stage_prepare()
            if "train" in stages and not self.failed:
                self.stage_train()
            if "eval" in stages and (not self.failed or self.args.continue_on_error):
                self.stage_eval()
            if "bench" in stages and (not self.failed or self.args.continue_on_error):
                self.stage_bench()
            if "summary" in stages:
                # 汇总与出图只读 models 与 runs 下的文件，前面阶段失败也应尽量出图。
                self.stage_summary()
        finally:
            self.monitor.stop()
        if self.monitor.last_error:
            self.errors.write(f"\n[monitor] {self.monitor.last_error}\n")
            self.errors.flush()
        self.status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.status["failed"] = self.failed
        (self.log_dir / "status.json").write_text(
            json.dumps(self.status, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log(f"pipeline end, failed={self.failed}, 监控图 {LIVE_DIR / 'live_training.png'}")
        self.master.close()
        self.errors.close()
        return 1 if self.failed else 0


def main():
    # 子进程输出按 UTF-8 解码后回写到控制台，编码不匹配时只替换字符，不让整条流水线崩掉。
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="check,prepare,train,eval,bench,summary")
    ap.add_argument("--archs", nargs="+", default=["mha", "gqa4", "mla"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--tokens", type=int, default=150_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.add_argument("--live-interval", type=int, default=30)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return Pipeline(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
