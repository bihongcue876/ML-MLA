# 项目运作参考数据

本目录存放本项目产生的全部数据。除 `data/tokens/` 外均可入库，因为体积小（合计约 2.2 MB）且是结果数字的原始记录。

## 结构

```text
data/
  tokens/     切分后的训练/验证/测试 token（不入库，可由语料重建）
  runs/       层级基准的每次运行记录
  results/    汇总表与图
  logs/       训练与流水线日志
```

## tokens/

| 文件 | 内容 |
| --- | --- |
| `train.bin` | 1.5 亿 token，`uint16`，300,000,000 字节 |
| `val.bin` | 100 万 token，2,000,000 字节 |
| `test.bin` | 100 万 token，2,000,000 字节 |
| `meta.json` | 词表大小、三个 split 的 token 数、切分参数 |

由 `python MLATrain/prepare_data.py` 从 `dataset-minimind/corpus/pretrain_t2t_mini.jsonl` 生成，固定随机种子，删除后重跑可得同一结果。该目录已加入 `.gitignore`。

## runs/

每次 `LayerTest/bench.py` 或 `LayerTest/bench_ctx.py` 产生一个子目录，形如 `layer_YYYYMMDD_HHMMSS` 或 `layerctx_YYYYMMDD_HHMMSS`，内含：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 本次运行的实现列表、序列长度、`d_c`、`n_heads`、块长、轮数 |
| `bench.jsonl` | 每条测量一行：阶段、延迟、峰值显存、缓存理论值与实测值 |
| `verify.jsonl` | 三种实现的输出相对误差，用于确认代数等价 |

重跑基准会新增目录而非覆盖旧目录，因此历史测点保留在原处。

## results/

```text
results/model/tables/   train_runs.csv、train_curve.csv、infer.csv、absorbed_reference.csv、cache_bench.json
results/model/paired/   配对评测的同 seed 差值 JSON，由 MLATrain/eval_paired.py 产生
results/model/archived/ 已作废配置的原始 jsonl（MLA 不启用 QK-Norm 的两组，权重已删除）
results/model/figs/     模型级 12 张图
results/layer/tables/   layer_bench.csv
results/layer/figs/     层级 4 张图
```

`archived/` 只保留训练与推理的 jsonl，因为那两组的权重已删、不再参与训练对照；`summarize.py` 读它生成 `absorbed_reference.csv`，作为吸收路径的唯一实测记录。表由 `scripts/model/summarize.py`、`scripts/layer/summarize_layer.py` 从原始 jsonl 生成；图由 `scripts/model/` 下 `plot_model.py`、`plot_tradeoff.py`、`plot_claims.py` 与 `scripts/layer/plot_layer.py` 从表生成。两者都可反复重跑，不改动 `runs/` 与 `models/`。

## logs/

| 路径 | 内容 |
| --- | --- |
| `run-logs/<时间戳>/master.log` | 流水线全部子进程输出 |
| `run-logs/<时间戳>/errors.log` | 失败阶段、命令、返回码与末 80 行输出 |
| `run-logs/<时间戳>/status.json` | 各阶段起止时间、返回码 |
| `live/live_training.png` | `run_all.py` 训练期间每 30 秒刷新的监控图 |

## 重建顺序

```bash
python MLATrain/prepare_data.py          # data/tokens/
python LayerTest/bench.py                # data/runs/layer_*
python LayerTest/bench_ctx.py --ctx-lens 16384 32768 --chunk 1024 --rounds 30
python scripts/model/summarize.py && python scripts/model/plot_model.py
python scripts/layer/summarize_layer.py && python scripts/layer/plot_layer.py
```
