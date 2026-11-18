# 训练产物

本目录存放从零预训练的模型，**权重不入库**（见仓库根 `.gitignore`），此文件只保留清单与重建方式。

## 目录约定

```text
models/<架构目录>/<run_id>/
  config.json    结构配置、超参数、参数量、数据规模
  model.pt       权重 + 结构配置 + 参数量，评测与推理只用这个
  last.pt        含优化器与随机状态，可 --resume 续训
  train.jsonl    训练记录，每 20 步一条
  val.jsonl      验证记录，每 500 万 token 一条
  infer.jsonl    推理指标（首 token 延迟、每 token 延迟、吞吐、峰值显存）
  diag.jsonl     仅 MLA 有，记录诊断量
```

`run_id` 形如 `<arch>_<seed>_<YYYYMMDD_HHMMSS>`，由 `train.py` 生成，含时间戳以避免覆盖。

## 现有 run

| 架构目录 | run_id（seed 42 / 43） | 参数量 | 最终验证 PPL（42 / 43） |
| --- | --- | --- | --- |
| `mha` | `mha_42_20261117_121454` / `mha_43_20261117_145413` | 68,630,784 | 9.1557 / 9.1246 |
| `gqa4` | `gqa4_42_20261117_005456` / `gqa4_43_20261117_031712` | 63,912,192 | 9.3272 / 9.2341 |
| `mla_qknorm` | `mla_42_20261117_202030` / `mla_43_20261117_231608` | 68,925,696 | 10.5606 / 10.3130 |

6 个 run 共用同一份 token 切分、同一 token 预算（1.5 亿）、同一优化器配置。GQA-4 的参数量比 MHA 少 6.88%，因为它只有 4 个 KV 头；两者共用 `model_gqa.py` 的同一个类，只差 `n_kv_heads` 取 4 还是 8。

目录名 `mla_qknorm` 来自 `train.py` 的分目录规则：`--arch mla` 默认启用 QK-Norm，产物写入 `mla_qknorm`，加 `--no-use-qk-norm` 才写入 `mla_plain`。`mla_plain` 那组权重已删除（吸收路径与 QK-Norm 互斥，该配置整体作废），只留推理记录在 `data/results/model/archived/`，汇总为 `data/results/model/tables/absorbed_reference.csv`。

## 重建

```bash
python MLATrain/prepare_data.py
python MLATrain/train.py --arch mha  --seed 42
python MLATrain/train.py --arch gqa4 --seed 42
python MLATrain/train.py --arch mla  --seed 42
```

单条在 RTX 4050 Laptop 上约 2–2.7 小时。`--arch mla` 默认带 QK-Norm，写入 `models/mla_qknorm/`；`--arch mha` 与 `--arch gqa4` 共用 `model_gqa.py`，只差 `n_kv_heads`。

## 使用

```bash
# 测试集 PPL 与推理指标
python MLATrain/eval.py --ckpt models/mha/mha_42_20261117_121454/model.pt

# 配对比较
python MLATrain/eval_paired.py --a models/mha/mha_42_*/model.pt --b models/mla_qknorm/mla_42_*/model.pt
```

汇总与出图不读权重，只读本目录的 jsonl 与 `data/results/` 下的汇总表，见 `scripts/model/summarize.py`。
