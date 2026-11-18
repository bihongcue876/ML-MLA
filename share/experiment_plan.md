# 实验方案

> 返回 [公共报告](public_report.md) · [readme](../readme.md)。配套：[测试结论](conclusion.md) · [参考结果](reference_results.md) · [操作手册](../guide.md)

## 关于 MLA 机制效果的小体量复现探索

单卡 6 GB 显存，64M 级模型，单层基准。目标是复现 MLA 的两项机制性质——KV 缓存压缩比与吸收式推理的时延代价——而不是复现 DeepSeek-V2 的训练配方或最终质量。

## 一、要回答的问题

1. MLA 的 KV 缓存相对 MHA 与 GQA-4 压缩到什么程度，压缩比是否与设计公式一致。
2. 吸收式推理在单请求、batch 1 条件下，相对朴素 MLA 是更快还是更慢；盈亏边界在哪里。
3. 在相同语料与相同 token 预算下，三种注意力骨架（MHA、GQA-4、MLA）的训练成本与困惑度差异有多大。

## 二、明确不回答的问题

- 不回答吸收式推理在服务端批量场景下是否划算。吸收的收益路径是缓存密度换批量吞吐，本项目全部测量固定在 batch 1，无该数据。
- 不以困惑度判定注意力结构的优劣。64M 级模型的能力上限由规模主导，PPL 只作同预算下的质量参照。
- 不覆盖超过 32K 的上下文、多卡、与官方权重的质量对齐。

## 三、模型级对照

### 3.1 三组骨架

| 组 | 注意力 | KV 头数 | 每 token 缓存元素 | 参数量 |
| --- | --- | --- | --- | --- |
| MHA | 每头独立 KV（广义 GQA 取 `n_kv_heads=8`） | 8 | 1536 | 68,630,784 |
| GQA-4 | 8 查询头共享 4 组 KV（`n_kv_heads=4`） | 4 | 768 | 63,912,192 |
| MQA（未训练） | 8 查询头共享 1 组 KV（`n_kv_heads=1`） | 1 | 192 | 60,373,248 |
| MLA | KV 低秩压缩 + 解耦 RoPE，启用 QK-Norm | — | 432 | 68,925,696 |

MHA、GQA、MQA 同属广义 GQA，共用 `MLATrain/model_gqa.py` 的同一个类，只差 `n_kv_heads` 这一个整数：取 8 是 MHA、取 4 是 GQA-4、取 1 是 MQA，缓存与投影形状随之线性变化。本轮只训练 MHA、GQA-4 与 MLA 三组；MQA 的参数量按同一公式外推，未实测。MLA 不属于这个族：它不减少 KV 组数，而是把每头 KV 压成低秩潜在向量，压缩的是维度而不是共享度。

共同骨架：`d_model=768`、`n_layers=8`、`n_heads=8`、`head_dim=96`、SwiGLU 中间维 2432、词表 6400、训练长度 512。MLA 取 `d_c=384=4·head_dim`、`d_rope=48=head_dim/2`，与 DeepSeek-V2 的 KV 侧比例一致（384/96=4、48/96=0.5）。

### 3.2 单一变量原则

三组之间只改注意力模块，其余全部固定：同一份 token 切分、同一 token 预算 1.5 亿、同一优化器配置（AdamW，lr 1e-3，warmup 500，wd 0.1，clip 1.0）、同一批大小（`batch 8 × grad-accum 2`，等效每步 8192 token）、同一验证间隔（每 500 万 token）。

每组跑两个 seed（42、43），差值小于 seed 极差时记为「未观测到显著差异」。

GQA-4 的参数量比 MHA 少 6.88%（KV 头数减半），这是结构差异的必然结果，因此在报告里称为「近参数规模」而非「同参数规模」。

### 3.3 指标

- 训练：墙钟时长、平均吞吐（token/s）、峰值显存、验证损失曲线。
- 质量：验证集与测试集困惑度。
- 推理：首 token 延迟、每 token 延迟、吞吐、峰值显存。序列长 512，20 个样本，每样本解码 64 步。

## 四、层级对照

### 4.1 三层实现

| 实现 | 缓存（每 token 每层） | 说明 |
| --- | --- | --- |
| MHA | 1024 元素 | 基准 |
| 朴素 MLA | 288 元素 | 缓存 `c` 与 `k_rope`，decode 时展开为完整键值 |
| 吸收 MLA | 288 元素 | 把 `W_UK`、`W_UV` 乘进查询与输出，decode 不再展开 |

三层权重名与形状完全一致，可用同一 `state_dict` 交叉验证；吸收与朴素两条路径的输出相对误差在 1e-07 量级，属浮点误差，即代数等价。

基准配置：`d_model=512`、`n_heads=8`、`head_dim=64`、`d_c=256`、`d_rope=32`、BF16、batch 1。

### 4.2 测量设置

- 短上下文：序列 128 / 512 / 2048，测 prefill 延迟与 decode 单步延迟。
- 长上下文：16K / 32K，prefill 必须分块（8 层 32768 见方 BF16 注意力矩阵需 17.2 GB，超过 6 GB 卡），块长 1024，最坏掩码占用 537 MB。
- decode 每点重复 30 轮取中位数，不用单次值。
- 扫描两个维度：`d_c ∈ {128, 192, 256, 384}` 与 `n_heads ∈ {8, 16, 32}`，其余固定。

### 4.3 盈亏判据

吸收把 decode 的解码计算从「先展开 `d_c → n_h·head_dim` 再做注意力」改为「在 `d_c` 宽度上直接做注意力」。单步 decode 下，吸收占优需要

```text
d_c < (4·n_h·head_dim − n_h·d_rope) / (2·n_h − 1)
```

在 `n_h=8`、`head_dim=64`、`d_rope=32` 下阈值为 179.2。MLA 的标准取值 `d_c=4·head_dim=256` 远超该阈值，因此在该配置下吸收不占优；这与实测一致（比值 1.34–1.35）。判据给的是量级边界，不是精确翻盘点。

## 五、测量规范

1. GPU 时钟：本机 `nvidia-smi -lgc` 因权限不足未能锁频（实测 SM 时钟约 2310 MHz，随负载浮动）。因此不跨时段比较绝对毫秒数，架构间时延一律用同进程交替测量的配对差值。
2. 每点重复多轮取中位数。
3. 缓存同时报理论值与实测值；实测持续大于理论 1% 说明 dtype 泄漏。
4. 键宽 `d_c + d_rope` 必须是 8 的倍数，否则 SDPA 走慢路径、延迟被放大一个数量级，该点作废并注明。
5. 分块 prefill 用显式掩码 `tril(diagonal=total − t)`，不用 `is_causal`，后者在带 `past` 时会关闭掩码导致看到未来 token。

## 六、与论文的偏差

MLA 的查询侧在 DeepSeek-V2 中还有一层低秩压缩（`W^DQ` 后接 `W^UQ`、`W^QR`）。本项目查询侧为满秩 `w_q`、`w_qr`，没有该分支。后果是查询侧参数量偏大、与论文结构不完全对齐；不影响 KV 缓存与吸收路径的时延，因为吸收只重写 KV 侧。

## 七、复现路径

```bash
python check_sanity.py
python MLATrain/prepare_data.py
python MLATrain/train.py --arch mha  --seed 42
python MLATrain/train.py --arch gqa4 --seed 42
python MLATrain/train.py --arch mla  --seed 42
python MLATrain/eval.py --ckpt models/mha/mha_42_*/model.pt
python LayerTest/bench.py
python LayerTest/bench_ctx.py --ctx-lens 16384 32768 --chunk 1024 --rounds 30
python scripts/model/summarize.py && python scripts/model/plot_model.py
python scripts/layer/summarize_layer.py && python scripts/layer/plot_layer.py
```
