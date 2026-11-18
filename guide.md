# 使用与改造手册

面向要改动、变化这个项目者：结构参数在哪、训练参数怎么调、换数据怎么做、测试程序怎么用、出错先看哪里。

## 1. 环境

| 项 | 值 |
| --- | --- |
| Python | 3.13 |
| torch | 2.10.0+cu126 |
| 依赖安装 | `uv sync`（torch 由 `pyproject.toml` 的 `pytorch-cu126` 索引解析） |
| 已验证硬件 | RTX 4050 Laptop，6140.5 MB 显存，驱动 566.24 |

本机 `uv run python` 在受限账户下会报 `拒绝访问 (os error 5)`，直接用虚拟环境解释器：`.\.venv\Scripts\python.exe`。

## 2. 模型结构

两个独立实现，共享 `MLATrain/model_common.py` 的 RMSNorm、SwiGLU、RoPE 与 `Transformer` 骨架。

### 2.1 MHA / GQA（`MLATrain/model_gqa.py`）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `d_model` | 768 | 隐层宽度 |
| `n_layers` | 8 | 层数 |
| `n_heads` | 8 | 查询头数 |
| `n_kv_heads` | 4 / 8 | 广义 GQA 的共享开关：4 = GQA-4，8 = MHA（等于 `n_heads`），1 = MQA |
| `head_dim` | 96 | 每头维度 |
| `hidden` | 2432 | SwiGLU 中间维 |
| `vocab_size` | 6400 | 与官方 tokenizer 一致 |
| `max_len` | 512 | 训练长度，RoPE 表按 `max_len + ROPE_HEADROOM` 建 |
| `theta` | 1e6 | RoPE 基频 |

该文件自带 QK-Norm（`q_norm`、`k_norm`），MHA 与 GQA-4 都有。缓存每 token 元素数 = `2 · n_kv_heads · head_dim`：MHA 为 1536，GQA-4 为 768，MQA 为 192。MHA、GQA、MQA 同属广义 GQA，共用 `GQAAttention`，只差 `n_kv_heads` 这一个整数：`train.py --arch mha` 取 8，MQA 未列入 choices，需在 `build_model` 里取 1。

### 2.2 MLA（`MLATrain/model_mla.py`）

| 参数 | 默认 | 与论文的对应 |
| --- | --- | --- |
| `d_model` | 768 | — |
| `n_heads` | 8 | — |
| `head_dim` | 96 | 每头 KV 维度 |
| `d_c` | 384 | KV 潜向量维度，= 4 · `head_dim` |
| `d_rope` | 48 | 解耦 RoPE 维度，= `head_dim / 2` |
| `use_qk_norm` | 类默认 False，`train.py` 传 True | 开启后 `q_norm`/`k_norm` 作用于 `head_dim`；开启即封死吸收模式 |

权重职责：`w_dkv` 把 `d_model` 压到 `d_c`；`w_uk`、`w_uv` 从 `d_c` 升到 `n_heads · head_dim`；`w_kr` 生成解耦 RoPE 的键；`w_q`、`w_qr` 生成查询的非 RoPE 与 RoPE 部分；`w_o` 输出投影。

**缓存每 token 元素数 = `d_c + d_rope` = 432**，压缩比 = `2 · n_heads · head_dim / (d_c + d_rope) = 1536 / 432 = 3.56`。

**与论文的偏差**：DeepSeek-V2 的查询侧还有一层低秩分支（`W^DQ: d → d_c'` 后接 `W^UQ: d_c' → n_h · head_dim`，`W^QR: d_c' → n_h · d_rope`）。本项目查询侧为满秩 `w_q`、`w_qr`，没有这层压缩。后果是查询侧参数量偏大，但不影响 KV 缓存与吸收路径的时延，因为吸收只重写 KV 侧。

### 2.3 吸收路径

`MLAAttention.forward_absorbed` 把 `w_uk`、`w_uv` 乘进查询与输出，使 decode 时不再显式展开 `d_c → n_h · head_dim`。两条路径共用同一组权重对象，因此可用同一 `state_dict` 交叉校验。模型级实测相对误差在 1e-07 量级，`check_sanity.py` 每次运行都核对。

### 2.4 改结构要改哪里

- 改宽度或层数：改 `model_gqa.py`、`model_mla.py` 里 `GQAModel`、`MLAModel` 的默认参数，或从 `train.py` 的 `build_model` 传入。改完必须重跑 `python check_sanity.py`，它核对参数量与缓存公式。
- 加一个新注意力变体：新建 `model_<name>.py`，实现一个 `nn.Module`，接收 `(x, past, start_pos)` 返回 `(out, new_past)`，并提供 `cache_elements_per_token()`；在 `train.py` 的 `build_model` 里加分支；在 `run_all.py` 的 `ARCH_DIR` 里注册目录名。
- 补查询低秩分支：在 `MLAAttention.__init__` 加 `w_dq`（`d_model → d_c_q`）与两个上行投影，把 `w_q`、`w_qr` 改成从 `d_c_q` 出发；`forward` 与 `forward_absorbed` 都要同步改，否则两条路径不再等价。

## 3. 训练参数

`MLATrain/train.py` 的默认值即为已验证配置。

| 参数 | 默认 | 调整建议 |
| --- | --- | --- |
| `--tokens` | 150,000,000 | 训练总 token 预算，决定步数（预算 ÷ 每步 token） |
| `--seq-len` | 512 | 与显存强相关；训练长度不测长上下文 |
| `--batch` | 8 | 每卡批大小 |
| `--grad-accum` | 2 | 等效每步 8192 token |
| `--lr` | 1e-3 | 配合 warmup 500 步 |
| `--weight-decay` | 0.1 | — |
| `--grad-clip` | 1.0 | — |
| `--val-every` | 5,000,000 | 每多少 token 验证一次 |
| `--ckpt-every` | 20,000,000 | 每多少 token 存一次 `last.pt` |
| `--grad-checkpoint` | 关 | 开启后省显存、降吞吐 |
| `--use-qk-norm` | 关 | 仅 MLA 有效；开启后不能再用吸收模式 |

显存不足时按此顺序降：`--batch 4 --grad-accum 4` → `--seq-len 256` → 开 `--grad-checkpoint`。显存富余时反过来加大 `--batch` 提吞吐。

`18310` 这个步数是预算除以每步 token 的商（150,000,000 ÷ 8192），不是收敛所需步数。

## 4. 换数据或换 tokenizer

1. 语料放 `dataset-minimind/corpus/`，格式为每行一个 JSON，取 `text` 字段。
2. tokenizer 放 `dataset-minimind/tokenizer/`，需含 `tokenizer.json`。
3. 改 `MLATrain/prepare_data.py` 的 `--src`、`--tok`、`--out` 与三个 split 的 token 数，然后重跑。
4. `data/tokens/meta.json` 记录词表大小与切分数量，`train.py`、`eval.py` 从该文件读 `vocab_size`。换 tokenizer 后旧模型权重不可复用。

默认产出：`train.bin` 3 亿字节（1.5 亿 token，`uint16`）、`val.bin`、`test.bin` 各 200 万字节（100 万 token）。

## 5. 层级基准

三层实现在 `LayerTest/`，接口一致：`(x, past, start_pos) → (out, new_past)`，`cache_elements_per_token()`。`common.py` 提供 `sdpa()`，支持 `full`/`causal`/`chunk` 三种掩码模式；`attn_mode(t, past)` 在 `t == 1` 时返回 `full`，否则按有无 `past` 选 `causal` 或 `chunk`。

### 5.1 短上下文（`bench.py`）

```bash
python LayerTest/bench.py                       # 默认 128 / 512 / 2048
python LayerTest/bench.py --seq-lens 256 1024 --out data/runs/try
```

输出 `config.json`、`bench.jsonl`、`verify.jsonl`。`verify` 行给三种实现的相对误差；`bench` 行给 prefill 与 decode 延迟、峰值显存、缓存理论值。

### 5.2 长上下文（`bench_ctx.py`）

```bash
python LayerTest/bench_ctx.py --ctx-lens 16384 32768 --chunk 1024 --rounds 30
python LayerTest/bench_ctx.py --d-c 192 --n-heads 8 --check-only
```

参数含义：`--ctx-lens` 目标上下文；`--chunk` 分块 prefill 的块长；`--decode-steps` 每轮解码步数；`--rounds` 重复轮数（取中位数）；`--n-layers` 层数；`--n-heads`、`--d-c` 覆盖基准配置；`--check-only` 只做等价性自检不出数。

32K 必须分块：8 层 32768 见方的 BF16 注意力矩阵需 17.2 GB，超过 6 GB 卡。块长 1024 时最坏掩码占用 537 MB。

### 5.3 无效测点判据

吸收路径的键宽为 `d_c + d_rope`。该值不是 8 的倍数时，SDPA 走慢路径，延迟被放大一个数量级。已记录的两个无效点：`d_c=171`（键宽 203）absorbed 150.8 ms、`n_heads=32` 在 32K 时 absorbed 198.8 ms。选 `d_c` 时保证 `d_c + d_rope` 能被 8 整除：模型级 `d_rope=48`，`d_c` 取 4 的倍数即可；层级基准 `d_rope=32`，可取 `d_c ∈ {128, 176, 192, 256, 384}`。`scripts/layer/plot_layer.py` 会剔除这类点并在图题注明。

### 5.4 测量规范

- 时钟：本轮 `nvidia-smi -lgc 2295,2295` 因权限不足未生效，实测 SM 时钟约 2310 MHz 且随负载浮动。同一检查点两次测量可差 5.8%，因此架构间时延只取 `eval_paired.py` 的同进程交替差值，绝对毫秒数不跨时段比较。
- 每点重复 `--rounds` 轮取中位数，不用单次值。
- decode 延迟是单层、batch 1、BF16、每 token 的耗时。
- 模型级 `bench_cache.py` 的 decode 用一对事件包围整段循环再除步数，与 `eval.py` 的 `per_token` 同口径；逐步单独计时会把每步的同步与采样开销算进去，三组会被同一个固定下限压平，与 `infer.csv` 对不上。
- 但两者的绝对毫秒仍不可比，原因有两个：`bench_cache.py` 三组模型同时驻留显存；更主要的是 cache 落盘为 FP32（见 `model_claim_cache` 的 raw 序列，MHA 为 3 B/元素），使 MHA 与 GQA-4 的 decode 每步多读一倍字节，实测比 `eval.py` 高 60–70%，而 MLA 的潜在向量主要是 BF16，反而低约 20%。因此 `model_claim_latency` 只用于图内三组相对比较。
- 缓存实测取 `cache_measured`，理论值取 `cache_theoretical`，两者差值持续超过 1% 说明 dtype 泄漏（见第 7 节）。

## 6. 汇总与出图

| 脚本 | 输入 | 输出 |
| --- | --- | --- |
| `scripts/model/summarize.py` | `models/*/*/*.jsonl`、`data/results/model/archived/*/*.jsonl` | `data/results/model/tables/{train_runs,train_curve,infer,absorbed_reference}.csv` |
| `scripts/model/plot_model.py` | 上述三张 CSV | 训练过程与结果对照 7 张 |
| `scripts/model/plot_tradeoff.py` | `infer.csv`、`cache_bench.json` | 困惑度比值、解码带宽、缓存档位 3 张 |
| `scripts/model/plot_claims.py` | `infer.csv`、`cache_bench.json` | 缓存与时延两条论证 2 张 |
| `scripts/model/compare.py` | `models/*/*/val.jsonl` | 终端按架构、按 token 对齐的 PPL 差值表 |
| `scripts/model/bench_cache.py` | `models/*/*/model.pt` | `data/results/model/tables/cache_bench.json` |
| `scripts/layer/summarize_layer.py` | `data/runs/*/bench.jsonl` | `data/results/layer/tables/layer_bench.csv` |
| `scripts/layer/plot_layer.py` | 上述 CSV | 层级 4 张 |

分工原则：`scripts/model/` 只处理模型级产物，`scripts/layer/` 只处理层级产物。图表脚本只读汇总表，不读原始 jsonl，避免两处解析逻辑漂移。

`summarize` 脚本可反复重跑，不修改 `data/runs/` 与 `models/`。区间字段一律用 `num()` 安全解析，空字符串视为缺失而不是 0。

## 7. 故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `CUDA out of memory` | 批或长度过大 | 按第 3 节阶梯降级 |
| 缓存实测比理论大 50% | autocast 下 RoPE 表或归一化让中间量落回 FP32 | 检查 `rope_tables` 是否按目标 dtype 注册 buffer，`RMSNorm.forward` 是否显式 `.to(x.dtype)` |
| 长上下文延迟异常高 | 键宽非 8 对齐，SDPA 走慢路径 | 按 5.3 换 `d_c` |
| 分块 prefill 结果与一次性不一致 | 掩码模式写错，用了 `is_causal` 而没有显式 mask | 用 `common.sdpa(..., mode="chunk")`，掩码取 `torch.ones(t, total, dtype=torch.bool).tril(diagonal=total - t)` |
| 图里中文变方框 | 缺 CJK 字体 | 图内一律用英文标签 |
| `summarize` 报 `could not convert string to float: ''` | 短基准与长基准字段不同，投影时混用了单一 schema | 用宽表字段集与 `num()` 合并解析 |
