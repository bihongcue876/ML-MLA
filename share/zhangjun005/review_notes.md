# 相对原报告的核对与采纳建议

整理：**zhangjun005**。定位以e3c695d的[原公共报告](../public_report.md)小节为准。本文件仅提出可分别审阅的建议；原文保持不变。独立技术稿是课程叙述提案，不表示原作者已接受全部改写。

## 保留的技术方向

原作者围绕缓存压缩和吸收代价做小规模机制实验；课程汇报增加“为什么需要MLA、技术怎样工作”的解释，仍沿综述、原理、方案、结果、局限展开。已有实验作为例证，不把审查仓库改成汇报主体。

## 建议逐项审阅

| 原报告位置 | 核对结果与建议 | 可查证据 |
| --- | --- | --- |
| §2.2、缓存定义 | 潜变量跨头共享；本模型每token每层384+48=432，不乘头数 | MLATrain/model_mla.py的缓存返回；cache_bench.json |
| §2.3、吸收查询宽度 | 吸收后的每头内容宽为d_c，含位置为d_c+d_rope；缩放保留原定义 | LayerTest/absorbed_mla.py；论文§2.1 |
| §2.3、prefill | 潜空间执行不只在decode数学成立，prefill并不必须先恢复完整KV | 两条吸收实现及矩阵结合律 |
| §2.3、按论文逐头QK-Norm | 官方参考代码是潜变量RMSNorm后上投影，与本地展开后逐头K归一化不同；不把本地吸收限制当MLA固有限制 | 官方DeepseekV2Attention，q_a_layernorm/kv_a_layernorm |
| §3.1、四方案训练 | 实际三种训练，MQA仅公式参考；也非严格等参数消融 | train_runs.csv六行；主报告参数表 |
| §3.2–3.3、单层/30轮/BF16 | 短序列四层、长序列一层；长run轮数2/3/10，协议不同；模型时延代码与精度描述并非全部一致 | data/runs各config；eval.py/eval_paired.py |
| §3.2、层级同权重比较 | 同权重同小输入仅用于FP32等价检查；BF16计时分别初始化实例并独立生成随机输入，只控制结构参数及输入形状，不称逐步配对同权重计时 | LayerTest/bench_ctx.py的check_equiv、main、push与timed_decode |
| §4.1、MLA极差 | 主组为0.2476，0.1592属于另一历史变体；两seed极差不作显著性检验 | train_runs.csv，10.5606−10.3130 |
| §4.1、PPL差异成因 | 记录当前配置下MLA高14.19%；共享低秩约束、实现差异和训练配方为可能原因，未经消融，不能确定唯一成因或说成普遍质量损失；主组走恢复路径，差距不归因于吸收计时 | 技术稿§4.2、train_runs.csv、train_curve.csv、model_mla.py、train.py |
| §4.1、训练墙钟 | wall_hours累计训练步耗时，不含完整验证与保存开销；吞吐取两seed均值 | train.py与train_runs.csv |
| §4.2、缓存比 | 同BF16 fixed：12/6/3.375 MiB，MHA/MLA=3.556；raw为18/9/3.75，比例4.8 | cache_bench.json的raw/fixed |
| §4.2、2.1–2.4×时延 | 独立infer快照不与同进程30轮或5轮配对合并；技术稿选30轮会话并限定条件 | cache_bench.json、infer.csv、paired目录 |
| §4.3、扫描分组 | 提案选同24步、3轮配置，各run取摘要中位数之比；d_c=256/16K为1.310，旧池化值1.340不冒称新测量 | evidence/layer_scan_selected.csv与layer_runs.csv |
| §4.3、prefill | 32K锚点记录的是从上一锚点继续推进的区间，不是从零到32K总耗时 | LayerTest/bench_ctx.py的push调用与pos推进 |
| §4.3、179.2门槛 | 所列式代入8/64/32得119.47，公式适用性还待核对，先撤下“吻合”论证 | experiment_plan.md所列公式 |
| §5.1、QK-Norm因果 | 本地主组有路径限制，但不同历史权重/测法不构成独立因果消融 | 主组、归档组和测量记录差异 |

表中路径相对仓库根目录；[证据表](evidence/证据表.md)和[manifest](evidence/manifest.json)给出输入文件、哈希和派生统计。原始记录没有更改；异常慢配置仍保留逐记录表，不按结果好坏删除。

## 论文与官方来源

- [DeepSeek-V2固定v5](https://arxiv.org/html/2405.04434v5)：§2.1、式(9)–(19)、Table 1、Appendix C/D.2。
- [官方参考实现](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py)：低维查询与KV潜变量先RMSNorm，再进入q_b_proj/kv_b_proj。2026-11-20读取，main可变；这份参考代码不等于部署优化内核。
- 令c'=RMSNorm(c)，之后k_i=W_i c'，仍可把W_i移到查询侧；本地RMSNorm(W_i c)的分母则依赖展开后的头和历史内容，不能直接用同一固定线性变换吸收。
- 论文§3.2.3的5.76×为整体部署比较，含FP8权重、KV平均6bit量化、更大batch及8张H800，不是本项目单请求消融的预期倍数。

## 需要与不需要补跑的边界

当前提案只增加讲解与离线派生统计，不需要训练或重跑旧GPU实验。0.965仅称三轮探索性观察，不声称稳定阈值；不报告4060新速度；不声称QK-Norm是唯一慢因。

如果以后提出稳定交叉点、4060速度、修改dtype/计时后的新性能，则分别补对应测量；修改注意力/掩码还需小张量等价与因果性自检。主模型重评需先找回权重及token资产，不以重训充当文档PR前置条件。
