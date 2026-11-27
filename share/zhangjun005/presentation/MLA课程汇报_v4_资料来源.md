# 第四版来源与比较口径

固定论文版本：[DeepSeek-V2 v5，2024-06-19](https://arxiv.org/html/2405.04434v5)。核对时使用该固定版本；PR提供原始来源链接，不依赖本机快照目录。

| 页码 | 依据 | 口径 |
| --- | --- | --- |
| 1–7 | 论文§2.1，式9–19，Appendix C | 推理背景、注意力、缓存与联合低秩表示；4GiB为理论估算 |
| 8–11 | 论文§2.1线性重排；自行设计的教学数值例子 | c=(1,2)/(2,1)/(1,1)，投影(x,y)到(x,y,x+y)；键、值同规则仅为教学。q=(1,1,1)重排为(2,2)，两路径内容分数均6/6/4。值权重另定0.2/0.3/0.5，不是这些分数的softmax；两路径输出均(1.3,1.2,2.5) |
| 12–14 | 论文§2.1、Appendix C | RoPE障碍、位置解耦与吸收推理流程；实际实现不一定全部采用此路径 |
| 15–17 | MLATrain/train.py、model_common.py、model_mla.py、model_gqa.py | 三种注意力替换共同Transformer骨架；错位token样本、因果掩码、交叉熵及AdamW；同预算但非严格等参数 |
| 18–19 | MLATrain/model_mla.py；官方modeling_deepseek.py | past逐层保存c与k_rope；主模型恢复历史K/V；直接查询投影、逐头QK-Norm与官方潜变量RMSNorm不同 |
| 20–21 | LayerTest/bench_ctx.py；MLATrain/eval.py；scripts/model/bench_cache.py | 小FP32等价检查同权重同输入；BF16计时分别初始化、独立随机输入；缓存字节为numel乘element_size；验证CE取指数为PPL |
| 22 | cache_bench.json fixed；train_runs.csv | BF16持久缓存字节，非整模型峰值显存；两seed PPL均值 |
| 23–24 | LayerTest/bench_ctx.py、naive_mla.py、absorbed_mla.py；四个layerctx配置及离线统计 | 历史4050单层batch1，固定结构参数和输入形状；计时没有逐步配对同权重同数值输入，结果为实现性能观察 |
| 25 | 论文Appendix D.2 Table 9 | 两规模MoE组质量对照，不同于简化模型PPL |
| 26 | 摘要、§3.2.3 | 93.3%及5.76倍对比DeepSeek67B；完整模型及服务系统比较 |
| 27 | §3.2.3；Table 9 | 服务吞吐机制；Table9元素数量比较另有模型规模，不能混为93.3% |
| 28–29 | 以上及留存训练、层测配置 | 技术总结；配置移附录，避免用参数清单代替实现过程 |

## 本版计时口径纠正

此前课件将层级性能比较描述为“同权重”，此说法过强，以本版为准。bench_ctx.py的check_equiv()确实使用load_state_dict复制权重，给两个FP32实例相同的小输入；但main()计时循环重新调用build()分别创建实例，push()和timed_decode()分别生成随机输入，没有复制该权重或配对输入数值。代码支持“同结构参数、同输入形状与分布”的计时比较，不支持“同权重同数值输入的逐步配对受控测量”。已保存的时间数据保留，降低其结论强度，不将修订说成重新验证。

timed_decode()对24步采用排序后索引len//2，即第13项作为轮摘要；课件继续沿用已有decode_med_ms，再对各实现的三轮摘要取中位数后相除。没有合并全部原始步骤后重新取中位数。数据源是2026-11-18的layerctx运行，末尾编号124423/124433/124443/124647对应潜宽384/256/192/128。GPU是RTX4050 Laptop约6GB；本机4060 8GB未重测。三轮且未锁时钟，不能据此宣布稳定加速阈值。

## Table 9 转录

| 指标 | 小MHA | 小MLA | 大MHA | 大MLA |
| --- | ---: | ---: | ---: | ---: |
| 激活参数B | 2.5 | 2.4 | 25.0 | 21.5 |
| 总参数B | 15.8 | 15.7 | 250.8 | 247.4 |
| KV元素/token（全层）K | 110.6 | 15.6 | 860.2 | 34.6 |
| BBH EM 3shot | 37.9 | 39.0 | 46.6 | 50.7 |
| MMLU Acc 5shot | 48.7 | 50.0 | 57.5 | 59.0 |
| C-Eval Acc 5shot | 51.6 | 50.9 | 57.9 | 59.2 |
| CMMLU Acc 5shot | 52.3 | 53.4 | 60.7 | 62.5 |

小组训练1.33T token，大组420B token。同规模组除注意力机制之外采用相同骨架，但总参数与激活参数不同。论文给出小组缓存约为MHA14%、大组约4%；这里是元素数，不涉及量化字节对比。

## 部署原文要点

§3.2.3：FP8参数，KV每元素平均6bits，利用更小缓存允许更大batch。按已部署DeepSeek67B服务的prompt及generation长度分布评测。单节点8H800，生成吞吐超过50K token/s，为DeepSeek67B最大生成吞吐5.76倍；输入吞吐超过100K token/s。该段未列固定输入输出长度或确切batch值，不补造。不是单请求延迟5.76倍改善，也不是纯MLA受控消融。

官方参考代码：[modeling_deepseek.py](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py)，main可变，本地快照读取日期2026-11-20。
原实验项目：[ML-MLA](https://github.com/bihongcue876/ML-MLA)。数据文件与四个run映射见 share/zhangjun005/evidence/manifest.json、layer_scan_selected.csv 和 mla_technology_report.md。
本轮贡献为结构核对、既有数据复算与课程讲解，未新训模型，也未新增GPU性能结果。

## PPL差异的解释

原因分析见[技术稿§4.2](../mla_technology_report.md#42-质量参照)：观测到的差距与未经消融的可能原因分别说明。曲线依据为data/results/model/tables/train_curve.csv留存汇总，不能替代缺失的原始日志。主模型质量差距不归因于独立层矩阵吸收计时。本次未重训或重评。
