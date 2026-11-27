<!-- ppt-master-schema: design-spec/v1 -->
# MLA课程汇报第四版 - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | MLA课程汇报第四版 |
| Canvas Format | PPT 16:9, 1280 × 720 |
| Page Count | 29 |
| Primary Language | zh-CN |
| Target Audience | 首次接触MLA、未了解实验代码的课程师生 |
| Communication Intent | 介绍MLA技术，先解释机制，再用我们的实验作例证 |
| Desired Audience Outcome | 能说明联合低秩KV、解耦RoPE、矩阵吸收各自解决的问题 |
| Core Message / Ask / Action | MLA用共享潜向量表示历史KV，再通过线性计算重排使用缓存 |
| Delivery Context | 约30–35分钟课堂汇报，原理后展开实现、实验设计与论文部署证据 |
| Artifact Afterlife | 无个人署名的可编辑PPTX与讲者备注；文献与数据保留引用 |
| Reading Mode | presentation |
| Content Strategy | 补齐推理背景与注意力基础；先动机后公式；实验自包含，逐步图解计算与实现 |
| Design Style | 浅底、深蓝与青绿，开放留白，线性计算图与少量数据图 |
| AI Image Acquisition Path | not applicable |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — workflow default |
| Custom Animations | disabled — agent draft decision |
| Narration Audio | disabled — workflow default |
| Created Date | 2026-11-20 |

修订依据：用户要求先总结问题、去除署名、补背景与过渡，改为面向首次接触MLA的课程听众。配色沿用初版。

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 左右64，上下40 |
| Content Area | 64, 40, 1152, 630 |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode Behavior**: 从推理瓶颈到三个机制，再讲实现与实验，最后联系论文部署效果；一页推进一个理解动作。
- **Visual style**: custom
- **Visual Style Behavior**: 浅底开放版面，深蓝标题，青绿强调共享潜空间；细线连真实计算依赖，灰蓝区分历史展开数据。
- **Theme**: 学术技术讲解
- **Tone**: 准确、平实、便于口述

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #F7F9FC | 页面背景 |
| Secondary background | #EAF0F6 | 缓存与计算区域 |
| Primary | #142840 | 标题与核心文字 |
| Accent | #008F86 | MLA、潜向量与核心收益 |
| Secondary accent | #57789E | 对照与内容键分支 |
| Body text | #24384D | 正文 |
| Secondary text | #617286 | 图注与来源 |
| Divider | #D5DFE8 | 分隔线 |

## IV. Typography System

### Font Plan

| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | 清晰黑体 | Microsoft YaHei | Arial | sans-serif |
| Body | 清晰黑体 | Microsoft YaHei | Arial | sans-serif |
| Formula | 数学 | Cambria Math | Cambria Math | serif |

- **Title stack**: Microsoft YaHei, Arial, sans-serif
- **Body stack**: Microsoft YaHei, Arial, sans-serif
- **Formula stack**: Cambria Math, serif

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 26 |
| Title | 40 |
| Subtitle | 30 |
| Annotation | 20 |
| Footer | 16 |
| Formula | 34 |
| Display | 60 |
| Label | 24 |

## V. Layout Principles

### Deck-wide Direction

- **Hierarchy direction**: 先结论标题，再图解，再一行限定条件。
- **Composition tendency**: 每页只有一个主图；原理页用计算依赖，结果页用统一坐标与直接标注。
- **Cross-page continuity**: 潜空间始终用青绿，历史展开KV用灰蓝；图形角色延续，构图随内容变化。
- **Spacing posture**: 机制页适度密集，封面和收束页留白。
- **Spacing anchors**: page margin 64px; block gap 28px; column gutter 36px; corner radius 8px; body leading 38px.

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## IX. Content Outline

#### Slide 01 - 多头潜注意力（MLA）

- **Audience move**: 从模型推理的问题进入MLA主题。
- **Relationships**: 长上下文推理与KV缓存压缩关联。
- **Title**: 多头潜注意力（MLA）
- **Core message**: DeepSeek-V2提出的KV缓存压缩与潜空间计算方法。
- **Content**: 研究背景；注意力与缓存；MLA原理；小规模实验。

#### Slide 02 - 大语言模型推理中的长上下文问题

- **Audience move**: 理解逐步生成与长上下文的推理负担。
- **Relationships**: 前缀、已生成文本和新token按时间顺序关联；多个请求各有上下文。
- **Title**: 大语言模型推理中的长上下文问题
- **Core message**: 生成过程需要反复使用已有上下文。
- **Content**: 自回归生成；token是文本单位；上下文增长；并发请求；KV缓存是其中的显存开销。

#### Slide 03 - 自注意力如何使用历史信息

- **Audience move**: 理解Q/K/V的职责和一次注意力计算。
- **Relationships**: 查询与历史键匹配，权重作用于对应值，再聚合输出。
- **Title**: 自注意力如何使用历史信息
- **Core message**: K决定匹配程度，V提供要聚合的信息。
- **Content**: q当前查询、k历史键、v历史值；softmax将打分变为权重；加权例子0.2v1+0.3v2+0.5v3；多头是多组投影并行。

#### Slide 04 - KV cache的作用与显存开销

- **Audience move**: 知道缓存保存什么以及增长量级。
- **Relationships**: 历史K/V跨生成步复用；请求数和序列长度影响存储。
- **Title**: KV cache的作用与显存开销
- **Core message**: 缓存避免重算历史投影，但随长度和并发增长。
- **Content**: prefill并行计算输入；decode追加当前KV；公式；32层32KV头、头宽128、BF16下8K需4GiB的理论示例，仅KV缓存。
- **Mathematical content**: M_{KV}=BTL(2n_{kv}d_h)s

#### Slide 05 - MHA、GQA与MQA

- **Audience move**: 区分共享KV头和共享潜表示。
- **Relationships**: 固定查询头下，完整KV头共享方式对比。
- **Title**: MHA、GQA与MQA
- **Core message**: 减少KV头数可减少缓存，MLA改变历史KV的表示。
- **Content**: MHA8组、GQA4组、MQA1组；固定8个查询头；各头宽度相同；不预设质量相同。

#### Slide 06 - MLA的联合低秩表示

- **Audience move**: 理解联合压缩、潜向量及各头投影。
- **Relationships**: 输入经下投影到共享c，再由各头不同上投影生成K和V。
- **Title**: MLA的联合低秩表示
- **Core message**: 一个较短的潜向量共同表示各头内容K和V的来源。
- **Content**: 下投影与上投影定义；共享c不乘头数；各头投影不同；训练结构约束而非任意MHA无损转换。
- **Mathematical content**: c_j=W^{DKV}h_j,\quad k^C_{j,i}=W^{UK}_ic_j,\quad v_{j,i}=W^{UV}_ic_j

#### Slide 07 - 从压缩缓存到潜空间计算

- **Audience move**: 理解为什么缓存压缩之后还需要计算重排。
- **Relationships**: 压缩缓存需要被读取；恢复历史KV和直接潜空间计算两种路径对比。
- **Title**: 从压缩缓存到潜空间计算
- **Core message**: 只存c节省存储，逐步恢复整段KV仍带来开销。
- **Content**: 每次decode访问多个历史c；直接恢复各头KV增加中间数据；矩阵吸收利用上投影线性改变顺序；先暂不考虑位置分支。

#### Slide 08 - 一个潜向量怎样生成较宽的键和值

- **Audience move**: 从具体计算理解一个潜向量怎样生成较宽的键和值。
- **Relationships**: order / link / contrast：同一输入沿两种计算路径，保持对应关系。
- **Title**: 一个潜向量怎样生成较宽的键和值
- **Core message**: 用2维潜向量和固定线性规则展示3维展开。
- **Content**: 示意c1=(1,2),c2=(2,1),c3=(1,1)，示意键/值上投影均为(x,y)映射到(x,y,x+y)。查询q=(1,1,1)。不是训练参数，未加入RoPE/缩放/softmax。

#### Slide 09 - 键侧：改算查询，就能得到相同分数

- **Audience move**: 从具体计算理解键侧：改算查询，就能得到相同分数。
- **Relationships**: order / link / contrast：同一输入沿两种计算路径，保持对应关系。
- **Title**: 键侧：改算查询，就能得到相同分数
- **Core message**: 同样的历史c，比较逐个展开键与只变换当前q。
- **Content**: 上路径c1/2/3展开k=(1,2,3)/(2,1,3)/(1,1,2)，与q点积得6/6/4。下路径先q变为(2,2)，直接与c点积也得6/6/4。无需T个历史3维键中间张量，仍读所有c；省去展开不是不用向量。

#### Slide 10 - 值侧：先加权潜向量，再展开一次

- **Audience move**: 从具体计算理解值侧：先加权潜向量，再展开一次。
- **Relationships**: order / link / contrast：同一输入沿两种计算路径，保持对应关系。
- **Title**: 值侧：先加权潜向量，再展开一次
- **Core message**: 用同一线性规则显示两条路径输出完全相同。
- **Content**: 示意权重0.2/0.3/0.5独立指定，不是上一页softmax结果。先展开v1/2/3再加权=(1.3,1.2,2.5)；先加权c=(1.3,1.2)，再展开一次也=(1.3,1.2,2.5)。省T次历史V展开，仍有每头聚合结果与输出。

#### Slide 11 - 多个头怎样共用历史潜向量

- **Audience move**: 从具体计算理解多个头怎样共用历史潜向量。
- **Relationships**: order / link / contrast：同一输入沿两种计算路径，保持对应关系。
- **Title**: 多个头怎样共用历史潜向量
- **Core message**: 把小例子推广到T个token与多个头，说明避开的张量。
- **Content**: 历史缓存T×dc共享，各头Wi不同；恢复路径历史K/V为H×T×dh，吸收路径当前H个查询变换和H个聚合结果投影。每头仍有自己的注意力权重与输出。不物化历史多头KV不等于所有张量不展开，不保证算力或时延同幅下降。

#### Slide 12 - 位置编码对矩阵吸收的影响

- **Audience move**: 理解位置必要性和普通RoPE与固定吸收的冲突。
- **Relationships**: 语序不同导致含义不同；随历史位置变化的旋转位于查询与键投影之间。
- **Title**: 位置编码对矩阵吸收的影响
- **Core message**: 位置信息必须保留，但不能直接破坏内容侧的固定线性重排。
- **Content**: 不同语序示例；RoPE是依位置旋转查询和键；中间旋转随j变化；问题是固定吸收，不是RoPE不能缓存。
- **Mathematical content**: (R_tq)^T(R_jW^K_ic_j)=q^TR_t^TR_jW^K_ic_j

#### Slide 13 - 解耦旋转位置编码

- **Audience move**: 理解内容和位置分支如何共同打分。
- **Relationships**: 内容与位置分支并行贡献分数；缓存共享c及共享位置键。
- **Title**: 解耦旋转位置编码
- **Core message**: 内容分支保持线性，位置交给小RoPE分支。
- **Content**: 内容点积加位置点积后softmax；位置查询逐头、位置键共享；cache=c和kR；dR位置维；缩放保持原定义。
- **Mathematical content**: s_j=\frac{(q^C)^Tk^C_j+(q^R)^Tk^R_j}{\sqrt{d_h+d_R}}

#### Slide 14 - MLA的推理流程

- **Audience move**: 把分散机制拼成一次decode流程。
- **Relationships**: 当前输入生成查询与缓存项；查询读取历史缓存形成权重；权重聚合c再投影输出。
- **Title**: MLA的推理流程
- **Core message**: 缓存、打分、聚合、输出构成完整计算过程。
- **Content**: 当前输入h；当前c/kR追加缓存；吸收内容查询与位置查询；softmax；潜空间加权；值与输出投影；历史无需整体展开。

#### Slide 15 - 我们的实验怎样组织

- **Audience move**: 从具体计算理解我们的实验怎样组织。
- **Relationships**: order / link / contrast：图中数据依次经过处理，区分训练、正确性检查与性能测量的对象。
- **Title**: 我们的实验怎样组织
- **Core message**: 用执行流程区分模型训练评估与独立层基准。
- **Content**: 同一token数据分别训练MHA/GQA4/MLA，再验证PPL和量缓存；另用固定结构参数及输入形状的独立层比较两路径计时。小FP32检查同权重同输入，实际BF16计时分别初始化并独立生成随机输入；前者训练语言模型，后者观察实现开销。

#### Slide 16 - 我们的语言模型怎样搭起来

- **Audience move**: 从具体计算理解我们的语言模型怎样搭起来。
- **Relationships**: order / link / contrast：图中数据依次经过处理，区分训练、正确性检查与性能测量的对象。
- **Title**: 我们的语言模型怎样搭起来
- **Core message**: 在完整输入输出链中定位可替换的注意力模块。
- **Content**: token编号到Embedding隐向量，每个残差块为Norm→注意力→残差再Norm→SwiGLU→残差，重复8层，末Norm和词表投影输出下一token概率；MHA/GQA/MLA只换注意力，骨架与训练方式相同。简化MLA查询直接投影。

#### Slide 17 - 一次训练步骤做了什么

- **Audience move**: 从具体计算理解一次训练步骤做了什么。
- **Relationships**: order / link / contrast：图中数据依次经过处理，区分训练、正确性检查与性能测量的对象。
- **Title**: 一次训练步骤做了什么
- **Core message**: 从错位token样本解释损失与梯度更新。
- **Content**: 从训练token流取513个编号，x前512，y后512；小例输入我/喜欢/机器，目标喜欢/机器/学习；因果掩码每位置只看自身前缀；前向输出词表logits，交叉熵目标下一token，反向传播后AdamW更新全部可训练投影；预算和参数移附录，学习到的低秩不是手工压缩MHA。

#### Slide 18 - 推理时，缓存怎样建立和更新

- **Audience move**: 从具体计算理解推理时，缓存怎样建立和更新。
- **Relationships**: order / link / contrast：图中数据依次经过处理，区分训练、正确性检查与性能测量的对象。
- **Title**: 推理时，缓存怎样建立和更新
- **Core message**: 用prefill三token到decode一token展示真实读写和形状。
- **Content**: prefill token a/b/c，经各层得到c1/c2/c3及r1/r2/r3写缓存；decode输入新token d得到c4/r4追加，读1..4全部cache完成注意力；B1 c由[1,3,384]到[1,4,384]，位置键[1,1,3,48]到[1,1,4,48]；每层各有一份。主模型naive临时恢复K/V，区别于理论吸收。

#### Slide 19 - 归一化位置与矩阵吸收

- **Audience move**: 理解归一化位置与矩阵吸收，并明确比较条件。
- **Relationships**: contrast / order / link，对照条件与计算步骤对应。
- **Title**: 归一化位置与矩阵吸收
- **Core message**: 归一化的位置决定上投影之后是否仍为线性映射。
- **Content**: 官方低维c先RMSNorm后上投影，主组每头展开K后归一化；后者分母依赖历史键，阻碍简单固定吸收；主模型质量与独立层吸收计时分开；并非RMSNorm与MLA固有冲突。

#### Slide 20 - 两种注意力计算怎样比较

- **Audience move**: 从具体计算理解两种注意力计算怎样比较。
- **Relationships**: order / link / contrast：图中数据依次经过处理，区分训练、正确性检查与性能测量的对象。
- **Title**: 两种注意力计算怎样比较
- **Core message**: 把copy权重、正确性检查、建历史、计时与记录串成操作顺序。
- **Content**: 小FP32检查复制同权重并使用同输入；实际计时另建分别初始化的BF16实例，使用独立随机输入，固定结构参数和输入形状，分块构建上下文，输入新随机隐向量更新cache，CUDA事件包围每步前向同步读取ms，每24步汇总每轮，3轮取摘要中位数再相除；同权重仅用于小输入等价检查，随机层不评语言。

#### Slide 21 - 图里的三个指标怎样得到

- **Audience move**: 从具体计算理解图里的三个指标怎样得到。
- **Relationships**: order / link / contrast：图中数据依次经过处理，区分训练、正确性检查与性能测量的对象。
- **Title**: 图里的三个指标怎样得到
- **Core message**: 从程序输出追踪到PPL、缓存容量与时延比。
- **Content**: 验证样本前向得到平均交叉熵exp为PPL，两seed值；prefill返回各层past，统一BF16累加numel×element_size为缓存字节，非峰值；单层CUDA事件测每个decode前向含追加，24步轮摘要再各3轮中位数比；训练/质量及性能来自留存结果，本轮仅离线整理。

#### Slide 22 - 缓存占用与模型质量

- **Audience move**: 理解缓存减少与当前质量结果及各自指标。
- **Relationships**: 同条件下MHA/GQA/MLA的缓存占用和PPL分别对比。
- **Title**: 缓存占用与模型质量
- **Core message**: 本配置MLA缓存明显减小，质量尚有差距。
- **Content**: batch1、512token、8层BF16下12/6/3.375MiB；PPL均值9.14015/9.28065/10.4368；两seed；不推出普遍质量结论。
- **Visualization**: cache-size：horizontal_bar_chart，12/6/3.375MiB，零基线0–14；quality-seeds：scatter_chart，两个seed分别9.1557/9.3272/10.5606和9.1246/9.2341/10.313，纵轴8.5–11，均值9.14015/9.28065/10.4368。来源model_cache.csv与train_runs.csv。
- **Native-ready**: cache-size=yes; quality-seeds=yes

#### Slide 23 - 矩阵吸收的运行时间

- **Audience move**: 读懂固定结构单层时延比及计时限制。
- **Relationships**: 潜宽变化与时延比关联；16K和32K对照。
- **Title**: 矩阵吸收的运行时间
- **Core message**: 压缩能否转为加速还依赖潜宽与内核。
- **Content**: 纵轴吸收除以恢复KV路径时间；大于1更慢；dc128/192/256/384；32K dc128为0.965；不是稳定阈值；随机输入无语言能力结论。
- **Visualization**: layer-ratio：line_chart，按dc128/192/256/384，16K=[1.1575406906,1.2348906192,1.3102790148,1.6018589926]，32K=[0.9646271255,1.2459356430,1.3510420271,1.6858682915]，纵轴0.8–1.8。来源layer_scan_selected.csv。
- **Native-ready**: layer-ratio=yes

#### Slide 24 - 矩阵吸收为什么没有稳定加速

- **Audience move**: 理解矩阵吸收为什么没有稳定加速及适用范围。
- **Relationships**: contrast / link，前后条件与结果对应。
- **Title**: 矩阵吸收为什么没有稳定加速
- **Core message**: 数学等价减少历史上投影，但会改变内维和中间张量。
- **Content**: 未吸收内容键宽64，加位置32共96；吸收后dc+32为160/224/288/416；LayerTest仍expand头维并cat物化K，V在潜聚合后上投影再输出，未融合；不同于主模型enable_gqa路径；缓存是持久状态，中间张量额外占用。

#### Slide 25 - 原论文中的质量对照实验

- **Audience move**: 理解原论文中的质量对照实验及适用范围。
- **Relationships**: contrast / link，前后条件与结果对应。
- **Title**: 原论文中的质量对照实验
- **Core message**: 论文在两个MoE规模比较，MLA整体更优但并非每项均上升。
- **Content**: AppendixD.2 Table9全部四项指标，小MoE~16B训练1.33T，大~250B训练420B；小MHA/MLA:BBH37.9/39.0,MMLU48.7/50.0,C-Eval51.6/50.9,CMMLU52.3/53.4；大46.6/50.7,57.5/59.0,57.9/59.2,60.7/62.5；同规模组骨架除注意力外相同，参数并非严格相等。
- **Native-ready**: paper-quality=yes

#### Slide 26 - DeepSeek-V2 的实际部署效果

- **Audience move**: 理解DeepSeek-V2 的实际部署效果及适用范围。
- **Relationships**: contrast / link，前后条件与结果对应。
- **Title**: DeepSeek-V2 的实际部署效果
- **Core message**: 论文的服务部署展示缓存节省与最大生成吞吐收益。
- **Content**: 相对DeepSeek67B，KV缓存减少93.3%，最大生成吞吐5.76倍；8H800单节点生成>50Ktoken/s，输入>100Ktoken/s；FP8权重，KV平均6bit，允许更大batch；按实际67B服务输入输出长度分布评估，无固定长度及具体batch数字；236B总21B激活，MoE与系统共同作用，不是MLA单因素消融。

#### Slide 27 - 缓存压缩如何改善服务吞吐

- **Audience move**: 理解缓存压缩如何改善服务吞吐及适用范围。
- **Relationships**: contrast / link，前后条件与结果对应。
- **Title**: 缓存压缩如何改善服务吞吐
- **Core message**: 缓存变小可提高可驻留请求数，服务吞吐与单请求时延不同。
- **Content**: 因果关系：每请求缓存变小，固定显存可容纳更多请求，允许更大batch，提高整体token吞吐；模型权重工作空间等也占显存，不按缓存比例线性外推；论文D.2 KV元素110.6K→15.6K约14%，860.2K→34.6K约4%，不是93.3%相同比较；笔记本B1不能复现H800服务吞吐。

#### Slide 28 - 总结

- **Audience move**: 复述MLA的完整技术关系及实验边界。
- **Relationships**: 低秩表示、解耦位置和矩阵吸收共同组成MLA；技术结论与实验外推区分。
- **Title**: 总结
- **Core message**: 联合压缩减少缓存，解耦位置保留线性结构，吸收改变计算顺序。
- **Content**: 三项机制的联系；我们的实验压缩3.556倍但时延不稳定；论文完整部署最大吞吐5.76倍须保留系统条件；目标是降低缓存与访存负担；质量速度依赖具体条件；论文和实验数据引用。

#### Slide 29 - 附录：实验配置与数据范围

- **Audience move**: 从具体计算理解附录：实验配置与数据范围。
- **Relationships**: order / link / contrast：图中数据依次经过处理，区分训练、正确性检查与性能测量的对象。
- **Title**: 附录：实验配置与数据范围
- **Core message**: 参数供复核，主讲依照流程理解实现。
- **Content**: 8层768宽8查询头head96/vocab6400/SwiGLU2432，149995520token/18310step/T512/seed42,43，MHA68.63M/GQA63.91M/MLA68.93M。层1H8dh64rope32dc128/192/256/384/BF16B1/16K32K/chunk1024/decode24/3轮/4050Laptop6GB。当前4060未重测，缺主组权重原始日志仅引用留存统计。

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Content**: 逐步指向计算图中的数据、操作、中间结果和输出，称我们的实验实现；资料出处单独保留，禁止仓库视角。
- **Total duration**: 约30–35分钟，28页主讲加1页附录，试讲调整
