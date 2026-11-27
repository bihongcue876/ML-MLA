"""Editable MLA course diagrams. Contributor: zhangjun005."""
from pathlib import Path
from html import escape
import json,sys
import drawing_utils as b
from drawing_utils import tx,rect,ln,group,box,header,takeaway
from drawing_utils import BG,NAVY,TEAL,BLUE,TEXT,MUTED,LINE,PALE,GREEN,FONT
R=Path(__file__).resolve().parent; OUT=R/'svg_output'; TEMPLATES=R/'templates'
b.OUT=OUT
roster=json.loads((R/'roster.json').read_text(encoding='utf-8'))
def para(x,y,lines,size=24,color=TEXT,leading=37,weight='normal'):
    return f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}">'+''.join(f'<tspan x="{x}" dy="{0 if i==0 else leading}">{escape(s)}</tspan>' for i,s in enumerate(lines))+'</text>'
def vec(x,y,values,w=44,h=42,c=TEAL,fill=GREEN,size=26):
    return ''.join(rect(x+i*(w+3),y,w,h,fill)+tx(x+i*(w+3)+w/2,y+h*.69,v,size,c,anchor='middle') for i,v in enumerate(values))
def arrow(x1,y1,x2,y2,c=TEAL):return ln(x1,y1,x2,y2,c,2,True)
def save(n,name,body,source='MLA 原理图解；数值为教学示意，位置分支稍后加入'):
    b.save(n,name,body,source)
    p=OUT/f'{n:02d}_{name}.svg';p.write_text(p.read_text(encoding='utf-8').replace(f'{n:02d} / 16',f'{n:02d} / 29'),encoding='utf-8')
def reuse(n,filename):
    """Copy an approved editable SVG template from this source tree."""
    source=TEMPLATES/filename
    (OUT/filename).write_bytes(source.read_bytes())
    print(n,filename)


def p08():
    a=header(8,'一个潜向量怎样生成较宽的键和值','原理图解','先用三个历史 token 做一个小例子：每个潜向量只有两个数。')
    s=tx(64,204,'历史潜向量 c',26,TEAL,'bold')+tx(420,204,'固定的线性上投影',26,NAVY,'bold')+tx(903,204,'展开后的 K 或 V',26,BLUE,'bold')
    for j,(c,k) in enumerate([([1,2],[1,2,3]),([2,1],[2,1,3]),([1,1],[1,1,2])]):
        y=249+j*95
        s+=tx(64,y+29,f'token {j+1}',22,MUTED)+vec(190,y,c,55,44)
        s+=arrow(318,y+22,414,y+22,BLUE)+box(424,y-5,362,54,'保留 x、y，补出 x+y',PALE,BLUE,24)
        s+=arrow(796,y+22,880,y+22,BLUE)+vec(894,y,k,65,44,BLUE,PALE)
    a+=group('expand-example',64,177,1152,329,s)
    a+=takeaway('linear-example','三个历史 token 都需要做一次上投影，才能得到三组较宽的 K/V。','教学示意令键、值使用同一规则；真实模型的键、值投影分别学习，潜宽也可大于单头宽。')
    save(8,'expand_example',a)

def p09():
    a=header(9,'键侧：改算查询，就能得到相同分数','原理图解','沿用上一页的向量与上投影；当前查询 q = (1, 1, 1)。')
    s=tx(64,200,'A. 先展开历史键',26,BLUE,'bold')+tx(64,278,'当前查询',22,MUTED)+vec(64,297,[1,1,1],43,42,BLUE,PALE)
    s+=tx(284,200,'缓存 c',22,TEAL)+tx(614,200,'展开的 k',22,BLUE)+tx(1030,200,'点积分数',22,NAVY)
    for j,(c,k,score) in enumerate([([1,2],[1,2,3],6),([2,1],[2,1,3],6),([1,1],[1,1,2],4)]):
        y=222+j*49;s+=vec(284,y,c,44,36)+arrow(385,y+18,595,y+18,BLUE)+vec(614,y,k,44,36,BLUE,PALE)+arrow(758,y+18,1004,y+18,BLUE)+tx(1076,y+28,str(score),30,BLUE,'bold',anchor='middle')
    s+=ln(64,382,1216,382)
    s+=tx(64,421,'B. 只变换当前查询',26,TEAL,'bold')
    s+=para(64,474,['x + y + (x+y) = 2x + 2y','所以查询 (1,1,1)','可以先变成 (2,2)'],22,leading=35)
    s+=vec(390,467,[2,2],58,48)+arrow(518,491,652,491)
    for j,(c,score) in enumerate([([1,2],6),([2,1],6),([1,1],4)]):
        y=428+j*49;s+=vec(670,y,c,47,36)+arrow(776,y+18,1004,y+18)+tx(1076,y+28,str(score),30,TEAL,'bold',anchor='middle')
    s+=tx(670,411,'直接读同一份 c',20,MUTED)
    a+=group('key-comparison',64,176,1152,403,s)
    a+=group('key-result',64,594,1152,66,tx(64,619,'两条路径都得到 6、6、4；下面只读 c，没有生成右上方的历史 k。',26,TEAL,'bold')+tx(64,650,'内容分数相同，沿用相同缩放与 softmax 后，注意力权重也相同。',20,MUTED))
    save(9,'key_without_expansion',a)

def p10():
    a=header(10,'值侧：先加权潜向量，再展开一次','原理图解','仍用同一上投影；示意权重为 0.2、0.3、0.5，单独指定以方便手算。')
    s=tx(64,204,'A. 三个历史值先展开，再加权',26,BLUE,'bold')
    for j,(v,w) in enumerate([([1,2,3],'0.2'),([2,1,3],'0.3'),([1,1,2],'0.5')]):
        y=237+j*53;s+=tx(64,y+28,w+' ×',24,BLUE)+vec(163,y,v,44,39,BLUE,PALE)
    s+=arrow(317,309,493,309,BLUE)+box(503,265,242,88,'加权相加',PALE,BLUE,26)+arrow(755,309,882,309,BLUE)+vec(892,286,['1.3','1.2','2.5'],80,47,BLUE,PALE)
    s+=ln(64,408,1216,408)+tx(64,449,'B. 三个潜向量先加权，只展开聚合结果',26,TEAL,'bold')
    s+=para(64,497,['0.2 × (1,2)','0.3 × (2,1)','0.5 × (1,1)'],24,leading=34)
    s+=arrow(258,530,363,530)+vec(373,506,['1.3','1.2'],82,47)+arrow(548,530,639,530)+box(649,491,211,78,'上投影一次',GREEN,TEAL,26)+arrow(870,530,882,530)+vec(892,506,['1.3','1.2','2.5'],80,47)
    a+=group('value-comparison',64,177,1152,397,s)
    a+=group('value-result',64,596,1152,66,tx(64,622,'相同输出 (1.3, 1.2, 2.5)，下面没有逐个生成历史 v₁、v₂、v₃。',26,TEAL,'bold')+tx(64,651,'示意权重不来自上一页分数；这里只隔离并展示线性加权的计算顺序。',20,MUTED))
    save(10,'value_without_expansion',a)

def p11():
    a=header(11,'多个头怎样共用历史潜向量','原理图解','推广到 T 个历史 token：共享的是缓存来源，各头的查询、权重与输出仍然不同。')
    s=tx(64,207,'一份历史缓存',26,TEAL,'bold')
    for j in range(4):s+=box(64,241+j*56,218,42,'c'+str(j+1),GREEN,TEAL,26)
    s+=tx(164,508,'… 共 T 行',22,MUTED,anchor='middle')
    for yy,head in [(234,'头1'),(400,'头2')]:
        s+=tx(419,yy+5,head+'：当前查询先做各自的变换',26,NAVY,'bold')
        s+=arrow(296,yy+62,409,yy+62)+box(419,yy+23,797,76,'变换后的查询与历史 c 打分，再对 c 加权',GREEN,TEAL,26)
        s+=tx(419,yy+130,'只对该头的聚合结果做值/输出投影',20,MUTED)
    a+=group('multihead-direct',64,180,1152,370,s)
    a+=takeaway('avoided-tensor','避开的是“头数 × 历史长度 × 头宽”的整段历史 K/V 中间张量。','仍会读取全部历史 c，也会产生每头的当前查询与聚合结果；数学重排不保证实际时延下降。')
    save(11,'multihead_without_history_kv',a,'原理：DeepSeek-V2 §2.1 / Appendix C；图中仅画两个头，省略位置分支')

def p15():
    a=header(15,'我们的实验怎样组织','实验实现','同一份 token 数据进入三个小语言模型，另设独立注意力层的计算对照。')
    s=box(64,235,226,82,'同一 token 数据',PALE,NAVY,26)
    for j,label in enumerate(['训练 MHA','训练 GQA-4','训练 MLA']):
        y=192+j*107;s+=arrow(302,276,411,y+37,BLUE)+box(421,y,256,74,label,PALE,BLUE,26)+arrow(687,y+37,800,289,BLUE)
    s+=box(810,244,406,91,'训练后测 PPL 与缓存',GREEN,TEAL,26)
    a+=group('model-experiment-flow',64,175,1152,306,s)
    s=ln(64,497,1216,497)+tx(64,540,'另一个对照：',26,NAVY,'bold')+tx(290,540,'固定层结构与输入形状',26,TEAL)+tx(638,540,'恢复 K/V 与吸收路径分别计时',26,TEXT)
    a+=group('layer-experiment-role',64,490,1152,62,s)
    a+=takeaway('experiment-separation','模型对照分别训练参数；层级对照固定结构，比较两种运算路径。','下面先看完整模型怎么搭建、训练和推理，再看如何测量这两个对照。')
    save(15,'our_experiment_flow',a,'我们的实验设计；训练与计时采用留存记录，本次仅作离线整理')

def p16():
    a=header(16,'我们的语言模型怎样搭起来','实验实现','把注意力模块放回完整模型：输入 token 编号，输出各位置的下一 token 预测。')
    s=box(64,216,193,76,'token 编号',PALE,NAVY,26)+arrow(267,254,305,254,BLUE)+box(315,216,232,76,'词嵌入向量',PALE,BLUE,26)+arrow(557,254,595,254,BLUE)
    s+=box(605,216,300,76,'8 个 Transformer 块',GREEN,TEAL,24)+arrow(915,254,953,254)+box(963,216,253,76,'词表预测分数',PALE,NAVY,26)
    s+=ln(755,302,755,337,TEAL,2)+ln(206,337,1099,337,TEAL,2)
    s+=tx(64,379,'展开其中一个块',26,NAVY,'bold')
    for j,label in enumerate(['归一化','注意力','加回输入','归一化 + 前馈','再加回输入']):
        x=64+j*234;w=216;s+=box(x,408,w,70,label,GREEN if j==1 else PALE,TEAL if j==1 else NAVY,24 if j!=3 else 22)
        if j<4:s+=arrow(x+w+3,443,x+232,443,BLUE)
    s+=tx(300,522,'替换这里：MHA / GQA-4 / MLA',26,TEAL,'bold')
    a+=group('model-assembly',64,179,1152,367,s)
    a+=takeaway('module-boundary','词嵌入、前馈网络和训练任务相同，比较集中在注意力模块。','MLA 保留联合压缩与解耦位置分支；我们的查询直接投影，省略论文的查询低秩分支。')
    save(16,'model_assembly',a,'实现依据：共同Transformer骨架与三种注意力模块；层内残差连接在备注中展开')

def p17():
    a=header(17,'一次训练步骤做了什么','实验实现','从 token 流取一段文本，让每个位置预测它后面的 token。')
    s=tx(64,207,'输入 x',26,NAVY,'bold')+tx(64,304,'目标 y',26,TEAL,'bold')
    for j,word in enumerate(['我','喜欢','机器']):s+=box(254+j*290,175,248,67,word,PALE,BLUE,28)
    for j,word in enumerate(['喜欢','机器','学习']):s+=box(254+j*290,273,248,67,word,GREEN,TEAL,28)+arrow(378+j*290,251,378+j*290,264)
    s+=tx(254,381,'目标序列比输入向后错开一位；因果掩码只允许看当前及之前位置。',24,MUTED)
    for j,label in enumerate(['模型前向预测','交叉熵比较目标','反向传播梯度','AdamW 更新参数']):
        x=64+j*294;s+=box(x,425,270,77,label,GREEN if j in [1,3] else PALE,TEAL if j in [1,3] else NAVY,24)
        if j<3:s+=arrow(x+275,463,x+289,463)
    a+=group('training-step',64,169,1152,362,s)
    a+=takeaway('learned-projection','MLA 的下投影和各头上投影，都在这个预测任务里通过梯度学习。','实际每次取513个token，前512作输入、后512作目标；图中文字仅是示意token。')
    save(17,'one_training_step',a,'我们的训练实现：错位取样、因果注意力、交叉熵与AdamW；预算见附录')

def p18():
    a=header(18,'推理时，缓存怎样建立和更新','实验实现','先处理整段输入建立历史，再逐步输入新 token，并把新状态接到缓存末尾。')
    s=tx(64,208,'prefill：输入 a、b、c',26,NAVY,'bold')+tx(683,208,'decode：输入新 token d',26,NAVY,'bold')
    for j,label in enumerate(['c₁ + r₁','c₂ + r₂','c₃ + r₃']):
        s+=box(64+j*177,252,162,76,label,GREEN,TEAL,26)
    for j,label in enumerate(['c₁+r₁','c₂+r₂','c₃+r₃','c₄+r₄']):
        s+=box(683+j*135,252,125,76,label,GREEN if j==3 else PALE,TEAL if j==3 else BLUE,22)
    s+=arrow(605,290,664,290)+tx(683,380,'追加新状态，然后读取第1到第4项参与注意力。',22,TEXT)
    s+=tx(64,380,'每层返回一份 past，供下一步复用。',24,TEXT)
    s+=para(64,439,['c：从 3 行潜向量变成 4 行，每行384维','r：共享位置键从 3 行变成 4 行，每行48维'],24,leading=45)
    s+=box(807,431,409,89,'主模型：临时恢复 K/V',PALE,BLUE,26)
    a+=group('cache-lifecycle',64,179,1152,359,s)
    a+=takeaway('cache-object','持久保存的是 c 与位置键 r；是否临时展开 K/V，取决于计算路径。','我们的主模型采用恢复路径；独立层实验再比较吸收路径，下一页说明原因。')
    save(18,'cache_lifecycle',a,'我们的MLA推理实现；图中a/b/c/d代表token编号，r即前文的位置键kᴿ')

def p20():
    a=header(20,'两种注意力计算怎样比较','实验实现','输入直接使用隐向量，不经过语言模型训练；正确性检查与计时分别执行。')
    s=tx(64,210,'1. 小输入检查计算对应',26,NAVY,'bold')+box(64,238,493,68,'复制同一权重，同输入比较 FP32 输出',PALE,BLUE,22)
    s+=tx(683,210,'2. 另建 BF16 计时实例',26,NAVY,'bold')+box(683,238,533,68,'分别初始化，使用同形状的随机输入',PALE,BLUE,24)
    s+=arrow(570,272,667,272,BLUE)+arrow(949,317,949,363)
    s+=tx(683,400,'3. 建立相同长度的历史',26,NAVY,'bold')+box(683,429,533,68,'分块输入隐向量，填好各自的缓存',GREEN,TEAL,24)
    s+=arrow(667,463,570,463)+tx(64,400,'4. 新输入到输出，测一次前向',26,NAVY,'bold')+box(64,429,493,68,'CUDA 计时 → 同步 → 记录毫秒',GREEN,TEAL,24)
    s+=tx(64,540,'每轮24个decode步骤，每种实现3轮；分别取轮摘要中位数后相除。',24,MUTED)
    a+=group('benchmark-procedure',64,178,1152,372,s)
    a+=takeaway('benchmark-controls','计时控制结构参数与输入形状；没有逐步配对同权重、同数值输入。','每步包含缓存更新；小输入精度检查不等于长序列验证，随机输入不衡量语言质量。')
    save(20,'benchmark_procedure',a,'我们的独立层测量实现；BF16，16K/32K，历史RTX4050 Laptop记录，完整配置见附录')

def p21():
    a=header(21,'图里的三个指标怎样得到','实验实现','每个图表都对应程序里的一个输出对象和一种统计方法。')
    s=''
    rows=[(222,'模型质量','验证文本 → 模型预测 → 平均交叉熵 → 取指数','得到 PPL，越低越好。'),(345,'缓存容量','prefill 返回 past → 逐层逐张量累加字节','统一 BF16，只统计缓存张量。'),(468,'计算时间','CUDA 事件测 decode → 汇总每轮 → 比较两路径','吸收耗时 / 恢复耗时，大于1更慢。')]
    for y,label,path,note in rows:
        s+=tx(64,y,label,30,TEAL,'bold')+tx(308,y,path,26,NAVY)+tx(308,y+43,note,22,MUTED)
    a+=group('metric-provenance',64,183,1152,340,s)
    a+=takeaway('measurement-source','先明确测量对象，再比较数值：PPL、缓存字节与单层时延各有含义。','本次使用留存结果复算与绘图，没有重跑训练或GPU性能测量。')
    save(21,'metrics_from_program',a,'我们的实验统计口径；PPL来自两种子记录，缓存字节按元素数×每元素字节数累加')

def p29():
    a=header(29,'附录：实验配置与数据范围','附录','参数供复核；主讲部分按构建、训练、缓存与测量流程理解实现。')
    s=tx(64,214,'语言模型对照',30,NAVY,'bold')+para(64,267,['8层，宽768，8查询头，每头96维','词表6400，SwiGLU中间维2432','约150M token，18,310步，序列512','种子42/43，MLA潜宽384、位置宽48','MHA / GQA-4 / MLA 参数量','68.63M / 63.91M / 68.93M'],24,leading=45)
    a+=group('model-settings',64,180,548,350,s)
    s=tx(683,214,'独立层对照',30,NAVY,'bold')+para(683,267,['1层，8头，每头64维，位置宽32','潜宽128 / 192 / 256 / 384','BF16，batch=1，16K / 32K历史','每块1024，每轮24步，各3轮','RTX4050 Laptop，约6GB，未锁时钟','本机4060 8GB未重测性能'],24,leading=45)
    a+=group('layer-settings',683,180,533,350,s)
    a+=takeaway('archive-limit','质量对照非严格等参数；两种子和三轮时延仅支持当前范围的观察。','主组权重和原始训练日志未留存，使用已保存汇总；统计修订不等于重新评测。')
    save(29,'appendix_settings',a,'数据与代码来源及原作者归属另见配套资料来源表')

if __name__=='__main__':
    start,end=map(int,sys.argv[1:3]) if len(sys.argv)>1 else (1,len(roster))
    OUT.mkdir(exist_ok=True)
    for n in range(start,end+1):
        kind,key=roster[n-1]
        if kind=='template':reuse(n,key)
        else:globals()[f'p{n:02d}']()
