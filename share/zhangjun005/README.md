# MLA课程整理补充材料

贡献者：**zhangjun005**。原项目、公共报告与历史实验保留原作者归属。本目录作为新增补充提案，不替换原作者文件；基于e3c695d整理，未检查此后的上游更新。

## 审阅入口

1. [核对说明](review_notes.md)：相对原公共报告的逐项建议、证据和结论范围。
2. [技术介绍稿](mla_technology_report.md)：背景、原理、实验设计和结果；§4.2补充“为什么我们的MLA验证PPL高于MHA”，区分观察结果和未经消融的可能原因。
3. [课程课件](presentation/README.md)：仅保留29页第四版PPTX、PDF预览、备注与来源表，PPT内无个人署名。
4. [课程讲稿入口与答辩提示](course_talk.md)：对应当前29页课件。
5. [离线证据表](evidence/证据表.md)：已有结果的复算、四张PNG/SVG图与输入映射。

## 脚本与复算

[build_evidence.py](build_evidence.py)重算数据和四张实验图。在仓库根目录执行：

~~~powershell
python -m pip install -r share/zhangjun005/requirements-analysis.txt
python share/zhangjun005/build_evidence.py
~~~

脚本通过自身位置定位仓库，只读取CSV、JSON和JSONL，不导入torch，不依赖CUDA、权重或语料；输出限于evidence/。共整理11个层级run的222条测量，保留run、测量协议及输入SHA256，未把文档修订称为重新测量。复算会更新manifest生成时间。

课件绘图脚本、SVG和备注在[presentation/source/](presentation/source/README.md)。[author_slides.py](presentation/source/author_slides.py)生成计算图；[build.py](presentation/source/build.py)调用另行安装的ppt-master检查和导出；[render_preview.py](presentation/source/render_preview.py)生成预览。源码已展开，不依赖本机ML-MLA-notes或旧版构建目录；不提交重复源码ZIP或工具临时输出。

本机使用Conda ml-mla完成离线复算和课件构建。性能数字仍属于历史RTX4050 Laptop实验，本机RTX4060 8GB未重训、重评或重跑性能基准。课程讲解及本次文档PR无需重任务。若以后报告修复后的新性能，应另做相应验证并标明环境。

本目录不包含权重、语料、环境锁文件或第三方源码快照。论文和参考代码通过来源链接查阅。署名用于PR与补充文档，PPT页面、备注及作者元数据不含个人署名。
