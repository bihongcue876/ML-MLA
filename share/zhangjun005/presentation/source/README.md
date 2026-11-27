# 课件绘图与导出源码

贡献者：**zhangjun005**。对应29页第四版课程汇报。PPT内不含个人署名，论文和历史实验归属见[资料来源](../MLA课程汇报_v4_资料来源.md)。所有必需的页面、模板和备注均在本目录，不依赖其他版本课件或本机ML-MLA-notes目录。

## 文件

| 文件或目录 | 用途 |
| --- | --- |
| author_slides.py | 生成数值图解、模型、训练、缓存与计时流程图；其余页面复制本目录的SVG模板 |
| drawing_utils.py | 可编辑文字、向量格、箭头、公式标记等绘图函数 |
| roster.json | 29页顺序及脚本生成/模板页的映射 |
| templates/ | 18页完整SVG模板，包含已核对的数据图和论文表格；可直接编辑 |
| svg_output/ | 29页最终设计源，导出时的页面依据 |
| notes/total.md | 29页讲者备注；同目录分文件由工具拆分生成 |
| design_spec.md / spec_lock.md | 页面规格、字体与配色约束 |
| build.py | 串行完成检查、备注拆分、SVG整理和PPTX导出 |
| render_preview.py | 从SVG生成逐页PNG、总览与PDF预览 |

所有向量和图表均可在SVG中修改。导出的图表/表格为可编辑形状，3组公式为Office Math；未使用Excel图表或PowerPoint表格对象。

## 修改和重建

使用Python 3.10及以上。author_slides.py及drawing_utils.py只用标准库，可直接运行；导出使用另行安装的[ppt-master](https://github.com/hugohe3/ppt-master)，依赖按该工具说明安装，本PR不复制第三方工具源码。

直接编辑svg_output/及notes/total.md可保留手工改动。如果修改Python绘图逻辑，运行以下命令会覆盖svg_output中的同名页；模板页应同时修改templates/，避免重画时被旧模板覆盖。

~~~powershell
python share/zhangjun005/presentation/source/author_slides.py
~~~

导出时传入自己安装的ppt-master根目录或skill目录，例如以下路径需替换为本机位置：

~~~powershell
python share/zhangjun005/presentation/source/build.py --ppt-master C:/tools/ppt-master
~~~

默认输出在source/exports/；追加--output指定目标PPTX。--redraw会先重画SVG并整理样式；--check-only只检查SVG；--preview在导出后生成PNG/PDF。build.py按工具要求依次运行，各步骤失败即停止，不包含训练、推理或性能基准入口。

只生成预览时，先安装requirements-preview.txt中的Playwright、Pillow。脚本优先使用PPT_PREVIEW_BROWSER指定的浏览器，其次使用Windows默认位置的Edge，否则使用Playwright安装的Chromium：

~~~powershell
python -m pip install -r share/zhangjun005/presentation/source/requirements-preview.txt
python -m playwright install chromium
python share/zhangjun005/presentation/source/render_preview.py
~~~

已安装Edge时可跳过Chromium下载。首次预览读svg_output/，导出整理后可追加--final读svg_final/。PDF/PNG是SVG浏览器渲染预览，未在PowerPoint/WPS实机渲染验证。

预览、导出、备份与工具验证日志已在本目录.gitignore中排除。PR保留最终课件和可读源码，不包含旧版课件或重复源码压缩包。质量结果仍属于历史4050实验，本机4060未重测；PPL原因分析见[技术稿§4.2](../../mla_technology_report.md#42-质量参照)。
