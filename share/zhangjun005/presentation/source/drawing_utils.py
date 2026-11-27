"""SVG drawing helpers for the MLA course deck. Contributor: zhangjun005."""
from pathlib import Path
from html import escape
import json,sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'svg_output'
BG='#F7F9FC';NAVY='#142840';TEAL='#008F86';BLUE='#57789E'
TEXT='#24384D';MUTED='#617286';LINE='#D5DFE8';PALE='#EAF0F6';GREEN='#E2F2EF'
FONT='Microsoft YaHei, Arial, sans-serif'

def tx(x,y,s,size=26,color=TEXT,weight='normal',anchor='start',family=FONT):
    return f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(str(s))}</text>'
def rect(x,y,w,h,fill=PALE,stroke='none',r=0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
def ln(x1,y1,x2,y2,c=LINE,sw=2,arrow=False,dash=False):
    a=f' marker-end="url(#arrow-{c[1:]})"' if arrow else ''
    d=' stroke-dasharray="6 5"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{sw}"{a}{d}/>'
def group(key,x,y,w,h,body):
    return f'<g id="{key}" data-pptx-bounds="{x} {y} {w} {h}">{body}</g>'
def box(x,y,w,h,label,fill=PALE,color=NAVY,size=26,stroke='none'):
    return rect(x,y,w,h,fill,stroke,8)+tx(x+w/2,y+h/2+size*.35,label,size,color,anchor='middle')
def formula(key,x,y,w,h,latex,preview,size=34):
    payload={'latex':latex.replace('@',chr(92)),'display':'block','font_size':size,'color':NAVY,'align':'center'}
    return f'<g id="{key}" data-pptx-replace-with="formula" data-pptx-x="{x}" data-pptx-y="{y}" data-pptx-width="{w}" data-pptx-height="{h}" data-pptx-bounds="{x} {y} {w} {h}"><metadata type="application/json"><![CDATA[{json.dumps(payload,ensure_ascii=False)}]]></metadata>'+tx(x+w/2,y+h/2+size*.33,preview,size,NAVY,anchor='middle',family='Cambria Math, serif')+'</g>'
def header(n,title,section,sub):
    return group('heading',64,28,1152,129,tx(64,48,'MLA  /  '+section,16,TEAL,'bold')+tx(64,106,title,40,NAVY,'bold')+tx(64,146,sub,20,MUTED))
def takeaway(key,text,note=None):
    s=ln(64,568,1216,568)+tx(64,608,text,26,TEAL,'bold')
    if note:s+=tx(64,641,note,20,MUTED)
    return group(key,64,565,1152,87,s)
def save(n,name,body,source='DeepSeek-V2（2024），§2.1',cover=False):
    defs='<defs>'+''.join(f'<marker id="arrow-{c[1:]}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><polygon points="0,0 8,4 0,8" fill="{c}"/></marker>' for c in [TEAL,BLUE,MUTED,NAVY])+'</defs>'
    footer=group('footer',64,664,1152,32,ln(64,665,1216,665)+tx(64,690,source,16,MUTED)+tx(1216,690,f'{n:02d} / 29',16,MUTED,anchor='end'))
    svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720" data-pptx-page-role="'+('cover' if cover else 'content')+'">'+defs+rect(0,0,1280,720,BG)+body+footer+'</svg>'
    (OUT/f'{n:02d}_{name}.svg').write_text(svg,encoding='utf-8');print(f'{n:02d}_{name}.svg')
