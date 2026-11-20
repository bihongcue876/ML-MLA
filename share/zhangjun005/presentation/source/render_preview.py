from pathlib import Path
from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw, ImageFont
import sys,os

root=Path(__file__).resolve().parent
out=root/'preview';out.mkdir(exist_ok=True)
source=root/('svg_final' if '--final' in sys.argv else 'svg_output')
with sync_playwright() as p:
    configured=os.environ.get('PPT_PREVIEW_BROWSER')
    edge=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
    browser=p.chromium.launch(**({'executable_path':configured} if configured else {'executable_path':str(edge)} if edge.exists() else {}),headless=True)
    page=browser.new_page(viewport={'width':1280,'height':720},device_scale_factor=1.5)
    for svg in sorted(source.glob('*.svg')):
        page.goto(svg.as_uri());page.evaluate('document.fonts.ready');page.screenshot(path=str(out/(svg.stem+'.png')))
    browser.close()
imgs=[Image.open(p).convert('RGB') for p in sorted(out.glob('*.png')) if p.stem[:2].isdigit()]
sheet=Image.new('RGB',(2000,((len(imgs)+2)//3)*400+40),'#DFE6EE')
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
for i,im in enumerate(imgs):
    x=20+(i%3)*660;y=20+(i//3)*400
    thumb=im.copy();thumb.thumbnail((640,360));sheet.paste(thumb,(x,y));ImageDraw.Draw(sheet).text((x,y+363),f'{i+1:02d} / MLA',font=font,fill='#24384D')
sheet.save(out/'contact_sheet.png')
for start in range(0,len(imgs),6):
    panel=Image.new('RGB',(2560,3*760),'#DFE6EE')
    for j,im in enumerate(imgs[start:start+6]):
        thumb=im.copy();thumb.thumbnail((1260,709))
        x=10+(j%2)*1280;y=10+(j//2)*760
        panel.paste(thumb,(x,y));ImageDraw.Draw(panel).text((x,y+712),str(start+j+1),font=font,fill='#24384D')
    panel.save(out/f'panel_{start+1:02d}_{min(start+6,len(imgs)):02d}.png')
imgs[0].save(out/'MLA课程汇报_预览版.pdf',save_all=True,append_images=imgs[1:],resolution=144)
print(out/'contact_sheet.png')
