"""Build the course deck with the separately installed ppt-master tools.

Contributor: zhangjun005. Runs document tools only; no model or GPU benchmark.
"""
from pathlib import Path
import argparse,os,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ppt-master',default=os.environ.get('PPT_MASTER_SKILL_DIR'),
                        help='ppt-master skill directory, or its repository root')
    parser.add_argument('--redraw',action='store_true',
                        help='Regenerate svg_output from Python diagrams and SVG templates; overwrites edits')
    parser.add_argument('--check-only',action='store_true',help='Stop after SVG quality check')
    parser.add_argument('--preview',action='store_true',help='Also generate PNG/PDF preview with Playwright')
    parser.add_argument('--output',type=Path,help='Copy exported PPTX here; default keeps it in source/exports')
    args=parser.parse_args()
    if not args.ppt_master:parser.error('Provide --ppt-master or PPT_MASTER_SKILL_DIR')
    skill=Path(args.ppt_master).expanduser().resolve()
    if not (skill/'scripts/svg_to_pptx.py').is_file():
        skill=skill/'skills/ppt-master'
    if not (skill/'scripts/svg_to_pptx.py').is_file():
        parser.error('The supplied directory does not contain the ppt-master skill')

    def run(script,*extra):
        subprocess.run([sys.executable,str(script),*map(str,extra)],check=True,cwd=ROOT)

    run(skill/'scripts/attribution_guard.py')
    if args.redraw:
        run(ROOT/'author_slides.py')
        run(skill/'scripts/compact_svg_styles.py',ROOT/'svg_output','--inplace')
    run(skill/'scripts/svg_quality_checker.py',ROOT,'--canonical-authoring','--stage','final','--json')
    if args.check_only:return
    run(skill/'scripts/total_md_split.py',ROOT)
    run(skill/'scripts/finalize_svg.py',ROOT)
    run(skill/'scripts/svg_to_pptx.py',ROOT,'--no-animations')
    exported=max((ROOT/'exports').glob('*.pptx'),key=lambda p:p.stat().st_mtime_ns)
    if args.output:
        target=args.output.expanduser().resolve()
        if target!=exported.resolve():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(exported,target)
        print('PPTX:',target)
    else:print('PPTX:',exported)
    if args.preview:run(ROOT/'render_preview.py','--final')

if __name__=='__main__':main()
