"""Offline reanalysis only. No training imports, network calls or GPU workloads."""
# Offline reanalysis author: zhangjun005. Historical experiments retain original attribution.
import csv
import hashlib
import json
import math
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
OUT = HERE / 'evidence'
FIGS = OUT / 'figures'
TABLES = REPO / 'data/results/model/tables'
INPUTS = set()
ORDER = ['MHA', 'GQA-4', 'MLA']
COLORS = ['#4169a1', '#3a9278', '#c77539']


def read_json(path):
    INPUTS.add(path)
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(name, rows):
    assert rows, name
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(name, fig):
    fig.savefig(FIGS / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(FIGS / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


def model_evidence():
    path = TABLES / 'train_runs.csv'
    INPUTS.add(path)
    with path.open(encoding='utf-8') as f:
        train = list(csv.DictReader(f))
    grouped = {a: [r for r in train if r['arch'] == a] for a in ORDER}
    means = {a: st.mean(float(r['final_val_ppl']) for r in rs) for a, rs in grouped.items()}
    quality = []
    for arch, rs in grouped.items():
        assert len(rs) == 2
        vals = [float(r['final_val_ppl']) for r in rs]
        quality.append(dict(arch=arch, seeds=';'.join(r['seed'] for r in rs),
            val_ppl_mean=means[arch], val_ppl_range=max(vals)-min(vals),
            val_pct_vs_mha=(means[arch]/means['MHA']-1)*100,
            train_tokens_per_s_mean=st.mean(float(r['avg_tokens_per_s']) for r in rs),
            accumulated_step_hours_mean=st.mean(float(r['wall_hours']) for r in rs),
            source=path.relative_to(REPO).as_posix()))
    write_csv('model_quality.csv', quality)
    bench = read_json(TABLES / 'cache_bench.json')
    cache = []
    for arch, elems in zip(ORDER, [1536, 768, 432]):
        for mode in ['raw', 'fixed']:
            r = next(r for r in bench['cache'][arch][mode] if r['seq_len'] == 512)
            assert r['cache_elements'] == elems * 512 * 8
            if mode == 'fixed': assert r['cache_bytes'] == elems * 512 * 8 * 2
            cache.append(dict(arch=arch, mode=mode, seq_len=512, layers=8, batch=1,
                elements_per_token_per_layer=elems, cache_mib=r['cache_mib'],
                bytes=r['cache_bytes'], source='data/results/model/tables/cache_bench.json'))
    write_csv('model_cache.csv', cache)
    latency = []
    for arch in ORDER[1:]:
        for phase in ['decode', 'prefill']:
            ds = [r[arch][phase+'_med']-r['MHA'][phase+'_med'] for r in bench['latency']['rounds']]
            mean = st.mean(ds)
            half = 1.96 * st.stdev(ds) / math.sqrt(len(ds))
            ref = bench['latency']['delta_vs_mha'][arch][phase+'_ms']
            assert len(ds) == ref['n'] == 30
            assert abs(mean-ref['mean']) < .00011
            assert max(abs(mean-half-ref['ci95'][0]),abs(mean+half-ref['ci95'][1])) < .00015
            base = st.mean(r['MHA'][phase+'_med'] for r in bench['latency']['rounds'])
            latency.append(dict(arch=arch, phase=phase, rounds=len(ds),
                mean_paired_delta_ms=mean, ci95_low_ms=mean-half, ci95_high_ms=mean+half,
                pct_of_mha_mean=100*mean/base, source='data/results/model/tables/cache_bench.json#latency.rounds'))
    write_csv('model_latency.csv', latency)

    fig, ax = plt.subplots(figsize=(7,4.5))
    fixed = [next(r['cache_mib'] for r in cache if r['arch']==a and r['mode']=='fixed') for a in ORDER]
    bars = ax.bar(ORDER,fixed,color=COLORS,width=.55)
    ax.bar_label(bars,fmt='%g MiB',padding=5)
    ax.set(ylabel='KV cache (MiB)',ylim=(0,14),title='Same dtype: cache compression is visible')
    ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.015,'Retained RTX 4050 measurements | BF16 fixed | batch 1 | 512 tokens | 8 layers',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1)); save('01_cache',fig)

    fig,ax = plt.subplots(figsize=(7,4.5))
    for i,a in enumerate(ORDER):
        vals = [float(r['final_val_ppl']) for r in grouped[a]]
        ax.scatter([i-.07,i+.07],vals,c=COLORS[i],s=50,zorder=3)
        ax.plot([i-.18,i+.18],[means[a]]*2,color=COLORS[i],lw=3)
        ax.annotate(f'{means[a]:.5f}',(i,max(vals)),xytext=(0,12),textcoords='offset points',ha='center')
    ax.set(xticks=range(3),xticklabels=ORDER,ylabel='Validation perplexity (lower is better)',
        title='Quality reference: this small trained variant',ylim=(8.8,10.95))
    ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.015,'CSV snapshots | two seeds | ~150M tokens | MLA uses per-head QK-Norm',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1)); save('02_quality',fig)

    fig,ax = plt.subplots(figsize=(7,4.5))
    vals=[r for r in latency if r['phase']=='decode']
    for i,r in enumerate(vals):
        m=r['mean_paired_delta_ms']
        ax.errorbar(m,i,xerr=[[m-r['ci95_low_ms']],[r['ci95_high_ms']-m]],fmt='o',capsize=5,color=COLORS[i+1])
        ax.annotate(f'{m:+.3f} ms',(m,i),xytext=(0,12),textcoords='offset points',ha='center')
    ax.axvline(0,color='grey',lw=1,ls='--')
    ax.set(yticks=range(2),yticklabels=[r['arch'] for r in vals],ylim=(-.5,1.5),
        xlabel='Mean paired decode difference vs MHA (ms/token)',title='One retained BF16 timing session; 30 rounds')
    ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.015,'RTX 4050 | batch 1 | 512 prompt + 128 decode | intervals: 1.96 SE across rounds',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1)); save('04_model_decode',fig)
    return quality,cache,latency


def layer_evidence():
    rows,inventory=[],[]
    for path in sorted((REPO / 'data/runs').glob('*/config.json')):
        rawpath=path.parent / 'bench.jsonl'
        if not rawpath.exists(): continue
        cfg=read_json(path)
        if cfg.get('kind') not in ['layer_bench','layer_ctx']: continue
        INPUTS.add(rawpath)
        raw=[json.loads(s) for s in rawpath.read_text(encoding='utf-8').splitlines() if s.strip()]
        base=cfg['base']; nl=cfg.get('n_layers',base.get('n_layers'))
        assert nl is not None
        checks=cfg.get('checks',[])
        ok=bool(checks) and all(max(c['out_rel'],c['cache_rel'])<.003 for c in checks)
        measured=[r for r in raw if r.get('kind')=='bench']
        inventory.append(dict(run_id=path.parent.name,kind=cfg['kind'],layers=nl,
            heads=base['n_heads'],d_c=base['d_c'],head_dim=base['head_dim'],d_rope=base['d_rope'],
            dtype=cfg.get('dtype'),torch=cfg.get('torch'),device=cfg.get('device'),
            chunk=cfg.get('chunk'),decode_steps=cfg.get('decode_steps'),rounds=cfg.get('rounds'),
            iters=cfg.get('iters'),measurement_rows=len(measured),
            stored_chunk_checks_pass=ok if checks else 'not_recorded',
            config=path.relative_to(REPO).as_posix(),jsonl=rawpath.relative_to(REPO).as_posix()))
        for lineno,r in enumerate(raw,1):
            if r.get('kind')!='bench': continue
            row=dict(run_id=path.parent.name,source_line=lineno,kind=cfg['kind'],
                layers=nl,d_model=base['d_model'],heads=base['n_heads'],head_dim=base['head_dim'],
                d_c=base['d_c'],d_rope=base['d_rope'],dtype=cfg.get('dtype'),
                torch=cfg.get('torch'),device=cfg.get('device'),chunk=cfg.get('chunk'),
                decode_steps=cfg.get('decode_steps'),configured_rounds=cfg.get('rounds'),
                round=r.get('round'),impl=r.get('impl'),ctx=r.get('ctx'),seq_len=r.get('seq_len'),
                actual_len=r.get('actual_len'),phase=r.get('phase'),latency_ms=r.get('latency_ms'),
                decode_med_ms=r.get('decode_med_ms'),prefill_ms=r.get('prefill_ms'),
                prefill_blocks=r.get('prefill_blocks'),cache_measured=r.get('cache_measured'),
                cache_theoretical=r.get('cache_theoretical'),jsonl=rawpath.relative_to(REPO).as_posix())
            if row['cache_measured'] is not None: assert row['cache_measured']==row['cache_theoretical']
            rows.append(row)
    write_csv('layer_runs.csv',inventory); write_csv('layer_measurements.csv',rows)
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['kind']=='layer_ctx': groups[(r['run_id'],r['ctx'])][r['impl']].append(r)
    ratios=[]
    for (run,ctx),im in sorted(groups.items()):
        if not all(i in im for i in ['naive_mla','absorbed_mla']): continue
        n,a=im['naive_mla'],im['absorbed_mla']; r=n[0]
        ratios.append(dict(run_id=run,ctx=ctx,layers=r['layers'],heads=r['heads'],d_c=r['d_c'],
            d_rope=r['d_rope'],chunk=r['chunk'],decode_steps=r['decode_steps'],
            n_naive=len(n),n_absorbed=len(a),
            decode_ratio=st.median(x['decode_med_ms'] for x in a)/st.median(x['decode_med_ms'] for x in n),
            incremental_prefill_ratio=st.median(x['prefill_ms'] for x in a)/st.median(x['prefill_ms'] for x in n)))
    write_csv('layer_ratios_by_run.csv',ratios)
    scan_ids=['layerctx_20261118_'+s for s in ['124423','124433','124443','124647']]
    scan=[r for r in ratios if r['run_id'] in scan_ids]
    assert len(scan)==8
    assert all((r['layers'],r['heads'],r['d_rope'],r['chunk'],r['decode_steps'],r['n_naive'],r['n_absorbed'])==(1,8,32,1024,24,3,3) for r in scan)
    write_csv('layer_scan_selected.csv',scan)
    fig,ax=plt.subplots(figsize=(7.5,5))
    for ctx,color,offset in [(16384,'#4169a1',-20),(32768,'#c77539',12)]:
        rs=sorted([r for r in scan if r['ctx']==ctx],key=lambda r:r['d_c'])
        ax.plot([r['d_c'] for r in rs],[r['decode_ratio'] for r in rs],'-o',color=color,label=f'{ctx//1024}K context')
        for r in rs:
            ax.annotate(f"{r['decode_ratio']:.3f}",(r['d_c'],r['decode_ratio']),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=9)
    ax.axhline(1,color='grey',ls='--',lw=1)
    ax.set(xlabel='Latent KV width d_c',ylabel='Median absorbed / median naive decode latency',
        title='Layer scan: a small observed crossing, not a stable threshold',ylim=(.85,1.85),xticks=[128,192,256,384])
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.025,'RTX 4050 | 1 layer | 8 heads | BF16 | 24 decode steps | 3 rounds per implementation',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1)); save('03_layer_scan',fig)
    return inventory,rows,scan


def main():
    FIGS.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.titlesize':12})
    quality,cache,latency=model_evidence()
    inventory,rows,scan=layer_evidence()
    lines=['# 课程汇报证据表（留存数据离线生成）','',
        '离线整理：**zhangjun005**。原始项目与历史测量归原作者。','',
        '全部图表属于旧实验的重新整理，未调用模型、训练或 GPU 基准。原始记录未改写。','',
        '## 缓存与质量','',
        '| 方案 | 每 token 每层元素 | BF16 fixed / 512 token / 8层 MiB | 验证 PPL 均值 | 两 seed 极差 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for q in quality:
        c=next(r for r in cache if r['arch']==q['arch'] and r['mode']=='fixed')
        lines.append(f"| {q['arch']} | {c['elements_per_token_per_layer']} | {c['cache_mib']:g} | {q['val_ppl_mean']:.5f} | {q['val_ppl_range']:.4f} |")
    lines += ['',f"MLA 验证 PPL 相对 MHA 高 {quality[2]['val_pct_vs_mha']:.4f}%；缓存 fixed 比为 12/3.375 = {12/3.375:.6f}，raw 比为 18/3.75 = 4.8。PPL 来自 CSV 快照，主组原始训练日志和权重缺失。",'',
        '## 同一会话内的模型时延','',
        '来源：cache_bench.json；BF16 autocast + dtype patches，batch 1，512 输入，128 decode，8样本，30轮，三个模型驻留于同进程。','',
        '| 方案 | 阶段 | 相对 MHA 配对差均值 ms | 轮间 1.96 SE 区间 ms | 相对 MHA 均值 |',
        '| --- | --- | ---: | --- | ---: |']
    for r in latency:
        lines.append(f"| {r['arch']} | {r['phase']} | {r['mean_paired_delta_ms']:+.4f} | [{r['ci95_low_ms']:.4f}, {r['ci95_high_ms']:.4f}] | {r['pct_of_mha_mean']:+.2f}% |")
    lines += ['', '区间从逐轮结果重算，只反映该会话的轮间变化；不涵盖硬件、代码与计时方法差异。decode 比较的是每轮内单步耗时摘要。','',
        '## 层级扫描：同配置协议，按 run 分组','',
        '| d_c | 16K absorbed/naive | 32K absorbed/naive | run |',
        '| ---: | ---: | ---: | --- |']
    for dc in [384,256,192,128]:
        rs=sorted([r for r in scan if r['d_c']==dc],key=lambda r:r['ctx'])
        lines.append(f"| {dc} | {rs[0]['decode_ratio']:.6f} | {rs[1]['decode_ratio']:.6f} | {rs[0]['run_id']} |")
    lines += ['',
        '以上均为一层、8头、BF16、chunk=1024、decode=24步、各实现3轮。各实现取轮摘要中位数再相除；不是配对比值均值。不同 d_c 来自不同会话，仍有时钟等不确定性。',
        '原报告 d_c=256 的16K比值1.340来自跨run池化（含16/24/32步协议）；本表只取124433，因此为1.310。这是分组变化，不是新实测。',
        '32K的prefill_ms只计从前一锚点继续推进的区间，不能叫从零到32K的总prefill。短序列是四层，单独保存在逐记录表中。',
        'd_c=171、heads=32等记录仍完整保留。本图只展示列出的四个8头扫描配置；不因内核退化就认定数据错误。','',
        '## 文件与追溯','',f'共读取{len(inventory)}个层级run，保留{len(rows)}条测量，没有按数值结果去重。','',
        '- model_quality.csv：三模型两seed汇总；源字段final_val_ppl / avg_tokens_per_s / wall_hours。',
        '- model_cache.csv：raw/fixed分列，元素与字节分列。',
        '- model_latency.csv：从latency.rounds重算配对差，核对原delta_vs_mha舍入值。',
        '- layer_runs.csv / layer_measurements.csv：层数、精度、PyTorch、chunk、decode步数、轮数和原文件行号。',
        '- layer_ratios_by_run.csv / layer_scan_selected.csv：逐run统计及主讲选择。',
        '- manifest.json：输入SHA256；PNG/SVG均由本脚本生成。',
        '', '复算：在ml-mla环境运行同目录上一级的build_evidence.py。脚本只用标准库和matplotlib，不导入torch或启动仓库流水线。']
    (OUT/'证据表.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    INPUTS.add(Path(__file__).resolve())
    manifest={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'mode':'offline_reanalysis_of_retained_results_no_gpu','historical_gpu':'RTX 4050 Laptop GPU',
        'local_analysis_env':'CPU-only offline analysis; no GPU measurements performed',
        'layer_run_count':len(inventory),'layer_measurement_rows':len(rows),
        'inputs':[{'path':p.relative_to(REPO).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(INPUTS)]}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Offline evidence ready: {OUT}; {len(inventory)} runs, {len(rows)} layer rows; 4 PNG + 4 SVG')


if __name__=='__main__':
    main()
