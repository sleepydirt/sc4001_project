"""Paper assets from saved evidence only: no model, network, or GPU imports."""
from collections import defaultdict
import csv
import json
from pathlib import Path
from .common import ROOT, STUDY, sha, write
from .evidence import panels_path, roots, validation_dir, split_path, resolve_path
from .selection import select_warmups

METRICS=('macro_exact','binary','argument_F1','schema_validity','tool_multiset_accuracy')
DEST=ROOT/'paper/generated'


class Evidence:
    def __init__(self): self.sources={}
    def read(self,path):
        path=Path(path); self.sources[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)]=sha(path)
        if path.suffix=='.jsonl':
            return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        return json.loads(path.read_text())


def latex(value):
    return str(value).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def table(dest,name,records):
    if not records:raise ValueError('Empty table: '+name)
    fields=list(records[0]); folder=dest/'tables'; folder.mkdir(parents=True,exist_ok=True)
    with (folder/f'{name}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    # Long source ledgers remain CSV; manuscript fragments contain compact publication columns.
    layouts={
        'warmups':[('model','Model'),('pool','Pool (%)'),('step','Update'),('macro_exact','Macro (%)'),('schema_validity','Schema (%)')],
        'pilot_test':[('model','Model'),('arm','Arm'),('parent_updates','Parent'),('lineage_updates','Lineage'),('macro_exact','Macro (%)'),('binary','Pooled (%)'),('schema_validity','Schema (%)')],
        'endpoints':[('model','Model'),('arm','Arm'),('additional_updates','Added updates'),('macro_exact','Macro (%)'),('binary','Pooled (%)'),('argument_F1','Arg. F1 (%)'),('schema_validity','Schema (%)'),('training_minutes','Minutes')],
        'dataset':[('tier','Tier'),('scope','Cohort'),('entries','Entries'),('requests','Requests'),('audio_files','WAVs'),('hours','Hours')],
        'splits':[('split','Split'),('components','Components'),('entries','Entries'),('requests','Requests'),('audio_files','WAVs'),('hours','Hours')],
    }
    if name not in layouts and name!='parameters':return
    if name=='parameters':
        keys=[('decoder_trainable','Decoder'),('projector_trainable','Projector'),('extra_separator_trainable','Separator'),('trainable_total','Total trainable'),('frozen_encoder','Frozen encoder'),('model_total','Total parameters')]
        headers=['Component']+[r['model'] for r in records]
        body=[[label]+[str(r[key]) for r in records] for key,label in keys]
    else:
        columns=layouts[name];headers=[label for _,label in columns];body=[]
        for row in records:
            values=[]
            for key,label in columns:
                value=row[key]
                if key in METRICS or key=='pool':value=f'{100*value:.2f}'
                elif key=='hours':value=f'{value:.4f}'
                elif key=='training_minutes':value=f'{value:.1f}'
                values.append(str(value))
            body.append(values)
    lines=['% Requires booktabs. Metric values are percentages in this fragment.',r'\small',r'\begin{tabular}{'+'l'*len(headers)+'}',r'\toprule',
           ' & '.join(latex(x) for x in headers)+r' \\',r'\midrule']
    lines += [' & '.join(latex(x) for x in row)+r' \\' for row in body]
    lines += [r'\bottomrule',r'\end{tabular}']
    (folder/f'{name}.tex').write_text('\n'.join(lines)+'\n')


def dataset_tables(e,dest):
    pilot=roots()['pilot']; records=e.read(pilot/'recordings.jsonl')
    def stats(rr):
        audio={p:d for r in rr for p,d in zip(r['audio_paths'],r['durations'])}
        return dict(components=len({r['component'] for r in rr}),entries=len(rr),
            requests=len({r['request_group'] for r in rr}),audio_files=len(audio),audio_references=sum(len(r['audio_paths']) for r in rr),
            hours=sum(sum(r['durations']) for r in rr)/3600,unique_audio_hours=sum(audio.values())/3600)
    result=[]
    for tier in range(1,9):
        rr=[r for r in records if r['tier']==tier]
        for scope,values in [('source',rr),('filtered',[r for r in rr if r['common_supported']])]:
            result.append(dict(tier=tier,scope=scope,**stats(values)))
    table(dest,'dataset',result)
    splits=[]
    for split in ('train','validation','test'):
        rr=e.read(split_path(pilot,split)); splits.append(dict(split=split,**stats(rr)))
    table(dest,'splits',splits)
    panel=e.read(panels_path(pilot))['small']
    data=e.read(split_path(pilot,'validation')); byid={r['id']:r for r in data}
    if len(set(panel))!=len(panel):raise ValueError('Duplicate panel IDs')
    selected=[byid[i] for i in panel]
    counts=dict(recording_entries=len(panel),distinct_requests=len({r['request_group'] for r in selected}),
        source_components=len({r['component'] for r in selected}),ids=panel,
        label='mini-validation: training diagnostic',interpretation='Low-cost progress signal; not a full-split or deployment estimate.')
    write(dest/'evidence/panel.json',counts)
    return result,splits,counts


def collect(e):
    r=roots(); pilot=r['pilot']; runs={}; warmup=[]; curves=[]; learning=[]
    selected=select_warmups(pilot)
    historical=e.read(pilot/'parent_selection.json')
    coverage=e.read(pilot/'train_subset_coverage.json')
    for model in ('samsone','mellow'):
        for fraction in (.1,.3,1.):
            name=f'{model}-warmup-{round(100*fraction):03d}'
            directory=pilot/'runs'/name
            status=e.read(directory/'status.json'); assert status['step']==640
            data=e.read(validation_dir(directory)/'000640_full.json')
            warmup.append(dict(model=model,pool=fraction,step=640,cohort='full_validation',
                **{k:data[k] for k in METRICS},selected=selected[model]['run']==name,
                historical_parent=historical[model]['run']==name,historical_parent_step=historical[model]['step'],
                **{k:coverage[str(fraction)][k] for k in ['components','request_groups','recordings','hours']}))
            runs[name]=(directory,model,'sft','warmup',0,0)
    for model in ('samsone','mellow'):
        for arm in ('sft','binary','typed'):
            name=f'{model}-continuation-{arm}'
            runs[name]=(pilot/'runs'/name,model,arm,'pilot64',0,historical[model]['step'])
    for model,arm,directory in [('samsone','sft',r['followup']/'runs/samsone-sft'),
            ('samsone','binary',r['followup']/'runs/samsone-binary'),
            ('samsone','typed',r['extension']/'runs/typed1024'),('mellow','sft',r['mellow']/'runs/mellow-sft')]:
        runs[f'{model}-{arm}-1024']=(directory,model,arm,'continuation1024',0,640)
    runs['samsone-sft-4096']=(r['extension']/'runs/sft4096','samsone','sft','sft4096',1024,1664)
    provenance=[]
    for name,(directory,model,arm,stage,base,parent_step) in runs.items():
        ledger=e.read(directory/'learning.jsonl'); cumulative=0
        for row in ledger:
            cumulative+=row['elapsed_seconds']
            learning.append(dict(run=name,model=model,arm=arm,stage=stage,update=row['step']+base,
                local_update=row['step'],loss=row['loss'],training_seconds=cumulative,
                elapsed_seconds=row['elapsed_seconds'],precision=row['precision'],kl=row.get('kl',''),
                reward_mean=row.get('reward_mean',''),informative_group_rate=1-row['equal_reward_group_rate'] if 'equal_reward_group_rate' in row else ''))
        time_at={x['step']:sum(y['elapsed_seconds'] for y in ledger[:x['step']]) for x in ledger if x['step']%64==0}
        for path in sorted(validation_dir(directory).glob('*.json')):
            step,kind=path.stem.split('_'); step=int(step); data=e.read(path)
            curves.append(dict(run=name,model=model,arm=arm,stage=stage,update=step+base,local_update=step,
                cohort='full_validation' if kind=='full' else 'mini_validation_training_diagnostic',
                training_seconds=time_at.get(step,0),recording_entries=data['n_recordings'],requests=data['n_requests'],
                **{k:data[k] for k in METRICS}))
        state=e.read(directory/'status.json')
        provenance.append(dict(run=name,model=model,arm=arm,stage=stage,local_updates=state['step'],
            parent_updates=parent_step,lineage_updates=parent_step+state['step'],
            sampler_offset=0 if stage=='warmup' else 640 if stage=='pilot64' else 704 if stage=='continuation1024' else 1728,
            policy=str(directory/'endpoint_policy.pt'),sha256=state['endpoint_sha256'],
            precision=ledger[0]['precision'],training_seconds=cumulative))
    pilot_test=[]
    for model in ('samsone','mellow'):
        for arm in ('parent','sft','binary','typed'):
            path=pilot/'evaluation'/f'{model}-{arm if arm=="parent" else arm+"-endpoint"}.json'
            data=e.read(path)
            pilot_test.append(dict(model=model,arm=arm,cohort='pilot_test',
                parent_pool=historical[model]['fraction'],parent_updates=historical[model]['step'],
                added_updates=0 if arm=='parent' else 64,lineage_updates=historical[model]['step']+(0 if arm=='parent' else 64),
                **{k:data[k] for k in METRICS}))
    endpoints=[dict(model=x['model'],arm=x['arm'],additional_updates=x['update'],cohort=x['cohort'],
        **{k:x[k] for k in METRICS},training_minutes=x['training_seconds']/60)
        for x in curves if x['cohort']=='full_validation' and x['stage'] in ('continuation1024','sft4096')
        and x['local_update']==(3072 if x['stage']=='sft4096' else 1024)]
    first=next(x for x in endpoints if x['model']=='samsone' and x['arm']=='sft' and x['additional_updates']==1024)
    for x in endpoints:
        if x['additional_updates']==4096:x['training_minutes']+=first['training_minutes']
    return dict(warmups=warmup,pilot_test=pilot_test,endpoints=endpoints,validation_curves=curves,
        training_curves=learning,provenance=provenance,selection=selected)


def figures(dest,data):
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/audio2tool-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    folder=dest/'figures';folder.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for ext in ('pdf','png'):fig.savefig(folder/f'{name}.{ext}',dpi=320,bbox_inches='tight')
        plt.close(fig)
    colors={'sft':'#0072B2','binary':'#D55E00','typed':'#009E73'}
    fig,axs=plt.subplots(2,3,figsize=(14,7.4),layout='constrained',sharex=True)
    pool_styles=[(.1,'#0072B2','o','-'),(.3,'#D55E00','s','--'),(1.,'#009E73','^','-.')]
    titles=['Training response NLL','Validation macro exact','Validation schema validity']
    for row,model in enumerate(('samsone','mellow')):
        for fraction,color,marker,style in pool_styles:
            name=f'{model}-warmup-{round(100*fraction):03d}'
            training=sorted([r for r in data['training_curves'] if r['run']==name],key=lambda r:r['update'])
            window=32
            axs[row,0].plot([r['update'] for r in training][window-1:],
                np.convolve([r['loss'] for r in training],np.ones(window)/window,'valid'),
                color=color,ls=style,lw=1.8,label=f'{100*fraction:g}% pool')
            validation=sorted([r for r in data['validation_curves'] if r['run']==name and r['cohort']=='full_validation'],key=lambda r:r['update'])
            for col,key in [(1,'macro_exact'),(2,'schema_validity')]:
                axs[row,col].plot([r['update'] for r in validation],[100*r[key] for r in validation],
                    color=color,ls=style,marker=marker,ms=6,lw=1.8,label=f'{100*fraction:g}% pool')
        for col,ax in enumerate(axs[row]):
            ax.set_title(f'{model.capitalize()} — {titles[col]}',fontsize=11,pad=10)
            ax.set_xlabel('Warm-up optimizer updates')
            ax.set_xlim(0,660);ax.set_xticks([0,128,256,384,512,640])
            ax.grid(alpha=.2);ax.set_axisbelow(True)
            ax.set_ylabel('Response NLL (32-update mean)' if col==0 else 'Macro exact (%)' if col==1 else 'Schema validity (%)')
            if col==1:ax.set_ylim(-.7,15)
            if col==2:ax.set_ylim(0,100)
            ax.legend(fontsize=8,loc='best',framealpha=.9)
    # Offset coincident Mellow markers without altering their measured coordinates:
    # hollow markers keep all three tied-zero series visible.
    for line in axs[1,1].lines:
        line.set_markerfacecolor('none')
    fig.suptitle('Supervised warm-up: training dynamics and full-validation performance',fontsize=14)
    save(fig,'warmups')
    for grpo in (False,True):
        fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,model in zip(axs,('samsone','mellow')):
            names=sorted({x['run'] for x in data['training_curves'] if x['model']==model and (x['arm']!='sft')==grpo})
            for name in names:
                rr=[x for x in data['training_curves'] if x['run']==name]; window=min(32,len(rr))
                ax.plot([x['update'] for x in rr][window-1:],np.convolve([x['loss'] for x in rr],np.ones(window)/window,'valid'),label=name.replace(model+'-',''))
            ax.set(title=model.capitalize(),xlabel='Stage/continuation optimizer updates',ylabel='GRPO objective (32-update mean)' if grpo else 'SFT response NLL (32-update mean)')
            ax.legend(fontsize=6)
        save(fig,'grpo_loss' if grpo else 'sft_loss')
    for metric,label,filename in [('macro_exact','Macro exact (%)','continuation1024'),('schema_validity','Schema validity (%)','continuation1024_schema')]:
        fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,model in zip(axs,('samsone','mellow')):
            for arm in ('sft','binary','typed'):
                rr=[x for x in data['validation_curves'] if x['stage']=='continuation1024' and x['model']==model and x['arm']==arm]
                if not rr:continue
                mini=[x for x in rr if x['cohort'].startswith('mini')];full=[x for x in rr if x['cohort']=='full_validation']
                ax.plot([x['update'] for x in mini],[100*x[metric] for x in mini],color=colors[arm],ls='--',label=arm+' mini-validation')
                ax.scatter([x['update'] for x in full],[100*x[metric] for x in full],color=colors[arm],marker='D',s=55,label=arm+' full validation')
            ax.set(title=model.capitalize(),xlabel='Additional optimizer updates',ylabel=label);ax.legend(fontsize=7)
        fig.suptitle('Mini-validation: training diagnostic; diamonds: full-validation endpoints',fontsize=10)
        save(fig,filename)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for model,color in [('samsone','#0072B2'),('mellow','#CC79A7')]:
        rr=[r for r in data['training_curves'] if r['stage']=='continuation1024' and r['model']==model and r['arm']=='sft']
        window=32
        axs[0].plot([r['update'] for r in rr][window-1:],np.convolve([r['loss'] for r in rr],np.ones(window)/window,'valid'),color=color,label=model.capitalize())
    for arm in ('binary','typed'):
        rr=[r for r in data['training_curves'] if r['stage']=='continuation1024' and r['model']=='samsone' and r['arm']==arm]
        if any(r['kl']=='' for r in rr):raise ValueError('Missing recorded GRPO KL')
        axs[1].plot([r['update'] for r in rr][31:],np.convolve([r['kl'] for r in rr],np.ones(32)/32,'valid'),color=colors[arm],label='Samsone '+arm+' GRPO')
    for ax,title,ylabel in zip(axs,['SFT response NLL','GRPO KL to frozen reference'],['Response NLL (32-update mean)','Sampled KL estimator (32-update mean)']):
        ax.set(title=title,xlabel='Additional optimizer updates',ylabel=ylabel)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('+1024 continuation training diagnostics',fontsize=12)
    save(fig,'continuation1024_training')
    fig,axs=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    rr=[x for x in data['endpoints'] if x['additional_updates']==1024]
    labels=[x['model']+' '+x['arm'] for x in rr]
    for ax,key,title in [(axs[0],'macro_exact','Full-validation macro exact (%)'),(axs[1],'training_minutes','Training compute (minutes)')]:
        ax.bar(labels,[x[key]*(100 if key=='macro_exact' else 1) for x in rr],color=[colors[x['arm']] for x in rr]);ax.set_ylabel(title);ax.tick_params(axis='x',rotation=25)
    save(fig,'continuation_endpoints')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for kind,style in [('mini_validation_training_diagnostic','--'),('full_validation','o-')]:
        rr=[x for x in data['validation_curves'] if x['stage']=='sft4096' and x['cohort']==kind]
        ax.plot([x['update'] for x in rr],[100*x['macro_exact'] for x in rr],style,label='mini-validation: training diagnostic' if kind.startswith('mini') else 'full validation')
    ax.set(xlabel='Additional updates from warmup (extension starts at 1024)',ylabel='Macro exact (%)');ax.legend(fontsize=8)
    save(fig,'sft4096')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    rr=data['pilot_test'];labels=[x['model']+' '+x['arm'] for x in rr]
    ax.bar(labels,[100*x['macro_exact'] for x in rr]);ax.tick_params(axis='x',rotation=35)
    ax.set(ylabel='Pilot test macro exact (%)',title='Historical +64 pilot: Samsone parent 10%/640; Mellow parent 100%/256')
    save(fig,'pilot64_test')


def qualitative(e,dest):
    bundle=e.read(roots()['extension']/'appendix_b/examples.json')
    sources=bundle['prediction_sources']; predictions={}
    for stage,source in sources.items():
        path=resolve_path(source['path']); rr=e.read(path)
        if sha(path)!=source['sha256']:raise ValueError('Qualitative prediction source changed')
        predictions[stage]={r['id']:r for r in rr}
    examples=[]
    for ex in bundle['examples']:
        for stage,prediction in ex['predictions'].items():
            if prediction!=predictions[stage][ex['record']['id']]:raise ValueError('Qualitative prediction differs')
        examples.append(ex)
    # Include a Mellow example with reference, raw prediction and actual historical checkpoint.
    pilot=roots()['pilot'];test={r['id']:r for r in e.read(split_path(pilot,'test'))}
    output=e.read(pilot/'evaluation/mellow-sft-endpoint.predictions.jsonl')[0]
    mellow=dict(record=test[output['id']],predictions={'mellow_sft64_historical':output},
        checkpoint=e.read(pilot/'evaluation/mellow-sft-endpoint.json')['checkpoint'],
        lineage='100%/256 warmup +64 = 320 updates; corrected-parent +64 is unrun')
    write(dest/'evidence/qualitative.json',dict(posthoc_selected=True,representative=False,
        examples=examples,mellow_example=mellow,prediction_sources=sources))
    lines=['# Qualitative appendix','', 'Post-hoc illustrative cases, not a representative accuracy sample. Transcripts are reader annotations; models received audio and catalogues.','']
    for ex in examples+[mellow]:
        record=ex['record'];lines += [f"## {record['id']}",'',f"Reference: `{json.dumps(record['calls'],ensure_ascii=False)}`",'']
        for stage,pred in ex['predictions'].items():
            lines += [f"{stage}; saved exact={pred['scores']['binary']}",'','```json',pred['output']['text'],'```','']
    lines += ['Mellow historical +64 parent: 100%/256; lineage endpoint 320.','All checkpoint hashes and prediction source hashes are in evidence/qualitative.json and tables/provenance.csv.']
    (dest/'qualitative.md').write_text('\n'.join(lines)+'\n')


def build(dest=DEST):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);(dest/'evidence').mkdir(exist_ok=True)
    e=Evidence(); dataset,splits,panel=dataset_tables(e,dest);data=collect(e)
    for name in ('warmups','pilot_test','endpoints','validation_curves','training_curves','provenance'):table(dest,name,data[name])
    parameters=e.read(ROOT/'nano_learning/resources/parameter_counts.json')
    table(dest,'parameters',[{k:r[k] for k in ('model','decoder_trainable','projector_trainable','extra_separator_trainable','trainable_total','frozen_encoder','model_total')} for r in parameters['records']])
    write(dest/'evidence/corrected_selection.json',data['selection'])
    figures(dest,data);qualitative(e,dest)
    # Additional protocol records required by the claim audit and reproduction.
    for root in roots().values():e.read(root/'execution_manifest.json')
    e.read(STUDY/'environment_provenance.json');e.read(STUDY/'split_audit.json');e.read(STUDY/'registry_audit.json')
    e.read(STUDY/'evaluation/paired_bootstrap.json')
    for model in ('samsone','mellow'):e.read(STUDY/f'evaluation/{model}-released.json')
    write(dest/'evidence/sources.json',{'sources':e.sources,'mode':'saved evidence only; no inference','table_tolerance':1e-10})
    (dest/'captions.md').write_text(f'''# Caption drafts

All results use seed 17 and the filtered Audio2Tool benchmark. Full-validation endpoints describe this benchmark, not deployment performance. CSV metrics are fractions; figure axes convert them to percentages.

- **Warmups (2×3):** rows Samsone/Mellow; columns response NLL (trailing 32-update mean), full-validation macro exact, and full-validation schema validity. All curves show 10/30/100% pools. Mellow ties at zero macro exact; 10% is selected by the smallest-pool tie-break, not a unique accuracy maximum. Full validation (2,141 entries / 1,105 requests), macro exact at updates 256, 512 and 640, independent released-weight parents, 10/30/100% component pools. Selection compares only update 640, ties favor the smallest pool. Both models select 10%/640.
- **SFT loss:** response NLL, trailing 32-update mean, separated from GRPO objectives. Local updates reset per stage; the SFT extension is displayed at continuation updates 1024–4096.
- **GRPO loss:** clipped group-relative objective including KL, trailing 32-update mean. Its scale is not comparable to SFT NLL. Pilot and +1024 runs use their respective recorded parents and schedules.
- **Pilot +64:** held-out pilot test (2,126 entries / 1,096 requests). Samsone parent 10%/640, endpoint lineage 704. Mellow historical parent 100%/256, endpoint lineage 320. These values do not measure corrected-parent Mellow +64.
- **+1024 trajectories:** mini-validation: training diagnostic. {panel['recording_entries']} recording entries, {panel['distinct_requests']} distinct requests, {panel['source_components']} source components. Every 64 updates; diamonds mark the full-validation endpoint. Independent branches from each model's 10%/640 warmup, endpoint lineage 1664. Mellow has SFT only.
- **+1024 endpoints:** full-validation macro exact and sum of saved optimizer-step elapsed seconds / 60. Timing excludes model loading, validation and checkpoint I/O; it is not end-to-end wall time. No full-split intermediate measurements are inferred from the mini panel.
- **SFT-4096:** Samsone extension inherits +1024 SFT weights, optimizer moments and scaler, with a new 3072-update scheduler. Mini-validation every 64 local updates; full validation every 512 local updates and endpoint. Lineage 4736. Endpoint training minutes sum both SFT segments.
- **Qualitative appendix:** verified saved raw predictions with IDs, references and provenance; post-hoc selection does not estimate prevalence.
''')
    return dict(output=str(dest),panel=panel,selection=data['selection'],sources=len(e.sources),dataset=dataset,splits=splits)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEST);args=p.parse_args()
    result=build(args.output);print(json.dumps({k:v for k,v in result.items() if k not in ('dataset','splits')},indent=2))

if __name__=='__main__':main()
