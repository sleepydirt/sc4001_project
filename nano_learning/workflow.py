"""Download, prepare, test and train the study with one consistent interface."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def call(module,*args):
    subprocess.run([sys.executable,'-m','nano_learning.'+module,*map(str,args)],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','test','train','warmup','pilot64','continuation1024','sft4096','evaluate'])
    parser.add_argument('--runs',type=Path,default=Path('artifacts/replication_runs'))
    args=parser.parse_args()
    from .common import STUDY,sha,write
    if args.stage=='prepare':
        call('prepare_models')
        for action in ['checkout','discover']:call('acquire',action)
        call('bulk_download')
        for action in ['metadata','verify']:call('eda',action)
        for model in ['samsone','mellow']:call('profile','--model',model)
        for module in ['dialogue','batch_check','probability_check','listening_audit']:call(module)
        call('eda','freeze')
    elif args.stage=='test':
        from .dataset import Dataset
        data=Dataset('train')
        example=STUDY/'integration_example.json'
        write(example,data.example(data.records[0]))
        subprocess.run([sys.executable,'-m','pytest','-q'],check=True,env={**os.environ,'NANO_AUDIO_INTEGRATION':'1','NANO_AUDIO_TRAINING_INTEGRATION_EXAMPLE':str(example)})
    elif args.stage in ('train','warmup','pilot64','continuation1024','sft4096'):
        from .selection import select_warmups
        if args.stage in ('train','warmup'):
          for model in ['samsone','mellow']:
            for fraction in [.1,.3,1.]:
                call('training','warmup','--model',model,'--fraction',fraction,'--output',args.runs/f'{model}-warmup-{round(fraction*100):03d}')
        parents=select_warmups(runs_root=args.runs.resolve())
        write(args.runs/'selected_parents.json',parents)
        for model,parent in parents.items():
          if args.stage in ('train','pilot64'):
            for arm in ['sft','binary','typed']:
                call('training','pilot64','--model',model,'--arm',arm,'--parent',parent['policy'],'--parent-sha256',parent['sha256'],'--output',args.runs/f'{model}-{arm}-64',*(['--enable-optional-mellow'] if model=='mellow' else []))
          if args.stage in ('train','continuation1024'):
            for arm in (['sft','binary','typed'] if model=='samsone' else ['sft']):
                call('training','continuation1024','--model',model,'--arm',arm,'--parent',parent['policy'],'--parent-sha256',parent['sha256'],'--output',args.runs/f'{model}-{arm}-1024')
        if args.stage not in ('train','sft4096'):return
        parent=args.runs/'samsone-sft-1024/endpoint_policy.pt'
        optimizer=args.runs/'samsone-sft-1024/checkpoint_001024/resume.pt'
        call('training','sft4096','--model','samsone','--parent',parent,'--parent-sha256',sha(parent),'--optimizer-resume',optimizer,'--optimizer-sha256',sha(optimizer),'--output',args.runs/'samsone-sft-4096')
    else:
        import csv
        records=[]
        for run in sorted(args.runs.iterdir()):
            checkpoint=run/'endpoint_policy.pt'
            if not checkpoint.is_file():continue
            output=run/'evaluation_validation.json'
            if not output.exists():call('inference','--model',run.name.split('-')[0],'--checkpoint',checkpoint,'--sha256',sha(checkpoint),'--output',output)
            result=json.loads(output.read_text())
            records.append(dict(run=run.name,macro_exact=result['macro_exact'],pooled_exact=result['binary'],argument_F1=result['argument_F1'],schema_validity=result['schema_validity']))
        if not records:raise ValueError('No completed runs found')
        with (args.runs/'results.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        print(json.dumps(records,indent=2))


if __name__=='__main__':main()
