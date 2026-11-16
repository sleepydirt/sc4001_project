"""Export paper tables from a newly prepared dataset and replication runs."""
import argparse
import csv
import json
from pathlib import Path
from .common import STUDY,sha,write
from .workflow import call


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table',choices=['dataset','parameters','warmup','pilot64','extensions'])
    parser.add_argument('--runs',type=Path,default=Path('artifacts/replication_runs'))
    parser.add_argument('--output',type=Path,default=Path('paper/replication'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    if args.table=='dataset':
        from .reporting import Evidence,dataset_tables
        dataset_tables(Evidence(),args.output)
        return
    if args.table=='parameters':
        from .policies import load_policy
        for model in ['samsone','mellow']:
            records.append(dict(model=model,**load_policy(model).parameter_manifest()))
        write(args.output/'parameters.json',records);return
    if args.table=='warmup':
        for model in ['samsone','mellow']:
            for pool in [10,30,100]:
                run=args.runs/f'{model}-warmup-{pool:03d}'
                result=json.loads((run/'validation/000640_full.json').read_text())
                records.append(dict(model=model,pool=pool,**{k:result[k] for k in ['macro_exact','binary','schema_validity']}))
    else:
        names=([f'{model}-{arm}-64' for model in ['samsone','mellow'] for arm in ['sft','binary','typed']]
               if args.table=='pilot64' else ['samsone-sft-1024','samsone-binary-1024','samsone-typed-1024','mellow-sft-1024','samsone-sft-4096'])
        checkpoints={name:dict(model=name.split('-')[0],policy=str((args.runs/name/'endpoint_policy.pt').resolve())) for name in names}
        if args.table=='pilot64':
            parents=json.loads((args.runs/'selected_parents.json').read_text())
            checkpoints.update({model+'-parent':dict(model=model,policy=value['policy']) for model,value in parents.items()})
        for item in checkpoints.values():item['sha256']=sha(item['policy'])
        if args.table=='pilot64':
            selection=args.output/'pilot64_checkpoints.json';write(selection,checkpoints)
            lock=STUDY/'evaluation_lock.json'
            if not lock.exists():call('lock_evaluation','--checkpoints',selection)
        for name,item in checkpoints.items():
            split='test' if args.table=='pilot64' else 'validation'
            output=args.output/f'{name}_{split}.json'
            if not output.exists():call('inference','--model',item['model'],'--checkpoint',item['policy'],'--sha256',item['sha256'],'--split',split,'--output',output)
            result=json.loads(output.read_text())
            records.append(dict(run=name,cohort=split,**{k:result[k] for k in ['macro_exact','binary','argument_F1','schema_validity']}))
    with (args.output/(args.table+'.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
