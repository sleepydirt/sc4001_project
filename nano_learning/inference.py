"""Explicit checkpoint evaluation; report generation never calls this module."""
import argparse
from pathlib import Path
from .common import STUDY, sha
from .evidence import historical_roots, load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['samsone','mellow'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if any(args.output.resolve().is_relative_to(root) for root in historical_roots().values()):
        raise ValueError('Use a new evaluation output; historical evidence is immutable')
    if args.output.exists() or args.output.with_suffix('.predictions.jsonl').exists():
        raise ValueError('Output exists; choose a new evaluation name')
    if sha(args.checkpoint)!=args.sha256:raise ValueError('Checkpoint hash mismatch')
    if args.split=='test':
        lock=load(STUDY/'evaluation_lock.json')
        if not any(c.get('sha256')==args.sha256 and c.get('model')==args.model for c in lock['checkpoints'].values()):
            raise ValueError('Checkpoint is not in the frozen test selection')
    from .dataset import Dataset
    from .policies import load_policy
    from .training import registry,seeded
    from .evaluate import evaluate
    seeded();policy=load_policy(args.model);policy.load_checkpoint(args.checkpoint)
    dataset=Dataset(args.split,allow_test=args.split=='test')
    result=evaluate(policy,dataset,registry(),args.output,primary_tiers=load(STUDY/'execution_manifest.json')['primary_tiers'])
    from .common import write
    result['checkpoint']={'model':args.model,'path':str(args.checkpoint),'sha256':args.sha256}
    write(args.output,result)

if __name__=='__main__':main()
