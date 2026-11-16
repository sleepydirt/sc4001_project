"""Freeze explicit checkpoints before new test inference, without running a model."""
import argparse
from pathlib import Path
from .common import STUDY, ARTIFACTS, sha,write
from .evidence import load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',type=Path,required=True,help='JSON mapping names to model, policy, sha256')
    args=parser.parse_args()
    historical=ARTIFACTS/'audio2tool_pilot/20261114_seed17'
    if STUDY==historical or (STUDY/'evaluation_lock.json').exists():raise ValueError('Existing/historical test lock is immutable')
    checkpoints=load(args.checkpoints)
    if not checkpoints:raise ValueError('Explicit checkpoints required')
    for name,item in checkpoints.items():
        if item['model'] not in ('samsone','mellow') or sha(item['policy'])!=item['sha256']:
            raise ValueError('Checkpoint identity mismatch: '+name)
    write(STUDY/'evaluation_lock.json',{'checkpoints':checkpoints,
        'final_test_lock_sha256':sha(STUDY/'final_test_lock.json'),'test_data_sha256':sha(STUDY/'data/test.jsonl'),
        'selection':'Explicit validation-selected checkpoints frozen before inference'})

if __name__=='__main__':main()
