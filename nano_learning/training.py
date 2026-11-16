"""Explicit, resumable Audio2Tool stages. Importing this module never trains."""
import argparse
import json
import os
from pathlib import Path
from .common import ROOT, STUDY, sha, digest, rows, write, append, tracking
from .evidence import panels_path, load, historical_roots, resolve_path
from .schedules import SPECS, validation_kinds


def seeded():
    import random, numpy as np, torch
    random.seed(17); np.random.seed(17); torch.manual_seed(17); torch.cuda.manual_seed_all(17)


def registry():
    from .task import Registry
    return Registry(STUDY / 'source_repo/tools_registry.csv')


def trainer_for(policy, model, arm, horizon, reg, path, metadata):
    from nano_audio.training.core import Trainer, TrainingConfig
    from .task import score
    grpo=arm in ('binary','typed')
    config=TrainingConfig.for_profile('local_pilot','grpo' if grpo else 'sft', total_updates=horizon,
        effective_batch_size=4,microbatch_size=1,max_new_tokens=512,
        precision='fp16' if model=='samsone' else 'bf16',
        reward_mode='verifiable_partial' if arm=='typed' else 'binary')
    return Trainer(policy,config,reward_fn=(lambda e,o:score(reg,e,o)[arm]) if grpo else None,
                   ledger_path=path/'learning.jsonl',metadata=metadata)


def prune_uncommitted(directory, step):
    for name in ('learning','exposure','rollouts','monitoring'):
        path=directory/f'{name}.jsonl'
        if not path.exists(): continue
        rr=rows(path); keep=[r for r in rr if r.get('step',r.get('update',0))<=step]
        lost=[r for r in rr if r.get('step',r.get('update',0))>step]
        if lost:
            import time
            for r in lost: append(directory/f'{name}.orphaned_{time.time_ns()}.jsonl',r)
            path.write_text(''.join(json.dumps(r)+'\n' for r in keep))


def install_group_sampling(policy, force_batch):
    from .sampling import generate_batch
    original=policy.generate; queue=[]
    def grouped(example, config=None):
        if not (config or {}).get('do_sample'): return original(example,config)
        if not queue:
            queue.extend((example['id'],o) for o in generate_batch(policy,[example]*4,config,force_batch=force_batch))
        identity,output=queue.pop(0)
        if identity!=example['id']: raise RuntimeError('Rollout group mixed prompts')
        return output
    policy.generate=grouped
    return queue


def run_stage(args):
    import fcntl, shutil, torch
    from dataclasses import asdict
    from .dataset import Dataset
    from .policies import load_policy
    from .evaluate import evaluate
    from .task import score
    spec=SPECS[args.stage]
    out=Path(args.output).resolve()
    if any(out.is_relative_to(r) for r in historical_roots().values()):
        raise ValueError('Historical evidence is read-only; choose a new output directory')
    if args.stage!='warmup' and not args.parent: raise ValueError('Explicit parent required')
    if args.stage in ('warmup','sft4096') and args.arm!='sft': raise ValueError('This stage is SFT only')
    if args.stage=='sft4096' and (args.model!='samsone' or not args.optimizer_resume):
        raise ValueError('SFT extension requires Samsone and the +1024 resume.pt optimizer')
    if args.model=='mellow' and args.stage=='pilot64' and not args.enable_optional_mellow:
        raise ValueError('Corrected Mellow +64 is disabled; explicitly enable the optional experiment')
    parent=resolve_path(args.parent) if args.parent else None
    if parent and sha(parent)!=args.parent_sha256: raise ValueError('Parent checksum mismatch')
    if args.stage=='sft4096' and sha(args.optimizer_resume)!=args.optimizer_sha256:
        raise ValueError('Optimizer checksum mismatch')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data=Dataset('train'); validation=Dataset('validation'); reg=registry()
        manifest=load(STUDY/'execution_manifest.json')
        components=load(STUDY/'train_subsets.json')['subsets'][str(args.fraction if args.stage=='warmup' else 1.0)]
        panel=load(panels_path(STUDY))['small']
        plan=dict(stage=args.stage,model=args.model,arm=args.arm,fraction=args.fraction,
            parent=str(parent) if parent else 'released',parent_sha256=args.parent_sha256,
            parent_updates=args.parent_updates,seed=17,**spec,
            train_sha256=data.sha256,validation_sha256=validation.sha256,panel_sha256=digest(panel),
            components_sha256=digest(components),optimizer_sha256=args.optimizer_sha256,
            sources={str(p.relative_to(ROOT)):sha(p) for folder in ['nano_learning','src/nano_audio']
                     for p in sorted((ROOT/folder).rglob('*.py'))})
        # JSON round trip normalizes tuples before comparing an existing freeze.
        plan=json.loads(json.dumps(plan)); frozen=out/'execution_manifest.json'
        if frozen.exists() and load(frozen)!=plan: raise ValueError('Resume configuration, data or source changed')
        write(frozen,plan)
        if (out/'status.json').exists() and load(out/'status.json')['status']=='completed': return
        seeded();torch.set_num_threads(4);policy=load_policy(args.model)
        if parent:policy.load_checkpoint(parent)
        trainer=trainer_for(policy,args.model,args.arm,spec['updates'],reg,out,{'manifest_sha256':sha(frozen)})
        pointer=out/'latest.json'
        if pointer.exists():
            latest=load(pointer); checkpoint=out/latest['directory']
            if sha(checkpoint/'policy')!=latest['policy_sha256']:raise ValueError('Resume policy changed')
            trainer.load_checkpoint(checkpoint)
        elif args.stage=='sft4096':
            state=torch.load(args.optimizer_resume,map_location='cpu',weights_only=False)
            if state['step']!=1024: raise ValueError('Expected +1024 optimizer')
            initial=list(trainer.scheduler.get_last_lr())
            trainer.optimizer.load_state_dict(state['optimizer']);trainer.scaler.load_state_dict(state['scaler'])
            for group,lr in zip(trainer.optimizer.param_groups,initial):group['lr']=lr
            write(out/'optimizer_restart.json',{'optimizer_moments_restored':True,'scheduler_reset':True,'initial_lrs':initial})
        prune_uncommitted(out,trainer.step)
        grpo=args.arm!='sft'
        queue=install_group_sampling(policy,args.stage=='continuation1024') if grpo else []
        run=tracking(out.name,'training',{**plan,**asdict(trainer.config)},root=out)
        def boundary():
            for kind in validation_kinds(args.stage,trainer.step):
                target=out/f'validation/{trainer.step:06d}_{kind}.json'
                if target.exists():continue
                result=evaluate(policy,validation,reg,target,ids=panel if kind=='small' else None,
                    primary_tiers=manifest['primary_tiers'])
                run.log({'update':trainer.step,f'validation/{kind}_macro_exact':result['macro_exact']})
        try:
            boundary()
            while trainer.step<spec['updates']:
                batch=data.batch(spec['offset']+trainer.step,components)
                row=trainer.grpo_step(batch) if grpo else trainer.sft_step(batch)
                append(out/'exposure.jsonl',{'step':trainer.step,'ids':[e['id'] for e in batch],
                    'components':[e['component'] for e in batch],'request_groups':[e['request_group'] for e in batch],
                    'sampled_completions':16 if grpo else 0,'response_tokens':row['tokens']})
                if grpo:
                    for ex,group in zip(batch,trainer.last_rollouts):
                        append(out/'rollouts.jsonl',{'step':trainer.step,'id':ex['id'],
                            'outputs':[r['rollout'] for r in group],'rewards':[r['reward'] for r in group],
                            'scores':[score(reg,ex,r['rollout']) for r in group]})
                metrics={'update':trainer.step,**{'train/'+k:v for k,v in row.items() if type(v) in (float,int)}}
                append(out/'monitoring.jsonl',metrics);run.log(metrics)
                if trainer.step%64==0 or trainer.step==spec['updates']:
                    if queue:raise RuntimeError('Incomplete rollout group')
                    checkpoint=out/f'checkpoint_{trainer.step:06d}'
                    trainer.save_checkpoint(checkpoint)
                    write(pointer,{'directory':checkpoint.name,'step':trainer.step,'policy_sha256':sha(checkpoint/'policy')})
                    boundary()
                    # Retain all complete checkpoints; optional deletion uses the manifest tool.
            shutil.copy2(checkpoint/'policy',out/'endpoint_policy.pt')
            write(out/'status.json',{'status':'completed','step':trainer.step,
                'lineage_updates':args.parent_updates+trainer.step,'endpoint_sha256':sha(out/'endpoint_policy.pt')})
        finally:run.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=SPECS);p.add_argument('--model',choices=['samsone','mellow'],required=True)
    p.add_argument('--arm',choices=['sft','binary','typed'],default='sft')
    p.add_argument('--fraction',type=float,choices=[.1,.3,1.],default=.1)
    p.add_argument('--parent');p.add_argument('--parent-sha256');p.add_argument('--parent-updates',type=int,default=640)
    p.add_argument('--optimizer-resume');p.add_argument('--optimizer-sha256');p.add_argument('--output',required=True)
    p.add_argument('--enable-optional-mellow',action='store_true')
    args=p.parse_args()
    if args.stage=='warmup':args.parent_updates=0
    elif args.stage=='sft4096':args.parent_updates=1664
    run_stage(args)

if __name__=='__main__':main()
