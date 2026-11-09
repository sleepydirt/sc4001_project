"""Executable research workflow; stages retain explicit measured/blocked status."""
from __future__ import annotations
import argparse, dataclasses, gc, json, os, random, signal, sys, time, traceback
from pathlib import Path
import yaml
from .utils import ROOT,append_json,digest,read_json,sha,write_json

def seeded(seed):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(8,os.cpu_count() or 1))

def profile_config(profile):return yaml.safe_load((ROOT/f'configs/profiles/{profile}.yaml').read_text())
def dataset_path(profile):return ROOT/f'artifacts/data/{profile}/examples.jsonl'
def records(profile,split=None):
    rows=[json.loads(l) for l in dataset_path(profile).read_text().splitlines() if l.strip()]
    for r in rows:
        if 'split' not in r:r['split']=r['base_split']
        if not Path(r['audio_path']).is_absolute():r['audio_path']=str(ROOT/r['audio_path'])
        elif not Path(r['audio_path']).exists() and '/artifacts/' in r['audio_path']:r['audio_path']=str(ROOT/'artifacts'/r['audio_path'].split('/artifacts/',1)[1])
        r['target_text']=json.dumps(r['target'],separators=(',',':'))
    return [r for r in rows if split is None or r['split']==split]
def families():
    from .data.task import schema_families
    return {f.family_id:f for f in schema_families()}
def scoring(example,rollout):
    from .evaluation.metrics import score_prediction
    return score_prediction(rollout['text'],example['canonical_target'],families()[example['schema_family_id']],example.get('order_matters',True),rollout.get('length_capped',False))
def load_policy(model,device,profile='smoke'):
    if model=='mellow_v0_published':
        from .models.mellow import MellowPolicy
        pool=sorted({r['audio_path'] for r in records(profile,'train')}) if dataset_path(profile).exists() else [str(ROOT/'vendor/mellow/resource/2.wav')]
        return MellowPolicy(device=device, asset_root=ROOT/'artifacts/assets', second_audio_pool=pool)
    from .models import SamsonePolicy
    return SamsonePolicy(model,device)

def smoke_text(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from .assets import download
    assets=download(offline=args.offline);path=assets['models']['smollm2_135m_base']['local_path']
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device
    m=AutoModelForCausalLM.from_pretrained(path).to(device).eval();tok=AutoTokenizer.from_pretrained(path);tok.pad_token=tok.eos_token
    out=[]
    for prompt in ['The capital of France is','A microphone records','To turn on a light,']:
        ids=tok(prompt,return_tensors='pt').to(device)
        with torch.no_grad():generated=m.generate(**ids,max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
        out.append({'prompt':prompt,'prompt_ids':ids.input_ids[0].tolist(),'generated_ids':generated[0].tolist(),'text':tok.decode(generated[0])})
    ids=tok('The capital of France is Paris.',return_tensors='pt').to(device)
    loss=m(**ids,labels=ids.input_ids).loss;loss.backward()
    dest=ROOT/'artifacts/smoke/text';m.save_pretrained(dest);tok.save_pretrained(dest)
    reloaded=AutoModelForCausalLM.from_pretrained(dest).to(device).eval()
    with torch.no_grad():reload_loss=reloaded(**ids,labels=ids.input_ids).loss.item()
    result={'status':'completed','outputs':out,'teacher_forced_loss':loss.item(),'backward_finite':all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None),'reload_loss':reload_loss,'reload_matches':abs(reload_loss-loss.item())<1e-6,'peak_memory_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None}
    write_json(ROOT/'artifacts/smoke/text.json',result);print(json.dumps(result,indent=2));return result

def smoke_audio(args):
    import torch
    p=load_policy(args.model,args.device,args.profile)
    path=str(ROOT/'vendor/samsone/assets/cow.wav');prompt='what can be heard in the audio? answer: '
    if args.model=='mellow_v0_published':raise RuntimeError('Use reproduce native worker for Mellow native example')
    # Same FP32 native path for strict parity, then research FP16 path separately.
    native=p.native(audio=path,prompt='what can be heard in the audio?',max_new_tokens=64)
    generated=p.generate({'audio_path':path,'_prompt':prompt},{'max_new_tokens':64})
    ids=p.encode('a cow is mooing.')+[p.eos_token_id]
    lp=p.completion_logprobs({'audio_path':path,'_prompt':prompt},ids);(-lp.mean()).backward()
    grads={n:float(v.grad.float().norm()) for n,v in p.model.named_parameters() if v.grad is not None}
    out={'status':'completed','native':native,'research':generated,'precision_difference':'native fp32; research fp16 autocast on cuda','prefix_match':native.startswith(generated['text']) or generated['text'].startswith(native),'loss':float(-lp.mean().detach()),'encoder_has_grad':any(v.grad is not None for v in p.model.audio_encoder.parameters()),'projector_nonzero_grad':any(v>0 for n,v in grads.items() if 'projector' in n),'decoder_nonzero_grad':any(v>0 for n,v in grads.items() if 'text_model' in n),'parameter_manifest':p.parameter_manifest()}
    p.save_checkpoint(ROOT/'artifacts/smoke/audio.pt',{'kind':'smoke'})
    before=p.generate({'audio_path':path,'_prompt':prompt},{'max_new_tokens':16});p.load_checkpoint(ROOT/'artifacts/smoke/audio.pt');after=p.generate({'audio_path':path,'_prompt':prompt},{'max_new_tokens':16});out['reload_prediction_matches']=before==after
    write_json(ROOT/'artifacts/smoke/audio.json',out);write_json(ROOT/'artifacts/architecture.json',p.parameter_manifest());print(json.dumps(out,indent=2));return out

def prepare(args):
    from .data.pipeline import prepare_data
    return prepare_data(profile=args.profile,output_dir=ROOT/f'artifacts/data/{args.profile}',seed=args.seed)

def audit(args):
    from .data.synthetic import audit_splits
    result=audit_splits(records(args.profile));write_json(ROOT/f'artifacts/data/{args.profile}/audit.json',result);return result

def run_dir(profile,run,seed):return ROOT/f'artifacts/runs/{profile}/{run}/seed_{seed}'
def train(args):
    import torch
    from .training.core import Trainer,TrainingConfig
    cfg=yaml.safe_load((ROOT/'configs/experiments/study.yaml').read_text())['runs'][args.run]
    prof=profile_config(args.profile);directory=run_dir(args.profile,args.run,args.seed);directory.mkdir(parents=True,exist_ok=True)
    data_hash=sha(dataset_path(args.profile));identity=digest({'run':cfg,'profile':prof,'data':data_hash,'seed':args.seed})
    done=directory/'stage.json'
    if done.exists() and read_json(done).get('identity')==identity and read_json(done).get('status')=='completed':return read_json(done)
    policy=load_policy(cfg['model'],args.device,args.profile)
    if cfg.get('parent'):
        parent_seed=17 if args.profile=='main' and cfg['parent'] in ['S_P1','M_P1','F_AUDIO'] else args.seed
        parent=run_dir(args.profile,cfg['parent'],parent_seed)/'checkpoint/policy'
        if not parent.exists():raise FileNotFoundError(f'Parent checkpoint missing. Run nano-audio train --run {cfg["parent"]} --profile {args.profile} --seed {args.seed}')
        policy.load_checkpoint(parent)
    rows=records(args.profile,'train');stage=cfg['stage'];objective='grpo' if stage=='grpo' else 'sft'
    if stage=='baseline':updates=0
    elif stage=='text':updates=prof['updates']['fresh_text']
    elif stage=='align':updates=prof['updates']['fresh_align']
    elif stage=='audio':updates=prof['updates']['fresh_audio']
    elif stage=='grpo':updates=prof['updates']['grpo']
    else:updates=prof['updates']['warmup' if args.run.endswith('P1') else 'continued_sft']
    if args.updates is not None:updates=args.updates
    if stage=='align':policy.decoder.requires_grad_(False)
    if stage=='text':
        policy.model.projector.requires_grad_(False);policy.model.sep_token.requires_grad_(False)
    configuration=TrainingConfig.for_profile(args.profile,objective,total_updates=max(1,updates),precision=getattr(policy,'research_precision','fp16' if policy.device.type=='cuda' else 'fp32'))
    reference=None
    if objective=='grpo' and cfg['model']=='mellow_v0_published':
        reference=load_policy(cfg['model'],args.device,args.profile);reference.model.load_state_dict(policy.model.state_dict(),strict=True)
    trainer=Trainer(policy,configuration,reference_policy=reference,reward_fn=lambda ex,out:scoring(ex,out)['reward'],ledger_path=directory/'learning.jsonl',metadata={'data_hash':data_hash,'identity':identity,'run':args.run,'parent':cfg.get('parent'),'condition':'mixed' if any(r['source']=='slurp' for r in rows) else 'constructed_only'})
    if (directory/'checkpoint/resume.pt').exists():trainer.load_checkpoint(directory/'checkpoint')
    stopping={'requested':False}
    def preempt(signum,frame):
        # Save only at a completed optimizer boundary, never a half-updated Adam state.
        stopping['requested']=True
    for sig in [signal.SIGTERM,signal.SIGUSR1]:signal.signal(sig,preempt)


    from contextlib import nullcontext

    tracking = nullcontext()
    if os.environ.get("WANDB_PROJECT"):
        import wandb

        tracking = wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name=f"{args.run}-{args.profile}-s{args.seed}-{os.getenv('SLURM_JOB_ID', 'local')}",
            config={
                **dataclasses.asdict(configuration),
                "run": args.run,
                "seed": args.seed,
                "model": cfg["model"],
                "data_sha256": data_hash,
                "parent_sha256": sha(parent) if cfg.get("parent") else None,
                "start_update": trainer.step,
            },
        )

    with tracking as wb:
        started = time.monotonic()
        if wb is not None:
            wb.define_metric("update")
            wb.define_metric("train/*", step_metric="update")
            wb.define_metric("dev/*", step_metric="update")
    
        # Indent the rest of train() inside this block, through its final return.
        # Deduplicate schema expansion for semantic-pool sampling; schema sampled independently.
        rng=random.Random(args.seed)
        pool={}
        for r in rows:pool.setdefault(r['utterance_group_id'],[]).append(r)
        groups=list(pool)
        sampled=[]
        for _ in range(updates):
            batch=[]
            for _ in range(configuration.effective_batch_size):
                group=rng.choice(groups);item=dict(rng.choice(pool[group]));item['tools']=list(item['tools']);rng.shuffle(item['tools']);batch.append(item)
            sampled.append(batch)
        probe=dict(rows[0])
        if stage=='text':probe['modality']='text'
        elif stage=='align':probe['_prompt']='transcribe the speech. answer: ';probe['target_text']=probe['transcript']
        probe_ids=policy.encode(probe['target_text'])+[policy.eos_token_id]
        with torch.no_grad():probe_before=float(-policy.completion_logprobs(probe,probe_ids).mean())
        while trainer.step<updates:
            batch=sampled[trainer.step]
            for ex in batch:
                if stage=='text':ex['modality']='text'
                elif stage=='align' and trainer.step%2==0:ex['_prompt']='transcribe the speech. answer: ';ex['target_text']=ex['transcript']
                elif stage=='audio' and trainer.step%4==0:ex['_prompt']='transcribe the speech. answer: ';ex['target_text']=ex['transcript']
            result=trainer.grpo_step(batch) if objective=='grpo' else trainer.sft_step(batch)
            print(json.dumps({'run':args.run,**result}),flush=True)
            if objective=='grpo':
                serial=[[{k:(v.cpu().tolist() if torch.is_tensor(v) else v) for k,v in row.items()} for row in group] for group in trainer.last_rollouts]
                append_json(directory/'rollouts.jsonl',{'step':trainer.step,'groups':serial})
            if trainer.step%configuration.checkpoint_every==0 or stopping['requested']:trainer.save_checkpoint(directory/'checkpoint')
            if wb is not None:
                metrics = {
                    f"train/{key}": value
                    for key, value in result.items()
                    if key != "step" and isinstance(value, (int, float))
                }
                metrics.update({
                    f"train/lr_group_{i}": lr
                    for i, lr in enumerate(result["learning_rates"])
                })
                wb.log({
                    "update": trainer.step,
                    "train/wall_minutes": (time.monotonic() - started) / 60,
                    **metrics,
                })
            if stopping['requested']:
                write_json(done,{'status':'interrupted','identity':identity,'next_command':f'nano-audio train --run {args.run} --profile {args.profile} --seed {args.seed}'})
                raise SystemExit(75)
        trainer.save_checkpoint(directory/'checkpoint')
        resumed=trainer.load_checkpoint(directory/'checkpoint')
        resume_verified=resumed['step']==trainer.step and resumed['sampler_position']==trainer.sampler_position
        with torch.no_grad():probe_after=float(-policy.completion_logprobs(probe,probe_ids).mean())
        write_json(directory/'probe.json',{'id':probe['id'],'target_ids':probe_ids,'before_response_nll':probe_before,'after_response_nll':probe_after,'interpretation':'teacher-forced diagnostic, not speech-understanding accuracy'})
        # Smoke held-out dev only. Main final evaluation is an explicit separate command.
        dev_pool=records(args.profile,'dev');dev_by_group={}
        for ex in dev_pool:dev_by_group.setdefault(ex['utterance_group_id'],ex)
        dev=list(dev_by_group.values())[:2 if args.profile=='smoke' else 16]
        predictions=[]
        for ex in dev:
            output=policy.generate(ex,{'max_new_tokens':128});predictions.append({'id':ex['id'],'utterance_group_id':ex['utterance_group_id'],'schema_family_id':ex['schema_family_id'],'source':ex['source'],'split':ex['split'],'output':output,**scoring(ex,output)})
        from .evaluation.metrics import aggregate
        write_json(directory/'dev_predictions.json',predictions)
        result={'status':'completed','identity':identity,'profile':args.profile,'run':args.run,'updates':trainer.step,'resume_verified':resume_verified,'parent':cfg.get('parent'),'parent_seed':parent_seed if cfg.get('parent') else None,'parent_checkpoint_sha256':sha(parent) if cfg.get('parent') else None,'data_hash':data_hash,'manifest':policy.parameter_manifest(),'dev_metrics':aggregate(predictions),'warning':'smoke infrastructure evidence only' if args.profile=='smoke' else None,'rl_signal':'unverified; inspect reward-bearing groups' if objective=='grpo' else None}
        write_json(done,result);return result

def overfit(args):
    from .training.diagnostics import overfit_eight
    if args.profile=='smoke' and args.updates is None:raise ValueError('Smoke scope does not run overfit by default; supply an explicit --updates cap or use local_pilot')
    policy=load_policy(args.model,args.device,args.profile)
    return overfit_eight(policy,records(args.profile,'train'),lambda ex,out:scoring(ex,out)['reward'],ROOT/f'artifacts/diagnostics/{args.profile}/eight',args.profile,args.updates or 200,args.seed)

def control_evaluation(args):
    from .evaluation.controls import materialize_controls,WhisperTextCascade
    import torch
    pool=records(args.profile,'dev');first=pool[0]
    other=next(r for r in pool if r['utterance_group_id']!=first['utterance_group_id'])
    root=ROOT/f'artifacts/reports/{args.profile}/controls'
    controls=materialize_controls(first,other['audio_path'],root,args.seed)
    policy=load_policy('samsone_134m_published',args.device,args.profile)
    policy.load_checkpoint(run_dir(args.profile,'S_P_SFT',args.seed)/'checkpoint/policy')
    rows=[]
    for condition,ex in [('clean',first),*controls.items()]:
        output=policy.generate(ex,{'max_new_tokens':128})
        rows.append({'condition':condition,'id':ex['id'],'utterance_group_id':ex['utterance_group_id'],'output':output,**scoring(ex,output)})
    del policy
    if torch.cuda.is_available():torch.cuda.empty_cache()
    text=load_policy('samsone_134m_fresh',args.device,args.profile)
    text.load_checkpoint(run_dir(args.profile,'F_TEXT',args.seed)/'checkpoint/policy')
    assets=read_json(ROOT/'artifacts/assets/manifest.json')
    cascade=WhisperTextCascade(assets['models']['whisper_tiny']['local_path'],text,str(text.device))
    output=cascade.generate(first,{'max_new_tokens':128})
    result={'scope':'smoke_only' if args.profile=='smoke' else 'fixed_control_subset','source':first['source'],'direct_audio_controls':rows,'cascade':{'text_decoder':'task-SFT official-pruned SmolLM2 F_TEXT','output':output,**scoring(first,output)}}
    write_json(root/'results.json',result);return result

def evaluate(args):
    if args.suite=='controls':return control_evaluation(args)
    import torch
    from .evaluation.metrics import aggregate,paired_bootstrap
    data=records(args.profile,'dev' if args.profile=='smoke' else 'test')
    limit=profile_config(args.profile).get('evaluation_limit_groups')
    if limit:
        groups=list(dict.fromkeys(r['utterance_group_id'] for r in data))[:limit]
        data=[r for r in data if r['utterance_group_id'] in groups]
        # Smoke runs exercise one rendering per schema condition per utterance.
        selected={}
        for r in data:
            suite='id_schema' if r['schema_family_id'].startswith('train_') else 'ood_structure' if r['schema_family_id'].startswith('structure_') else 'ood_names'
            selected.setdefault((r['utterance_group_id'],suite),r)
        data=list(selected.values())
        # Pair the SAME dev audio with a familiar catalog; no final test schemas.
        familiar=families()['train_01'];paired=[]
        for ex in data:
            ex['evaluation_suite']='dev_schema_ood'
            row=dict(ex);row.update(id=ex['id']+'_id_schema',schema_family_id=familiar.family_id,tools=familiar.tools,target=familiar.encode(ex['canonical_target']),evaluation_suite='id_schema')
            paired.extend([ex,row])
        data=paired
    results={}
    run_root=ROOT/f'artifacts/runs/{args.profile}'
    for stage in sorted(run_root.glob('*/seed_*/stage.json')):
        info=read_json(stage)
        if info.get('status')!='completed':continue
        checkpoint=stage.parent/'checkpoint/policy'
        if not checkpoint.exists():continue
        policy=load_policy(info['manifest']['model_id'],args.device,args.profile);policy.load_checkpoint(checkpoint)
        rows=[]
        for ex in data:
            suite=ex.get('evaluation_suite') or ('id_schema' if ex['schema_family_id'].startswith('train_') else 'ood_structure' if ex['schema_family_id'].startswith('structure_') else 'ood_names')
            for modality in ['audio','text']:
                item=dict(ex)
                if modality=='text':item['modality']='text'
                output=policy.generate(item,{'max_new_tokens':128})
                rows.append({'id':ex['id'],'utterance_group_id':ex['utterance_group_id'],'schema_family_id':ex['schema_family_id'],'source':ex['source'],'split':ex['split'],'suite':suite,'modality':modality,'output':output,**scoring(ex,output)})
        write_json(stage.parent/'research_predictions.json',rows)
        scores={}
        for suite in sorted({r['suite'] for r in rows}):
            for modality in ['audio','text']:
                scores[suite+'/'+modality]=aggregate([r for r in rows if r['suite']==suite and r['modality']==modality])
        results[stage.parent.parent.name+'/'+stage.parent.name]=scores
        del policy;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result={'suite':args.suite,'profile':args.profile,'evaluated_partition':'dev' if args.profile=='smoke' else 'test','results':results,'human_speech_primary':{'status':'unavailable' if not any(r['source']=='slurp' for r in data) else 'pending','reason':'No completed held-out human-speech OOD evaluation'},'scope':'smoke_only' if args.profile=='smoke' else 'research','final_test':'not used for smoke model selection'}
    write_json(ROOT/f'artifacts/reports/{args.profile}/evaluation.json',result);return result

def reproduce(args):
    from .evaluation.native import claims_registry,write_claims
    claims=claims_registry(ROOT/'vendor/samsone')
    assets=read_json(ROOT/'artifacts/assets/manifest.json')
    enriched=[]
    for claim in claims:
        asset=assets['models'].get(claim.model_id,{})
        source=claim.source_url
        if claim.model_id.startswith('samsone'):source=source.replace('/blob/main/','/blob/'+assets['sources']['samsone']['revision']+'/')
        enriched.append(dataclasses.replace(claim,source_url=source,checkpoint_sha256=asset.get('sha256') if isinstance(asset.get('sha256'),str) else None,reason=claim.reason or 'Published benchmark scoring not executed under user-selected minimal smoke scope. Native inference is reported separately.'))
    write_claims(ROOT/'artifacts/reproduction/claims.jsonl',enriched)
    return {'status':'partial','claims':len(claims),'reason':'Native smoke outputs are separate; benchmark assets/scorers require explicit acquisition. No benchmark score inferred.'}

def report(args):
    from .reporting.report import build_report
    return build_report(args.profile)

def run_study(args):
    stages=[('inspect-env',lambda: __import__('nano_audio.environment',fromlist=['inspect']).inspect(ROOT/'artifacts/environment.json')),('download',lambda:__import__('nano_audio.assets',fromlist=['download']).download(args.profile)),('smoke-text',lambda:smoke_text(args)),('smoke-audio',lambda:smoke_audio(args)),('prepare-data',lambda:prepare(args)),('audit-data',lambda:audit(args)),('reproduce',lambda:reproduce(args))]
    if args.profile!='smoke':stages.append(('overfit',lambda:overfit(args)))
    runs=['S_P0','S_P1','S_P_SFT','S_P_GRPO','M_P0','M_P1','M_P_SFT','M_P_GRPO','F_TEXT','F_ALIGN','F_AUDIO','F_SFT','F_GRPO']
    for seed in ([17,42,123] if args.profile=='main' else [args.seed]):
        for run in runs:
            # Seed variation is conditional on one common family initialization.
            if args.profile=='main' and seed!=17 and run not in ['S_P_SFT','S_P_GRPO','M_P_SFT','M_P_GRPO','F_SFT','F_GRPO']:continue
            ns=argparse.Namespace(**vars(args));ns.run=run;ns.seed=seed;stages.append((run,lambda ns=ns:train(ns)))
    states=[]
    for name,fn in stages:
        start=time.time()
        try:out=fn();state={'stage':name,'status':'completed','output':out}
        except Exception as exc:state={'stage':name,'status':'blocked','error':repr(exc),'traceback':traceback.format_exc(),'next_command':f'nano-audio '+(f'train --run {name}' if name in runs else name)+f' --profile {args.profile}'}
        state['elapsed_seconds']=time.time()-start;states.append(state);write_json(ROOT/f'artifacts/stages/{args.profile}.json',states);print(json.dumps({'stage':name,'status':state['status']}),flush=True)
        gc.collect()
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
    evaluate(args);report(args);return {'stages':[{k:s.get(k) for k in ['stage','status','error']} for s in states]}

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    commands=['inspect-env','resolve-models','download','smoke-text','smoke-audio','prepare-data','audit-data','reproduce','train','evaluate','run-study','report','overfit','audit-external']
    for command in commands:
        p=sub.add_parser(command);p.add_argument('--profile',choices=['smoke','local_pilot','main'],default='smoke' if command!='run-study' else 'local_pilot');p.add_argument('--device',default='auto');p.add_argument('--seed',type=int,default=17);p.add_argument('--model',default='samsone_134m_published' if command!='smoke-text' else 'smollm2_135m_base');p.add_argument('--offline',action='store_true');p.add_argument('--output',default='artifacts/environment.json');p.add_argument('--config',default='configs/experiments/study.yaml');p.add_argument('--run',default='S_P1');p.add_argument('--updates',type=int);p.add_argument('--suite',default='research');p.add_argument('--models',nargs='+',default=['samsone_134m_published','mellow_v0_published']);p.add_argument('--input');p.add_argument('--catalog')
    args=parser.parse_args();os.chdir(ROOT);seeded(args.seed)
    try:
        if args.command=='inspect-env':from .environment import inspect;out=inspect(args.output)
        elif args.command=='resolve-models':from .assets import resolve;out=resolve(offline=args.offline)
        elif args.command=='download':from .assets import download;out=download(args.profile,offline=args.offline)
        elif args.command=='audit-external':
            from .evaluation.external import audit_audio2tool
            if not args.input or not args.catalog:raise ValueError('--input and --catalog are required')
            out=audit_audio2tool(args.input,args.catalog,args.output)
        else:out={'smoke-text':smoke_text,'smoke-audio':smoke_audio,'prepare-data':prepare,'audit-data':audit,'train':train,'reproduce':reproduce,'evaluate':evaluate,'report':report,'run-study':run_study,'overfit':overfit}[args.command](args)
        print(json.dumps(out,indent=2,default=str))
    except Exception as exc:
        print(json.dumps({'status':'failed','command':args.command,'error':str(exc),'type':type(exc).__name__}),file=sys.stderr);traceback.print_exc();raise SystemExit(1)
if __name__=='__main__':main()
