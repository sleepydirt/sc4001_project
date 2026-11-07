"""One-hour, exact-reward development attempt. No final test access."""
import argparse,copy,datetime,gc,json,random,signal,time
from pathlib import Path
import torch
from nano_audio.cli import seeded,records,scoring,load_policy,families
from nano_audio.training.core import Trainer,TrainingConfig
from nano_audio.utils import ROOT,write_json,append_json,sha

OUT=ROOT/'artifacts/grpo_attempt_20261107/strong_init'
DEADLINE=datetime.datetime(2026,11,7,9,22,35,tzinfo=datetime.timezone.utc).timestamp()
START=time.time();STOP=False

def stop(sig,frame):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
def remaining():return DEADLINE-time.time()
def emit(obj):
    obj={'wall_elapsed_seconds':round(time.time()-START,2),'remaining_seconds':round(remaining(),2),**obj}
    append_json(OUT/'events.jsonl',obj);print(json.dumps(obj),flush=True)
def generated(policy,ex,phase,sample=False):
    output=policy.generate(ex,{'max_new_tokens':128,'do_sample':sample})
    score=scoring(ex,output)
    row={'phase':phase,'id':ex['id'],'group':ex['utterance_group_id'],'transcript_for_log_only':ex['transcript'],'gold_for_log_only':ex['target'],'output':output,'score':score}
    append_json(OUT/'generations.jsonl',row)
    with (OUT/'generations.log').open('a') as f:
        f.write(f'\nPHASE {phase} | {ex["id"]} | reward={score["reward"]} valid={score["valid"]}\nREQUEST: {ex["transcript"]}\nGOLD: {json.dumps(ex["target"])}\nGENERATION: {output["text"]}\n')
    return score

def evaluate(policy,phase):
    counts={}
    for label,pool in [('train',TRAIN_PROBES),('dev_id',DEV_ID),('dev_ood',DEV_OOD)]:
        if STOP or remaining()<120:break
        scores=[generated(policy,ex,phase+'/'+label) for ex in pool]
        counts[label]={'n':len(scores),'exact':sum(r['reward'] for r in scores)/len(scores),'valid':sum(r['valid'] for r in scores)/len(scores)}
    emit({'event':'evaluation','phase':phase,'scores':counts});return counts

def gate(policy,phase):
    mixed=0;correct=0
    for ex in TRAIN_PROBES:
        if STOP or remaining()<180:break
        rewards=[generated(policy,ex,phase,True)['reward'] for _ in range(4)]
        correct+=sum(rewards);mixed+=int(0<sum(rewards)<4)
    emit({'event':'gate','phase':phase,'correct':correct,'mixed_groups':mixed})
    return mixed>0

def config(objective,updates):
    return TrainingConfig.for_profile('local_pilot',objective,total_updates=updates,precision='fp16',effective_batch_size=4)
def batch(rng):return [copy.deepcopy(rng.choice(POOL[g])) for g in rng.sample(list(POOL),4)]
def save(trainer,label):
    trainer.save_checkpoint(OUT/label);emit({'event':'checkpoint','label':label,'updates':trainer.step})

seeded(17);OUT.mkdir(parents=True,exist_ok=True)
train=records('smoke','train');dev=records('smoke','dev')
POOL={}
for ex in train:POOL.setdefault(ex['utterance_group_id'],[]).append(ex)
TRAIN_PROBES=[v[0] for v in list(POOL.values())[:8]]
# All eight operations get checked; four fixed held-out dev utterances.
unique={}
for ex in dev:unique.setdefault(ex['utterance_group_id'],ex)
DEV_OOD=list(unique.values())[:4]
f=families()['train_01'];DEV_ID=[]
for ex in DEV_OOD:
    row=copy.deepcopy(ex);row.update(id=ex['id']+'_id',schema_family_id=f.family_id,tools=f.tools,target=f.encode(ex['canonical_target']))
    DEV_ID.append(row)
write_json(OUT/'protocol.json',{'deadline_utc':'2026-11-07T09:22:35Z','parent':'first_attempt/sft_control_checkpoint/policy; no RL weights inherited','start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':17,'source':'constructed_only','model':'samsone_134m_published','data_sha256':sha(ROOT/'artifacts/data/smoke/examples.jsonl'),'reward':'strict complete-program 0/1; no formatting bonus','gate':'At least7/8 exact greedy training probes and one mixed sampled reward group','warmup_max_updates':256,'continuation_max_updates':64,'effective_batch':4,'G':4,'temperature':1,'train_groups':list(POOL),'dev_groups':[x['utterance_group_id'] for x in DEV_OOD],'test_access':False,'selection':'development diagnostics only; report every evaluation, not only best','shared_parent':'common/checkpoint','sampling':'same deterministic prompt schedule for both continuation branches'})
p=load_policy('samsone_134m_published','cuda');p.load_checkpoint(ROOT/'artifacts/grpo_attempt_20261107/sft_control_checkpoint/policy')
current=None;result={}
try:
    result['initial']=evaluate(p,'initial')
    warm=Trainer(p,config('sft',256),ledger_path=OUT/'warmup.jsonl',metadata={'phase':'shared_sft','data':sha(ROOT/'artifacts/data/smoke/examples.jsonl')});current=warm
    rng=random.Random(1701);ready=False
    for _ in range(256):
        if STOP or remaining()<1500:break
        metrics=warm.sft_step(batch(rng));emit({'event':'update','phase':'shared_sft',**metrics})
        if warm.step%32==0:
            result['warmup_'+str(warm.step)]=evaluate(p,'warmup_'+str(warm.step))
            save(warm,'warmup_checkpoint')
            if gate(p,'gate_'+str(warm.step)) and result['warmup_'+str(warm.step)]['train']['exact']>=7/8:
                ready=True;break
    save(warm,'common_checkpoint');result['warmup_updates']=warm.step;result['mixed_reward_gate']=ready
    result['common']=evaluate(p,'common')
    common_path=OUT/'common_checkpoint/policy'
    # Fixed parent, reference and eligible pool, regardless of gate result.
    del warm;current=None;gc.collect();torch.cuda.empty_cache()
    rl=Trainer(p,config('grpo',64),reward_fn=lambda ex,out:scoring(ex,out)['reward'],ledger_path=OUT/'grpo.jsonl',metadata={'phase':'grpo','parent_sha256':sha(common_path)});current=rl
    rng=random.Random(1702);rl_metrics=[]
    for _ in range(64):
        if STOP or remaining()<420:break
        examples=batch(rng);metrics=rl.grpo_step(examples);rl_metrics.append(metrics)
        for ex,group in zip(examples,rl.last_rollouts):
            for row in group:
                record={k:(v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k,v in row.items()}
                append_json(OUT/'rollouts.jsonl',{'step':rl.step,'example_id':ex['id'],**record})
                out=row['rollout'];score=scoring(ex,out)
                append_json(OUT/'generations.jsonl',{'phase':'grpo','step':rl.step,'id':ex['id'],'transcript_for_log_only':ex['transcript'],'gold_for_log_only':ex['target'],'output':out,'score':score})
                with (OUT/'generations.log').open('a') as f:
                    f.write(f'\nGRPO UPDATE {rl.step} | reward={row["reward"]} | valid={score["valid"]}\nREQUEST: {ex["transcript"]}\nGOLD: {json.dumps(ex["target"])}\nGENERATION: {out["text"]}\n')
        emit({'event':'update','phase':'grpo',**metrics})
        if rl.step%8==0:
            result['grpo_'+str(rl.step)]=evaluate(p,'grpo_'+str(rl.step));save(rl,'grpo_checkpoint')
    save(rl,'grpo_checkpoint');result['grpo_final']=evaluate(p,'grpo_final');steps=rl.step
    result['grpo_updates']=steps;result['reward_bearing_updates']=sum(m['reward_mean']>0 for m in rl_metrics);result['mixed_reward_updates']=sum(m['equal_reward_group_rate']<1 for m in rl_metrics)
    del rl;current=None;gc.collect();torch.cuda.empty_cache()
    # Exposure-matched SFT control from exactly the same common initialization.
    p.load_checkpoint(common_path);control=Trainer(p,config('sft',64),ledger_path=OUT/'sft_control.jsonl',metadata={'phase':'continued_sft','parent_sha256':sha(common_path)});current=control;rng=random.Random(1702)
    for _ in range(steps):
        if STOP or remaining()<100:break
        metrics=control.sft_step(batch(rng));emit({'event':'update','phase':'sft_control',**metrics})
    save(control,'sft_control_checkpoint');result['sft_control_updates']=control.step;result['sft_control_final']=evaluate(p,'sft_control_final')
    result['status']='completed' if not STOP else 'wall_limit'
except Exception as e:
    import traceback
    result.update(status='failed',error=repr(e),traceback=traceback.format_exc());emit({'event':'failure',**result})
finally:
    if current is not None:
        try:save(current,'last_checkpoint')
        except Exception as e:result['checkpoint_error']=repr(e)
    result.update(elapsed_seconds=time.time()-START,remaining_seconds=remaining(),scope='single-seed constructed-speech development attempt; no final test')
    write_json(OUT/'result.json',result);emit({'event':'finished',**result})
