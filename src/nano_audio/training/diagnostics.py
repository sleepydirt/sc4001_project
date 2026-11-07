"""Train-only eight-example overfit diagnostic, isolated from paired experiment."""
from pathlib import Path
import json,random
from .core import Trainer,TrainingConfig
from nano_audio.utils import append_json,write_json

def overfit_eight(policy,records,reward_fn,output,profile='local_pilot',cap=200,seed=17):
    if cap<1:raise ValueError('Overfit diagnostic needs a positive update cap')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    groups={}
    for ex in records:
        if ex.get('split',ex.get('base_split'))!='train':raise ValueError('Overfit diagnostic accepts training examples only')
        groups.setdefault(ex['utterance_group_id'],ex)
    pool=list(groups.values())[:8]
    if len(pool)!=8:raise ValueError('Need eight independent training groups')
    config=TrainingConfig.for_profile(profile,total_updates=cap,effective_batch_size=1,precision=getattr(policy,'research_precision','fp16' if policy.device.type=='cuda' else 'fp32'))
    trainer=Trainer(policy,config,ledger_path=output/'learning.jsonl',metadata={'diagnostic':'eight_train_groups','seed':seed,'ids':[e['id'] for e in pool]})
    if (output/'checkpoint/resume.pt').exists():trainer.load_checkpoint(output/'checkpoint')
    curve=[]
    while trainer.step<cap:
        trainer.sft_step([pool[trainer.step%8]])
        if trainer.step%8==0 or trainer.step==cap:
            predictions=[]
            for ex in pool:
                out=policy.generate(ex,{'max_new_tokens':128});predictions.append({'id':ex['id'],'output':out,'exact':bool(reward_fn(ex,out))})
            row={'update':trainer.step,'exact_successes':sum(p['exact'] for p in predictions),'n':8,'predictions':predictions}
            append_json(output/'curve.jsonl',row);curve.append(row);trainer.save_checkpoint(output/'checkpoint')
            if row['exact_successes']>=7:break
    result={'status':'completed','diagnostic_only':True,'updates':trainer.step,'criterion_reached':curve[-1]['exact_successes']>=7,'last':curve[-1]}
    write_json(output/'stage.json',result);return result
