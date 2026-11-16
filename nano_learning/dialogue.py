"""Full chronological history + final-user audio, with public role boundaries."""
from __future__ import annotations
import json
from pathlib import Path
import soundfile as sf
import numpy as np
from .common import STUDY,write,rows,digest,sha
from .task import compact,Registry

def adapt(raw,paths):
    if len(paths)<2 or raw['chat_history'][-1]['role']!='user':raise ValueError('Invalid dialogue turn structure')
    arrays=[];boundaries=[];offset=0
    for turn,path in zip(raw['chat_history'][:-1],paths[:-1]):
        audio,sr=sf.read(path,dtype='float32',always_2d=True)
        audio=audio.mean(1)
        if sr!=16000:
            from scipy.signal import resample_poly
            from math import gcd
            divisor=gcd(sr,16000)
            audio=resample_poly(audio,16000//divisor,sr//divisor).astype('float32')
        arrays.append(audio)
        boundaries.append([turn['role'],round(offset/16000,6),round((offset+len(audio))/16000,6)])
        offset+=len(audio)
    key=digest({'source_sha256':[sha(p) for p in paths[:-1]],'concatenation':'chronological-mono-polyphase16k-no-gap-v1'})
    out=STUDY/'derived_dialogue'/f'{key}.wav';out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():sf.write(out,np.concatenate(arrays),16000,subtype='FLOAT')
    prompt=(' audio segment 1 is the complete chronological history. its turn boundaries in seconds are '+compact(boundaries)+
        '. audio segment 2 is the final user request. use the complete history to resolve it. answer: ')
    return [str(out),str(paths[-1])],prompt

def verify_adapters():
    from .acquire import download
    from .common import tracking
    from .policies import load_policy
    from .accounting import GPULedger
    import torch,gc
    reg=Registry(STUDY/'source_repo/tools_registry.csv')
    candidates=[r for r in rows(STUDY/'source_rows.jsonl') if r['tier']==7 and r['calls'] and
        sum(t['duration'] for t in r['raw']['chat_history'][:-1])<=10 and r['raw']['chat_history'][-1]['duration']<=10]
    row=min(candidates,key=lambda r:r['id'])
    designation=STUDY/'engineering_panel.json';panel=json.loads(designation.read_text())
    if row['request_group'] not in panel['request_groups']:
        panel['request_groups'].append(row['request_group']);write(designation,panel)
    # Explicit training designation precedes download/inference.
    paths=[download(remote) for remote in row['audio_remotes']]
    paths,role_prompt=adapt(row['raw'],paths)
    ex={'id':row['id'],'domain':row['domain'],'calls':row['calls'],'audio_paths':paths,'audio_path':paths[0],
        '_prompt':reg.prompt(row['domain'])+role_prompt,'target_text':compact({'calls':row['calls']}),'completion_cap':512}
    for model in ('samsone','mellow'):
        run=tracking(model+'-dialogue-engineering','engineering',{'training_only':True,'full_history':True})
        ledger=GPULedger();ledger.begin('engineering-dialogue-'+model)
        try:
            p=load_policy(model)
            ids=p.encode(ex['target_text'])+[p.eos_token_id]
            loss=-p.completion_logprobs(ex,ids).mean();loss.backward()
            if not all(torch.isfinite(x.grad).all() for x in p.model.parameters() if x.grad is not None):
                raise FloatingPointError('Dialogue gradient nonfinite')
            with torch.no_grad(),p._autocast():prefix=p.build_prefix(ex)
            profile_path=STUDY/'interface_profile.json';profiles=json.loads(profile_path.read_text())
            profiles[model]['dialogue_adapter_verified']=True
            profiles[model]['dialogue_check']={'training_request_group':row['request_group'],
                'source_turn_hashes':[sha(path) for path in paths],'prefix_tokens':prefix.shape[1],
                'history_seconds':sf.info(paths[0]).duration,'final_seconds':sf.info(paths[1]).duration,
                'roles_and_boundaries':role_prompt,'finite_backward':True}
            write(profile_path,profiles);run.log({'engineering/dialogue_finite_backward':1})
            del p,prefix;gc.collect();torch.cuda.empty_cache()
        finally:ledger.end();run.finish()

if __name__=='__main__':verify_adapters()
