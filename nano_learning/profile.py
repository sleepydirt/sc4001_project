"""Training-designated real audio checks before allocating long learning runs."""
from __future__ import annotations
import argparse
import copy
import gc
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch
from .common import ROOT,STUDY,rows,write,digest,sha,tracking
from .task import Registry,compact,score
from .policies import load_policy
from .accounting import GPULedger
from nano_audio.training.core import Trainer,TrainingConfig,capture_rng_state,restore_rng_state,_identity

def training_examples():
    path=STUDY/'engineering_panel.json'
    if path.exists():return json.loads(path.read_text())['examples']
    reg=Registry(STUDY/'source_repo/tools_registry.csv')
    allrows=rows(STUDY/'source_rows.jsonl')
    available={r['remote']:r for r in rows(STUDY/'file_inventory.jsonl') if r['ok']}
    examples=[]
    for d in ('wearables','smart_home','smart_car'):
        rr=[r for r in allrows if r['domain']==d and r['tier']==1 and r['calls'] and all(p in available for p in r['audio_remotes'])]
        rr.sort(key=lambda r:r['id'])
        if not rr:raise RuntimeError('Need verified training-only audio in every domain before profiling')
        for r in rr[:2]:
            examples.append({'id':r['id'],'request_group':r['request_group'],'domain':d,
                'audio_path':available[r['audio_remotes'][0]]['path'],'calls':r['calls'],
                'target_text':compact({'calls':r['calls']}),'_prompt':reg.prompt(d),'completion_cap':512})
    write(path,{'designation':'MUST BE TRAIN ONLY; frozen before any GPU prediction',
        'request_groups':sorted({x['request_group'] for x in examples}),'examples':examples})
    return examples

def main(model):
    torch.set_num_threads(4)
    torch.manual_seed(17)
    examples=training_examples()
    run=tracking(model+'-engineering','engineering',{'model':model,'seed':17,'training_only':True})
    ledger=GPULedger()
    # Starts BEFORE loading: all GRPO prerequisite GPU work is charged.
    ledger.begin('engineering-'+model)
    start=time.perf_counter()
    try:
        policy=load_policy(model)
        config=TrainingConfig.for_profile('local_pilot','sft',total_updates=4,
            precision='fp16' if model=='samsone' else 'bf16',max_new_tokens=512)
        # Near-limit gradients on every complete catalog, trainable projection active.
        checks=[]
        for ex in examples[::2]:
            ledger.reserve_unit(100)
            torch.cuda.reset_peak_memory_stats()
            ids=policy.encode(ex['target_text'])+[policy.eos_token_id]
            with policy._autocast():prefix=policy.build_prefix(ex)
            prefix_length=prefix.shape[1]
            with torch.no_grad(),policy._autocast():
                tokens=torch.tensor([ids],device=policy.device)
                inputs=torch.cat((prefix,policy.decoder.get_input_embeddings()(tokens)),dim=1)
                full=policy.decoder(inputs_embeds=inputs,attention_mask=torch.ones(inputs.shape[:2],device=policy.device,dtype=torch.long),use_cache=False).logits
                expected=full[:,prefix_length-1:prefix_length+len(ids)-1].float().log_softmax(-1).gather(-1,tokens.unsqueeze(-1)).squeeze(-1)
                actual=policy.completion_logprobs(ex,ids)
                error=float((expected-actual).abs().max())
                if error>1e-3:raise ValueError('Selective/full likelihood mismatch')
            del full,inputs,prefix,expected,actual
            filler=policy.encode(' ')[0]
            near=ids[:-1]+[filler]*(511-len(ids[:-1]))+[policy.eos_token_id]
            before=time.perf_counter();loss=-policy.completion_logprobs(ex,near).mean();loss.backward()
            gradients=[p.grad for p in policy.model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise FloatingPointError('Near-limit backward nonfinite')
            checks.append({'domain':ex['domain'],'prefix_tokens':prefix_length,'completion_tokens':len(near),
                'near_limit_backward_seconds':time.perf_counter()-before,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
                'full_likelihood_max_error':error,'finite_gradients':True})
            policy.model.zero_grad(set_to_none=True)
            print(model,'backward check',checks[-1],flush=True)
        reference=copy.deepcopy(policy)
        reference.model.requires_grad_(False);reference.model.eval()
        identity=_identity(reference.model)
        ex=examples[0];ids=policy.encode(ex['target_text'])+[policy.eos_token_id]
        with torch.no_grad():
            old=policy.completion_logprobs(ex,ids);ref=reference.completion_logprobs(ex,ids)
            if not torch.equal(old,ref):raise ValueError('Complete reference likelihood differs at initialization')
        trainer=Trainer(policy,config,metadata={'engineering_version':1})
        training_metrics=trainer.sft_step([examples[0]])
        if _identity(reference.model)!=identity:raise ValueError('Reference mutated')
        out=STUDY/'engineering'/model
        trainer.save_checkpoint(out/'resume_check')
        expected_rng=capture_rng_state()
        with torch.no_grad():expected=policy.completion_logprobs(ex,ids).detach().clone()
        trainer.sft_step([examples[1]])
        trainer.load_checkpoint(out/'resume_check')
        with torch.no_grad():actual=policy.completion_logprobs(ex,ids)
        if not torch.equal(expected,actual):raise ValueError('Save/load did not restore exact likelihood')
        actual_rng=capture_rng_state()
        if not torch.equal(expected_rng['torch'],actual_rng['torch']):raise ValueError('RNG continuity failed')
        # Validate batched decoding independently on each family before activation.
        from .sampling import generate_batch
        short={**ex,'completion_cap':512}
        serial=policy.generate(short,{'do_sample':False,'max_new_tokens':16})
        batched=generate_batch(policy,[short,short],{'do_sample':False,'max_new_tokens':16},force_batch=True)
        if any(x['ids']!=serial['ids'] for x in batched):raise ValueError('Batch generation differs from serial')
        # One complete GRPO update exercises G4 generation, old/reference, backward.
        del trainer,reference;gc.collect();torch.cuda.empty_cache()
        grpo_config=TrainingConfig.for_profile('local_pilot','grpo',total_updates=1,
            precision=config.precision,max_new_tokens=512,reward_mode='verifiable_partial')
        reg=Registry(STUDY/'source_repo/tools_registry.csv')
        trainer=Trainer(policy,grpo_config,reward_fn=lambda e,o:score(reg,e,o)['typed'])
        original=policy.generate;queue=[]
        def grouped(e,c=None):
            if not (c or {}).get('do_sample'):return original(e,c)
            if not queue:queue.extend(generate_batch(policy,[e]*4,c))
            return queue.pop(0)
        policy.generate=grouped
        ledger.reserve_unit(300)
        grpo_metrics=trainer.grpo_step([ex])
        policy.generate=original
        profile={'model':model,'native_context_cap':policy.decoder.config.max_position_embeddings,
            'effective_context_cap':policy.context_cap,'completion_cap':512,
            'native_clip_seconds':30 if model=='samsone' else 10,
            'audio_prefix_tokens':checks[0]['prefix_tokens']-len(policy.encode(ex['_prompt'])),
            'dialogue_adapter_verified':False,'checks':checks,'reference_frozen_independent':True,
            'save_load_likelihood_and_rng_continuity':True,'batch_runtime_verified':True,
            'sft_metrics':training_metrics,'grpo_check_metrics':grpo_metrics,
            'reserved_gpu_seconds':time.perf_counter()-start,'parameter_manifest':policy.parameter_manifest()}
        path=STUDY/'interface_profile.json'
        profiles=json.loads(path.read_text()) if path.exists() else {}
        profiles[model]=profile;write(path,profiles)
        run.log({'engineering/finite_backward':1,'engineering/resume_passed':1,
            'engineering/batch_passed':1,'engineering/seconds':profile['reserved_gpu_seconds'],
            **{'engineering/grpo_'+k:v for k,v in grpo_metrics.items() if type(v) in (int,float)}})
        print('Profile complete',model,flush=True)
    finally:
        ledger.end();run.finish()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['samsone','mellow'],required=True);a=p.parse_args();main(a.model)
