"""Check batched positions/probabilities with declared mixed-precision tolerances."""
import gc
import json
import torch
from .common import STUDY,write,tracking
from .profile import training_examples
from .policies import load_policy
from .accounting import GPULedger

def main():
    torch.set_num_threads(4);examples=training_examples();results={}
    for model in ('samsone','mellow'):
        run=tracking(model+'-batch-likelihood','engineering',{'training_only':True,
            'max_mean_vocabulary_KL':.001,'max_chosen_token_logprob_difference':.05 if model=='samsone' else .25})
        ledger=GPULedger();ledger.begin('engineering-batch-likelihood-'+model)
        try:
            p=load_policy(model);chosen=[examples[0],examples[2],examples[4],examples[4]]
            with torch.no_grad(),p._autocast():
                prefixes=[p.build_prefix(e) for e in chosen]
                ids=[torch.tensor(p.encode(e['target_text'])+[p.eos_token_id],device=p.device) for e in chosen]
                sequences=[torch.cat((prefix,p.decoder.get_input_embeddings()(t[None])),dim=1) for prefix,t in zip(prefixes,ids)]
                maximum=max(x.shape[1] for x in sequences)
                batch=torch.zeros((len(chosen),maximum,sequences[0].shape[-1]),dtype=sequences[0].dtype,device=p.device)
                mask=torch.zeros((len(chosen),maximum),dtype=torch.long,device=p.device)
                for i,x in enumerate(sequences):batch[i,-x.shape[1]:]=x[0];mask[i,-x.shape[1]:]=1
                positions=(mask.cumsum(-1)-1).clamp(min=0)
                hidden=p.decoder.model(inputs_embeds=batch,attention_mask=mask,position_ids=positions,use_cache=False).last_hidden_state
                checks=[]
                for i,(e,seq,tokens) in enumerate(zip(chosen,sequences,ids)):
                    n=len(tokens)
                    batched=p.decoder.lm_head(hidden[i:i+1,-n-1:-1]).float().log_softmax(-1)
                    states=p.decoder.model(inputs_embeds=seq,attention_mask=torch.ones(seq.shape[:2],dtype=torch.long,device=p.device),use_cache=False).last_hidden_state
                    serial=p.decoder.lm_head(states[:,-n-1:-1]).float().log_softmax(-1)
                    kl=float((serial.exp()*(serial-batched)).sum(-1).mean().clamp(min=0))
                    error=float((serial-batched).gather(-1,tokens[None,:,None]).abs().max())
                    checks.append({'id':e['id'],'domain':e['domain'],'vocabulary_mean_KL':kl,
                        'chosen_token_max_logprob_error':error,'finite':bool(torch.isfinite(batched).all())})
                    if kl>.001 or error>(.05 if model=='samsone' else .25) or not torch.isfinite(batched).all():
                        raise ValueError('Batched distribution/position check exceeds predeclared precision tolerance: '+str(checks[-1]))
            results[model]={'checks':checks,'passed':True,'greedy_token_identity_claim':False,
                'evaluation_protocol':'fixed batches up to four, grouped by public catalog and prefix length; unconstrained greedy',
                'rollout_protocol':'serial unconstrained T1/top_p1/top_k0; same complete old/reference likelihood scoring'}
            write(STUDY/'batch_probability_validation.json',results)
            run.log({'engineering/maximum_mean_vocabulary_KL':max(x['vocabulary_mean_KL'] for x in checks),
                'engineering/maximum_logprob_difference':max(x['chosen_token_max_logprob_error'] for x in checks)})
            del p,prefixes,sequences,batch,hidden,serial,batched,states;gc.collect();torch.cuda.empty_cache()
        finally:
            try:run.finish()
            finally:ledger.end()
    write(STUDY/'decoding_protocol.json',{'version':'fixed-eval-b4-serial-rollout-v1',
        'evaluation_batch_size':4,'evaluation_greedy':True,'rollout_batch_size':1,'group_size':4,
        'bitwise_native_serial_equivalence':False,'numerical_validation':results,
        'decision_basis':'Serial full-validation time would dominate overnight execution; one fixed batched protocol is used for every compared endpoint before any learning/checkpoint selection or final-test inference.'})

if __name__=='__main__':main()
