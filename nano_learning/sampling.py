"""Validated left-padded generation, with a single frozen completion cap."""
import torch

def generate_batch(policy,examples,config=None,*,force_batch=False):
    cap=examples[0].get('completion_cap',policy.completion_cap)
    if any(e.get('completion_cap',policy.completion_cap)!=cap for e in examples):
        raise ValueError('One frozen completion cap required per generation batch')
    config={'max_new_tokens':cap,'do_sample':False,**(config or {})}
    if config['max_new_tokens']>cap:raise ValueError('Generation exceeds frozen completion cap')
    if config['do_sample']:config.update(top_k=0,top_p=1.,temperature=1.)
    if not force_batch and not getattr(policy,'batching_enabled',False):
        return [policy._serial_generate(ex,config) for ex in examples]
    policy.model.eval()
    with torch.no_grad(),policy._autocast():
        prefixes=[policy.build_prefix(ex) for ex in examples];length=max(p.shape[1] for p in prefixes)
        policy._check_context(length+config['max_new_tokens'])
        batch=torch.zeros((len(examples),length,prefixes[0].shape[-1]),device=policy.device,dtype=prefixes[0].dtype)
        mask=torch.zeros((len(examples),length),device=policy.device,dtype=torch.long)
        for i,prefix in enumerate(prefixes):batch[i,-prefix.shape[1]:]=prefix[0];mask[i,-prefix.shape[1]:]=1
        sequences=policy.decoder.generate(inputs_embeds=batch,attention_mask=mask,
            eos_token_id=policy.eos_token_id,pad_token_id=policy.eos_token_id,**config).tolist()
    result=[]
    for ids in sequences:
        eos=policy.eos_token_id in ids
        if eos:ids=ids[:ids.index(policy.eos_token_id)+1]
        result.append({'ids':ids,'text':policy.decode(ids),'eos':eos,
            'length_capped':not eos and len(ids)>=config['max_new_tokens']})
    return result
