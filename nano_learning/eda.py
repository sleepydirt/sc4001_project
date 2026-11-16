"""Complete release audit, request-provenance graph and guarded grouped split."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .common import ARTIFACTS, ROOT, STUDY, REVISION, rows, write, sha, digest, tracking
from .task import Registry, GoldError, compact, typed

def jsonl(path, values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for value in values:
            f.write(json.dumps(value,allow_nan=False)+'\n')
    tmp.replace(path)

def normalized(text):
    return ' '.join(re.findall(r'\w+',text.casefold()))

def clean(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list): return [clean(v) for v in value]
    return value

def registry():
    return Registry(STUDY/'source_repo/tools_registry.csv')

def metadata():
    reg=registry();source=STUDY/'source_repo'
    files=json.loads((STUDY/'remote_files.json').read_text())
    fileset=set(files)
    allrows=[];issues=[];inventory=[]
    for remote in files:
        path=source/remote
        if not remote.endswith(('.json','.csv','.md')): continue
        if not path.exists(): raise FileNotFoundError(path)
        inventory.append({'remote':remote,'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
    jsonl(STUDY/'metadata_inventory_complete.jsonl',inventory)
    for entry in inventory:
        if '_data/' not in entry['remote']: continue
        for raw in json.loads(Path(entry['path']).read_text()):
            raw=clean(raw);tier=int(re.search(r'tier(\d+)',raw['tier'])[1])
            key=f"t{tier}:{raw['domain']}:{raw['query_idx']}"
            rid=f"t{tier}:row{raw['id']}"
            paths=([x['path'] for x in raw['chat_history']] if tier==7 else [raw['path']])
            remotes=['public/'+p.removeprefix('public/') for p in paths]
            flags=['gold:'+x for x in reg.audit_flags(raw)]
            if tier==7:
                history=raw['chat_history']
                if len(history)<2 or history[-1]['role']!='user':flags.append('dialogue_final_turn_not_user')
                elif normalized(history[-1]['content'])!=normalized(raw['user_request']):flags.append('dialogue_final_request_text_disagrees')
            if any(p not in fileset for p in remotes): flags.append('audio_reference_missing_from_release')
            try: calls=reg.gold(raw)
            except GoldError as e:
                flag='gold:'+re.sub(r'0x[0-9a-f]+','address',str(e))
                if not flags:flags.append(flag)
                calls=None
            if flags:calls=None
            item={'id':rid,'request_group':key,'tier':tier,'domain':raw['domain'],
                'query_idx':raw['query_idx'],'source_metadata':entry['remote'],'source_row_id':raw['id'],
                'transcript':raw['query'],'normalized_request':normalized(raw['query']),
                'speaker':raw.get('speaker_id') or raw.get('user_speaker_id') or 'unavailable',
                'speakers':([raw.get('user_speaker_id'),raw.get('agent_speaker_id')] if tier==7 else [raw.get('speaker_id')]),
                'speaker_source':raw.get('speaker_source') or raw.get('user_speaker_source'),
                'source_endpoint':raw.get('source_endpoint'),'audio_remotes':remotes,
                'calls':calls,'order_matters':False,'flags':flags,'raw':raw}
            allrows.append(item)
            if flags: issues.append({'id':rid,'request_group':key,'flags':flags,
                'expected_tool_call':raw['expected_tool_call'],'extracted_params':raw.get('extracted_params'),
                'additional_tool_calls':raw.get('additional_tool_calls')})
    linked=defaultdict(list)
    for r in allrows:linked[r['request_group']].append(r)
    discordant=[]
    for key,rr in linked.items():
        texts={r['normalized_request'] for r in rr};targets={digest(r['calls']) for r in rr if r['calls']}
        if len(texts)>1 or len(targets)>1:
            discordant.append({'request_group':key,'texts':len(texts),'valid_targets':len(targets),'ids':[r['id'] for r in rr]})
    write(STUDY/'linked_recording_consistency.json',{'groups':len(linked),'discordant_groups':discordant,
        'group_size_distribution':dict(Counter(str(len(v)) for v in linked.values()))})
    if discordant:raise ValueError('Linked recordings have unresolved text/target disagreement; adjudicate before freezing')
    jsonl(STUDY/'source_rows.jsonl',allrows);jsonl(STUDY/'label_issues.jsonl',issues)
    defaults=[]
    for (domain,name),tool in reg.tools.items():
        for k,spec in tool['parameters'].items():
            if 'default' in spec and not reg.validate_value(spec,spec['default']):
                defaults.append({'domain':domain,'tool':name,'parameter':k,'spec':spec})
    write(STUDY/'registry_audit.json',{'version':'public-registry-v1','hash':reg.hash,
        'tools':len(reg.tools),'domains':dict(Counter(d for d,n in reg.tools)),
        'invalid_or_unsupported_defaults':defaults,'unknown_custom_types':['Goal'],
        'all_aliases_bijective':True,'gold_adjudication':'none; prediction-independent exclusions',
        'call_order':'unordered bijective multiset; public release does not annotate execution dependencies',
        'default_semantics':'registry default insertion; no speech-dependent additional argument invention'})
    print('Source audit:',len(allrows),'rows; label-supported',sum(r['calls'] is not None for r in allrows),flush=True)
    return allrows

def verify_local(remote):
    import soundfile as sf
    path=STUDY/'source_repo'/remote
    try:
        if path.stat().st_size < 200 and path.read_bytes().startswith(b'version https://git-lfs'):
            raise ValueError('LFS pointer still present')
        audio,sr=sf.read(path,dtype='float32',always_2d=True)
        if not len(audio) or not np.isfinite(audio).all(): raise ValueError('Empty/nonfinite waveform')
        actual=sha(path)
        if expected_oids and actual!=expected_oids[remote]:raise ValueError('Pinned LFS SHA-256 mismatch')
        return {'remote':remote,'path':str(path),'sha256':actual,'pinned_lfs_sha256':expected_oids.get(remote),'bytes':path.stat().st_size,
            'duration':len(audio)/sr,'frames':len(audio),'sample_rate':sr,'channels':audio.shape[1],
            'peak':float(np.abs(audio).max()),'rms':float(np.sqrt(np.mean(audio**2))),
            'clipped_fraction':float(np.mean(np.abs(audio)>=.999)),'ok':True}
    except Exception as e: return {'remote':remote,'ok':False,'error':str(e)}

def verify():
    global expected_oids
    listing=subprocess.check_output(['git','-C',str(STUDY/'source_repo'),'lfs','ls-files','-l'],text=True)
    expected_oids={parts[2]:parts[0] for line in listing.splitlines() if len(parts:=line.split(maxsplit=2))==3}
    files=json.loads((STUDY/'remote_files.json').read_text())
    audios=[p for p in files if p.endswith('.wav')]
    output=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,item in enumerate(pool.map(verify_local,audios),1):
            output.append(item)
            if i%2000==0: print('Local waveform verification',i,'/',len(audios),flush=True)
    jsonl(STUDY/'file_inventory_verified.jsonl',output)
    jsonl(STUDY/'audio_qc.jsonl',output)
    failures=[r for r in output if not r['ok']]
    write(STUDY/'audio_verification.json',{'files':len(output),'successful':len(output)-len(failures),
        'failures':failures,'hours':sum(r.get('duration',0) for r in output)/3600,
        'bytes':sum(r.get('bytes',0) for r in output)})
    if failures: raise RuntimeError(f'{len(failures)} released WAVs have not been verified')

class Union:
    def __init__(self,keys): self.parent={k:k for k in keys}
    def root(self,key):
        if self.parent[key]!=key: self.parent[key]=self.root(self.parent[key])
        return self.parent[key]
    def join(self,a,b):
        a,b=self.root(a),self.root(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)

def token_audit(source_rows,reg):
    import sys
    from transformers import AutoTokenizer
    sys.path.insert(0,str(ROOT/'vendor/samsone/src'))
    from samsone.inference.models import _build_pruned_token_map
    assets=json.loads((ARTIFACTS/'assets/manifest.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(assets['models']['smollm2_135m_base']['local_path'],local_files_only=True)
    mapping=_build_pruned_token_map(tokenizer);inverse={v:k for k,v in mapping.items()}
    keep=set(mapping.values())
    from .encoding import retained_encode
    catalog={d:reg.prompt(d) for d in ('smart_car','smart_home','wearables')}
    public={d:{'text':p,'tokens':len(tokenizer.encode(p,add_special_tokens=False)),
        'samsone_tokens_supported':all(i in keep for i in tokenizer.encode(p,add_special_tokens=False))} for d,p in catalog.items()}
    write(STUDY/'catalogs.json',public)
    targets={}
    for r in source_rows:
        if r['calls'] is None: continue
        text=compact({'calls':r['calls']})
        ids=tokenizer.encode(text,add_special_tokens=False)
        try:
            samsone_ids=retained_encode(tokenizer,inverse,text);supported=True
            samsone_roundtrip=typed(json.loads(tokenizer.decode([mapping[i] for i in samsone_ids])))==typed({'calls':r['calls']})
        except ValueError:samsone_ids=[];supported=False;samsone_roundtrip=False
        mellow_roundtrip=typed(json.loads(tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)))==typed({'calls':r['calls']})
        targets[r['id']]={'target_text':text,'target_tokens':len(samsone_ids)+1,
            'mellow_target_tokens':len(ids)+1,'retained_fallback_used':any(i not in keep for i in ids),
            'samsone_supported':supported,'roundtrip_by_model':{'samsone':samsone_roundtrip,'mellow':mellow_roundtrip}}
    return public,targets,tokenizer

def freeze():
    source_rows=rows(STUDY/'source_rows.jsonl');reg=registry()
    verified={r['remote']:r for r in rows(STUDY/'file_inventory_verified.jsonl')}
    if any(not r['ok'] for r in verified.values()): raise ValueError('Audio verification incomplete')
    public,targets,tokenizer=token_audit(source_rows,reg)
    profiles=json.loads((STUDY/'interface_profile.json').read_text())
    union=Union({r['request_group'] for r in source_rows});edges=[]
    buckets=defaultdict(list)
    for r in source_rows:
        buckets[('stable_source_index',r['domain'],r['query_idx'])].append(r['request_group'])
        buckets[('identical_normalized_request',r['normalized_request'])].append(r['request_group'])
        for remote in r['audio_remotes']:
            if verified[remote]['ok']:buckets[('identical_audio',verified[remote]['sha256'])].append(r['request_group'])
    for key,groups in buckets.items():
        groups=sorted(set(groups))
        for b in groups[1:]:
            union.join(groups[0],b);edges.append({'a':groups[0],'b':b,'reason':key[0],'evidence':list(key[1:])})
    jsonl(STUDY/'component_edges.jsonl',edges)
    inspected=set(json.loads((ROOT/'nano_learning/resources/inspected_requests.json').read_text())['request_groups'])
    quarantined={union.root(k) for k in inspected if k in union.parent}
    previous_panel=STUDY/'pretraining_revision_quarantine.json'
    revision_quarantine=set()
    if previous_panel.exists():
        revision_quarantine={union.root(k) for k in json.loads(previous_panel.read_text())['request_groups'] if k in union.parent}
        quarantined.update(revision_quarantine)
    engineering=json.loads((STUDY/'engineering_panel.json').read_text())['request_groups']
    forced_train={union.root(k) for k in engineering}
    interfaces={m:[] for m in ('samsone','mellow')};recordings=[]
    for row in source_rows:
        r={k:v for k,v in row.items() if k!='raw'}
        r['component']=union.root(r['request_group'])
        r['audio_paths']=[verified[p]['path'] for p in r['audio_remotes']]
        r['durations']=[verified[p]['duration'] for p in r['audio_remotes']]
        r['audio_hashes']=[verified[p]['sha256'] for p in r['audio_remotes']]
        r['hours']=sum(r['durations'])/3600
        r['official_primary_gold']={'name':row['raw']['tool_name'],'arguments':row['raw'].get('extracted_params') or {}}
        prompt_tokens=public[r['domain']]['tokens']
        if r['tier']==7 and r['calls'] and sum(r['durations'][:-1])<=30 and r['durations'][-1]<=30:
            from .dialogue import adapt
            r['adapted_audio_paths'],r['audio_role_prompt']=adapt(row['raw'],r['audio_paths'])
            import soundfile as sf
            r['adapted_durations']=[sf.info(p).duration for p in r['adapted_audio_paths']]
            prompt_tokens=len(tokenizer.encode(public[r['domain']]['text']+r['audio_role_prompt'],add_special_tokens=False))
        target=targets.get(r['id'],{})
        r.update(target)
        r['support']={}
        for model in interfaces:
            flags=list(r['flags']);p=profiles[model]
            prefix_tokens=104 if model=='samsone' and r['tier']==7 else p['audio_prefix_tokens']
            if model=='samsone' and (not public[r['domain']]['samsone_tokens_supported'] or not target.get('samsone_supported',True)):
                flags.append('vocabulary')
            if target and not target['roundtrip_by_model'][model]:flags.append('serialization_roundtrip')
            target_length=target.get('target_tokens') if model=='samsone' else target.get('mellow_target_tokens')
            if target and target_length>p['completion_cap']:flags.append('completion_cap')
            if prompt_tokens+prefix_tokens+p['completion_cap']>p['effective_context_cap']:
                flags.append('catalog_context')
            if r['tier']==7:
                # Complete history concatenation is explicit; no role transcript oracle.
                history,last=r.get('adapted_durations',[sum(r['durations'][:-1]),r['durations'][-1]])
                if history>p['native_clip_seconds'] or last>p['native_clip_seconds']:flags.append('full_history_audio_limit')
                if not p.get('dialogue_adapter_verified',False):flags.append('full_history_adapter_unverified')
            elif max(r['durations'])>p['native_clip_seconds']+1e-6:flags.append('native_audio_duration')
            if any(verified[x]['rms']<1e-5 for x in r['audio_remotes']):flags.append('silent_audio')
            r['support'][model]=not flags
            interfaces[model].append({'id':r['id'],'tier':r['tier'],'domain':r['domain'],
                'eligible':not flags,'flags':flags,'prompt_tokens':prompt_tokens,
                'target_tokens':target_length,'audio_prefix_tokens':prefix_tokens,
                'native_context_cap':p['native_context_cap'],'effective_context_cap':p['effective_context_cap'],
                'completion_cap':p['completion_cap'],'durations':r['durations']})
        r['common_supported']=all(r['support'].values())
        recordings.append(r)
    for model,items in interfaces.items():jsonl(STUDY/f'interface/{model}.jsonl',items)
    lengths=[{'model':m,**{k:v for k,v in r.items() if k in ('id','tier','domain','prompt_tokens','target_tokens','audio_prefix_tokens','completion_cap')}} for m,rr in interfaces.items() for r in rr]
    with (STUDY/'lengths.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(lengths[0]));writer.writeheader();writer.writerows(lengths)
    groups=defaultdict(list)
    for r in recordings:groups[r['component']].append(r)
    components=[{'id':c,'requests':sorted({r['request_group'] for r in rr}),
        'rows':[r['id'] for r in rr],'strata':sorted({f"{r['tier']}:{r['domain']}" for r in rr if r['common_supported']}),
        'tools':sorted({r['domain']+':'+x['name'] for r in rr if r['calls'] for x in r['calls']}),
        'quarantined':c in quarantined} for c,rr in sorted(groups.items())]
    jsonl(STUDY/'request_components.jsonl',components)
    for c in components:
        combos={r['domain']+':'+','.join(sorted(x['name'] for x in r['calls'])) for r in groups[c['id']] if r['common_supported']}
        c['allocation_strata']=[*c['strata'],*['tool:'+x for x in c['tools']],*['combination:'+x for x in sorted(combos)]]
    jsonl(STUDY/'request_components.jsonl',components)
    rng=random.Random(17011);order=components.copy();rng.shuffle(order)
    # Greedy multi-label stratification. Whole components remain indivisible.
    totals=Counter(s for c in components for s in c['allocation_strata']);counts={s:Counter() for s in ('train','validation','test')}
    splits={};sizes=Counter();fractions={'train':.7,'validation':.15,'test':.15}
    order.sort(key=lambda c:min((totals[s] for s in c['allocation_strata']),default=10**9))
    for c in order:
        choices=['train'] if c['id'] in forced_train else ['train','validation'] if c['quarantined'] else ['train','validation','test']
        def cost(split):
            strat=sum((counts[split][s]+1)/(fractions[split]*totals[s]+1e-9) for s in c['allocation_strata'])/max(1,len(c['allocation_strata']))
            return strat+.05*(sizes[split]+1)/(fractions[split]*len(components))
        split=min(choices,key=cost);splits[c['id']]=split;sizes[split]+=1
        counts[split].update(c['allocation_strata'])
    for r in recordings:r['split']=splits[r['component']]
    jsonl(STUDY/'recordings.jsonl',recordings)
    for split in fractions:
        jsonl(STUDY/f'data/{split}.jsonl',[r for r in recordings if r['split']==split and r['common_supported']])
    train=[c for c in components if splits[c['id']]=='train' and c['strata']]
    rng=random.Random(17012);rng.shuffle(train)
    # Coverage-first deterministic prefix, then randomized remaining components.
    covered=set();first=[];rest=[]
    for c in train:
        if set(c['strata'])-covered:first.append(c);covered.update(c['strata'])
        else:rest.append(c)
    training_order=first+rest
    subsets={str(f):[c['id'] for c in training_order[:max(len(first),math.ceil(len(train)*f))]] for f in (.1,.3,1.)}
    write(STUDY/'train_subsets.json',{'unit':'component','seed':17012,'coverage_floor_components':len(first),'ordered_train_components':[c['id'] for c in training_order],'subsets':subsets})
    subset_coverage={}
    for fraction,ids in subsets.items():
        selected=set(ids);rr=[r for r in recordings if r['common_supported'] and r['component'] in selected]
        subset_coverage[fraction]={'components':len(ids),'actual_fraction':len(ids)/len(train),
            'request_groups':len({r['request_group'] for r in rr}),'recordings':len(rr),'hours':sum(r['hours'] for r in rr),
            'tools':dict(Counter(r['domain']+':'+c['name'] for r in rr for c in r['calls'])),
            'tier_domain_rows':dict(Counter(f"{r['tier']}:{r['domain']}" for r in rr)),
            'unique_programs':len({digest(r['calls']) for r in rr})}
    write(STUDY/'train_subset_coverage.json',subset_coverage)
    validation=[r for r in recordings if r['split']=='validation' and r['common_supported']]
    bycell=defaultdict(list)
    for r in validation:bycell[(r['tier'],r['domain'])].append(r)
    small=[]
    for cell,rr in sorted(bycell.items()):
        chosen={}
        for r in sorted(rr,key=lambda x:digest({'seed':17013,'id':x['id']})):
            chosen.setdefault(r['component'],r['id'])
        small.extend(list(chosen.values())[:2])
    write(STUDY/'validation_panels.json',{'small':small,'full':[r['id'] for r in validation],
        'small_rule':'up to two independent components per tier/domain, one recording per selected component',
        'small_and_full_are_distinct_denominators':True})
    coverage=[];funnel=[]
    for tier in range(1,9):
        for domain in public:
            rr=[r for r in recordings if r['tier']==tier and r['domain']==domain]
            for model in ('samsone','mellow','common'):
                included=[r for r in rr if r['common_supported']] if model=='common' else [r for r in rr if r['support'][model]]
                test_components={r['component'] for r in included if r['split']=='test'}
                coverage.append({'tier':tier,'domain':domain,'model':model,'source_rows':len(rr),
                    'included_rows':len(included),'source_hours':sum(r['hours'] for r in rr),'included_hours':sum(r['hours'] for r in included),
                    'test_components':len(test_components),'reporting':'primary' if len(test_components)>=20 else 'exploratory' if len(test_components)>=5 else 'descriptive'})
            for model in interfaces:
                relevant=[x for x in interfaces[model] if x['tier']==tier and x['domain']==domain]
                for reason,n in Counter(x['flags'][0] if x['flags'] else 'included' for x in relevant).items():
                    funnel.append({'tier':tier,'domain':domain,'model':model,'first_failure':reason,'rows':n})
    for name,rr in [('coverage',coverage),('exclusion_funnel',funnel)]:
        with (STUDY/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
    audit={}
    for key in ('component','normalized_request'):
        seen={s:{r[key] for r in recordings if r['split']==s} for s in fractions}
        audit[key+'_overlaps']={a+'_'+b:len(seen[a]&seen[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    hashsets={s:{h for r in recordings if r['split']==s for h in r['audio_hashes']} for s in fractions}
    audit['audio_hash_overlaps']={a+'_'+b:len(hashsets[a]&hashsets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    if any(n for k,v in audit.items() for n in v.values()):raise ValueError('Cross-split leakage')
    speakers={s:{speaker for r in recordings if r['split']==s for speaker in r.get('speakers',[]) if speaker} for s in fractions}
    speaker_disjoint=[r['id'] for r in recordings if r['split']=='test' and r['common_supported'] and
        all(r.get('speakers',[])) and not set(r['speakers'])&speakers['train']]
    write(STUDY/'speaker_disjoint_secondary_panel.json',{'ids':speaker_disjoint,
        'feasible':bool(speaker_disjoint),'rule':'all annotated target/history speakers absent from common and excluded training source rows; unknown mixed speakers omitted',
        'selection_frozen_before_predictions':True})
    known_tools={r['domain']+':'+c['name'] for r in recordings if r['split']=='train' and r['common_supported'] for c in r['calls']}
    unknown_test=[{'id':r['id'],'tools':[r['domain']+':'+c['name'] for c in r['calls'] if r['domain']+':'+c['name'] not in known_tools]}
        for r in recordings if r['split']=='test' and r['common_supported'] and any(r['domain']+':'+c['name'] not in known_tools for c in r['calls'])]
    audit.update(sizes=dict(sizes),fractions={s:sizes[s]/len(components) for s in fractions},
        historical_prediction_open_components=len(quarantined),historical_test_overlap=any(splits[c]=='test' for c in quarantined),
        pretraining_revision_quarantine_components=len(revision_quarantine),
        allocation_strata='tier/domain plus public tool identities and complete tool-name combinations',
        speaker_overlaps={a+'_'+b:len(speakers[a]&speakers[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]},
        speaker_disjoint_claim=False,largest_component_requests=max(len(c['requests']) for c in components),
        speaker_disjoint_secondary_rows=len(speaker_disjoint),
        test_tools_without_training_coverage=unknown_test,
        source_to_split_rows={s:sum(r['split']==s for r in recordings) for s in fractions},
        common_to_split_rows={s:sum(r['split']==s and r['common_supported'] for r in recordings) for s in fractions},
        cross_tier_rule='same public stable original query_idx and domain, conservatively grouped across tiers; exact text/audio edges added',
        near_duplicate_rule='casefolded Unicode word sequence identity; no outcome-dependent embedding threshold',
        label_access='mechanical structural audit across all labels; final test remains prediction-unseen')
    write(STUDY/'split_audit.json',audit);write(STUDY/'splits.json',{'seed':17011,'component_assignments':splits})
    write(STUDY/'supported_scope.json',{'common_rows':sum(r['common_supported'] for r in recordings),
        'per_model_rows':{m:sum(r['support'][m] for r in recordings) for m in interfaces},'coverage':coverage})
    write(STUDY/'final_test_lock.json',{'status':'prediction_unopened','recordings_sha256':sha(STUDY/'recordings.jsonl'),
        'split_sha256':sha(STUDY/'splits.json'),'interface_sha256':sha(STUDY/'interface_profile.json'),
        'reward_version':'a2t-typed-v1','registry_hash':reg.hash,
        'test_ids':[r['id'] for r in recordings if r['split']=='test' and r['common_supported']]})
    write(STUDY/'cache_and_storage_plan.json',{'free_bytes':shutil.disk_usage(STUDY).free,
        'feature_disk_limit_per_model':32*2**30,'feature_device_LRU_entries':12,
        'feature_location':'before trainable projection; shared immutable features with independent policy/reference projections',
        'checkpoint_retention':'latest two complete resumes plus every warmup endpoint, best, shared parent/reference, continuation endpoints'})
    # Freeze the reporting scope and data identities for the supported stage runner.
    tier_components=defaultdict(set)
    for c in components:
        if splits[c['id']]=='test':
            for stratum in c['strata']:tier_components[int(stratum.split(':')[0])].add(c['id'])
    write(STUDY/'execution_manifest.json',{'version':'audio2tool-clean-pipeline-v1','seed':17,
        'split_seed':17011,'warmup_updates':640,'primary_tiers':sorted(t for t,cs in tier_components.items() if len(cs)>=20),
        'data_hashes':{str(p.relative_to(STUDY)):sha(p) for p in [STUDY/'data/train.jsonl',STUDY/'data/validation.jsonl',STUDY/'data/test.jsonl',STUDY/'validation_panels.json']},
        'selection_rule':'640 endpoint full-validation macro exact; ties prefer smallest pool'})
    print('Frozen common supported rows',sum(r['common_supported'] for r in recordings),'components',len(components),'splits',dict(sizes),flush=True)

fileset=set()
expected_oids={}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['metadata','verify','freeze']);a=p.parse_args()
    fileset=set(json.loads((STUDY/'remote_files.json').read_text()))
    {'metadata':metadata,'verify':verify,'freeze':freeze}[a.command]()
