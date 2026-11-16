"""Non-predictive original/adapted token lengths and all candidate caps."""
import csv
import json
from collections import Counter,defaultdict
from transformers import AutoTokenizer
from .common import ARTIFACTS, ROOT,STUDY,rows,write
from .task import Registry

def main():
    reg=Registry(STUDY/'source_repo/tools_registry.csv')
    assets=json.loads((ARTIFACTS/'assets/manifest.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(assets['models']['smollm2_135m_base']['local_path'],local_files_only=True)
    original_prompts={}
    for domain in ('smart_car','smart_home','wearables'):
        text='Return the requested tool calls. Tools: '+json.dumps([r for r in reg.rows if r['domain']==domain],ensure_ascii=False,separators=(',',':'))
        original_prompts[domain]=len(tokenizer.encode(text,add_special_tokens=False))
    recording={r['id']:r for r in rows(STUDY/'recordings.jsonl')};original_targets={}
    for rid,r in recording.items():
        if not r['calls']:continue
        calls=[]
        for c in r['calls']:
            tool=reg.tools[(r['domain'],c['name'])]
            calls.append({'name':tool['name'],'arguments':{tool['parameters'][k]['public_name']:v for k,v in c['arguments'].items()}})
        original_targets[rid]=len(tokenizer.encode(json.dumps({'calls':calls},ensure_ascii=False,separators=(',',':')),add_special_tokens=False))+1
    lengths=[];candidates=[]
    for model in ('samsone','mellow'):
        interface=rows(STUDY/f'interface/{model}.jsonl')
        for x in interface:
            lengths.append({'id':x['id'],'model':model,'tier':x['tier'],'domain':x['domain'],
                'original_complete_public_registry_prompt_tokens':original_prompts[x['domain']],
                'adapted_complete_catalog_prompt_tokens':x['prompt_tokens'],'audio_prefix_tokens':x['audio_prefix_tokens'],
                'original_canonical_output_tokens':original_targets.get(x['id']),
                'adapted_canonical_output_tokens':x['target_tokens']})
        for context in (2048,4096,8192):
            for completion in (128,256,512):
                counts=Counter()
                for x in interface:
                    nonlength=[f for f in x['flags'] if f not in ('completion_cap','catalog_context')]
                    supported=(not nonlength and x['target_tokens'] is not None and x['target_tokens']<=completion and
                        x['prompt_tokens']+x['audio_prefix_tokens']+completion<=context)
                    if supported:counts[(x['tier'],x['domain'])]+=1
                for tier in range(1,9):
                    for domain in ('smart_car','smart_home','wearables'):
                        candidates.append({'model':model,'tier':tier,'domain':domain,'context_cap':context,
                            'completion_cap':completion,'supported_rows':counts[(tier,domain)]})
    for name,data in [('original_adapted_lengths.csv',lengths),('candidate_caps.csv',candidates)]:
        with (STUDY/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    coverage=list(csv.DictReader((STUDY/'coverage.csv').open()))
    for r in coverage:
        model=r['model'];tier=int(r['tier'])
        supported=[x for x in recording.values() if x['tier']==tier and x['split']=='test' and
            (x['common_supported'] if model=='common' else x['support'][model])]
        count=len({x['component'] for x in supported});r['tier_test_components']=count
        r['reporting']='primary' if count>=20 else 'exploratory' if count>=5 else 'descriptive'
    with (STUDY/'coverage.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
    write(STUDY/'candidate_cap_decision.json',{'context_cap':8192,'completion_cap':512,
        'evidence':['candidate_caps.csv','original_adapted_lengths.csv','interface_profile.json'],
        'reason':'Full lossless public catalogs and every currently supported complete target fit the selected native-cap interface; longest supported catalogs and 512-token backward passes verified on both models. No sample-specific oracle caps.',
        'original_length_reference':'Full public registry rows without aliases or lowercase-ASCII JSON encoding; for audit only, never an oracle subset',
        'human_listening_performed':False})

if __name__=='__main__':main()
