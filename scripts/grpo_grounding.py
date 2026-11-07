"""Fixed dev audio controls, diagnostic only; never used to tune training."""
import copy,json,sys
from pathlib import Path
from nano_audio.cli import records,load_policy,scoring,families,seeded
from nano_audio.evaluation.controls import materialize_controls
from nano_audio.utils import ROOT,write_json
out=ROOT/'artifacts/grpo_attempt_20261107'
if len(sys.argv)>1:out=out/sys.argv[1]
seeded(1717);p=load_policy('samsone_134m_published','cuda');p.load_checkpoint(out/'grpo_checkpoint/policy')
groups={}
for ex in records('smoke','dev'):groups.setdefault(ex['utterance_group_id'],ex)
pool=list(groups.values());f=families()['train_01'];rows=[]
for i,ex in enumerate(pool):
    ex=copy.deepcopy(ex);ex.update(schema_family_id=f.family_id,tools=f.tools,target=f.encode(ex['canonical_target']))
    variants=materialize_controls(ex,pool[(i+1)%len(pool)]['audio_path'],out/'audio_controls',1717)
    for condition,item in [('clean',ex),('silence',variants['silence']),('mismatch',variants['mismatch'])]:
        output=p.generate(item,{'max_new_tokens':128});score=scoring(ex,output)
        rows.append({'id':ex['id'],'group':ex['utterance_group_id'],'condition':condition,'original_request_for_log_only':ex['transcript'],'output':output,'score_against_original_request':score})
        with (out/'generations.log').open('a') as h:h.write(f'\nAUDIO CONTROL {condition} reward_against_original={score["reward"]}\nORIGINAL REQUEST: {ex["transcript"]}\nGENERATION: {output["text"]}\n')
write_json(out/'audio_controls/results.json',{'scope':'fixed constructed dev diagnostic, no optimization','scores':{c:{'n':len([r for r in rows if r['condition']==c]),'correct_against_original':sum(r['score_against_original_request']['reward'] for r in rows if r['condition']==c)} for c in ['clean','silence','mismatch']},'rows':rows})
