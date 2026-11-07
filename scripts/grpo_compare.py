"""Fixed-seed sampled comparison of the common parent and both continuations."""
import copy,datetime,json,random,time
from pathlib import Path
import torch
from nano_audio.cli import seeded,records,scoring,load_policy,families
from nano_audio.utils import ROOT,write_json,append_json
OUT=ROOT/'artifacts/grpo_attempt_20261107'
import sys
if len(sys.argv)>1:OUT=OUT/sys.argv[1]
DEADLINE=datetime.datetime(2026,11,7,9,22,35,tzinfo=datetime.timezone.utc).timestamp()
seeded(1717)
train=records('smoke','train');dev=records('smoke','dev');pools=[]
for split,data in [('train',train),('dev_id',dev)]:
    groups={}
    for ex in data:groups.setdefault(ex['utterance_group_id'],ex)
    for ex in groups.values():
        row=copy.deepcopy(ex)
        if split=='dev_id':
            f=families()['train_01'];row.update(schema_family_id=f.family_id,tools=f.tools,target=f.encode(ex['canonical_target']))
        pools.append((split,row))
p=load_policy('samsone_134m_published','cuda');result={}
for label,folder in [('common','common_checkpoint'),('grpo','grpo_checkpoint'),('sft_control','sft_control_checkpoint')]:
    if time.time()>DEADLINE-120:break
    p.load_checkpoint(OUT/folder/'policy');torch.manual_seed(1717);torch.cuda.manual_seed_all(1717)
    rows=[]
    for split,ex in pools:
        if time.time()>DEADLINE-90:break
        for sample in range(4):
            output=p.generate(ex,{'do_sample':True,'max_new_tokens':128});score=scoring(ex,output)
            row={'checkpoint':label,'split':split,'group':ex['utterance_group_id'],'sample':sample,'request_for_log_only':ex['transcript'],'gold_for_log_only':ex['target'],'output':output,'score':score}
            rows.append(row);append_json(OUT/'comparison_generations.jsonl',row)
            with (OUT/'generations.log').open('a') as f:f.write(f'\nCOMPARISON {label} {split} sample={sample} reward={score["reward"]}\nREQUEST: {ex["transcript"]}\nGENERATION: {output["text"]}\n')
    result[label]={split:{'n':len([r for r in rows if r['split']==split]),'correct':sum(r['score']['reward'] for r in rows if r['split']==split)} for split in ['train','dev_id']}
    print(json.dumps({label:result[label]}),flush=True)
    write_json(OUT/'comparison.json',result)
# Utterance-cluster bootstrap of sampled exact rates; uncertainty is descriptive.
allrows=[json.loads(l) for l in (OUT/'comparison_generations.jsonl').read_text().splitlines()]
intervals={}
for split in ['train','dev_id']:
    for opponent in ['common','sft_control']:
        def averages(label):
            d={}
            for r in allrows:
                if r['checkpoint']==label and r['split']==split:d.setdefault(r['group'],[]).append(r['score']['reward'])
            return {k:sum(v)/len(v) for k,v in d.items()}
        left,right=averages('grpo'),averages(opponent);keys=sorted(set(left)&set(right))
        if not keys:continue
        diffs=[left[k]-right[k] for k in keys];rng=random.Random(1717)
        boots=sorted(sum(rng.choices(diffs,k=len(diffs)))/len(diffs) for _ in range(2000))
        intervals[split+'/grpo-minus-'+opponent]={'difference':sum(diffs)/len(diffs),'low':boots[49],'high':boots[1949],'utterance_groups':len(diffs)}
write_json(OUT/'comparison_intervals.json',intervals)
