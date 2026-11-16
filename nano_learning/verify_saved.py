"""Rescore stored text and compare summaries. Never generates model outputs."""
import json
from .common import ROOT, STUDY, rows, sha, write
from .evidence import roots,validation_dir,split_path
from .evaluate import aggregate,official_score,METRICS
from .task import Registry,score


def verify():
    reg=Registry(STUDY/'source_repo/tools_registry.csv')
    datasets={split:{r['id']:r for r in rows(split_path(STUDY,split))} for split in ('validation','test')}
    sources=[];r=roots()
    for model in ('samsone','mellow'):
        for fraction in (10,30,100):sources.append(('validation',validation_dir(STUDY/f'runs/{model}-warmup-{fraction:03d}')/'000640_full.json'))
        for stage in ('released','parent','sft-endpoint','binary-endpoint','typed-endpoint'):
            sources.append(('test',STUDY/f'evaluation/{model}-{stage}.json'))
    for group,name,step in [('followup','samsone-sft',1024),('followup','samsone-binary',1024),
        ('extension','typed1024',1024),('extension','sft4096',3072),('mellow','mellow-sft',1024)]:
        sources.append(('validation',validation_dir(r[group]/'runs'/name)/f'{step:06d}_full.json'))
    checked=[]
    for split,path in sources:
        saved=json.loads(path.read_text());prediction_path=path.with_suffix('.predictions.jsonl');predictions=rows(prediction_path)
        assert sha(prediction_path)==saved['prediction_sha256'],path
        for pred in predictions:
            row=datasets[split][pred['id']];actual=score(reg,row,pred['output'])
            for key,value in actual.items():assert abs(value-pred['scores'][key])<=1e-12,(path,pred['id'],key)
            if 'official_compatibility' in pred:
                actual=official_score(reg,row,pred['output'])
                for key,value in actual.items():
                    assert abs(value-pred['official_compatibility'][key])<=1e-12,(path,pred['id'],key)
        actual=aggregate(predictions,saved['primary_tiers'])
        for key in ['macro_exact',*METRICS,'n_recordings','n_requests','n_components']:
            assert abs(actual[key]-saved[key])<=1e-12,(path,key)
        checked.append({'path':str(path.relative_to(ROOT)),'prediction_sha256':sha(prediction_path),'recordings':len(predictions),'passed':True})
    result={'scorer_sha256':sha(ROOT/'nano_learning/task.py'),'tolerance':1e-12,
        'endpoints':checked,'recordings_rescored':sum(x['recordings'] for x in checked),'inference':False}
    write(ROOT/'paper/generated/evidence/scoring_regression.json',result)
    return result

if __name__=='__main__':
    result=verify();print('Verified',len(result['endpoints']),'endpoints and',result['recordings_rescored'],'saved predictions')
