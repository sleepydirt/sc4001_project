from __future__ import annotations
from collections import defaultdict
import json
import time
import numpy as np
from .common import STUDY,write,append,digest,sha
from .task import score

METRICS=['binary','schema_validity','tool_multiset_accuracy','argument_F1','parse_validity']

def aggregate(predictions,primary_tiers=None):
    groups=defaultdict(list)
    for row in predictions:groups[(row['request_group'],row['tier'],row['domain'],row['component'])].append(row)
    requests=[]
    for (group,tier,domain,component),rr in groups.items():
        requests.append({'request_group':group,'tier':tier,'domain':domain,'component':component,
            **{m:float(np.mean([r['scores'][m] for r in rr])) for m in METRICS}})
    cells=defaultdict(list)
    for r in requests:cells[(r['tier'],r['domain'])].append(r)
    cell_metrics={f'{t}:{d}':{'n_requests':len(rr),'n_components':len({r['component'] for r in rr}),
        **{m:float(np.mean([r[m] for r in rr])) for m in METRICS}} for (t,d),rr in cells.items()}
    tiers=set(primary_tiers) if primary_tiers is not None else {r['tier'] for r in requests}
    selected=[v for k,v in cell_metrics.items() if int(k.split(':')[0]) in tiers]
    result={'n_recordings':len(predictions),'n_requests':len(requests),'n_components':len({r['component'] for r in requests}),
        'cells':cell_metrics,'macro_exact':float(np.mean([x['binary'] for x in selected])) if selected else 0.,
        **{m:float(np.mean([r[m] for r in requests])) if requests else 0. for m in METRICS},'requests':requests}
    result['length_capped_rate']=float(np.mean([r['output']['length_capped'] for r in predictions])) if predictions else 0.
    result['primary_tiers']=sorted(tiers)
    compatible=[r['official_compatibility'] for r in predictions if r.get('official_compatibility',{}).get('single_call_supported')]
    result['official_single_call_compatibility']={'n_recordings':len(compatible),
        **{k:float(np.mean([r[k] for r in compatible])) if compatible else None for k in ('tool_accuracy','parameter_f1','exact')},
        'target_semantics':'original public extracted_params, no default expansion; official normalization differs from typed metric'}
    if any('training_tool_coverage_complete' in r for r in predictions):
        for known in (True,False):
            rr=[r for r in predictions if r.get('training_tool_coverage_complete')==known]
            grouped=defaultdict(list)
            for r in rr:grouped[r['request_group']].append(r['scores']['binary'])
            result['training_tools_seen' if known else 'rare_unseen_training_tool_challenge']={
                'n_requests':len(grouped),'pooled_request_exact':float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None}
    return result

def official_score(registry,row,output):
    if len(row['calls'])!=1:return {'single_call_supported':False}
    import importlib.util,sys
    from .common import ROOT,SCORER_REVISION
    from .task import strict_json
    module=sys.modules.get('a2t_pinned_official_metrics')
    if module is None:
        source=ROOT/'vendor/audio2tool'
        provenance=json.loads((source/'PROVENANCE.json').read_text())
        if provenance['revision']!=SCORER_REVISION or sha(source/'metrics.py')!=provenance['sha256']:
            raise ValueError('Official scorer revision changed')
        spec=importlib.util.spec_from_file_location('a2t_pinned_official_metrics',source/'metrics.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    name='';args={}
    try:
        if output['length_capped']:raise ValueError('Capped')
        value=strict_json(output['text'])
        if set(value)!={'calls'} or len(value['calls'])!=1:raise ValueError('Not single-call output')
        call=value['calls'][0];tool=registry.tools[(row['domain'],call['name'].lower())]
        name=tool['name'];args={tool['parameters'].get(k.lower(),{}).get('public_name',k):v for k,v in call['arguments'].items()}
    except (ValueError,KeyError,TypeError,AttributeError):pass
    gold=row['official_primary_gold']
    return {'single_call_supported':True,'tool_accuracy':float(module.compute_tool_accuracy(name,gold['name'])),
        'parameter_f1':module.compute_parameter_accuracy(args,gold['arguments'])['f1'],
        'exact':float(module.compute_exact_match(name,args,gold['name'],gold['arguments']))}

def evaluate(policy,dataset,reg,out,*,ids=None,ledger=None,batch_size=4,primary_tiers=None):
    from .sampling import generate_batch
    if dataset.split=='test' and not (STUDY/'evaluation_lock.json').exists():raise PermissionError('Test inference lock absent')
    out.parent.mkdir(parents=True,exist_ok=True)
    selected=dataset.records if ids is None else [dataset.byid[i] for i in ids]
    rare_ids={r['id'] for r in json.loads((STUDY/'split_audit.json').read_text()).get('test_tools_without_training_coverage',[])}
    prediction_path=out.with_suffix('.predictions.jsonl')
    previous={}
    if prediction_path.exists():
        from .common import rows
        previous={r['id']:r for r in rows(prediction_path)}
    pending=[r for r in selected if r['id'] not in previous]
    # Public catalog/tier grouping avoids padding to car length for every domain.
    pending.sort(key=lambda r:(r['domain'],len(r.get('audio_role_prompt','')),r['id']))
    chunks=[];grouped=defaultdict(list)
    for r in pending:
        ex=dataset.example(r)
        grouped[(r['domain'],len(policy.encode(ex['_prompt'])),len(ex['audio_paths']))].append(r)
    for key,rr in sorted(grouped.items()):
        for start in range(0,len(rr),batch_size):chunks.append(rr[start:start+batch_size])
    started=time.perf_counter()
    for chunk in chunks:
        if ledger:ledger.reserve_unit(60.)
        protocol=json.loads((STUDY/'decoding_protocol.json').read_text())
        if protocol['evaluation_batch_size']!=batch_size:raise ValueError('Evaluation batch protocol changed')
        outputs=generate_batch(policy,[dataset.example(r) for r in chunk],force_batch=True)
        for row,output in zip(chunk,outputs):
            value={'id':row['id'],'component':row['component'],'request_group':row['request_group'],
                'tier':row['tier'],'domain':row['domain'],'output':output,
                'training_tool_coverage_complete':row['id'] not in rare_ids,
                'scores':score(reg,row,output),'official_compatibility':official_score(reg,row,output)}
            append(prediction_path,value);previous[row['id']]=value
    predictions=[previous[r['id']] for r in selected]
    result=aggregate(predictions,primary_tiers)
    result.update(elapsed_seconds=time.perf_counter()-started,dataset_sha256=dataset.sha256,
        panel_ids_hash=digest([r['id'] for r in selected]),prediction_sha256=sha(prediction_path))
    write(out,result)
    return result

def bootstrap_difference(left,right,seed=17015,n=2000):
    a={r['request_group']:r for r in left['requests']};b={r['request_group']:r for r in right['requests']}
    if set(a)!=set(b):raise ValueError('Paired bootstrap requires identical request groups')
    components=defaultdict(list)
    for key in a:components[a[key]['component']].append((a[key],b[key]))
    if left.get('primary_tiers')!=right.get('primary_tiers'):raise ValueError('Different primary tier definitions')
    keys=sorted(components);rng=np.random.default_rng(seed);values=[]
    for _ in range(n):
        sampled=rng.choice(keys,len(keys),replace=True);cells=defaultdict(list)
        for k in sampled:
            for x,y in components[k]:
                if x['tier'] in left['primary_tiers']:cells[(x['tier'],x['domain'])].append(y['binary']-x['binary'])
        values.append(float(np.mean([np.mean(v) for v in cells.values()])))
    diff=right['macro_exact']-left['macro_exact']
    return {'macro_exact_difference':diff,'ci95':np.quantile(values,[.025,.975]).tolist(),
        'bootstrap_unit':'split component; recordings averaged within request group','replicates':n,
        'seed':seed,'seed_uncertainty_estimated':False}
