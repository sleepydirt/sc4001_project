"""Held-out Audio2Tool consistency audit; never repair labels from predictions.

The public release can disagree between tool_name/expected_tool_call/functions.
Transfer catalogs must be preregistered by domain, never retrieved per gold label.
"""
import ast,json
from collections import Counter
from pathlib import Path
from nano_audio.utils import digest,sha,write_json

SOURCE='https://huggingface.co/datasets/RVtech/Audio2Tool'

def parse_call(text):
    node=ast.parse(text.strip(),mode='eval').body
    if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.args:
        raise ValueError('Expected named tool with literal keyword arguments')
    arguments={}
    for kw in node.keywords:
        if kw.arg is None or kw.arg in arguments:raise ValueError('Duplicate or expanded argument')
        arguments[kw.arg]=ast.literal_eval(kw.value)
    return {'name':node.func.id,'arguments':arguments}

def audit_record(row,domain_catalogs):
    if row.get('split','test')!='test':raise ValueError('Audio2Tool is evaluation-only')
    if float(row.get('duration',0))<=0 or float(row['duration'])>8:return None,'duration_outside_task_window'
    embedded={f.get('tool_name',f.get('name')):f for f in row.get('functions',[])}
    try:call=parse_call(row['expected_tool_call'])
    except (SyntaxError,ValueError,TypeError,KeyError):return None,'unsupported_expected_call_syntax'
    if call['name']!=row.get('tool_name') or call['name'] not in embedded:return None,'tool_registry_label_disagreement'
    params=row.get('extracted_params',{})
    if isinstance(params,str):
        try:params=json.loads(params)
        except ValueError:return None,'invalid_parameter_labels'
    if params!=call['arguments']:return None,'parameter_label_disagreement'
    additional=row.get('additional_tool_calls')
    if isinstance(additional,str):
        try:additional=json.loads(additional)
        except ValueError:return None,'unsupported_additional_calls'
    if additional:return None,'additional_call_adapter_requires_explicit_mapping'
    catalog=domain_catalogs.get(row['domain'])
    if not catalog:return None,'domain_outside_preregistered_catalog'
    names={f.get('name',f.get('tool_name')) for f in catalog}
    if call['name'] not in names:return None,'tool_outside_preregistered_catalog'
    # Preserve native names and labels. Lowercase research bindings need an
    # independently audited mapping; this stage does not silently invent one.
    return {'id':str(row['id']),'source':'audio2tool','split':'test','utterance_group_id':'audio2tool_'+str(row['query_idx']),
            'audio_path':row['path'],'duration_seconds':row['duration'],'domain':row['domain'],'tier':row['tier'],
            'tools':catalog,'native_target':call,'native_expected_call':row['expected_tool_call'],
            'catalog_hash':digest(catalog),'protocol':'audited_native_transfer_candidate_v1'},None

def audit_audio2tool(input_path,catalog_path,output_path):
    catalog=json.loads(Path(catalog_path).read_text())
    text=Path(input_path).read_text();rows=json.loads(text) if text.lstrip().startswith('[') else [json.loads(l) for l in text.splitlines() if l.strip()]
    accepted=[];excluded=[];reasons=Counter()
    for row in rows:
        record,reason=audit_record(row,catalog)
        if reason:excluded.append({'id':str(row.get('id')),'reason':reason});reasons[reason]+=1
        else:accepted.append(record)
    result={'source_url':SOURCE,'input_sha256':sha(input_path),'preregistered_catalog_sha256':sha(catalog_path),
            'status':'audited','native_score':None,'research_score':None,'accepted_candidates':accepted,
            'excluded':excluded,'exclusion_counts':dict(reasons),'training_allowed':False,
            'limitation':'Candidate audit only. Official native scorer and lowercase research bindings remain separate execution gates.'}
    write_json(output_path,result);return result
