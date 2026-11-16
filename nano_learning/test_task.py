import copy
import csv
import json
import pytest
from .task import Registry, GoldError, compact, score, source_program, strict_json

@pytest.fixture
def reg(tmp_path):
    p=tmp_path/'registry.csv'
    fields=['tool_id','domain','category','tool_name','signature','description','argument_defaults','argument_constraints']
    examples=[
        ['1','d','c','setThing','setThing(flag: Boolean, count: Int, value: String, data: Map?)','Sets a thing.',"flag=False, count=1, value='Case', data=null",'count: [0-10]'],
        ['2','d','c','stop','stop()','Stops.','',''],
    ]
    with p.open('w') as f:
        w=csv.writer(f);w.writerow(fields);w.writerows(examples)
    return Registry(p)

def example(reg):
    calls=reg.bind('d',[{'name':'setthing','arguments':{'flag':True,'count':3,'value':'Case','data':{'nested':[1,True,'x']}}}])
    return {'domain':'d','calls':calls,'order_matters':False}

def out(calls,**kwargs):return {'text':compact({'calls':calls}),'length_capped':False,**kwargs}

def test_lossless_unicode_surface():
    value={'Text':'AbC, café \" true null \\ 你好 😄','bool':False}
    s=compact(value)
    assert all(not c.isupper() for c in s)
    assert json.loads(s)==value

def test_literal_safety_and_json_spellings():
    assert source_program('f(a=true,b="true",c=null)')[0]['arguments']=={'a':True,'b':'true','c':None}
    for text in ['__import__("os").system("x")','f(a=g())','f(**x)','f(a=1,a=2)','x=1','f(a=nan)']:
        with pytest.raises((ValueError,SyntaxError)):source_program(text)
    with pytest.raises(ValueError):strict_json('{"x":1,"x":2}')

def test_exact_and_default_equivalence(reg):
    ex=example(reg);assert score(reg,ex,out(ex['calls']))['binary']==1
    ex={'domain':'d','calls':reg.bind('d',[{'name':'setThing','arguments':{}}])}
    assert score(reg,ex,out([{'name':'setthing','arguments':{}}]))['binary']==1

@pytest.mark.parametrize('mutation',['missing_call','extra_call','wrong_tool','bool_number','int_string','casing','nested_bool_number','extra_field','unknown_arg','truncated','duplicate_key'])
def test_adversarial_failures(reg,mutation):
    ex=example(reg);calls=copy.deepcopy(ex['calls']);output=None
    if mutation=='missing_call':calls=[]
    elif mutation=='extra_call':calls.append({'name':'stop','arguments':{}})
    elif mutation=='wrong_tool':calls[0]['name']='stop'
    elif mutation=='bool_number':calls[0]['arguments']['flag']=1
    elif mutation=='int_string':calls[0]['arguments']['count']='3'
    elif mutation=='casing':calls[0]['arguments']['value']='case'
    elif mutation=='nested_bool_number':calls[0]['arguments']['data']['nested'][1]=1
    elif mutation=='extra_field':calls[0]['extra']=1
    elif mutation=='unknown_arg':calls[0]['arguments']['unknown']=1
    elif mutation=='truncated':output=out(calls,length_capped=True)
    elif mutation=='duplicate_key':output={'text':'{"calls":[],"calls":[]}','length_capped':False}
    assert score(reg,ex,output or out(calls))['binary']==0

def test_partial_invalid_other_field_and_extra_penalty(reg):
    ex=example(reg);calls=copy.deepcopy(ex['calls']);calls[0]['arguments']['count']='wrong'
    s=score(reg,ex,out(calls));assert 0<s['typed']<.5 and s['schema_validity']==0
    calls[0]['arguments']['extra']='bad';assert score(reg,ex,out(calls))['typed']<s['typed']

def test_repeated_calls_bijective_and_order(reg):
    a=example(reg)['calls'][0];b=copy.deepcopy(a);b['arguments']['count']=5
    ex={'domain':'d','calls':[a,b]}
    assert score(reg,ex,out([b,a]))['binary']==1
    assert score(reg,ex,out([a,a]))['binary']==0
    ex['order_matters']=True
    assert score(reg,ex,out([b,a]))['binary']==0

def test_invalid_gold_is_error(reg):
    ex=example(reg);ex['calls'][0]['arguments']['count']=True
    with pytest.raises(GoldError):score(reg,ex,out([]))

def test_no_call_count_limit(reg):
    calls=[{'name':'stop','arguments':{}} for _ in range(7)]
    assert score(reg,{'domain':'d','calls':calls},out(calls))['binary']==1
