import sys
import pytest
from nano_learning import workflow


@pytest.mark.parametrize('stage,expected',[
    ('warmup',{'warmup'}),('pilot64',{'pilot64'}),
    ('continuation1024',{'continuation1024'}),('sft4096',{'sft4096'})])
def test_individual_training_stage_dispatch(tmp_path,monkeypatch,stage,expected):
    from nano_learning import selection,common
    calls=[]
    monkeypatch.setattr(workflow,'call',lambda *args:calls.append(args))
    monkeypatch.setattr(selection,'select_warmups',lambda **kwargs:{m:dict(policy='parent.pt',sha256='test') for m in ['samsone','mellow']})
    monkeypatch.setattr(common,'sha',lambda path:'test')
    monkeypatch.setattr(sys,'argv',['workflow',stage,'--runs',str(tmp_path)])
    workflow.main()
    assert {args[1] for args in calls}==expected
    assert len(calls)=={'warmup':6,'pilot64':6,'continuation1024':4,'sft4096':1}[stage]
