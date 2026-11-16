import json
import pytest
from .accounting import GPULedger,BudgetStop

def test_budget_survives_resume_and_includes_stalls(tmp_path,monkeypatch):
    clock=[100.]
    monkeypatch.setattr('nano_learning.accounting.time.time',lambda:clock[0])
    ledger=GPULedger(tmp_path,limit=20,margin=2)
    ledger.begin('binary')
    clock[0]+=5;ledger.tick();ledger.end()
    resumed=GPULedger(tmp_path,limit=20,margin=2)
    assert resumed.remaining()==15
    resumed.begin('typed');clock[0]+=4;resumed.tick()
    assert resumed.remaining()==11
    with pytest.raises(BudgetStop):resumed.guard(10)
    resumed.end()
    assert json.loads((tmp_path/'gpu_budget.json').read_text())['charged_seconds']==9
    with pytest.raises(ValueError):GPULedger(tmp_path,limit=21).remaining()

def test_crash_charges_reserved_work_conservatively(tmp_path,monkeypatch):
    clock=[110.]
    monkeypatch.setattr('nano_learning.accounting.time.time',lambda:clock[0])
    state={'limit_seconds':20.,'charged_seconds':3.,'by_arm':{'old':3.},'sessions':[],
        'active':{'pid':999999999,'label':'old','start_wall':100.,'last_wall':103.,'reserved_until':109.}}
    (tmp_path/'gpu_budget.json').write_text(json.dumps(state))
    ledger=GPULedger(tmp_path,limit=20,margin=2);ledger.begin('new')
    assert ledger.remaining()==11
    ledger.end()

def test_concurrent_lease_rejected(tmp_path):
    a=GPULedger(tmp_path,limit=20,margin=2);a.begin('a')
    try:
        with pytest.raises(RuntimeError):GPULedger(tmp_path,limit=20,margin=2).begin('b')
    finally:a.end()
