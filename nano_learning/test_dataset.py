import json
import pytest
from . import dataset

def test_test_loader_guard_and_train_component_schedule(tmp_path,monkeypatch):
    monkeypatch.setattr(dataset,'STUDY',tmp_path)
    (tmp_path/'data').mkdir()
    records=[]
    for i in range(5):
        for voice in range(2):
            records.append({'id':f'{i}-{voice}','component':f'c{i}','request_group':f'q{i}',
                'domain':'d','audio_paths':['a.wav'],'target_text':'{}'})
    (tmp_path/'data/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (tmp_path/'catalogs.json').write_text(json.dumps({'d':{'text':'public catalog'}}))
    with pytest.raises(PermissionError):dataset.Dataset('test')
    train=dataset.Dataset('train');components=[f'c{i}' for i in range(5)]
    a=train.batch(0,components);b=train.batch(0,components)
    assert [r['id'] for r in a]==[r['id'] for r in b]
    assert len({r['component'] for r in a})==4
    assert set(r['component'] for r in a+train.batch(1,components)[:1])==set(components)
    with pytest.raises(ValueError):train.batch(0,['foreign-component'])
    assert all(r['_prompt']=='public catalog' for r in a)
