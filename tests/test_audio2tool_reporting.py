import json
from pathlib import Path
import pytest
from nano_learning.selection import select_endpoint, select_warmups
from nano_learning.schedules import validation_kinds
from nano_learning.evidence import validation_fields, validation_dir, roots
from nano_learning.common import ROOT, sha


@pytest.fixture(autouse=True)
def historical_reporting_inputs(request):
    historical_tests={'test_historical_selection_is_immutable','test_schedule_matches_recorded_checks','test_paper_hash_and_declared_numeric_discrepancies'}
    if getattr(request.node,'originalname',request.node.name) in historical_tests and not (roots()['pilot']/'parent_selection.json').exists():
        pytest.skip('Historical paper reporting inputs are optional for a fresh replication')


def candidate(step,score,pool,cohort='full_validation'):
    return dict(step=step,macro_exact=score,fraction=pool,cohort=cohort)


def test_endpoint_selection_ignores_early_peaks_and_diagnostics():
    rows=[candidate(256,.99,1.),candidate(640,.2,.1),candidate(640,.2,.3),candidate(640,1.,1.,'mini')]
    assert select_endpoint(rows)['fraction']==.1
    with pytest.raises(ValueError):select_endpoint([candidate(256,.9,.1)])
    with pytest.raises(ValueError):select_endpoint([candidate(640,float('nan'),.1)])


def test_historical_selection_is_immutable():
    p=roots()['pilot']/'parent_selection.json';before=sha(p)
    result=select_warmups()
    assert all(r['fraction']==.1 and r['step']==640 for r in result.values())
    old=json.loads(p.read_text())
    assert old['mellow']['fraction']==1 and old['mellow']['step']==256
    assert sha(p)==before


def test_compatibility_reader_does_not_rewrite_keys():
    old={'dev/full_macro_exact':.2,'update':64}
    assert validation_fields(old)=={'validation/full_macro_exact':.2,'update':64}
    assert 'dev/full_macro_exact' in old


@pytest.mark.parametrize('stage,run',[
    ('warmup',('pilot','samsone-warmup-010')),('warmup',('pilot','mellow-warmup-100')),
    ('pilot64',('pilot','mellow-continuation-sft')),
    ('continuation1024',('followup','samsone-sft')),
    ('continuation1024',('followup','samsone-binary')),
    ('continuation1024',('mellow','mellow-sft')),
    ('continuation1024',('extension','typed1024')),('sft4096',('extension','sft4096'))])
def test_schedule_matches_recorded_checks(stage,run):
    from nano_learning.schedules import SPECS
    recorded={p.stem for p in validation_dir(roots()[run[0]]/'runs'/run[1]).glob('*.json')}
    expected={f'{step:06d}_{kind}' for step in range(SPECS[stage]['updates']+1) for kind in validation_kinds(stage,step)}
    assert recorded==expected


def test_paper_hash_and_declared_numeric_discrepancies():
    from nano_learning.audit import build_register
    result=build_register()
    assert result['numeric_cells']==212
    mismatches={(r['table'],r['row'],r['column']) for r in result['discrepancies']}
    assert mismatches=={*{('tab:a2t-pilot-test',i,3) for i in range(5,9)},
        ('tab:audio2tool-hours',4,4),('tab:audio2tool-hours',9,4)}
    panel=json.loads((ROOT/'paper/generated/evidence/panel.json').read_text())
    assert (panel['recording_entries'],panel['distinct_requests'],panel['source_components'])==(40,40,36)


def test_no_removed_import_dependencies():
    import ast
    for folder in ['nano_learning','src/nano_audio','maintenance']:
        for path in (ROOT/folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
                assert not any(n.split('.')[0] in ('phase2','phase3','phase4','semi_pilot') for n in names),path


def test_active_local_import_targets_exist():
    import ast
    for prefix,folder in [('nano_learning',ROOT/'nano_learning'),('nano_audio',ROOT/'src/nano_audio')]:
        for path in folder.rglob('*.py'):
            parts=[prefix,*path.relative_to(folder).with_suffix('').parts]
            package=parts[:-1]
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node,ast.ImportFrom):continue
                if node.level:
                    module='.'.join(package[:len(package)-node.level+1]+((node.module or '').split('.') if node.module else []))
                else:module=node.module or ''
                if module.startswith('nano_learning.') or module.startswith('nano_audio.'):
                    base=ROOT if module.startswith('nano_learning.') else ROOT/'src'
                    target=base.joinpath(*module.split('.'))
                    assert target.with_suffix('.py').exists() or (target/'__init__.py').exists(),(path,module)


def test_scorer_and_trainer_bytes_preserved():
    index=json.loads((ROOT/'nano_learning/resources/scoring_baseline.json').read_text())
    for name in ['nano_learning/task.py','src/nano_audio/training/core.py']:
        assert sha(ROOT/name)==index[name]
