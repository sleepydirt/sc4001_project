"""Report tests use only local artifact fixtures; no model imports/downloads."""
import json
from pathlib import Path

import pytest
from nano_audio.reporting import report as reporting


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, 'ROOT', tmp_path)
    return tmp_path


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def ledger(root, run, rows, profile='smoke', seed=17):
    path = root/f'artifacts/runs/{profile}/{run}/seed_{seed}/learning.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def test_empty_report_does_not_invent_evidence(root):
    result = reporting.build_report('smoke')
    assert not result['experiment_complete']
    assert not result['positive_rl_claim_supported']
    assert result['runs'] == []
    assert result['human_speech_primary']['status'] == 'unavailable'
    assert result['hardware'] is None
    assert all(x['status'] == 'implementation_files_missing' for x in result['implementation_audit'])
    assert (root/'artifacts/reports/smoke/research_report.md').is_file()
    assert json.loads((root/'artifacts/reports/smoke/research_report.json').read_text())['profile'] == 'smoke'


def test_all_zero_reward_and_equal_groups_preclude_rl_claim(root):
    ledger(root, 'S_P_GRPO', [dict(step=1, objective='grpo', profile='smoke',
           reward_mean=0., equal_reward_group_rate=1., loss=.2, kl=.4,
           elapsed_seconds=2., peak_memory_bytes=512)])
    result = reporting.build_report('smoke')
    run = result['runs'][0]
    assert run['all_groups_zero_reward_advantage']
    assert run['all_sampled_rewards_zero']
    assert run['rl_signal'] == 'no_reward_advantage'
    assert run['evidence_class'] == 'smoke_only'
    assert not run['positive_rl_claim_supported']
    assert run['elapsed_seconds'] == 2
    assert run['peak_memory_bytes'] == 512
    assert run['curves']
    for curve in run['curves']:
        assert '<svg' in (root/'artifacts/reports/smoke'/curve).read_text()
    text = (root/'artifacts/reports/smoke/research_report.md').read_text()
    assert 'every recorded GRPO group has zero reward advantage' in text


def test_equal_nonzero_rewards_still_zero_advantage(root):
    ledger(root, 'F_GRPO', [dict(step=1, objective='grpo', reward_mean=1., equal_reward_group_rate=1.)])
    run = reporting.build_report('smoke')['runs'][0]
    assert run['all_groups_zero_reward_advantage']
    assert not run['all_sampled_rewards_zero']


def test_reward_variation_is_not_improvement_and_profiles_seeds_separate(root):
    rows = [dict(step=1, objective='grpo', reward_mean=.5, equal_reward_group_rate=0.)]
    ledger(root, 'S_P_GRPO', rows, seed=17)
    ledger(root, 'S_P_GRPO', rows, seed=42)
    ledger(root, 'MAIN_ONLY', rows, profile='main')
    result = reporting.build_report('smoke')
    assert len(result['runs']) == 2
    assert all(r['rl_signal'] == 'reward_variation_observed_not_improvement' for r in result['runs'])
    assert all(not r['positive_rl_claim_supported'] for r in result['runs'])
    assert {r['seed'] for r in result['runs']} == {'seed_17', 'seed_42'}


def test_artifact_inventory_hardware_predictions_and_warnings(root):
    write(root, 'artifacts/environment.json', {'cuda': {'name': 'recorded device'}})
    write(root, 'artifacts/runs/smoke/S_P1/seed_17/stage.json',
          {'status':'completed', 'profile':'smoke', 'updates':1, 'dev_metrics':{'exact_accuracy':.25}})
    write(root, 'artifacts/runs/smoke/S_P1/seed_17/dev_predictions.json', [{'id':'one'}])
    checkpoint = root/'artifacts/runs/smoke/S_P1/seed_17/checkpoint/policy'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'not a pickle: do not load')
    path = ledger(root, 'S_P1', [dict(step=1,objective='sft',loss=3.), dict(step=1,objective='sft',loss=2.)])
    with path.open('a') as handle:
        handle.write('{incomplete')
    result = reporting.build_report('smoke')
    assert result['hardware']['cuda']['name'] == 'recorded device'
    assert result['runs'][0]['dev_prediction_count'] == 1
    assert result['runs'][0]['checkpoints'][0]['bytes'] == checkpoint.stat().st_size
    assert len(result['warnings']) == 2
    assert result['runs'][0]['dev_metrics']['exact_accuracy'] == .25
    assert not result['implementation_complete']


def test_invalid_profile_does_not_write(root):
    with pytest.raises(ValueError):
        reporting.build_report('../elsewhere')
    assert not (root/'artifacts').exists()
