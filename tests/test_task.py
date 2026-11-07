"""Offline protocol tests: no checkpoints, synthesis or network required."""
from copy import deepcopy
import json
import sys
import pytest
from nano_audio.data.task import SchemaFamily, TaskError, strict_json, build_prompt, schema_families, OPERATIONS
from nano_audio.data.synthetic import generate_groups, expand_groups, audit_splits, composition_reserve, exclude_cross_partition_duplicates
from nano_audio.data.slurp import parse_quantity, map_annotation
from nano_audio.data.cache import feature_cache_key, add_noise
from nano_audio.data.synthesis import audio_quality, discover_voices
from nano_audio.evaluation import score_prediction, simulate, aggregate, paired_bootstrap

PROGRAMS = [
    [{'operation': 'set_power', 'arguments': {'device': 'kitchen_light', 'state': 'on'}}],
    [{'operation': 'set_brightness', 'arguments': {'device': 'kitchen_light', 'percent': 25}}],
    [{'operation': 'set_volume', 'arguments': {'device': 'speaker', 'percent': 50}}],
    [{'operation': 'set_temperature', 'arguments': {'device': 'thermostat', 'celsius': 20}}],
    [{'operation': 'start_timer', 'arguments': {'label': 'tea', 'seconds': 90}}],
    [{'operation': 'cancel_timer', 'arguments': {'label': 'tea'}}],
    [{'operation': 'set_playback', 'arguments': {'device': 'speaker', 'state': 'paused'}}],
    [{'operation': 'query_device_state', 'arguments': {'device': 'radio'}}],
]

@pytest.mark.parametrize('family', schema_families())
@pytest.mark.parametrize('program', PROGRAMS)
def test_full_catalog_binding_roundtrip(family, program):
    target = family.encode(program)
    assert family.decode(json.dumps(target)) == program
    assert score_prediction(json.dumps(target), program, family)['reward'] == 1
    assert len(family.tools) == 8
    for tool in family.tools:
        assert tool['description'].isascii() and tool['description'].islower()

@pytest.mark.parametrize('raw', ['{"calls":[],"calls":[]}', '{"a":{"x":1,"x":2}}', 'NaN', 'Infinity', '{} trailing', '```json\n{}\n```'])
def test_strict_json_rejects(raw):
    with pytest.raises(TaskError): strict_json(raw)

def test_wrong_types_extra_keys_and_calls():
    f = SchemaFamily(renamed=False)
    gold = PROGRAMS[1]
    for value in (True, 25.0, '25', -1, 101):
        bad = f.encode(gold)
        bad['calls'][0]['arguments']['percent'] = value
        assert not score_prediction(json.dumps(bad), gold, f)['valid']
    for edit in ('extra', 'missing', 'unknown', 'enum', 'capability', 'top', 'three'):
        target = f.encode(PROGRAMS[0])
        a = target['calls'][0]['arguments']
        if edit == 'extra': a['hidden'] = 1
        if edit == 'missing': del a['device']
        if edit == 'unknown': target['calls'][0]['name'] = 'other'
        if edit == 'enum': a['state'] = 'yes'
        if edit == 'capability': target['calls'][0]['name'] = 'set_volume'; a.clear(); a.update(device='kitchen_light', percent=10)
        if edit == 'top': target['explanation'] = 'ok'
        if edit == 'three': target['calls'] *= 3
        assert not score_prediction(json.dumps(target), PROGRAMS[0], f)['valid']

def test_extra_side_effects_order_and_truncation():
    f = SchemaFamily(renamed=False)
    on = PROGRAMS[0][0]
    off = deepcopy(on); off['arguments']['state'] = 'off'
    gold = [on, off]
    assert not score_prediction(json.dumps(f.encode([off, on])), gold, f)['exact']
    assert score_prediction(json.dumps(f.encode([off, on])), gold, f, order_matters=False)['exact']
    assert not score_prediction(json.dumps(f.encode([on, off])), [off], f)['exact']
    assert not score_prediction(json.dumps(f.encode([on])), [on], f, truncated=True)['reward']
    trace = simulate(gold)
    assert len(trace['events']) == 2 and trace['state']['devices']['kitchen_light']['power'] == 'off'
    query = PROGRAMS[-1][0]
    assert simulate([query])['events'][0]['result'] == {}

def test_presented_names_nested_and_unit_constraints():
    f = SchemaFamily('new', nested=True, units=True)
    assert not score_prediction(json.dumps(SchemaFamily(renamed=False).encode(PROGRAMS[0])), PROGRAMS[0], f)['valid']
    out = f.encode(PROGRAMS[4])
    assert out['calls'][0]['arguments']['payload']['field_1'] == 1.5
    out['calls'][0]['arguments']['payload']['field_1'] = .001
    assert not score_prediction(json.dumps(out), PROGRAMS[4], f)['valid']
    assert 'enabled means on' in f.tools[0]['description']

@pytest.mark.parametrize('profile,count', [('smoke',32),('local_pilot',448),('main',5500)])
def test_generation_is_deterministic_and_audited(profile, count):
    groups = generate_groups(profile)
    assert len(groups) == count and groups == generate_groups(profile)
    assert audit_splits(groups, composition_reserve())['status'] == 'passed'
    for split in ('train','dev','test'):
        assert {c['operation'] for r in groups if r['split'] == split for c in r['canonical_target']} == set(OPERATIONS)


def test_schema_voice_expansion_and_no_prompt_metadata():
    groups = generate_groups()
    rows = expand_groups(groups, schema_families())
    assert len({r['recording_id'] for r in rows}) == 48
    assert audit_splits(rows)['groups'] == 32
    row = rows[0]
    before = build_prompt(row)
    row = {**row, 'secret_gold': 'leak', 'transcript': 'never serialize this', 'provenance': {'secret': True}}
    assert build_prompt(row) == before
    assert before.isascii() and before == before.lower()
    assert 'never serialize this' in build_prompt(row, transcript=True)
    short = {**row, 'tools': row['tools'][:-1]}
    with pytest.raises(TaskError): build_prompt(short)
    class Tokenizer:
        def encode(self, text): return list(text)
    with pytest.raises(TaskError): build_prompt(row, tokenizer=Tokenizer(), max_positions=20)

@pytest.mark.parametrize('key', ['utterance_group_id','recording_id','audio_sha256','counterfactual_group_id','source_utterance_id','transcript'])
def test_split_audit_rejects_every_sibling_identity(key):
    a,b = deepcopy(generate_groups()[:2])
    a['split'], b['split'] = 'train','test'
    a[key] = b[key] = 'same shared identity'
    with pytest.raises(TaskError): audit_splits([a,b])

def test_duplicate_exclusion_removes_whole_group():
    a,b,c = deepcopy(generate_groups()[:3])
    b['split'] = 'test'; b['transcript'] = a['transcript'].upper()
    c['utterance_group_id'] = a['utterance_group_id']
    kept, excluded = exclude_cross_partition_duplicates([a,b,c])
    assert not kept and len(excluded) == 2

def test_forbidden_composition():
    op,device,value = composition_reserve()[0]
    row = deepcopy(generate_groups()[0])
    row['canonical_target'] = [{'operation':op,'arguments':{'device':device,'celsius' if op == 'set_temperature' else 'percent':value}}]
    with pytest.raises(TaskError): audit_splits([row], composition_reserve())

def test_quantity_units_and_conservative_slurp():
    assert parse_quantity('one minute','seconds') == 60
    assert parse_quantity('twenty five percent','percent') == 25
    assert parse_quantity('68 degrees fahrenheit','celsius') == 20
    for value in ('twenty twenty percent','some percent','25','nan percent'):
        with pytest.raises((TaskError, ValueError, OverflowError)): parse_quantity(value,'percent')
    row = {'intent':'iot_hue_lighton','tokens':[{'id':0,'surface':'kitchen light'}], 'entities':[{'type':'device_type','span':[0]}]}
    program,reason = map_annotation(row, {'kitchen light':'kitchen_light'})
    assert reason is None and program == PROGRAMS[0]
    row['entities'] = []
    assert map_annotation(row, {'kitchen light':'kitchen_light'})[1] == 'missing_or_ambiguous_device'
    row['intent'] = 'music_play'
    assert map_annotation(row,{})[1] == 'unsupported_intent'

def test_group_weighting_bootstrap_partial_metrics():
    f = SchemaFamily(renamed=False)
    good = score_prediction(json.dumps(f.encode(PROGRAMS[0])), PROGRAMS[0], f)
    bad = score_prediction('{}', PROGRAMS[0], f)
    rows = [{**good,'utterance_group_id':'a'}]*10 + [{**bad,'utterance_group_id':'b'}]
    assert aggregate(rows)['exact'] == .5
    assert paired_bootstrap({'a':[1,1],'b':[0]}, {'a':[0],'b':[0]}, samples=100)['difference'] == .5
    wrong = deepcopy(PROGRAMS[0]); wrong[0]['arguments']['state'] = 'off'
    score = score_prediction(json.dumps(f.encode(wrong)), PROGRAMS[0], f)
    assert score['tool_accuracy'] and score['argument_f1'] == .5 and not score['reward']

def test_audio_noise_quality_cache_invalidation(tmp_path):
    import numpy as np
    x = np.sin(np.arange(24000) / 20) * .1
    quality = audio_quality(x,24000)
    assert quality['duration_seconds'] == 1
    noisy = add_noise(x,15,17)
    assert np.array_equal(noisy,add_noise(x,15,17))
    assert np.isclose(10*np.log10(np.mean(x*x)/np.mean((noisy-x)**2)),15)
    for bad in (np.zeros(10),np.array([float('nan')]),np.ones(200000)):
        with pytest.raises(TaskError): audio_quality(bad,24000)
    identity = ['audio','v1','weights',{'pool':4},'float32']
    key = feature_cache_key(*identity)
    for index in range(5):
        changed = identity.copy(); changed[index] = 'changed'
        assert feature_cache_key(*changed) != key
    assert discover_voices(tmp_path) == []
