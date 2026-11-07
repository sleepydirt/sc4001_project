"""Program-first deterministic generation and split expansion (no audio side effects)."""
from collections import Counter, defaultdict
from copy import deepcopy
import random
import re
from .task import DEFAULT_DEVICES, OPERATIONS, TaskError, compact, digest

PROFILES = {'smoke': (16, 8, 8), 'local_pilot': (256, 64, 128), 'main': (4000, 500, 1000)}

def normalized_text(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))

def composition_reserve(seed=17):
    """Reserve 10% of device/value tuples without losing any constituent in train."""
    tuples = [(op, d, v) for op in ('set_brightness', 'set_volume', 'set_temperature')
              for d, caps in DEFAULT_DEVICES.items() if op in caps
              for v in ([10, 25, 50, 75, 90] if op != 'set_temperature' else [16, 18, 20, 22, 24])]
    rng = random.Random(seed)
    rng.shuffle(tuples)
    held = []
    for item in tuples:
        remaining = [t for t in tuples if t not in held and t != item]
        if len(held) >= round(.1 * len(tuples)):
            break
        if any(t[:2] == item[:2] for t in remaining) and any((t[0], t[2]) == (item[0], item[2]) for t in remaining):
            held.append(item)
    return held

def tuples_for(program):
    return [(c['operation'], c['arguments']['device'], c['arguments'][k]) for c in program
            for k in ('percent', 'celsius') if k in c['arguments']]

def _call(op, rng):
    args = {}
    for key, spec in OPERATIONS[op].items():
        if key == 'device':
            args[key] = rng.choice([d for d, caps in DEFAULT_DEVICES.items() if op in caps])
        elif type(spec) is list:
            args[key] = rng.choice(spec)
        else:
            args[key] = rng.choice({'percent': [10, 25, 50, 75, 90], 'temperature': [16, 18, 20, 22, 24], 'seconds': [30, 60, 90, 120, 300], 'label': ['tea', 'pasta', 'laundry']}[spec])
    return {'operation': op, 'arguments': args}

def render_call(call):
    op, a = call['operation'], call['arguments']
    d = a.get('device', '').replace('_', ' ')
    if op == 'set_power': return f"turn {d} {a['state']}"
    if op == 'set_brightness': return f"set {d} brightness to {a['percent']} percent"
    if op == 'set_volume': return f"set {d} volume to {a['percent']} percent"
    if op == 'set_temperature': return f"set {d} to {a['celsius']} degrees celsius"
    if op == 'start_timer': return f"start {a['label']} timer for {a['seconds']} seconds"
    if op == 'cancel_timer': return f"cancel {a['label']} timer"
    if op == 'set_playback': return f"set {d} playback to {a['state']}"
    return f'query the state of {d}'

def generate_groups(profile='smoke', seed=17, counts=None):
    counts = counts or dict(zip(('train', 'dev', 'test'), PROFILES[profile]))
    forbidden = composition_reserve(seed)
    rng = random.Random(seed)
    groups, seen = [], set()
    prefixes = {'train': ['please', 'could you', 'now'], 'dev': ['i would like you to', 'kindly'], 'test': ['would you mind helping me', 'my request is to']}
    for split, count in counts.items():
        kinds = ['single'] * round(count * .75) + ['correction'] * round(count * .15)
        kinds = (kinds + ['two_action'] * count)[:count]
        rng.shuffle(kinds)
        for index, kind in enumerate(kinds):
            for attempt in range(10000):
                op = list(OPERATIONS)[index % 8]
                program = [_call(op, rng)]
                if kind == 'two_action': program.append(_call(rng.choice(list(OPERATIONS)), rng))
                if split != 'test' and any(t in forbidden for t in tuples_for(program)): continue
                text = render_call(program[0])
                if kind == 'correction':
                    old = _call(op, rng)
                    if old == program[0]: continue
                    text = render_call(old) + ', no, ' + text
                elif kind == 'two_action': text += ', then ' + render_call(program[1])
                text = rng.choice(prefixes[split]) + ' ' + rng.choice(['', 'please ', 'for me ', 'right now ', 'if possible ', 'when ready ', 'at once ', 'as requested ', 'just ', 'quickly ', 'again ', 'today ', 'this time ', 'now ', 'kindly ', 'for now ']) + text + rng.choice(['', ' please', ' for me', ' now', ' right now', ' if possible', ' thank you', ' thanks', ' when ready', ' at once', ' as requested', ' today', ' this time', ' for now', ' kindly', ' again'])
                norm = normalized_text(text)
                if norm not in seen: break
            else:
                raise TaskError('unique template pool exhausted; reduce quota or add templates')
            seen.add(norm)
            group_id = digest({'seed': seed, 'split': split, 'text': text})[:20]
            groups.append({'utterance_group_id': group_id, 'split': split, 'source': 'synthetic',
                           'transcript': text, 'canonical_target': program, 'task_kind': kind,
                           'order_matters': kind == 'two_action', 'template_family': split,
                           'provenance': {'generation_seed': seed, 'label_source': 'programmatic', 'task_version': 'task-v1'},
                           'forbidden_training_tuples': forbidden})
    audit_splits(groups, forbidden)
    return groups

def expand_groups(groups, families, voices=None):
    """Expand already assigned groups. Test audio receives familiar + held-out catalogs."""
    voices = voices or {'train': ['voice_train_a', 'voice_train_b'], 'dev': ['voice_dev'], 'test': ['voice_test']}
    if any(set(voices[a]) & set(voices[b]) for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]):
        raise TaskError('voice partition overlap')
    records = []
    for group in groups:
        for family in families:
            if family.partition != group['split'] and not (group['split'] == 'test' and family.partition == 'train'): continue
            for voice in ([group['speaker_id']] if group.get('audio_path') else voices[group['split']]):
                if voice not in voices[group['split']]: raise TaskError('recorded voice has wrong partition')
                r = deepcopy(group)
                r.update({'speaker_id': voice, 'schema_family_id': family.family_id, 'schema_partition': family.partition,
                          'binding_id': family.family_id, 'tools': family.tools, 'target': family.encode(group['canonical_target']),
                          'recording_id': digest([group['utterance_group_id'], voice])[:20],
                          'id': digest([group['utterance_group_id'], voice, family.family_id])[:20]})
                records.append(r)
    audit_splits(records)
    return records

def audit_splits(records, forbidden=()):
    owners = defaultdict(set)
    for r in records:
        split = r['split']
        for key in ('utterance_group_id', 'recording_id', 'audio_sha256', 'counterfactual_group_id', 'source_utterance_id'):
            if r.get(key): owners[(key, r[key])].add(split)
        if r.get('transcript'): owners[('text', normalized_text(r['transcript']))].add(split)
        if split == 'train':
            if r.get('schema_partition', 'train') != 'train': raise TaskError('held-out schema in training')
            blocked = forbidden or r.get('forbidden_training_tuples', ())
            if any(tuple(t) in [tuple(x) for x in blocked] for t in tuples_for(r['canonical_target'])):
                raise TaskError('forbidden composition in training')
    overlaps = [key for key, splits in owners.items() if len(splits) > 1]
    if overlaps: raise TaskError('split overlap: ' + repr(overlaps[:5]))
    return {'status': 'passed', 'records': len(records), 'groups': len({r['utterance_group_id'] for r in records}), 'manifest_sha256': digest(records)}

def exclude_cross_partition_duplicates(records):
    owners = defaultdict(set)
    for r in records:
        owners[normalized_text(r['transcript'])].add(r['split'])
    bad_groups = {r['utterance_group_id'] for r in records if len(owners[normalized_text(r['transcript'])]) > 1}
    return ([r for r in records if r['utterance_group_id'] not in bad_groups],
            [{'utterance_group_id': g, 'reason': 'cross_partition_duplicate'} for g in sorted(bad_groups)])
