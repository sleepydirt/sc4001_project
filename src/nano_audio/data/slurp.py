"""Conservative SLURP span mapping. Never derive missing arguments from free text.

Mapping table: iot_hue_lighton/off -> set_power(state on/off),
iot_hue_lightdim -> set_brightness only with explicit numeric percent entity,
iot_hue_lightchange -> excluded (color unsupported), music/volume -> excluded
without explicit device and absolute value. Aliases must be supplied by caller.
Official train/devel/test partitions remain train/dev/test; exclusions are counted.
"""
from collections import Counter
from decimal import Decimal, InvalidOperation
import re
from .task import SchemaFamily, TaskError

WORDS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
WORDS.update(dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10))))

def parse_quantity(text, kind):
    text = text.lower().strip().replace('-', ' ')
    units = {'percent': {'percent': 1, '%': 1}, 'seconds': {'seconds': 1, 'second': 1, 'minutes': 60, 'minute': 60, 'hours': 3600, 'hour': 3600}, 'celsius': {'degrees celsius': 1, 'celsius': 1, 'degrees fahrenheit': 'f', 'fahrenheit': 'f'}}[kind]
    for unit in sorted(units, key=len, reverse=True):
        match = re.fullmatch(r'(.+?)\s*' + re.escape(unit), text)
        if not match: continue
        number = match[1].strip()
        try: value = float(Decimal(number))
        except InvalidOperation:
            tokens = number.split()
            if not tokens or any(t not in WORDS for t in tokens): raise TaskError('unsupported quantity')
            if len(tokens) > 2 or len(tokens) == 2 and not (WORDS[tokens[0]] >= 20 and WORDS[tokens[0]] % 10 == 0 and 0 < WORDS[tokens[1]] < 10):
                raise TaskError('ambiguous number words')
            value = sum(WORDS[t] for t in tokens)
        value = (value - 32) * 5 / 9 if units[unit] == 'f' else value * units[unit]
        if kind != 'celsius':
            if not value.is_integer(): raise TaskError('nonintegral quantity')
            value = int(value)
        return value
    raise TaskError('explicit supported unit required')

def annotation_entities(row):
    out = []
    tokens = {t['id']: t['surface'] for t in row.get('tokens', [])}
    for entity in row.get('entities', []):
        span = entity.get('span', [])
        if not span or any(i not in tokens for i in span): raise TaskError('invalid entity span')
        out.append((entity['type'], ' '.join(tokens[i] for i in span).lower()))
    return out

def map_annotation(row, device_aliases, family=None):
    family = family or SchemaFamily(renamed=False)
    intent = row.get('intent')
    if intent not in ('iot_hue_lighton', 'iot_hue_lightoff', 'iot_hue_lightdim'):
        return None, 'unsupported_intent'
    try:
        entities = annotation_entities(row)
        # Locations alone do not determine a device; aliases must cover the exact entity.
        devices = [device_aliases[v] for k, v in entities if k in ('device', 'device_name', 'device_type') and v in device_aliases]
        if len(devices) != 1: return None, 'missing_or_ambiguous_device'
        op = 'set_brightness' if intent == 'iot_hue_lightdim' else 'set_power'
        args = {'device': devices[0]}
        if op == 'set_power': args['state'] = 'on' if intent.endswith('lighton') else 'off'
        else:
            quantities = [v for k, v in entities if k in ('percent', 'percentage', 'brightness', 'amount')]
            if len(quantities) != 1: return None, 'missing_or_ambiguous_absolute_value'
            args['percent'] = parse_quantity(quantities[0], 'percent')
        allowed = {'device', 'device_name', 'device_type', 'percent', 'percentage', 'brightness', 'amount'}
        if any(k not in allowed for k, _ in entities): return None, 'unsupported_entity_constraint'
        program = [{'operation': op, 'arguments': args}]
        family.encode(program)
        return program, None
    except (TaskError, KeyError, ValueError, OverflowError):
        return None, 'invalid_annotation_or_quantity'

def map_slurp(rows, split, device_aliases, dataset_revision, family=None):
    split = {'devel': 'dev', 'train': 'train', 'test': 'test', 'dev': 'dev'}.get(split)
    if split is None or not dataset_revision: raise TaskError('official split and pinned revision required')
    records, exclusions = [], Counter()
    for row in rows:
        program, reason = map_annotation(row, device_aliases, family)
        if reason:
            exclusions[reason] += 1
            continue
        for recording in row.get('recordings', []):
            records.append({'source': 'slurp', 'split': split, 'source_utterance_id': str(row['slurp_id']),
                            'utterance_group_id': 'slurp_' + str(row['slurp_id']), 'transcript': row['sentence'],
                            'canonical_target': program, 'task_kind': 'single', 'order_matters': True,
                            'audio_path': recording['file'], 'provenance': {'dataset_revision': dataset_revision, 'label_source': 'human_annotations'}})
        if not row.get('recordings'): exclusions['missing_recordings'] += 1
    return records, dict(exclusions)
