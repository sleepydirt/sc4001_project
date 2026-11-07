"""Versioned task protocol. No optional dependencies or model imports."""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import math
import random

VERSION = 'task-v1'
OPERATIONS = {
    'set_power': {'device': 'device', 'state': ['on', 'off']},
    'set_brightness': {'device': 'device', 'percent': 'percent'},
    'set_volume': {'device': 'device', 'percent': 'percent'},
    'set_temperature': {'device': 'device', 'celsius': 'temperature'},
    'start_timer': {'label': 'label', 'seconds': 'seconds'},
    'cancel_timer': {'label': 'label'},
    'set_playback': {'device': 'device', 'state': ['playing', 'paused', 'stopped']},
    'query_device_state': {'device': 'device'},
}
DEFAULT_DEVICES = {
    'kitchen_light': ['set_power', 'set_brightness', 'query_device_state'],
    'bedroom_light': ['set_power', 'set_brightness', 'query_device_state'],
    'speaker': ['set_power', 'set_volume', 'set_playback', 'query_device_state'],
    'radio': ['set_power', 'set_volume', 'set_playback', 'query_device_state'],
    'thermostat': ['set_power', 'set_temperature', 'query_device_state'],
    'heater': ['set_power', 'set_temperature', 'query_device_state'],
}
DESCRIPTIONS = ['set device power on or off.', 'set light brightness.', 'set audio volume.',
                'set device temperature.', 'start or replace a labelled countdown.',
                'cancel a labelled countdown.', 'set media playback state.', 'read device state.']

class TaskError(ValueError):
    pass

def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()

def strict_json(text):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise TaskError('duplicate key: ' + k)
            out[k] = v
        return out
    try:
        return json.loads(text, object_pairs_hook=pairs,
                          parse_constant=lambda x: (_ for _ in ()).throw(TaskError('nonfinite number')))
    except (ValueError, TypeError, RecursionError) as exc:
        raise TaskError(str(exc)) from exc

def validate(value, schema):
    kind = schema['type']
    ok = {'object': lambda: type(value) is dict, 'string': lambda: type(value) is str,
          'integer': lambda: type(value) is int,
          'number': lambda: type(value) in (int, float) and math.isfinite(value)}[kind]()
    if not ok:
        raise TaskError('wrong type: expected ' + kind)
    if kind == 'object':
        props = schema['properties']
        if set(value) - set(props) or set(schema['required']) - set(value):
            raise TaskError('missing or extra arguments')
        for k, v in value.items():
            validate(v, props[k])
    if 'enum' in schema and value not in schema['enum']:
        raise TaskError('invalid enum')
    if 'minimum' in schema and value < schema['minimum'] or 'maximum' in schema and value > schema['maximum']:
        raise TaskError('out of range')

def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}

@dataclass
class SchemaFamily:
    family_id: str = 'train_01'
    partition: str = 'train'
    renamed: bool = True
    nested: bool = False
    units: bool = False
    devices: dict = None
    temperature_range: tuple = (5, 35)

    def __post_init__(self):
        self.devices = deepcopy(DEFAULT_DEVICES if self.devices is None else self.devices)
        if not self.devices or any(not n.isascii() or n.lower() != n for n in self.devices):
            raise TaskError('device vocabulary must be lowercase ascii')
        if self.temperature_range[0] >= self.temperature_range[1]:
            raise TaskError('invalid temperature range')

    def name(self, operation):
        return self.family_id + '_action_' + str(list(OPERATIONS).index(operation)) if self.renamed else operation

    def key(self, operation, key):
        return 'field_' + str(list(OPERATIONS[operation]).index(key)) if self.renamed else key

    def transform(self, key, value, inverse=False):
        if not self.units:
            return value
        if key == 'celsius':
            result = (value - 32) * 5 / 9 if inverse else value * 9 / 5 + 32
            return round(result, 9)
        if key in ('seconds', 'percent'):
            result = value * (60 if key == 'seconds' else 100) if inverse else value / (60 if key == 'seconds' else 100)
            if inverse:
                if abs(result - round(result)) > 1e-7:
                    raise TaskError('nonintegral canonical quantity')
                return int(round(result))
            return result
        if key == 'state':
            table = {'on': 'enabled', 'off': 'disabled', 'playing': 'active', 'paused': 'held', 'stopped': 'ended'}
            return {v: k for k, v in table.items()}[value] if inverse else table[value]
        return value

    @property
    def tools(self):
        tools = []
        for index, (op, args) in enumerate(OPERATIONS.items()):
            props = {}
            bindings = []
            for key, spec in args.items():
                exposed = self.key(op, key)
                if type(spec) is list:
                    rule = {'type': 'string', 'enum': [self.transform(key, v) for v in spec]}
                elif spec == 'device':
                    rule = {'type': 'string', 'enum': [d for d, caps in self.devices.items() if op in caps]}
                elif spec == 'label':
                    rule = {'type': 'string', 'enum': ['tea', 'pasta', 'laundry']}
                else:
                    low, high = {'percent': (0, 100), 'temperature': self.temperature_range, 'seconds': (1, 86400)}[spec]
                    rule = {'type': 'number' if self.units or spec == 'temperature' else 'integer',
                            'minimum': self.transform(key, low), 'maximum': self.transform(key, high)}
                props[exposed] = rule
                unit = {'celsius': 'fahrenheit' if self.units else 'celsius',
                        'seconds': 'minutes' if self.units else 'seconds',
                        'percent': 'fraction from zero to one' if self.units else 'percent'}.get(key, key)
                bindings.append(exposed + ' is ' + unit + ('; enabled means on; disabled means off; active means playing; held means paused; ended means stopped' if key == 'state' and self.units else ''))
            params = obj(props)
            if self.nested:
                params = obj({'payload': params})
            tools.append({'name': self.name(op), 'description': DESCRIPTIONS[index] + ' ' + '; '.join(bindings), 'parameters': params})
        return tools

    def encode(self, program):
        calls = []
        # Validate canonical arguments under an identity catalog first.
        identity = SchemaFamily('canonical', renamed=False, devices=self.devices, temperature_range=self.temperature_range)
        canonical_tools = {t['name']: t for t in identity.tools}
        for call in program:
            if set(call) != {'operation', 'arguments'} or call['operation'] not in OPERATIONS:
                raise TaskError('invalid canonical call')
            op, args = call['operation'], call['arguments']
            validate(args, canonical_tools[op]['parameters'])
            mapped = {self.key(op, k): self.transform(k, v) for k, v in args.items()}
            calls.append({'name': self.name(op), 'arguments': {'payload': mapped} if self.nested else mapped})
        result = {'calls': calls}
        self.decode(result)
        return result

    def decode(self, output):
        if type(output) is str:
            output = strict_json(output)
        if type(output) is not dict or set(output) != {'calls'} or type(output['calls']) is not list or not 1 <= len(output['calls']) <= 2:
            raise TaskError('expected one or two calls')
        lookup = {self.name(op): op for op in OPERATIONS}
        schemas = {t['name']: t['parameters'] for t in self.tools}
        result = []
        for call in output['calls']:
            if type(call) is not dict or set(call) != {'name', 'arguments'} or type(call['name']) is not str or call['name'] not in lookup:
                raise TaskError('unknown or malformed tool')
            name = call['name']
            validate(call['arguments'], schemas[name])
            args = call['arguments']['payload'] if self.nested else call['arguments']
            op = lookup[name]
            canonical = {k: self.transform(k, args[self.key(op, k)], inverse=True) for k in OPERATIONS[op]}
            # Unit transforms can expose nonintegral values; canonical range/type remains authoritative.
            identity = SchemaFamily('identity', renamed=False, devices=self.devices, temperature_range=self.temperature_range)
            validate(canonical, next(t['parameters'] for t in identity.tools if t['name'] == op))
            result.append({'operation': op, 'arguments': canonical})
        return result


def schema_families():
    return ([SchemaFamily(f'train_{i:02}', 'train') for i in range(1, 5)] +
            [SchemaFamily(f'dev_{i:02}', 'dev', nested=bool(i % 2), units=True) for i in range(1, 3)] +
            [SchemaFamily(f'names_{i:02}', 'test') for i in range(1, 5)] +
            [SchemaFamily(f'structure_{i:02}', 'test', nested=True, units=bool(i % 2)) for i in range(1, 5)])


def build_prompt(record, transcript=False, tokenizer=None, max_positions=2048, audio_positions=0, completion_positions=128):
    tools = record['tools']
    if len(tools) != 8 or len({t['name'] for t in tools}) != 8:
        raise TaskError('complete eight-tool catalog required')
    public = {'instruction': record.get('instruction', 'return the requested calls as json using the supplied tools.'), 'tools': tools}
    if transcript:
        public['transcript'] = record['transcript']
    prompt = compact(public).lower()
    if not prompt.isascii():
        raise TaskError('public prompt must be lowercase ascii')
    if tokenizer is not None and len(tokenizer.encode(prompt)) + audio_positions + completion_positions > max_positions:
        raise TaskError('complete catalog exceeds context budget')
    return prompt
