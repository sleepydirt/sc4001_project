"""Generic public-registry binding and strict typed call-program reward, v1.

No gold-driven catalogs, transcript prompts, call-count cap or executable parsing.
"""
from __future__ import annotations
import ast
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
from scipy.optimize import linear_sum_assignment
from .common import digest

VERSION = 'a2t-typed-v1'
INTERFACE_VERSION = 'a2t-registry-json-v1'

class GoldError(ValueError):
    pass

def strict_json(text):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError('duplicate JSON key')
            d[k] = v
        return d
    def constant(x):
        raise ValueError('nonfinite JSON constant')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)

def literal(node):
    # Accept JSON spellings in source pseudo-Python calls, never in string values.
    if isinstance(node, ast.Name) and node.id in ('true', 'false', 'null'):
        return {'true': True, 'false': False, 'null': None}[node.id]
    if isinstance(node, ast.Name):
        raise ValueError('Unquoted nonliteral source value')
    if isinstance(node, ast.List):
        return [literal(x) for x in node.elts]
    if isinstance(node, ast.Tuple):
        return [literal(x) for x in node.elts]
    if isinstance(node, ast.Dict):
        keys = [literal(k) for k in node.keys]
        if any(type(k) is not str for k in keys) or len(set(keys)) != len(keys):
            raise ValueError('non-string or duplicate source dictionary keys')
        return {k: literal(v) for k, v in zip(keys, node.values)}
    value = ast.literal_eval(node)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite source value')
    return value

def source_program(text, *, bare_arguments=None):
    tree = ast.parse(text.strip(), mode='exec')
    nodes = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr):
            raise ValueError('Only literal function invocations allowed')
        node = statement.value
        nodes.extend(node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node])
    result = []
    for node in nodes:
        if isinstance(node, ast.Name) and bare_arguments is not None:
            result.append({'name': node.id, 'arguments': bare_arguments})
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.args:
            keys = [k.arg for k in node.keywords]
            if None in keys or len(keys) != len(set(keys)):
                raise ValueError('Expanded or repeated source keyword')
            result.append({'name': node.func.id, 'arguments': {k.arg: literal(k.value) for k in node.keywords}})
        else:
            raise ValueError('Incomplete or nonliteral source call')
    if not result:
        raise ValueError('Empty source program')
    return result

def compact(value):
    # Universal lowercase ASCII JSON surface, reversible even for uppercase/Unicode
    # argument values. Both model families receive exactly this representation.
    text = json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    return ''.join('\\u%04x' % ord(c) if 'A' <= c <= 'Z' else c for c in text)

def typed(value):
    if isinstance(value, dict):
        return ('dict', tuple(sorted((k, typed(v)) for k, v in value.items())))
    if isinstance(value, list):
        return ('list', tuple(typed(x) for x in value))
    return (type(value).__name__, value)

class Registry:
    def __init__(self, path):
        self.rows = list(csv.DictReader(Path(path).open()))
        self.tools = {}
        self.ids = {}
        for row in self.rows:
            domain, alias = row['domain'], row['tool_name'].lower()
            key = (domain, alias)
            if key in self.tools:
                raise ValueError('Registry name alias collision')
            signature = row['signature']
            name, params = signature.split('(', 1)
            if name != row['tool_name'] or not params.endswith(')'):
                raise ValueError('Registry signature/name disagreement')
            parameters = {}
            for part in params[:-1].split(','):
                if not part.strip():
                    continue
                k, t = part.strip().split(':', 1)
                if k.lower() in parameters:
                    raise ValueError('Registry key alias collision')
                parameters[k.lower()] = {'public_name': k, 'type': t.strip()}
            defaults = {}
            if row['argument_defaults'].strip():
                defaults = source_program('f(' + row['argument_defaults'] + ')')[0]['arguments']
            constraints = {k.lower(): v.strip() for k, v in re.findall(r'(\w+):\s*\[([^\]]*)\]', row['argument_constraints'])}
            for alias_key, spec in parameters.items():
                public_key = spec['public_name']
                if public_key in defaults:
                    spec['default'] = defaults[public_key]
                if alias_key in constraints:
                    constraint = constraints[alias_key]
                    match = re.fullmatch(r'\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*', constraint)
                    if match and spec['type'].rstrip('?') in ('Int', 'Float'):
                        spec.update(minimum=float(match[1]), maximum=float(match[2]))
                    elif ':' not in constraint:
                        values = [x.strip() for x in constraint.split(',')]
                        if spec['type'].rstrip('?') == 'Boolean':
                            spec['enum'] = [x.lower() == 'true' for x in values]
                        else:
                            spec['enum'] = values
                    else:
                        spec['compound_constraint'] = constraint
            self.tools[key] = {'name': row['tool_name'], 'alias': alias, 'parameters': parameters, 'raw': row}
            self.ids[(domain, int(row['tool_id']))] = self.tools[key]
        self.hash = digest(self.rows)

    def validate_value(self, spec, value):
        t = spec['type']
        if value is None:
            return t.endswith('?')
        base = t.rstrip('?')
        if base == 'Boolean':
            valid = type(value) is bool
        elif base == 'Int':
            valid = type(value) is int
        elif base == 'Float':
            valid = type(value) in (int, float) and math.isfinite(value)
        elif base.startswith('List<'):
            subtype = base[5:-1]
            valid = type(value) is list and all(self.validate_value({'type': subtype, **({'enum': spec['enum']} if 'enum' in spec else {})}, x) for x in value)
            return valid
        elif base == 'Map':
            valid = type(value) is dict and all(type(k) is str for k in value)
            try:
                json.dumps(value, allow_nan=False)
            except (ValueError, TypeError):
                valid = False
        elif base == 'Goal':
            return False  # No public Goal structure supplied; never infer from labels.
        else:
            valid = type(value) is str
        if not valid:
            return False
        if 'enum' in spec and value not in spec['enum']:
            return False
        if 'minimum' in spec and not spec['minimum'] <= value <= spec['maximum']:
            return False
        if base == 'HexCode' and re.fullmatch(r'#[0-9a-fA-F]{6}', value) is None:
            return False
        if base == 'HsbColor':
            try:
                h, s, b = [float(x) for x in value.split(',')]
                if not (0 <= h <= 360 and 0 <= s <= 100 and 0 <= b <= 100):
                    return False
            except (ValueError, TypeError):
                return False
        return True

    def bind(self, domain, calls, *, strict=True):
        result = []
        for call in calls:
            if type(call) is not dict or set(call) != {'name', 'arguments'} or type(call['name']) is not str or type(call['arguments']) is not dict:
                raise ValueError('Invalid call object')
            tool = self.tools.get((domain, call['name'].lower()))
            if tool is None:
                raise ValueError('Unknown tool')
            args = {}
            for key, value in call['arguments'].items():
                if type(key) is not str or key.lower() in args:
                    raise ValueError('Invalid or aliased duplicate argument')
                args[key.lower()] = value
            # Defaults define omission equivalence. Optional without a default -> null.
            for key, spec in tool['parameters'].items():
                if key not in args:
                    if 'default' in spec:
                        args[key] = spec['default']
                    elif spec['type'].endswith('?'):
                        args[key] = None
                    elif strict:
                        raise ValueError('Missing required argument')
            if strict:
                if set(args) - set(tool['parameters']):
                    raise ValueError('Extra argument')
                if not all(self.validate_value(tool['parameters'][k], v) for k, v in args.items()):
                    raise ValueError('Argument type/range/enum invalid or unsupported')
            result.append({'name': tool['alias'], 'arguments': args})
        return result

    def gold(self, row):
        params = row.get('extracted_params') or {}
        if isinstance(params, str):
            params = literal(ast.parse(params, mode='eval').body)
        if type(params) is not dict:
            raise GoldError('invalid_extracted_params')
        try:
            primary = source_program(row['expected_tool_call'], bare_arguments=params)
            if primary[0]['name'] != row['tool_name']:
                raise GoldError('primary_tool_disagreement')
            if typed(primary[0]['arguments']) != typed(params):
                raise GoldError('call_and_extracted_params_disagree')
            # Embedded function identities are audited, never used for candidates.
            embedded = {f.get('tool_name') for f in row.get('functions', [])}
            if embedded and any(c['name'] not in embedded for c in primary):
                raise GoldError('inline_registry_name_disagreement')
            by_id = self.ids.get((row['domain'], int(row['tool_id'])))
            if by_id is None or by_id['name'] != row['tool_name']:
                raise GoldError('tool_id_registry_disagreement')
            extra = []
            for source in row.get('additional_tool_calls') or []:
                # A bare additional tool name lacks its argument annotation. Defaults
                # cannot recover speech-specific values. Allow true zero-arg tools.
                if '(' not in source:
                    tool = self.tools.get((row['domain'], source.lower()))
                    if tool is None or tool['parameters']:
                        raise GoldError('additional_call_arguments_unannotated')
                    source += '()'
                extra.extend(source_program(source))
            return self.bind(row['domain'], primary + extra, strict=True)
        except GoldError:
            raise
        except (ValueError, SyntaxError, TypeError, KeyError) as e:
            raise GoldError(str(e)) from e

    def audit_flags(self,row):
        """Independent flags; the exclusion funnel separately uses the first one."""
        flags=[];domain=row['domain'];name=row['tool_name']
        by_id=self.ids.get((domain,int(row['tool_id'])))
        if by_id is None or by_id['name']!=name:flags.append('tool_id_registry_disagreement')
        if (domain,name.lower()) not in self.tools:flags.append('tool_name_not_in_registry')
        functions=row.get('functions') or []
        if functions and name not in {f.get('tool_name') for f in functions}:flags.append('inline_registry_name_disagreement')
        params=row.get('extracted_params') or {}
        if isinstance(params,str):
            try:params=literal(ast.parse(params,mode='eval').body)
            except (ValueError,SyntaxError):flags.append('invalid_extracted_params');params={}
        primary=[]
        try:
            primary=source_program(row['expected_tool_call'],bare_arguments=params)
            if primary[0]['name']!=name:flags.append('primary_tool_disagreement')
            if typed(primary[0]['arguments'])!=typed(params):flags.append('call_and_extracted_params_disagree')
        except (ValueError,SyntaxError,TypeError):flags.append('unparseable_primary_call')
        for c in primary:
            try:self.bind(domain,[c],strict=True)
            except ValueError as e:flags.append('schema:'+str(e))
        for text in row.get('additional_tool_calls') or []:
            try:
                if '(' not in text:
                    tool=self.tools.get((domain,text.lower()))
                    if tool is None or tool['parameters']:
                        flags.append('additional_call_arguments_unannotated');continue
                    text+='()'
                extra=source_program(text)
                self.bind(domain,extra,strict=True)
            except (ValueError,SyntaxError,TypeError) as e:flags.append('additional_call_invalid:'+type(e).__name__)
        return list(dict.fromkeys(flags))

    def catalog(self, domain):
        tools = []
        for (d, alias), tool in self.tools.items():
            if d != domain:
                continue
            # Every tool, all public constraints/defaults, descriptive semantics.
            constraints = {k.lower(): v.strip() for k, v in re.findall(r'(\w+):\s*\[([^\]]*)\]', tool['raw']['argument_constraints'])}
            tools.append([alias, {k: [s['type'].lower(), s.get('default'), constraints.get(k, '')]
                for k, s in tool['parameters'].items()}, tool['raw']['description'].lower()])
        return tools

    def prompt(self, domain):
        return ('listen to the complete audio and return only json: {"calls":[{"name":"tool","arguments":{}}]}. '
            'use lowercase tool and parameter names. preserve argument values exactly; json unicode escapes preserve uppercase. '
            'include every requested final call, ignore superseded corrections and background speech. '
            'catalog entries are [name,parameters,description]; each parameter is [type,default,constraint]. '
            'omitted parameters take the listed defaults. tools: ' + compact(self.catalog(domain)) + ' answer: ')

def leaves(value, path=()):
    if isinstance(value, dict) and value:
        return [x for k, v in sorted(value.items()) for x in leaves(v, path + (('key', k),))]
    if isinstance(value, list) and value:
        return [x for i, v in enumerate(value) for x in leaves(v, path + (('index', i),))]
    return [(path, typed(value))]

def score(registry, example, output):
    # Revalidate gold first: corrupt gold is a hard error, never a zero reward.
    try:
        gold = registry.bind(example['domain'], example['calls'], strict=True)
    except (KeyError, ValueError) as e:
        raise GoldError('Frozen gold invalid: ' + str(e)) from e
    result = dict(binary=0., typed=0., tool_multiset_accuracy=0., argument_F1=0., schema_validity=0., parse_validity=0.)
    if output.get('length_capped'):
        return result
    try:
        parsed = strict_json(output['text'])
        if type(parsed) is not dict or set(parsed) != {'calls'} or type(parsed['calls']) is not list or not parsed['calls']:
            return result
        predicted = registry.bind(example['domain'], parsed['calls'], strict=False)
        result['parse_validity'] = 1.
    except (ValueError, TypeError, KeyError):
        return result
    try:
        registry.bind(example['domain'], parsed['calls'], strict=True)
        result['schema_validity'] = 1.
    except ValueError:
        pass
    if Counter(c['name'] for c in predicted) != Counter(c['name'] for c in gold):
        return result
    ordered = example.get('order_matters', False)
    if ordered and [c['name'] for c in predicted] != [c['name'] for c in gold]:
        return result
    result['tool_multiset_accuracy'] = 1.
    # Bijective assignment per repeated tool. Maximize correctly typed leaf matches.
    pairs = list(zip(range(len(predicted)), range(len(gold)))) if ordered else []
    if not ordered:
        for name in sorted({c['name'] for c in gold}):
            pi = [i for i, c in enumerate(predicted) if c['name'] == name]
            gi = [i for i, c in enumerate(gold) if c['name'] == name]
            matrix = [[len(set(leaves(predicted[p]['arguments'])) & set(leaves(gold[g]['arguments']))) for g in gi] for p in pi]
            a, b = linear_sum_assignment(matrix, maximize=True)
            pairs.extend((pi[p], gi[g]) for p, g in zip(a, b))
    correct = sum(len(set(leaves(predicted[p]['arguments'])) & set(leaves(gold[g]['arguments']))) for p, g in pairs)
    npred = sum(len(leaves(c['arguments'])) for c in predicted)
    ngold = sum(len(leaves(c['arguments'])) for c in gold)
    f1 = 2*correct/(npred+ngold) if npred+ngold else 1.
    exact = bool(result['schema_validity']) and all(typed(predicted[p]['arguments']) == typed(gold[g]['arguments']) for p, g in pairs)
    result.update(binary=float(exact), typed=1. if exact else .5*f1, argument_F1=f1)
    return result
