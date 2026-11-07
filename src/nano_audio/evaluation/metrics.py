"""Strict binary rewards and utterance-weighted research summaries."""
from collections import Counter, defaultdict
import random
from statistics import mean
from nano_audio.data.task import TaskError, compact
from .simulator import simulate

def score_prediction(prediction, target, family, order_matters=True, truncated=False):
    family.encode(target)  # corrupt gold must fail loudly
    try:
        calls = family.decode(prediction)
        valid = True
    except (TaskError, KeyError, TypeError, OverflowError):
        calls, valid = [], False
    normalize = lambda p: p if order_matters else sorted(p, key=compact)
    exact = valid and not truncated and normalize(calls) == normalize(target)
    def atoms(program):
        return Counter((i if order_matters else c['operation'], c['operation'], k, compact(v))
                       for i, c in enumerate(program) for k, v in c['arguments'].items())
    pred, gold = atoms(calls), atoms(target)
    overlap = sum((pred & gold).values())
    precision = overlap / sum(pred.values()) if pred else 0.0
    recall = overlap / sum(gold.values()) if gold else 0.0
    selection = [c['operation'] for c in calls]
    expected = [c['operation'] for c in target]
    return {'valid': valid, 'exact': bool(exact), 'reward': int(exact),
            'tool_accuracy': (selection == expected if order_matters else sorted(selection) == sorted(expected)),
            'argument_precision': precision, 'argument_recall': recall,
            'argument_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'joint_argument_accuracy': valid and normalize(calls) == normalize(target),
            'ordered_program_success': valid and not truncated and calls == target,
            'truncated': truncated, 'trace': simulate(calls, family=family) if valid else None}

def reward(prediction, target, family, **kwargs):
    return score_prediction(prediction, target, family, **kwargs)['reward']

def aggregate(rows):
    """Rows contain utterance_group_id plus numeric score fields; average siblings first."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['utterance_group_id']].append(row)
    keys = ('exact', 'valid', 'tool_accuracy', 'argument_precision', 'argument_recall', 'argument_f1', 'joint_argument_accuracy', 'ordered_program_success')
    return {'recordings': len(rows), 'groups': len(groups), **{k: mean(mean(float(r[k]) for r in g) for g in groups.values()) if groups else None for k in keys}}

def paired_bootstrap(left, right, seed=17, samples=2000):
    """Inputs map group IDs to lists of schema/voice scores. Never bootstrap rows."""
    if set(left) != set(right) or not left or samples < 1:
        raise TaskError('paired nonempty groups and positive sample count required')
    differences = [mean(left[k]) - mean(right[k]) for k in sorted(left)]
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    return {'difference': mean(differences), 'low': draws[int(.025 * (samples - 1))], 'high': draws[int(.975 * (samples - 1))], 'groups': len(differences)}
