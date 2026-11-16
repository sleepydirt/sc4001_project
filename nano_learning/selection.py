"""Corrected warmup selection; historical parent_selection.json is immutable."""
import argparse
import math
from .common import STUDY, write
from .evidence import load, validation_dir

RULE = 'step-640 full-validation macro exact; exact ties prefer the smallest pool'


def select_endpoint(candidates):
    eligible = [r for r in candidates if r['step'] == 640 and r['cohort'] == 'full_validation']
    if not eligible or any(not math.isfinite(r['macro_exact']) for r in eligible):
        raise ValueError('Finite step-640 full-validation measurements required')
    if len({r['fraction'] for r in eligible}) != len(eligible):
        raise ValueError('Duplicate pool endpoint')
    return max(eligible, key=lambda r: (r['macro_exact'], -r['fraction']))


def select_warmups(study=STUDY, *, runs_root=None):
    runs_root = runs_root if runs_root is not None else study / 'runs'
    result = {}
    for model in ('samsone', 'mellow'):
        candidates = []
        for fraction in (.1, .3, 1.):
            name = f'{model}-warmup-{round(100*fraction):03d}'
            run = runs_root / name
            state = load(run / 'status.json')
            if state['status'] != 'completed' or state['step'] != 640:
                raise ValueError(f'Incomplete warmup: {name}')
            measurement = load(validation_dir(run) / '000640_full.json')
            candidates.append(dict(model=model, run=name, fraction=fraction, step=640,
                cohort='full_validation', macro_exact=measurement['macro_exact'],
                schema_validity=measurement['schema_validity'],
                policy=str(run / 'endpoint_policy.pt'), sha256=state['endpoint_sha256'], rule=RULE))
        result[model] = select_endpoint(candidates)
    return result


def main():
    from pathlib import Path
    p=argparse.ArgumentParser(); p.add_argument('--output', required=True)
    p.add_argument('--runs-root',type=Path,help='Directory containing the six completed warmup runs')
    a=p.parse_args()
    if Path(a.output).resolve().is_relative_to(STUDY):
        raise ValueError('Write corrected selection outside immutable historical study')
    write(a.output, select_warmups(runs_root=a.runs_root))

if __name__ == '__main__': main()
