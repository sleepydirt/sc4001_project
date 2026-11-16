from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get('AUDIO2TOOL_ARTIFACTS', ROOT / 'artifacts')).resolve()
STUDY = Path(os.environ.get('AUDIO2TOOL_STUDY', ARTIFACTS / 'audio2tool_pilot/20261114_seed17')).resolve()
RUNTIME = STUDY / 'runtime'
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))  # Existing pinned training packages take precedence.
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
REVISION = 'f1388da9a3189541ab82adac88824a0661670c43'
SCORER_REVISION = 'b9d5383ca30945fbafb965a04bea40117e1e8a82'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')

def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def tracking(name, job_type, config=None, *, root=None):
    import re
    import wandb
    keyfile = ROOT / 'nano_learning/api_key.txt'
    if keyfile.exists():
        content = keyfile.read_text().strip()
        candidates = re.findall(r'(?:wandb_v1_)?[A-Za-z0-9_]{40,}', content)
        if len(candidates) != 1:
            raise ValueError('The W&B key file must contain exactly one API key')
        os.environ['WANDB_API_KEY'] = candidates[0]
    os.environ['WANDB_SILENT'] = 'true'
    root = Path(root) if root is not None else STUDY
    directory = root / 'wandb'
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'team88_sc4001'),
        entity=os.environ.get('WANDB_ENTITY'), name=name,
        id=digest({'study': str(root), 'run': name})[:16], resume='allow',
        group='audio2tool-' + root.name, job_type=job_type,
        config=config or {}, dir=str(directory), mode=os.environ.get('WANDB_MODE', 'offline'),
        settings=wandb.Settings(console='off', init_timeout=60))
    run.define_metric('update')
    for prefix in ('train', 'validation', 'diagnostic', 'compute'):
        run.define_metric(prefix + '/*', step_metric='update')
    append(root / 'wandb_runs.jsonl', {'name': name, 'url': run.url, 'id': run.id, 'time': time.time()})
    print('W&B:', run.url, flush=True)
    return run
