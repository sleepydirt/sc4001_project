"""Read-only compatibility boundary for immutable historical artifact names."""
from pathlib import Path
import json
from .common import ROOT, ARTIFACTS, STUDY


def split_path(study, split):
    path = Path(study) / 'data' / f'{split}.jsonl'
    if not path.exists() and split == 'validation':
        return Path(study) / 'data/dev.jsonl'
    return path


def validation_dir(run):
    path = Path(run) / 'validation'
    return path if path.exists() else Path(run) / 'dev'


def validation_fields(record):
    """Normalize names on a copy; never rewrite historical W&B/local records."""
    return {('validation/' + k[4:] if k.startswith('dev/') else k): v for k, v in record.items()}


def resolve_path(path):
    path = Path(path)
    parts = path.parts
    if 'artifacts' in parts:
        return ARTIFACTS.joinpath(*parts[parts.index('artifacts') + 1:])
    return path if path.is_absolute() else ROOT / path


def load(path):
    return json.loads(Path(path).read_text())


def roots():
    return {'pilot': STUDY,
            'followup': ARTIFACTS / 'audio2tool_followup/20261114_seed17',
            'extension': ARTIFACTS / 'audio2tool_extension/20261115_seed17',
            'mellow': ARTIFACTS / 'audio2tool_mellow_followup/20261115_seed17'}


def historical_roots():
    result=roots()
    result['pilot']=ARTIFACTS / 'audio2tool_pilot/20261114_seed17'
    return result


def panels_path(study=STUDY):
    path=Path(study)/'validation_panels.json'
    return path if path.exists() else Path(study)/'development_panels.json'
