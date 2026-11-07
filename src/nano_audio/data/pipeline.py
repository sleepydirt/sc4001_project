"""CLI entry point for deterministic, real speech preparation."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from urllib.request import Request, urlopen
from .task import TaskError, schema_families, digest
from .synthetic import generate_groups, expand_groups, audit_splits
from .synthesis import synthesize_groups

ROOT = Path(__file__).resolve().parents[3]

def acquire_kokoro(asset_dir, manifest_path=None):
    """Fetch only pinned checkpoint/config and four discovered English voices."""
    root = Path(asset_dir)
    manifest_path = Path(manifest_path or ROOT / 'artifacts/assets/manifest.json')
    item = json.loads(manifest_path.read_text())['models']['kokoro']
    revision = item['revision']
    available = sorted(f for f in item['files'] if f.startswith(('voices/af_', 'voices/am_')) and f.endswith('.pt'))
    if len(available) < 4: raise TaskError('release has fewer than four disjoint English voices')
    # Prefer distinct female/male voices while deriving names exclusively from pinned manifest.
    available = sorted(available, key=lambda f: (not f.startswith('voices/af_'), f))
    selected = available[:2] + [f for f in available[2:] if f.startswith('voices/am_')][:2]
    if len(selected) != 4: selected = available[:4]
    checkpoint = [f for f in item['files'] if f.startswith('kokoro') and f.endswith('.pth')]
    if len(checkpoint) != 1: raise TaskError('unique Kokoro checkpoint missing from pinned manifest')
    files = ['config.json', checkpoint[0], *selected]
    root.mkdir(parents=True, exist_ok=True)
    for filename in files:
        destination = root / filename
        if destination.exists(): continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{item['repo_id']}/resolve/{revision}/{filename}"
        temporary = destination.with_suffix(destination.suffix + '.partial')
        print('acquiring pinned tts asset: ' + filename, flush=True)
        with urlopen(Request(url, headers={'User-Agent':'nano-audio-research/0.1'}), timeout=60) as source, temporary.open('wb') as sink:
            while chunk := source.read(1024 * 1024): sink.write(chunk)
        temporary.replace(destination)
    hashes = {}
    for filename in files:
        h = hashlib.sha256()
        with (root / filename).open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
        hashes[filename] = h.hexdigest()
    (root / 'source_manifest.json').write_text(json.dumps({'repo_id':item['repo_id'],'revision':revision,'license':item.get('license'),'sha256':hashes},indent=2))
    ids = [Path(v).stem for v in selected]
    return revision, {'train':ids[:2], 'dev':[ids[2]], 'test':[ids[3]]}

def _ensure_g2p(target):
    """Install the small official spaCy model into the run's asset directory only."""
    target = Path(target)
    sys.path.insert(0, str(target))
    import importlib.util
    if importlib.util.find_spec('en_core_web_sm') is not None: return
    url = 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
    target.mkdir(parents=True, exist_ok=True)
    wheel = target.parent / 'en_core_web_sm-3.8.0-py3-none-any.whl'
    if not wheel.exists():
        with urlopen(Request(url, headers={'User-Agent':'nano-audio-research/0.1'}), timeout=60) as response:
            wheel.write_bytes(response.read())
    # This venv may have uv but no pip. Unpack this pure Python official wheel,
    # preserving dist-info so spaCy's package check sees its installed metadata.
    import zipfile
    with zipfile.ZipFile(wheel) as archive:
        if any(Path(n).is_absolute() or '..' in Path(n).parts for n in archive.namelist()): raise TaskError('unsafe model wheel')
        archive.extractall(target)
    import importlib
    importlib.invalidate_caches()
    (target / 'source_manifest.json').write_text(json.dumps({'url':url,'sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'version':'3.8.0'},indent=2))

def prepare_data(profile='smoke', output_dir=None, seed=17):
    """Acquire minimal TTS assets, synthesize profile quotas, write examples.jsonl.

    Returns manifest with paths, source availability and split audit. Synthetic
    examples never stand in for unavailable human speech. SLURP metadata is acquired
    separately by acquire_slurp_metadata; real records can enter via attach_local_audio.
    """
    if output_dir is None: raise TaskError('explicit output directory required')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    groups = generate_groups(profile, seed)
    (output / 'groups.json').write_text(json.dumps(groups, indent=2))
    assets = output / 'tts_assets'
    revision, voices = acquire_kokoro(assets)
    _ensure_g2p(assets / 'python')
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    print('synthesizing profile groups on cpu: ' + str(len(groups)), flush=True)
    recordings = synthesize_groups(groups, assets, output / 'audio', voices, revision, device='cpu', seed=seed, speeds=(1.0,1.15,1.3,1.5))
    examples = expand_groups(recordings, schema_families(), voices)
    audit = audit_splits(examples)
    path = output / 'examples.jsonl'
    path.write_text(''.join(json.dumps(r, separators=(',',':'))+'\n' for r in examples))
    manifest = {'profile':profile,'seed':seed,'status':'completed','condition':'constructed_only',
                'human_speech':{'status':'unavailable','reason':'official SLURP audio archive not acquired; metadata alone is not speech'},
                'independent_groups':len(groups),'recordings':len(recordings),'schema_examples':len(examples),
                'voices':voices,'tts_revision':revision,'examples_path':str(path.resolve()),
                'examples_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'audit':audit}
    (output / 'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output / 'audit.json').write_text(json.dumps(audit,indent=2))
    return manifest
