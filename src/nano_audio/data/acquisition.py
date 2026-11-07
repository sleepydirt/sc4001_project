"""Small pinned official SLURP annotation acquisition; no implicit audio archive download."""
from pathlib import Path
import hashlib
import json
import re
from urllib.request import Request, urlopen
from .task import TaskError

SLURP_REPO = 'https://github.com/pswietojanski/slurp'
SLURP_AUDIO = 'https://zenodo.org/records/4274930/files/slurp_real.tar.gz'

def _fetch(url, limit=32 * 1024 * 1024):
    with urlopen(Request(url, headers={'User-Agent': 'nano-audio-research/0.1'}), timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit: raise TaskError('metadata exceeds explicit byte budget')
    return data

def acquire_slurp_metadata(output_dir, revision=None, offline=False):
    """Fetch train/devel/test JSONL and licenses; return source manifest.

    Pin repo commit via GitHub API if revision is omitted. offline=True verifies
    all local hashes and never contacts the network. Audio remains explicitly blocked
    until an existing local audio root or individually hosted files are supplied.
    """
    root = Path(output_dir)
    path = root / 'source_manifest.json'
    if offline:
        manifest = json.loads(path.read_text())
        for name, info in manifest['files'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != info['sha256']: raise TaskError('source checksum mismatch')
        return manifest
    if revision is None:
        revision = json.loads(_fetch('https://api.github.com/repos/pswietojanski/slurp/commits/master'))['sha']
    if not re.fullmatch('[0-9a-f]{40}', revision): raise TaskError('immutable SLURP commit required')
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ('train.jsonl', 'devel.jsonl', 'test.jsonl'):
        url = f'https://raw.githubusercontent.com/pswietojanski/slurp/{revision}/dataset/slurp/{name}'
        data = _fetch(url)
        # Validate the acquired annotation structure before writing.
        for line in data.splitlines():
            row = json.loads(line)
            if not {'slurp_id', 'sentence', 'intent', 'entities', 'tokens', 'recordings'} <= row.keys(): raise TaskError('unexpected SLURP annotation format')
        (root / name).write_bytes(data)
        files[name] = {'url': url, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'rows': len(data.splitlines())}
    manifest = {'source': 'slurp', 'source_url': SLURP_REPO, 'revision': revision, 'files': files,
                'text_license': 'cc-by-4.0', 'audio_license': 'cc-by-nc-4.0',
                'audio': {'status': 'blocked', 'url': SLURP_AUDIO,
                          'reason': 'official distribution is a multi-gigabyte gzip tar archive, without an individual-file index; selected random access is unavailable',
                          'next_action': 'supply an already extracted slurp_real directory, or explicitly authorize the official archive download'}}
    path.write_text(json.dumps(manifest, indent=2))
    return manifest

def attach_local_audio(records, audio_root):
    """Resolve only selected recordings, audit task window; count missing/rejected audio."""
    from collections import Counter
    import soundfile as sf
    import numpy as np
    from .synthesis import audio_quality
    root = Path(audio_root).resolve()
    result, excluded = [], Counter()
    for r in records:
        path = (root / Path(r['audio_path']).name).resolve()
        if path.parent != root or not path.exists():
            excluded['missing_audio'] += 1
            continue
        try:
            samples, sr = sf.read(path)
            if samples.ndim == 2: samples = samples.mean(axis=1)
            quality = audio_quality(samples, sr)
        except (RuntimeError, TaskError):
            excluded['invalid_or_overlong_audio'] += 1
            continue
        result.append({**r, 'audio_path': str(path), 'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'sample_rate': sr, **quality})
    return result, dict(excluded)
