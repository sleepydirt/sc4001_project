"""Pinned all-tier metadata discovery and resumable, hash-verified acquisition."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from huggingface_hub import HfApi, hf_hub_download
from .common import ROOT, STUDY, REVISION, SCORER_REVISION, write, append, rows, sha

def download(remote):
    # Prefer the existing exact-revision cache; new files go into study storage.
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache('RVtech/Audio2Tool', remote, repo_type='dataset', revision=REVISION)
    if isinstance(cached, str) and Path(cached).is_file():
        return cached
    return hf_hub_download('RVtech/Audio2Tool', remote, repo_type='dataset', revision=REVISION,
                           cache_dir=STUDY / 'hf_cache')

def discover():
    STUDY.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    existing = STUDY / 'remote_files.json'
    files = json.loads(existing.read_text()) if existing.exists() else api.list_repo_files('RVtech/Audio2Tool', repo_type='dataset', revision=REVISION)
    write(existing, files)
    selected = [p for p in files if p.endswith(('.json', '.jsonl', '.csv', '.md'))]
    inventory = STUDY / 'metadata_inventory.jsonl'
    done = {r['remote'] for r in rows(inventory)} if inventory.exists() else set()
    priority = [p for p in selected if 'query_' not in p]
    for remote in priority:
        if remote in done:
            continue
        path = download(remote)
        append(inventory, {'remote': remote, 'path': path, 'sha256': sha(path), 'bytes': Path(path).stat().st_size})
        done.add(remote)
        print(remote, Path(path).stat().st_size, flush=True)
    remaining = [p for p in selected if p not in done]
    def metadata(remote):
        path = download(remote)
        return {'remote': remote, 'path': path, 'sha256': sha(path), 'bytes': Path(path).stat().st_size}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, result in enumerate(pool.map(metadata, remaining), 1):
            append(inventory, result)
            if i % 500 == 0:
                print('Metadata verified', i, '/', len(remaining), flush=True)
    write(STUDY / 'sources.json', {'dataset': 'RVtech/Audio2Tool', 'revision': REVISION,
        'scorer_revision': SCORER_REVISION, 'files': len(files), 'metadata_files': len(selected),
        'training_sources': ['RVtech/Audio2Tool'], 'external_data_training': False})

def verify_file(remote):
    import numpy as np
    import soundfile as sf
    start = time.monotonic()
    try:
        path = download(remote)
        audio, sr = sf.read(path, dtype='float32', always_2d=True)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError('Empty or nonfinite audio')
        return {'remote': remote, 'path': path, 'sha256': sha(path), 'bytes': Path(path).stat().st_size,
            'frames': len(audio), 'sample_rate': sr, 'channels': audio.shape[1], 'duration': len(audio)/sr,
            'peak': float(np.abs(audio).max()), 'rms': float(np.sqrt(np.mean(audio**2))),
            'clipped_fraction': float(np.mean(np.abs(audio) >= .999)), 'ok': True,
            'seconds': time.monotonic()-start}
    except Exception as e:
        return {'remote': remote, 'ok': False, 'error': type(e).__name__ + ': ' + str(e), 'seconds': time.monotonic()-start}

def audio(workers):
    files = json.loads((STUDY / 'remote_files.json').read_text())
    audio_files = [p for p in files if p.endswith(('.wav', '.flac', '.mp3', '.ogg'))]
    inventory = STUDY / 'file_inventory.jsonl'
    done = {r['remote']: r for r in rows(inventory)} if inventory.exists() else {}
    remaining = [p for p in audio_files if not done.get(p, {}).get('ok')]
    print('Audio inventory:', len(audio_files), 'remaining:', len(remaining), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(verify_file, p): p for p in remaining}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            append(inventory, result)
            if i % 100 == 0 or not result['ok']:
                print('Verified', i, '/', len(remaining), 'last_ok', result['ok'], flush=True)

def checkout():
    import os, subprocess
    repo=STUDY/'source_repo'
    if not repo.exists():
        env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'}
        subprocess.run(['git','clone','https://huggingface.co/datasets/RVtech/Audio2Tool',str(repo)],check=True,env=env)
        subprocess.run(['git','-C',str(repo),'checkout','--detach',REVISION],check=True,env=env)
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=REVISION:
        raise ValueError('Existing source checkout is not the pinned revision')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['checkout', 'discover', 'audio'])
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    {'checkout':checkout,'discover':discover,'audio':lambda:audio(a.workers)}[a.command]()

if __name__=='__main__':main()
