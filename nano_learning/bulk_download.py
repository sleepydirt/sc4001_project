"""Authenticated Git/LFS acquisition; no credentials in arguments or artifacts."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import time
from huggingface_hub import get_token
from .common import STUDY, REVISION, rows, sha, write

def main():
    repo = STUDY / 'source_repo'
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Source repository revision mismatch')
    verified = {r['remote']: r for r in (rows(STUDY/'file_inventory.jsonl') if (STUDY/'file_inventory.jsonl').exists() else []) if r['ok']}
    seeded = 0
    for remote, info in verified.items():
        pointer = repo / remote
        if not pointer.exists() or pointer.stat().st_size > 200:
            continue
        text = pointer.read_text()
        if not text.startswith('version https://git-lfs.github.com/spec/v1'):
            continue
        oid = next(x.split(':', 1)[1] for x in text.splitlines() if x.startswith('oid sha256:'))
        if oid != info['sha256'] or sha(info['path']) != oid:
            raise ValueError('Cached file differs from Git LFS content hash')
        destination = repo / '.git/lfs/objects' / oid[:2] / oid[2:4] / oid
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            destination.unlink()
        if not destination.exists():
            os.link(Path(info['path']).resolve(), destination)
        seeded += 1
    print('Seeded verified existing LFS objects:', seeded, flush=True)
    env = os.environ.copy()
    env['GIT_LFS_SKIP_SMUDGE'] = '1'
    env['GIT_LFS_PROGRESS'] = str(STUDY/'lfs_progress.log')
    token = get_token()
    if token:
        env.update(GIT_CONFIG_COUNT='3', GIT_CONFIG_KEY_0='http.extraHeader',
            GIT_CONFIG_VALUE_0='Authorization: Bearer '+token,
            GIT_CONFIG_KEY_1='lfs.concurrenttransfers', GIT_CONFIG_VALUE_1='16',
            GIT_CONFIG_KEY_2='lfs.transfer.maxretries', GIT_CONFIG_VALUE_2='10')
    for attempt in range(12):
        result = subprocess.run(['git','-C',str(repo),'lfs','fetch','origin',REVISION],env=env)
        if result.returncode == 0:
            subprocess.run(['git','-C',str(repo),'lfs','checkout'],env=env,check=True)
            write(STUDY/'bulk_acquisition.json', {'revision': REVISION, 'complete': True,
                'seeded_objects': seeded, 'authentication_available': bool(token), 'attempts': attempt+1})
            return
        print('LFS acquisition waiting for rate-limit window before resume', flush=True)
        time.sleep(300)  # Worker backoff, while the research agent continues other work.
    raise RuntimeError('LFS download did not complete after resumable retries')

if __name__ == '__main__':
    main()
