"""Restore the public reporting bundle without overwriting historical evidence."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile
from .common import ROOT, sha


def destination(root, name):
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or '\\' in name
            or not path.parts or path.parts[0] != 'artifacts'
            or str(path) != name):
        raise ValueError('Unsafe evidence path: ' + name)
    target = Path(root).joinpath(*path.parts)
    for part in [target, *target.parents]:
        if part.is_symlink():
            raise ValueError('Symlink in evidence destination: ' + name)
    return target


def restore(root=ROOT, bundle=None, manifest=None):
    root = Path(root)
    bundle = Path(bundle) if bundle else root / 'evidence/reporting.zip'
    manifest = Path(manifest) if manifest else root / 'evidence/manifest.json'
    spec = json.loads(manifest.read_text())
    if sha(bundle) != spec['archive_sha256']:
        raise ValueError('Evidence archive checksum mismatch')
    entries = spec['entries']
    names = [entry['path'] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate manifest paths')
    pending = []
    with zipfile.ZipFile(bundle) as archive:
        if sorted(archive.namelist()) != sorted(names):
            raise ValueError('Archive members differ from manifest')
        # Validate every destination and member before writing anything.
        for entry in entries:
            name = entry['path']; target = destination(root, name)
            info = archive.getinfo(name)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Non-file archive member: ' + name)
            content = archive.read(name)
            if len(content) != entry['size'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise ValueError('Evidence member checksum mismatch: ' + name)
            if target.exists():
                if not target.is_file() or sha(target) != entry['sha256']:
                    raise ValueError('Existing evidence differs; refusing overwrite: ' + name)
            else:
                pending.append((name, target))
        for name, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(archive.read(name))
    return dict(restored=len(pending),already_present=len(entries)-len(pending),files=len(entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='Fresh mirror root containing evidence/, or current repository')
    args = parser.parse_args()
    print(json.dumps(restore(args.root), indent=2))


if __name__ == '__main__':
    main()
