import hashlib
import json
import zipfile
from pathlib import Path
import pytest
from nano_learning.restore_evidence import restore
from maintenance.package import allowed


def bundle(tmp_path, members):
    archive=tmp_path/'bundle.zip';manifest=tmp_path/'manifest.json'
    with zipfile.ZipFile(archive,'w') as stream:
        for name,content in members.items():stream.writestr(name,content)
    spec=dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),entries=[
        dict(path=name,size=len(content),sha256=hashlib.sha256(content).hexdigest())
        for name,content in members.items()])
    manifest.write_text(json.dumps(spec));return archive,manifest


def test_restore_is_idempotent_and_never_overwrites(tmp_path):
    archive,manifest=bundle(tmp_path,{'artifacts/run/a.json':b'{}'})
    assert restore(tmp_path,archive,manifest)['restored']==1
    assert restore(tmp_path,archive,manifest)['already_present']==1
    target=tmp_path/'artifacts/run/a.json';target.write_text('different')
    with pytest.raises(ValueError,match='refusing overwrite'):restore(tmp_path,archive,manifest)
    assert target.read_text()=='different'


def test_restore_preflights_every_member_before_writing(tmp_path):
    archive,manifest=bundle(tmp_path,{'artifacts/new.json':b'{}','artifacts/existing.json':b'{}'})
    (tmp_path/'artifacts').mkdir();(tmp_path/'artifacts/existing.json').write_text('history')
    with pytest.raises(ValueError,match='refusing overwrite'):restore(tmp_path,archive,manifest)
    assert not (tmp_path/'artifacts/new.json').exists()


@pytest.mark.parametrize('name',['../escape','/tmp/escape','artifacts/../../escape','artifacts\\escape','PAPER.tex'])
def test_restore_rejects_unsafe_members(tmp_path,name):
    archive,manifest=bundle(tmp_path,{name:b'x'})
    with pytest.raises(ValueError,match='Unsafe'):restore(tmp_path,archive,manifest)


def test_restore_rejects_symlinks_and_corruption(tmp_path):
    archive,manifest=bundle(tmp_path,{'artifacts/a.json':b'{}'})
    (tmp_path/'elsewhere').mkdir();(tmp_path/'artifacts').symlink_to(tmp_path/'elsewhere')
    with pytest.raises(ValueError,match='Symlink'):restore(tmp_path,archive,manifest)
    archive.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):restore(tmp_path,archive,manifest)


def test_public_mirror_excludes_local_resources():
    for name in ['trash/old.py','artifacts/run/learning.jsonl','.venv/config',
                 'nano_learning/api_key.txt','.env','vendor/mellow/.git/config',
                 'grpo_trial.slurm','slurm-old.out','archives/old.tar.gz','dist/old.zip',
                 'src/nano_audio.egg-info/PKG-INFO']:
        assert not allowed(name),name
    for name in ['PAPER.tex','.gitignore','nano_learning/reporting.py',
                 'evidence/reporting.zip','vendor/audio2tool/LICENSE']:
        assert allowed(name),name
