"""Resolve immutable revisions before using any remote model assets."""
from pathlib import Path
import subprocess
from .utils import ROOT,read_json,write_json,sha,gitrev
HF_MODELS={'smollm2_135m_base':'HuggingFaceTB/SmolLM2-135M','whisper_tiny':'openai/whisper-tiny','kokoro':'hexgrad/Kokoro-82M','mellow_v0_published':'soham97/mellow'}
def resolve(root=ROOT/'artifacts/assets',offline=False):
    from huggingface_hub import HfApi
    root=Path(root);path=root/'manifest.json'
    if offline:return read_json(path)
    old=read_json(path) if path.exists() else {}; models=old.get('models',{})
    for key,repo in HF_MODELS.items():
        if key not in models:
            info=HfApi().model_info(repo);models[key]={'repo_id':repo,'revision':info.sha,'source_url':'https://huggingface.co/'+repo,'license':getattr(info.card_data,'license',None),'files':[s.rfilename for s in info.siblings]}
    sources={name:{'revision':gitrev(ROOT/'vendor'/name),'url':url} for name,url in [('samsone','https://github.com/SamsungLabs/samsone'),('mellow','https://github.com/soham97/mellow')]}
    manifest={'models':models,'sources':sources};write_json(path,manifest);return manifest

def download(profile='smoke',root=ROOT/'artifacts/assets',offline=False):
    from huggingface_hub import snapshot_download
    root=Path(root);m=resolve(root,offline)
    for key in ['smollm2_135m_base','whisper_tiny']:
        item=m['models'][key]
        p=snapshot_download(item['repo_id'],revision=item['revision'],allow_patterns=['*.json','*.safetensors','*.txt','*.model'],local_files_only=offline)
        item['local_path']=p;item['sha256']={str(f.relative_to(p)):sha(f) for f in Path(p).rglob('*') if f.is_file()}
    import sys
    sys.path.insert(0,str(ROOT/'vendor/samsone/src'))
    from samsone.inference.download import get_checkpoint_path
    from samsone import Samsone134M
    spec=Samsone134M.spec
    cp=get_checkpoint_path(spec.checkpoint_filename,cache_dir=root/'samsone') if not offline else root/'samsone'/spec.checkpoint_filename
    if not cp.exists():raise FileNotFoundError(cp)
    m['models']['samsone_134m_published']={'local_path':str(cp.resolve()),'sha256':sha(cp),'source_revision':m['sources']['samsone']['revision'],'source_url':'https://github.com/SamsungLabs/samsone/releases/tag/v1.0.0','license':'not specified in inspected release/source metadata'}
    write_json(root/'manifest.json',m);return m
