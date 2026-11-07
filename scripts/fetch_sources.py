"""Fetch only the two official upstreams; reuse/verify manifest-pinned revisions."""
import json,re,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=root/'artifacts/assets/manifest.json'
sources=json.loads(manifest.read_text()).get('sources',{}) if manifest.exists() else {}
for name,url in [('samsone','https://github.com/SamsungLabs/samsone'),('mellow','https://github.com/soham97/mellow')]:
    path=root/'vendor'/name;revision=sources.get(name,{}).get('revision')
    if revision is not None and not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Invalid upstream revision')
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['git','clone',url,str(path)],check=True)
        if revision:subprocess.run(['git','-C',str(path),'checkout','--detach',revision],check=True)
    elif revision:
        current=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if current!=revision:raise RuntimeError(f'{path} is at {current}, expected {revision}; inspect before changing this checkout')
