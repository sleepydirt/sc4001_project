from __future__ import annotations
import hashlib, json, os, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def digest(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str)+'\n');tmp.replace(path)
def read_json(path): return json.loads(Path(path).read_text())
def append_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(obj,default=str)+'\n')
def gitrev(path):return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
