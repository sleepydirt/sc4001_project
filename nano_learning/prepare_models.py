"""Explicit acquisition of hash-pinned model files; never invoked by reports."""
import argparse
from pathlib import Path
from .common import ROOT,ARTIFACTS,sha,write
from .evidence import load


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--offline',action='store_true');args=parser.parse_args()
    from huggingface_hub import snapshot_download,hf_hub_download
    import subprocess,urllib.request
    spec=load(ROOT/'nano_learning/resources/model_assets.json')
    for name,source in spec['sources'].items():
        checkout=ROOT/'vendor'/name
        if not checkout.exists():
            if args.offline:raise FileNotFoundError(checkout)
            subprocess.run(['git','clone',source['url'],str(checkout)],check=True)
            subprocess.run(['git','-C',str(checkout),'checkout','--detach',source['revision']],check=True)
        revision=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
        if revision!=source['revision']:raise ValueError('Vendor revision mismatch: '+name)
    folder=ARTIFACTS/'assets';folder.mkdir(parents=True,exist_ok=True)
    for name,item in spec['models'].items():
        if isinstance(item['sha256'],dict):
            path=Path(snapshot_download(item['repo_id'],revision=item['revision'],allow_patterns=list(item['sha256']),local_files_only=args.offline))
            for filename,expected in item['sha256'].items():
                if sha(path/filename)!=expected:raise ValueError('Weight content mismatch: '+name+'/'+filename)
        elif name=='mellow_v0_published':
            path=Path(hf_hub_download(item['repo_id'],filename=item['filename'],revision=item['revision'],local_files_only=args.offline))
        else:
            path=folder/'samsone'/item['filename'];path.parent.mkdir(exist_ok=True)
            if not path.exists():
                if args.offline:raise FileNotFoundError(path)
                url=item['source_url'].replace('/tag/','/download/')+'/'+item['filename']
                temp=path.with_suffix('.download');urllib.request.urlretrieve(url,temp);temp.replace(path)
        if isinstance(item['sha256'],str) and sha(path)!=item['sha256']:raise ValueError('Weight content mismatch: '+name)
        item['local_path']=str(path.resolve())
    write(folder/'manifest.json',spec)

if __name__=='__main__':main()
