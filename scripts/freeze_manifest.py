"""Aggregate executed evidence without promoting absent stages to completed."""
from pathlib import Path
from nano_audio.utils import ROOT,read_json,write_json,sha,digest

def freeze():
    assets=read_json(ROOT/'artifacts/assets/manifest.json')
    env=read_json(ROOT/'artifacts/environment.json')
    data={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'artifacts/data').rglob('*.json*') if p.is_file()}
    runs={str(p.relative_to(ROOT)):read_json(p) for p in (ROOT/'artifacts/runs').rglob('stage.json')}
    out={'study':'Learning dynamics of supervised and reinforcement post-training for nano speech-to-tool models','execution_scope':'minimal smoke only per user','seed':17,'assets':assets,'environment':env,'architecture':read_json(ROOT/'artifacts/architecture.json') if (ROOT/'artifacts/architecture.json').exists() else None,'locks':{p.name:sha(p) for p in [ROOT/'uv.lock',ROOT/'constraints-linux-py312.txt']},'data_hashes':data,'runs':runs,'protocol':{'task_version':'task-v1','grpo':'population_std_sequence_mean_v1','pooling':'official50','audio_preprocessing':'native30s pad 16khz; requests <=8s','max_context':2048,'max_completion':128,'decoding':'unconstrained greedy','encoder_frozen':True},'config_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'configs').rglob('*.yaml')}}
    out['source_snapshot']=read_json(ROOT/'artifacts/source_snapshot.json') if (ROOT/'artifacts/source_snapshot.json').exists() else None
    out['research_decisions']=read_json(ROOT/'artifacts/research_decisions.json')
    out['native_mellow']=read_json(ROOT/'artifacts/reproduction/mellow_assets.json')
    out['validation_logs']={p.name:p.read_text() for p in [ROOT/'artifacts/offline-tests.log',ROOT/'artifacts/policy-integration-tests.log']}
    out['run_parent_checksums']={}
    for name,run in runs.items():
        if run.get('parent'):
            parent=ROOT/f"artifacts/runs/{run['profile']}/{run['parent']}/seed_17/checkpoint/policy"
            if parent.exists():out['run_parent_checksums'][name]={'parent':str(parent.relative_to(ROOT)),'sha256':sha(parent)}
    out['completion']={'smoke_orchestration':'completed','full_study_implementation':'incomplete; see SERVER_HANDOFF.md','full_experiment':'not executed'}
    write_json(ROOT/'artifacts/experiment_manifest.json',out)
if __name__=='__main__':freeze()
