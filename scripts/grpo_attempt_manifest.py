import json
from nano_audio.utils import ROOT,sha,write_json
out=ROOT/'artifacts/grpo_attempt_20261107'
manifest={'models':json.loads((ROOT/'artifacts/assets/manifest.json').read_text()),'data_sha256':sha(ROOT/'artifacts/data/smoke/examples.jsonl'),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'scripts').glob('*grpo*.py'))+list((ROOT/'src/nano_audio/training').glob('*.py'))+[ROOT/'src/nano_audio/models/policy.py',ROOT/'src/nano_audio/evaluation/metrics.py',ROOT/'src/nano_audio/data/task.py']},'lock_sha256':{p.name:sha(p) for p in [ROOT/'uv.lock',ROOT/'constraints-linux-py312.txt']},'checkpoint_graph':{},'reward':'strict complete semantic program success0/1 task-v1','generation':'unconstrained; sampled temperature1 top_k0 top_p1; greedy fixed evaluation; completion max128','selection':'development-only, two versioned common initializations; no final tests','shaping':'not used; decision deferred by user','budget_start_utc':'2026-11-07T08:22:35Z','deadline_utc':'2026-11-07T09:22:35Z'}
for base in [out,out/'strong_init']:
    label='first' if base==out else 'strong_init'
    manifest['checkpoint_graph'][label]={}
    for name in ['common_checkpoint','grpo_checkpoint','sft_control_checkpoint']:
        path=base/name/'policy'
        if path.exists():manifest['checkpoint_graph'][label][name]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
write_json(out/'manifest.json',manifest)
