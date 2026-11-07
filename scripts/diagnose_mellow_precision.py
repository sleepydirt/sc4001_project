import json
import torch
from nano_audio.cli import load_policy,records,seeded
from nano_audio.utils import ROOT,write_json
seeded(17)
p=load_policy('mellow_v0_published','cuda','smoke');ex=records('smoke','train')[0];ids=p.encode(ex['target_text'])+[p.eos_token_id]
rows=[]
for dtype in [torch.float32,torch.float16,torch.bfloat16]:
    p.model.zero_grad(set_to_none=True)
    try:
        with torch.autocast('cuda',dtype=dtype,enabled=dtype!=torch.float32):
            prefix=p.build_prefix(ex);scores=p.completion_logprobs(ex,ids);loss=-scores.mean()
        finite=bool(torch.isfinite(scores).all())
        if finite:loss.backward()
        rows.append({'dtype':str(dtype),'prefix_finite':bool(torch.isfinite(prefix).all()),'prefix_abs_max':prefix.abs().max().item(),'scores_finite':finite,'loss':loss.item() if finite else None,'gradients_finite':all(bool(torch.isfinite(v.grad).all()) for v in p.model.parameters() if v.grad is not None) if finite else None})
    except Exception as e:rows.append({'dtype':str(dtype),'error':repr(e)})
write_json(ROOT/'artifacts/reproduction/mellow_precision.json',rows);print(json.dumps(rows,indent=2))
