import torch,json
from nano_audio.utils import ROOT,write_json
root=ROOT/'artifacts/grpo_attempt_20261107'
import sys
if len(sys.argv)>1:root=root/sys.argv[1]
a=torch.load(root/'common_checkpoint/policy',map_location='cpu',weights_only=False)['model']
b=torch.load(root/'grpo_checkpoint/policy',map_location='cpu',weights_only=False)['model']
result={}
for label,prefix in [('frozen_encoder','audio_encoder.'),('projector','projector.'),('separator','sep_token'),('decoder','text_model.')]:
    keys=[k for k in a if k.startswith(prefix)]
    changed=[k for k in keys if not torch.equal(a[k],b[k])]
    result[label]={'state_tensors':len(keys),'changed_tensors':len(changed),'maximum_absolute_change':max(((a[k]-b[k]).abs().max().item() for k in keys if a[k].is_floating_point()),default=0)}
assert result['frozen_encoder']['changed_tensors']==0
assert result['projector']['changed_tensors']>0 and result['decoder']['changed_tensors']>0
write_json(root/'weight_changes.json',result);print(json.dumps(result,indent=2))
