"""Real optimizer/scaler/RNG reload check without adding a training update."""
import torch
from nano_audio.cli import load_policy,records,run_dir
from nano_audio.training.core import Trainer,TrainingConfig
from nano_audio.utils import ROOT,read_json,write_json
p=load_policy('samsone_134m_published','auto');directory=run_dir('smoke','S_P1',17)
header=torch.load(directory/'checkpoint/resume.pt',map_location='cpu',weights_only=False)
trainer=Trainer(p,TrainingConfig(**header['config']),metadata=header['metadata'])
ex=records('smoke','train')[0];ids=p.encode(ex['target_text'])+[p.eos_token_id]
trainer.load_checkpoint(directory/'checkpoint')
with torch.no_grad():first=p.completion_logprobs(ex,ids).cpu()
rng_first=torch.rand(8)
trainer.load_checkpoint(directory/'checkpoint')
with torch.no_grad():second=p.completion_logprobs(ex,ids).cpu()
rng_second=torch.rand(8)
result={'status':'completed','step':trainer.step,'sampler_position':trainer.sampler_position,'optimizer_state_entries':len(trainer.optimizer.state),'scaler':trainer.scaler.state_dict(),'rng_continuity':torch.equal(rng_first,rng_second),'logprob_continuity':torch.equal(first,second),'additional_optimizer_updates':0}
assert result['rng_continuity'] and result['logprob_continuity']
write_json(ROOT/'artifacts/smoke/resume.json',result)
