"""Offline causal-position fixture plus explicit real-model contract tests."""
import os
import pytest
import torch
from types import SimpleNamespace
from nano_audio.models.policy import SamsonePolicy

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb=torch.nn.Embedding(7,4);self.head=torch.nn.Linear(4,7)
    def get_input_embeddings(self):return self.emb
    def forward(self,inputs_embeds,**kw):return SimpleNamespace(logits=self.head(inputs_embeds.cumsum(1)))

def test_causal_completion_alignment():
    p=SamsonePolicy.__new__(SamsonePolicy);p.device=torch.device('cpu');p.decoder=Decoder()
    prefix=torch.randn(1,3,4,requires_grad=True);p.build_prefix=lambda ex:prefix
    ids=torch.tensor([[2,4]])
    actual=p.completion_logprobs({},ids)
    logits=p.decoder(inputs_embeds=torch.cat([prefix,p.decoder.emb(ids)],1)).logits
    expected=logits[:,2:4].log_softmax(-1).gather(-1,ids[:,:,None]).squeeze(-1)
    torch.testing.assert_close(actual,expected)
    (-actual.mean()).backward();assert prefix.grad.abs().sum()>0

def test_public_prompt_whitelist():
    p=SamsonePolicy.__new__(SamsonePolicy)
    record={'tools':[{'name':'some_tool','description':'Set A Device'}]}
    assert p.prompt(record)==p.prompt({**record,'target':'SECRET','transcript':'SECRET','binding_id':'SECRET'})
    assert p.prompt(record).islower()

@pytest.mark.integration
@pytest.mark.skipif(os.getenv('NANO_AUDIO_INTEGRATION')!='1',reason='Requires pinned downloaded models')
def test_real_policy_roundtrip_and_fresh_architecture():
    published=SamsonePolicy(device='cpu');fresh=SamsonePolicy('samsone_134m_fresh',device='cpu')
    assert published.fingerprint()==fresh.fingerprint()
    text='{"calls":[{"name":"train_01_action_0","arguments":{"field_0":"kitchen_light","field_1":"off"}}]}'
    assert published.decode(published.encode(text))==text
    assert published.model.text_model.model.get_input_embeddings().weight is published.decoder.get_output_embeddings().weight
