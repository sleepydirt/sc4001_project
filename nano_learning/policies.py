"""Pilot adapters: native capacity, lossless text and bounded frozen features."""
from __future__ import annotations
from collections import OrderedDict
import copy
import json
import os
from pathlib import Path
import random
import sys
import torch
from .common import ROOT, STUDY, ARTIFACTS, sha, digest, write
from nano_audio.models.policy import SamsonePolicy
from nano_audio.models.mellow import MellowPolicy

class FeatureStore:
    def __init__(self, family, device_limit=12, disk_limit=32*2**30):
        self.root = STUDY / 'features' / family
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory = OrderedDict()
        self.device_limit = device_limit
        self.disk_limit = disk_limit
        self.disk_bytes = sum(p.stat().st_size for p in self.root.glob('*.pt'))
        self.hits = self.misses = 0

    def __deepcopy__(self, memo):
        # Shared immutable detached encoder outputs only, never projections.
        return self

    def get(self, key, compute, device):
        if key in self.memory:
            self.hits += 1
            result = self.memory.pop(key)
            self.memory[key] = result
            return result
        path = self.root / (key + '.pt')
        if path.exists():
            result = torch.load(path, map_location='cpu', weights_only=True)
            os.utime(path, None)
            self.hits += 1
        else:
            self.misses += 1
            result = tuple(x.detach().cpu().contiguous() for x in compute())
            tmp = path.with_suffix('.tmp')
            torch.save(result, tmp)
            size = tmp.stat().st_size
            if self.disk_bytes + size > self.disk_limit:
                for old in sorted(self.root.glob('*.pt'), key=lambda p: p.stat().st_mtime):
                    self.disk_bytes -= old.stat().st_size
                    old.unlink()
                    if self.disk_bytes + size <= self.disk_limit:
                        break
            tmp.replace(path)
            self.disk_bytes += size
        result = tuple(x.to(device) for x in result)
        self.memory[key] = result
        while len(self.memory) > self.device_limit:
            self.memory.popitem(last=False)
        return result

class PilotMixin:
    context_cap = 8192
    completion_cap = 512

    def parameter_manifest(self):
        manifest=super().parameter_manifest()
        initialization=manifest.pop('provenance')
        manifest['provenance']={'initialization':initialization,'new_task_training_source':'Audio2Tool only',
            'task_audio_slots':'whole single clip plus fixed silent Mellow slot; dialogue whole history plus final user clip',
            'transcript_in_direct_audio_prompt':False}
        manifest['pilot_interface']={'context_cap':self.context_cap,'completion_cap':self.completion_cap,
            'rollout_serial_decoding':True,'evaluation_decoding':'fixed-batches-up-to-four','lossless_json_values':True}
        return manifest

    def completion_logprobs(self, example, ids, mask=None):
        tokens = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(1, -1)
        cap = example.get('completion_cap', self.completion_cap)
        if tokens.shape[1] > cap:
            raise ValueError('Completion exceeds frozen cap')
        with self._autocast():
            prefix = self.build_prefix(example)
            inputs = torch.cat((prefix, self.decoder.get_input_embeddings()(tokens)), dim=1)
            if inputs.shape[1] > self.context_cap:
                raise ValueError('Native context exceeded in scoring')
            attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device)
            def forward(x):
                return self.decoder.model(inputs_embeds=x, attention_mask=attention, use_cache=False).last_hidden_state
            if torch.is_grad_enabled() and inputs.requires_grad and inputs.shape[1] > 4096:
                from torch.utils.checkpoint import checkpoint
                states = checkpoint(forward, inputs, use_reentrant=False)
            else:
                states = forward(inputs)
            # Score only response positions; avoid full-catalog vocabulary logits.
            states = states[:, prefix.shape[1]-1:prefix.shape[1]+tokens.shape[1]-1]
            logits = self.decoder.lm_head(states).float()
            result = logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            result = result * torch.as_tensor(mask, device=self.device).reshape_as(result)
        return result

    def _check_context(self, length):
        if length > self.context_cap:
            raise ValueError(f'Context {length} exceeds {self.context_cap}')

    def generate(self, example, config=None):
        config = {'max_new_tokens': example.get('completion_cap', self.completion_cap), 'do_sample': False, **(config or {})}
        if config['max_new_tokens'] > example.get('completion_cap', self.completion_cap):
            raise ValueError('Generation cap differs from task cap')
        return super().generate(example, config)

    def save_checkpoint(self, path, training_state=None):
        # Atomic, architecture-independent pilot format. RNG belongs to Trainer.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        torch.save({'model': self.model.state_dict(), 'training_state': training_state or {},
            'manifest': self.parameter_manifest(), 'pilot': {'context_cap': self.context_cap,
                'completion_cap': self.completion_cap, 'audio2': 'fixed-silence-or-declared-history'}}, tmp)
        tmp.replace(path)

    def load_checkpoint(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck['manifest']['architecture_fingerprint'] != self.parameter_manifest()['architecture_fingerprint']:
            raise ValueError('Checkpoint family/architecture mismatch')
        self.model.load_state_dict(ck['model'], strict=True)
        return ck['training_state']

class PilotSamsone(PilotMixin, SamsonePolicy):
    def encode(self,text):
        from .encoding import retained_encode
        ids=retained_encode(self.tokenizer,self.inverse,text)
        if self.decode(ids)!=text:raise ValueError('Tokenizer round trip failed')
        return ids

    def cached_audio(self, path):
        path = Path(path)
        key = digest({'audio': sha(path), 'encoder': self.encoder_hash,
            'preprocessing': 'native-16k-mono-480000-official-fixed50', 'dtype': 'fp16-compute-native-feature-dtype-storage', 'version': 1})
        return self.store.get(key, lambda: self.encode_audio([path]), self.device)

    def build_prefix(self, example):
        ids = torch.tensor([self.encode(self.prompt(example))], device=self.device)
        text = self.decoder.get_input_embeddings()(ids)
        if example.get('modality') == 'text':
            result = text
        else:
            paths = example.get('audio_paths', [example['audio_path']])
            pieces = []
            for path in paths:
                features, mask = self.cached_audio(path)
                audio, mask = self.model.projector(features, mask)
                audio = audio[mask.bool()].unsqueeze(0)
                sep = self.model.sep_token.view(1, 1, -1)
                pieces.extend([sep, audio, sep])
            result = torch.cat(pieces + [text], dim=1)
        self._check_context(result.shape[1] + example.get('completion_cap', self.completion_cap))
        return result

class PilotMellow(PilotMixin, MellowPolicy):
    def audio_paths(self, example):
        paths = example.get('audio_paths')
        if paths:
            if len(paths) > 2:
                raise ValueError('Mellow supports at most two declared complete clips')
            return (str(paths[0]), str(paths[1] if len(paths) == 2 else self.silence))
        return str(example['audio_path']), str(example.get('audio2', self.silence))

    def cached_audio(self, example):
        result = []
        for path in self.audio_paths(example):
            key = digest({'audio': sha(path), 'encoder': self.encoder_hash,
                'preprocessing': 'upstream-repeat-32k-10s-wholeclip-no-cropping',
                'seed': 17, 'dtype': 'fp32', 'version': 1})
            def compute(path=path):
                state = random.getstate()
                random.seed(17)
                try:
                    self.model.audio_encoder.base.eval()
                    with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
                        return (self.model.audio_encoder.base(self.wrapper.preprocess_audio([path], resample=True).squeeze(1))['embedding'],)
                finally:
                    random.setstate(state)
            result.append(self.store.get(key, compute, self.device)[0])
        return tuple(result)

    def build_prefix(self, example):
        if example.get('modality') == 'text':
            ids = torch.tensor([self.encode(self.prompt(example))], device=self.device)
            result = self.decoder.get_input_embeddings()(ids)
        else:
            result = super().build_prefix(example)
        self._check_context(result.shape[1]+example.get('completion_cap', self.completion_cap))
        return result

def load_policy(model, device='cuda'):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    silence = STUDY / 'silence_1s.wav'
    if not silence.exists():
        import soundfile as sf
        import numpy as np
        silence.parent.mkdir(parents=True, exist_ok=True)
        sf.write(silence, np.zeros(16000, dtype='float32'), 16000, subtype='PCM_16')
    if model == 'samsone':
        assets=json.loads((ARTIFACTS/'assets/manifest.json').read_text())
        info=assets['models']['samsone_134m_published']
        if sha(info['local_path'])!=info['sha256']:raise ValueError('Released Samsone weight hash mismatch')
        policy = PilotSamsone(device=device)
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(assets['models']['smollm2_135m_base']['local_path'],local_files_only=True)
        if policy.tokenizer.get_vocab()!=tokenizer.get_vocab():raise ValueError('Samsone tokenizer differs from pinned foundation component')
    elif model == 'mellow':
        # Construct the pinned v0 interface directly.
        policy = load_mellow(device)
        policy.silence = silence
        assets = json.loads((ARTIFACTS/'assets/manifest.json').read_text())
        policy.encoder_hash = assets['models']['mellow_v0_published']['sha256']
        policy._feature_cache.clear()
    else:
        raise ValueError('Unknown pilot model')
    policy.context_cap = min(8192, policy.decoder.config.max_position_embeddings)
    policy.store = FeatureStore(model)
    # Released weak-model greedy paths diverged across batched GEMM shapes in
    # training-only boundary checks. Use serial native decoding in every arm and
    # evaluation, rather than silently changing the decoder distribution.
    policy.batching_enabled = False
    policy._serial_generate = policy.generate
    policy.model.eval()
    return policy

def load_mellow(device):
    """Exact local v0 construction without the legacy loader's historical writes."""
    import subprocess
    sys.path.insert(0, str(ROOT/'vendor/mellow'))
    from mellow.wrapper import MellowWrapper
    from mellow.model.model import get_model_class
    from transformers import AutoTokenizer
    assets = json.loads((ARTIFACTS/'assets/manifest.json').read_text())
    vendor = ROOT/'vendor/mellow'
    revision = subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
    if revision != assets['sources']['mellow']['revision']:
        raise ValueError('Mellow vendor revision mismatch')
    info = assets['models']['mellow_v0_published']
    if sha(info['local_path']) != info['sha256']:
        raise ValueError('Mellow released weight hash mismatch')
    component = assets['models']['smollm2_135m_base']
    for filename, expected in component['sha256'].items():
        if sha(Path(component['local_path'])/filename) != expected:
            raise ValueError('Mellow foundation component hash mismatch')
    wrapper = MellowWrapper.__new__(MellowWrapper)
    wrapper.parent_path = vendor/'mellow'
    wrapper.config_path = str(wrapper.parent_path/'config/v0.yaml')
    wrapper.args = wrapper.read_config_as_args(wrapper.config_path)
    args = wrapper.args
    args.model['decoder']['text_decoder'] = component['local_path']
    model_cls = get_model_class(args.model['model_type'])
    wrapper.model = model_cls(audioenc_name=args.model['encoder']['audioenc_name'],
        d_in=args.model['encoder']['out_emb'],text_decoder=component['local_path'],
        prefix_length=args.model['decoder']['prefix_length'],d_out=args.model['encoder']['d_proj'])
    state = torch.load(info['local_path'],map_location='cpu',weights_only=True)
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]:v for k,v in state.items()}
    wrapper.model.load_state_dict(state,strict=True)
    # Upstream stores its Python config module inside HTSAT. Replace only that
    # immutable attribute with the same public values so a full reference can be
    # independently deep-copied; weights and forward behavior are unchanged.
    from types import ModuleType, SimpleNamespace
    for module in wrapper.model.modules():
        config = getattr(module,'config',None)
        if isinstance(config,ModuleType):
            values={k:copy.deepcopy(v) for k,v in vars(config).items()
                if not k.startswith('_') and isinstance(v,(str,int,float,bool,list,dict,tuple,type(None)))}
            module.config=SimpleNamespace(**values)
    wrapper.tokenizer = AutoTokenizer.from_pretrained(component['local_path'],local_files_only=True)
    wrapper.tokenizer.add_special_tokens({'pad_token':'!'})
    target = torch.device(device)
    wrapper.use_cuda = target.type == 'cuda'
    wrapper.device = (target.index or 0) if wrapper.use_cuda else 'cpu'
    wrapper.model.to(target).eval()
    policy = PilotMellow(wrapper,second_audio_pool=(),seed=17)
    policy.assets = assets
    return policy
