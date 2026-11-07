"""Differentiable adapter around the exact released Samsone modules.

Microbatch one is deliberate: no padding ambiguity and identical generation/scoring
positions. Frozen encoder features are cached BEFORE the trainable projector.
"""
from __future__ import annotations
import copy,json,sys
from pathlib import Path
import torch
from ..utils import ROOT,read_json,write_json,sha,digest

class SamsonePolicy:
    def __init__(self,model_id='samsone_134m_published',device='auto',asset_root=None):
        sys.path.insert(0,str(ROOT/'vendor/samsone/src'))
        from samsone import Samsone134M
        from samsone.models.text.lm import HuggingFaceLM
        from samsone.models.audio import WhisperAudioEncoder
        self.model_id=model_id;self.device=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device)
        self.asset_root=Path(asset_root or ROOT/'artifacts/assets');self.assets=read_json(self.asset_root/'manifest.json')
        cp=self.assets['models']['samsone_134m_published']['local_path']
        # Keep optimizer master weights FP32; CUDA operations use FP16 autocast.
        self.native=Samsone134M(device=self.device,dtype=torch.float32,checkpoint_path=cp,max_new_tokens=128)
        self.model=self.native.model;self.tokenizer=self.model.text_model.tokenizer
        if model_id=='samsone_134m_fresh':
            expected=self.fingerprint()
            from samsone.inference.models import _build_pruned_token_map
            map_path=self.asset_root/'fresh_prune_map.json';write_json(map_path,_build_pruned_token_map(self.tokenizer))
            self.model.text_model=HuggingFaceLM(self.assets['models']['smollm2_135m_base']['local_path'],prune_tokens_map_path=str(map_path),load_pretrained_weights=True)
            self.model.audio_encoder.encoder=WhisperAudioEncoder(self.assets['models']['whisper_tiny']['local_path'],load_pretrained_weights=True)
            self.model.projector.init_weights()
            with torch.no_grad():self.model.sep_token.normal_()
            self.model.to(self.device)
            assert self.fingerprint()==expected,'Fresh architecture does not match release'
            self.provenance={'decoder':'component text-pretrained SmolLM2 selected by official map','encoder':'component pretrained whisper-tiny','projector':'random','separator':'random','published_weights_retained':False}
        else:self.provenance={'all_weights':'official Samsone release checkpoint'}
        self.decoder=self.model.text_model.model
        self.mapping=self.model.text_model.tokens_map;self.inverse={v:k for k,v in self.mapping.items()}
        self.eos_token_id=self.inverse[self.tokenizer.eos_token_id]
        self.decoder.generation_config.eos_token_id=self.eos_token_id;self.decoder.generation_config.pad_token_id=self.eos_token_id
        self.model.requires_grad_(True);self.model.audio_encoder.requires_grad_(False);self.model.eval()
        self.encoder_hash=digest({n:sha(cp) if model_id.endswith('published') else self.assets['models']['whisper_tiny']['revision'] for n in ['encoder']})
        self.feature_cache=ROOT/'artifacts/features';self.feature_cache.mkdir(parents=True,exist_ok=True)
        self._memory_cache={}

    def fingerprint(self):
        return digest({'shapes':{k:list(v.shape) for k,v in self.model.state_dict().items()},'map':self.model.text_model.tokens_map,'tied':self.model.text_model.model.get_input_embeddings().weight is self.model.text_model.model.get_output_embeddings().weight})
    def encode(self,text):
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        try:result=[self.inverse[t] for t in ids]
        except KeyError as e:raise ValueError(f'Unavailable pruned token in {text!r}: {e}') from e
        if self.decode(result)!=text:raise ValueError('Tokenizer round trip failed')
        return result
    def decode(self,ids):return self.tokenizer.decode([self.mapping[int(t)] for t in ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    def _autocast(self):return torch.autocast(self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda')
    def encode_audio(self,waveforms,sample_rates=None):
        # Native path preserves exact 30s padding and released 50-token pooling.
        paths=waveforms if isinstance(waveforms,list) else [waveforms]
        with torch.no_grad(),self._autocast():
            f=self.native._prepare_audio([Path(p) for p in paths]);return self.model.audio_encoder(f)
    def cached_audio(self,path):
        path=Path(path);key=digest({'audio':sha(path),'encoder':self.encoder_hash,'preprocessing':'native-16k-mono-480000-v1','pooling':'official-fixed50','dtype':'float32-storage-fp16-compute'})
        if key not in self._memory_cache:
            file=self.feature_cache/(key+'.pt')
            if file.exists():features,mask=torch.load(file,map_location=self.device,weights_only=True)
            else:
                features,mask=self.encode_audio([path]);features=features.float().detach();mask=mask.detach();torch.save((features.cpu(),mask.cpu()),file)
            self._memory_cache[key]=(features.to(self.device),mask.to(self.device))
        return self._memory_cache[key]
    def prompt(self,example):
        if '_prompt' in example:return example['_prompt']
        # Only public fields; never serialize target/transcript/provenance.
        return (example.get('instruction','return the requested calls as json using the supplied tools.')+' tools:'+json.dumps(example['tools'],separators=(',',':'),ensure_ascii=True)+' answer: ').lower()
    def build_prefix(self,example):
        prompt=self.prompt(example)
        if example.get('modality')=='text':prompt='transcript: '+example['transcript'].lower().strip()+' '+prompt
        tokens=torch.tensor([self.encode(prompt)],device=self.device)
        text=self.decoder.get_input_embeddings()(tokens)
        if example.get('modality')=='text':result=text
        else:
            features,mask=self.cached_audio(example['audio_path'])
            if example.get('audio_control')=='silence':
                features,mask=self.cached_audio(example['silence_path'])
            audio,mask=self.model.projector(features,mask)
            audio=audio[mask.bool()].unsqueeze(0)
            sep=self.model.sep_token.view(1,1,-1)
            result=torch.cat([sep,audio,sep,text],dim=1)
        if result.shape[1]+128>min(2048,self.decoder.config.max_position_embeddings):raise ValueError(f'Complete catalog exceeds context: {result.shape[1]}+128')
        return result
    def completion_logprobs(self,example,completion_ids,completion_mask=None):
        ids=torch.as_tensor(completion_ids,device=self.device,dtype=torch.long).reshape(1,-1)
        with self._autocast():
            prefix=self.build_prefix(example);emb=self.decoder.get_input_embeddings()(ids)
            inputs=torch.cat([prefix,emb],dim=1)
            out=self.decoder(inputs_embeds=inputs,attention_mask=torch.ones(inputs.shape[:2],device=self.device,dtype=torch.long),use_cache=False).logits
            # First answer token is predicted by the last PREFIX position.
            logits=out[:,prefix.shape[1]-1:prefix.shape[1]+ids.shape[1]-1,:].float()
            result=logits.log_softmax(-1).gather(-1,ids.unsqueeze(-1)).squeeze(-1)
        return result
    def generate(self,example,generation_config=None):
        config={'max_new_tokens':128,'do_sample':False,**(generation_config or {})}
        if config.get('do_sample'):config.update(top_k=0,top_p=1.0,temperature=1.0)
        self.model.eval()
        with torch.no_grad(),self._autocast():
            prefix=self.build_prefix(example)
            ids=self.decoder.generate(inputs_embeds=prefix,attention_mask=torch.ones(prefix.shape[:2],device=self.device,dtype=torch.long),eos_token_id=self.eos_token_id,pad_token_id=self.eos_token_id,**config)[0].tolist()
        eos=self.eos_token_id in ids
        if eos:ids=ids[:ids.index(self.eos_token_id)+1]
        return {'ids':ids,'text':self.decode(ids),'eos':eos,'length_capped':not eos and len(ids)>=config['max_new_tokens']}
    def parameter_manifest(self):
        return {'model_id':self.model_id,'architecture_fingerprint':self.fingerprint(),'vocabulary_map_hash':digest(self.mapping),'total_parameters':sum(p.numel() for p in self.model.parameters()),'trainable_parameters':sum(p.numel() for p in self.model.parameters() if p.requires_grad),'trainable_names':[n for n,p in self.model.named_parameters() if p.requires_grad],'provenance':self.provenance,'audio_encoder_frozen':not any(p.requires_grad for p in self.model.audio_encoder.parameters()),'tied_embeddings':self.decoder.get_input_embeddings().weight is self.decoder.get_output_embeddings().weight}
    def save_checkpoint(self,path,training_state=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
        torch.save({'model':self.model.state_dict(),'training_state':training_state or {},'manifest':self.parameter_manifest()},tmp);tmp.replace(path)
    def load_checkpoint(self,path):
        ck=torch.load(path,map_location=self.device,weights_only=False)
        if ck['manifest']['architecture_fingerprint']!=self.fingerprint():raise ValueError('Checkpoint architecture mismatch')
        self.model.load_state_dict(ck['model'],strict=True);return ck['training_state']
