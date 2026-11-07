"""Speech grounding controls and the explicitly counted two-stage ASR cascade."""
from pathlib import Path
from copy import deepcopy
import hashlib
import numpy as np
import soundfile as sf
from nano_audio.data.cache import add_noise
from nano_audio.utils import write_json,sha

def materialize_controls(example,other_audio,output_dir,seed=17):
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    samples,sr=sf.read(example['audio_path']);samples=samples.mean(1) if samples.ndim==2 else samples
    out={}
    for kind in ['silence','mismatch','noise_15db','noise_5db']:
        row=deepcopy(example)
        if kind=='mismatch':path=Path(other_audio)
        else:
            path=root/(example['id']+'_'+kind+'.wav')
            signal=np.zeros_like(samples) if kind=='silence' else add_noise(samples,int(kind.split('_')[1].replace('db','')),seed)
            sf.write(path,signal,sr,subtype='FLOAT')
        row.update(audio_path=str(path.resolve()),audio_sha256=sha(path),control=kind,noise_seed=seed if kind.startswith('noise') else None)
        out[kind]=row
    write_json(root/(example['id']+'_controls.json'),out);return out

class WhisperTextCascade:
    def __init__(self,whisper_path,text_policy,device='cpu'):
        import torch
        from transformers import WhisperForConditionalGeneration,WhisperProcessor
        self.asr=WhisperForConditionalGeneration.from_pretrained(whisper_path).to(device).eval()
        self.processor=WhisperProcessor.from_pretrained(whisper_path)
        self.text_policy=text_policy;self.device=device
    def generate(self,example,generation_config=None):
        import torch,time
        from scipy.signal import resample_poly
        from math import gcd
        audio,sr=sf.read(example['audio_path']);audio=audio.mean(1) if audio.ndim==2 else audio
        if sr!=16000:
            g=gcd(sr,16000);audio=resample_poly(audio,16000//g,sr//g)
        inputs=self.processor(audio,sampling_rate=16000,return_tensors='pt').to(self.device)
        start=time.perf_counter()
        with torch.no_grad():ids=self.asr.generate(**inputs,language='en',task='transcribe',max_new_tokens=128)
        transcript=self.processor.batch_decode(ids,skip_special_tokens=True)[0]
        record={**example,'transcript':transcript,'modality':'text'}
        result=self.text_policy.generate(record,generation_config)
        return {**result,'asr_transcript':transcript,'asr_ids':ids[0].tolist(),'asr_parameters':sum(p.numel() for p in self.asr.parameters()),'text_parameters':sum(p.numel() for p in self.text_policy.decoder.parameters()),'two_stage_elapsed_seconds':time.perf_counter()-start}
