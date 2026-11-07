"""ASR diagnostic on a fixed TRAINING sample; never a test-set filter."""
import json,time
from pathlib import Path
import torch,soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from transformers import WhisperProcessor,WhisperForConditionalGeneration
from nano_audio.utils import ROOT,read_json,write_json

def main():
    torch.set_num_threads(4)
    assets=read_json(ROOT/'artifacts/assets/manifest.json');model_path=assets['models']['whisper_tiny']['local_path']
    processor=WhisperProcessor.from_pretrained(model_path);model=WhisperForConditionalGeneration.from_pretrained(model_path).to('cuda').eval()
    examples=[json.loads(l) for l in (ROOT/'artifacts/data/smoke/examples.jsonl').read_text().splitlines()]
    unique={}
    for r in examples:
        if r['split']=='train':unique.setdefault(r['utterance_group_id'],r)
    output=[]
    for row in list(unique.values())[:2]:
        samples,sr=sf.read(row['audio_path']);g=gcd(sr,16000);samples=resample_poly(samples,16000//g,sr//g)
        inputs=processor(samples,sampling_rate=16000,return_tensors='pt').to('cuda')
        with torch.no_grad():ids=model.generate(**inputs,language='en',task='transcribe',max_new_tokens=128)
        output.append({'id':row['id'],'reference':row['transcript'],'asr_diagnostic':processor.batch_decode(ids,skip_special_tokens=True)[0],'ids':ids[0].tolist(),'audio_path':row['audio_path']})
    write_json(ROOT/'artifacts/data/smoke/asr_audit.json',{'status':'completed','scope':'fixed training examples, not used for filtering','outputs':output})
if __name__=='__main__':main()
