"""Outcome-independent audio sample with ASR proxy checks, never eligibility rules."""
from __future__ import annotations
from collections import defaultdict
import json
import math
import re
import time
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration,WhisperProcessor
from .common import ARTIFACTS, ROOT,STUDY,rows,write,sha,tracking,digest

def main():
    grouped=defaultdict(list)
    for r in rows(STUDY/'source_rows.jsonl'):grouped[(r['tier'],r['domain'])].append(r)
    chosen=[]
    for key,rr in sorted(grouped.items()):
        # Stable source-only stratum selection includes excluded/long/mixed speech.
        unique={r['request_group']:r for r in rr}
        chosen.append(min(unique.values(),key=lambda r:digest({'seed':17014,'request':r['request_group']})))
    run=tracking('audio-fidelity-audit','audio_qc',{'selection':'one deterministic request per tier/domain; no outcome filtering','human_listening':False})
    start=time.perf_counter();assets=json.loads((ARTIFACTS/'assets/manifest.json').read_text())
    info=assets['models']['whisper_tiny'];processor=WhisperProcessor.from_pretrained(info['local_path'],local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(info['local_path'],local_files_only=True).cuda().eval()
    output=[]
    for r in chosen:
        transcriptions=[]
        for remote in r['audio_remotes']:
            path=STUDY/'source_repo'/remote;wave,sr=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(1)
            if sr!=16000:
                from scipy.signal import resample_poly
                divisor=math.gcd(sr,16000);wave=resample_poly(wave,16000//divisor,sr//divisor).astype('float32')
            texts=[]
            # Inspect the whole clip in chronological non-overlapping ASR windows.
            for offset in range(0,len(wave),30*16000):
                features=processor(wave[offset:offset+30*16000],sampling_rate=16000,return_tensors='pt').input_features.cuda()
                with torch.no_grad():ids=model.generate(features,language='en',task='transcribe',max_new_tokens=256)
                texts.append(processor.batch_decode(ids,skip_special_tokens=True)[0])
            transcriptions.append({'remote':remote,'sha256':sha(path),'asr':' '.join(texts)})
        output.append({'id':r['id'],'tier':r['tier'],'domain':r['domain'],'request_group':r['request_group'],
            'source_transcript':r['transcript'],'gold_calls':r['calls'],'source_flags':r['flags'],
            'asr_windows':transcriptions,'listening_human_review':'not_performed'})
    write(STUDY/'listening_audit.json',{'status':'ASR-assisted proxy audit; human listening pending',
        'selection':'one deterministic request per tier/domain, chosen without model predictions',
        'samples':output,'human_listening_performed':False,'asr_model':info,'eligibility_uses_asr_errors':False,
        'gpu_wall_seconds':time.perf_counter()-start,'limitations':'ASR cannot certify spoken-label fidelity, especially mixed speech, correction scope or source-label ambiguity'})
    run.log({'audio_qc/asr_audited_requests':len(output),'audio_qc/gpu_seconds':time.perf_counter()-start});run.finish()

if __name__=='__main__':main()
