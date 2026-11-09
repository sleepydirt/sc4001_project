"""Lazy, local-only Kokoro synthesis. Model acquisition is a separate explicit step."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import re
from .task import TaskError, digest

def discover_voices(snapshot_dir):
    return sorted(p.stem for p in (Path(snapshot_dir) / 'voices').glob('*.pt') if p.stem.startswith(('af_', 'am_')))

def audio_quality(samples, sample_rate, max_seconds=8):
    import numpy as np
    x = np.asarray(samples)
    if x.ndim != 1 or not len(x) or sample_rate <= 0 or not np.isfinite(x).all():
        raise TaskError('invalid audio samples')
    duration = len(x) / sample_rate
    rms = float(np.sqrt(np.mean(x.astype('float64') ** 2)))
    clipping = float(np.mean(np.abs(x) >= .999))
    if duration > max_seconds: raise TaskError('audio exceeds task window; regenerate without truncation')
    if rms < 1e-5 or clipping > .01: raise TaskError('silent or clipped audio')
    return {'duration_seconds': duration, 'rms': rms, 'clipped_fraction': clipping,
            'completion_audit': 'requires_listening', 'transcript_audit': 'requires_listening'}

def synthesize_groups(groups, snapshot_dir, output_dir, voices, revision, device='cpu', seed=17, speeds=(1.0, 1.15, 1.3)):
    """Return unexpanded recordings; reject overlong outputs after bounded speed retries.

    snapshot_dir must contain config.json, exactly one kokoro*.pth and the selected
    voices/*.pt from revision. No model or voice download is performed by this API.
    output_dir receives WAV files and synthesis_manifest.json. Existing files are
    reused only if every synthesis input hash matches. Never truncate a waveform.
    """
    if not re.fullmatch('[0-9a-f]{40}', revision): raise TaskError('immutable 40-character model revision required')
    root, out = Path(snapshot_dir), Path(output_dir)
    checkpoints = list(root.glob('kokoro*.pth'))
    if len(checkpoints) != 1 or not (root / 'config.json').exists():
        raise TaskError('missing pinned kokoro config.json and unique kokoro*.pth')
    if set(voices) != {'train', 'dev', 'test'} or len(voices['train']) != 2 or any(len(voices[s]) != 1 for s in ('dev', 'test')):
        raise TaskError('require two train voices and one each dev/test')
    if any(set(voices[a]) & set(voices[b]) for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]): raise TaskError('voice leakage')
    available = discover_voices(root)
    if any(v not in available for vv in voices.values() for v in vv): raise TaskError('selected voice absent from pinned release')
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    assets = {p.name: sha(p) for p in [checkpoints[0], root / 'config.json']}
    assets.update({f'voices/{v}.pt': sha(root / 'voices' / (v + '.pt')) for vv in voices.values() for v in vv})
    import torch
    import numpy as np
    import soundfile as sf
    from kokoro import KModel, KPipeline
    torch.manual_seed(seed)
    model = KModel(repo_id='hexgrad/Kokoro-82M', config=str(root / 'config.json'), model=str(checkpoints[0])).to(device).eval()
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', model=model, device=device)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for group in groups:
        for voice in voices[group['split']]:
            inputs = {'group': group['utterance_group_id'], 'text': group['transcript'], 'voice': voice, 'revision': revision,
                      'assets': assets, 'speeds': speeds, 'seed': seed, 'version': 'kokoro-synthesis-v1'}
            key = digest(inputs)
            wav, meta = out / (key + '.wav'), out / (key + '.json')
            quality = None
            speed = None
            if wav.exists() and meta.exists():
                old = json.loads(meta.read_text())
                if old.get('audio_sha256') == sha(wav):
                    samples, sr = sf.read(wav)
                    quality = audio_quality(samples, sr)
                    speed = old['speed']
            if quality is None:
                voice_path = str(root / 'voices' / (voice + '.pt'))
                for speed in speeds:
                    chunks = list(pipeline(group['transcript'], voice=voice_path, speed=speed))
                    samples = np.concatenate([c[2].detach().cpu().numpy() for c in chunks if c[2] is not None]) if chunks else np.array([])
                    try: quality = audio_quality(samples, 24000)
                    except TaskError:
                        if speed == speeds[-1]: raise
                        continue
                    break
                sf.write(wav, samples, 24000, subtype='PCM_16')
                decoded, sr = sf.read(wav)
                quality = audio_quality(decoded, sr)
            row = deepcopy(group)
            row.update({'speaker_id': voice, 'recording_id': digest([group['utterance_group_id'], voice])[:20],
                        'audio_path': str(wav.resolve()), 'audio_sha256': sha(wav), 'sample_rate': 24000, **quality})
            row['provenance'] = {**row.get('provenance', {}), 'tts_model': 'hexgrad/Kokoro-82M', 'tts_revision': revision,
                                 'tts_seed': seed, 'tts_voice': voice, 'tts_speed': speed, 'tts_asset_hashes': assets}
            meta.write_text(json.dumps({'audio_sha256': row['audio_sha256'], 'speed': speed, 'inputs': inputs}, indent=2))
            records.append(row)
            print('synthesized ' + str(len(records)) + ': ' + voice + ' ' + str(round(row['duration_seconds'], 2)) + 's', flush=True)
    (out / 'synthesis_manifest.json').write_text(json.dumps(records, indent=2))
    return records
