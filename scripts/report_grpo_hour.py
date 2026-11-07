import json,re
from pathlib import Path
from nano_audio.utils import ROOT,write_json,sha
out=ROOT/'artifacts/grpo_attempt_20261107'
import sys
if len(sys.argv)>1:out=out/sys.argv[1]
result=json.loads((out/'result.json').read_text())
rl=[json.loads(l) for l in (out/'grpo.jsonl').read_text().splitlines()]
report={'result':result,'rl_updates':len(rl),'mean_rollout_reward':sum(r['reward_mean'] for r in rl)/len(rl) if rl else None,'mixed_groups':sum(round(4*(1-r['equal_reward_group_rate'])) for r in rl),'total_groups':sum(r['examples'] for r in rl),'max_kl':max((r['kl'] for r in rl),default=None),'parent_checkpoint_sha256':sha(out/'common_checkpoint/policy'),'scope':'single-seed tiny constructed-speech development run; final tests untouched','comparison':json.loads((out/'comparison.json').read_text()) if (out/'comparison.json').exists() else None,'comparison_intervals':json.loads((out/'comparison_intervals.json').read_text()) if (out/'comparison_intervals.json').exists() else None}
write_json(out/'summary.json',report)
lines=['# One-hour exact-reward GRPO attempt','','Exact semantic 0/1 rewards were retained. No format bonus, partial credit, constrained generation or gold answer in rollout prompts.','',f"Shared SFT warm-up: {result.get('warmup_updates')} additional updates. GRPO: {result.get('grpo_updates')} updates. Exposure-matched SFT control: {result.get('sft_control_updates')} updates.",'','| Checkpoint | Training probes | Development familiar schema | Development unseen schema |','|---|---:|---:|---:|']
for label,key in [('Initial smoke checkpoint','initial'),('Shared initialization','common'),('GRPO final','grpo_final'),('Continued SFT final','sft_control_final')]:
    row=result.get(key,{})
    def cell(k):
        v=row.get(k);return f"{round(v['exact']*v['n'])}/{v['n']} ({100*v['exact']:.1f}%)" if v else 'unavailable'
    lines.append(f'| {label} | {cell("train")} | {cell("dev_id")} | {cell("dev_ood")} |')
lines+=['',f"Reward-bearing variation: {report['mixed_groups']}/{report['total_groups']} prompt groups. Mean online rollout reward: {report['mean_rollout_reward']}. Maximum logged KL: {report['max_kl']}.",'','A nonzero GRPO gradient demonstrates a working learning signal, not an accuracy advantage. Compare the common parent, GRPO and continued SFT separately. All training costs include shared warm-up; the SFT control is exposure-matched, not compute-matched.','',f"Elapsed runner time: {result.get('elapsed_seconds')} seconds. Status: {result.get('status')}.",'','## Fixed-seed sampled comparison','',json.dumps(report['comparison'],indent=2),'','Utterance-cluster bootstrap intervals (descriptive; small dependent development suite):','',json.dumps(report['comparison_intervals'],indent=2),'','## Logs and provenance','','- generations.log: readable requests, gold labels for logging only, generated responses and rewards.','- generations.jsonl / rollouts.jsonl: exact generated IDs, scores, old/reference token log probabilities.','- comparison_generations.jsonl: fixed-seed common/GRPO/SFT sampled comparison.','- events.jsonl / console.log: chronological training and evaluation.','- protocol.json: pool IDs, dataset hash, seeds, limits and scope.','- *_checkpoint/: weights, optimizer, scheduler, scaler, RNG and reference checkpoints.','','## Limits','','This is synthetic speech with 16 training utterance groups and a small development set. Improvements on repeated training examples are not human-speech grounding or unseen-schema generalization. Reward shaping remains deferred at the user’s request. The full study and statistical claims remain unexecuted.','','Rationale and sources: [reward_rationale.md](reward_rationale.md).']
(out/'report.md').write_text('\n'.join(lines)+'\n')
# Standalone SVG learning curves, same axes for greedy exact rates.
events=[json.loads(l) for l in (out/'events.jsonl').read_text().splitlines()]
curves={k:[] for k in ['train','dev_id','dev_ood']}
for r in events:
    match=re.fullmatch('grpo_(\\d+)',r.get('phase',''))
    if r.get('event')=='evaluation' and (match or r.get('phase')=='common'):
        step=int(match[1]) if match else 0
        for k in curves:
            if k in r['scores']:curves[k].append((step,r['scores'][k]['exact']))
colors=['#2563eb','#d97706','#059669'];svg=['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="360" viewBox="0 0 800 360"><rect width="800" height="360" fill="white"/><text x="40" y="25" font-size="17">Exact greedy success during GRPO (tiny development diagnostic)</text><path d="M60 55V290H740" fill="none" stroke="black"/>']
for (label,pts),color in zip(curves.items(),colors):
    points=' '.join(f'{60+x/max(1,len(rl))*680},{290-y*230}' for x,y in pts)
    svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for x,y in pts:svg.append(f'<circle cx="{60+x/max(1,len(rl))*680}" cy="{290-y*230}" r="3" fill="{color}"/>')
    svg.append(f'<text x="{70+colors.index(color)*210}" y="335" fill="{color}">{label}</text>')
svg+=['<text x="360" y="313">GRPO updates</text><text x="25" y="65">1</text><text x="25" y="290">0</text></svg>'];(out/'learning.svg').write_text(''.join(svg))
