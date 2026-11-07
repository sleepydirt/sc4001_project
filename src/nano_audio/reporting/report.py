"""Conservative, dependency-free reporting of measured research artifacts.

Never loads a model, unpickles a checkpoint, or executes an experiment. Source
presence is an implementation inventory, not verification against the spec.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ..utils import ROOT, write_json


PROFILES = ('smoke', 'local_pilot', 'main')


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _relative(path):
    return str(path.relative_to(ROOT))


def _read(path, warnings, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as error:
        warnings.append(f'{_relative(path)}: {error}')
        return default


def _jsonl(path, warnings):
    rows = []
    if path.exists():
        for index, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError('expected object')
                rows.append(value)
            except ValueError as error:
                warnings.append(f'{_relative(path)}:{index}: {error}')
    return rows


def _inventory(paths):
    result = []
    for path in sorted(set(paths)):
        if path.is_file():
            # Hash small text provenance files; do not re-read multi-GB checkpoints.
            row = {'path': _relative(path), 'bytes': path.stat().st_size}
            if path.suffix in ('.json', '.jsonl', '.yaml', '.yml', '.toml', '.txt', '.lock', '.slurm'):
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    for block in iter(lambda: handle.read(1024*1024), b''):
                        digest.update(block)
                row['sha256'] = digest.hexdigest()
            result.append(row)
    return result


def _svg(path, rows, metric):
    points = [(r.get('step'), r.get(metric)) for r in rows]
    points = [(float(x), float(y)) for x, y in points if _number(x) and _number(y)]
    if not points:
        return False
    points.sort()
    xs, ys = zip(*points)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    def coordinate(x, y):
        return 65 + 510*(x-xmin)/max(1, xmax-xmin), 205-155*(y-ymin)/(ymax-ymin or 1)
    polyline = ' '.join(f'{a:.2f},{b:.2f}' for a, b in (coordinate(x,y) for x,y in points))
    dots = ''.join(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="3" fill="#2563eb"/>' for a,b in (coordinate(x,y) for x,y in points))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="620" height="265" viewBox="0 0 620 265" role="img" aria-label="{html.escape(metric)} versus optimizer update">
<rect width="620" height="265" fill="white"/><g font-family="sans-serif" font-size="12" fill="#111">
<text x="65" y="24">{html.escape(metric)} (recorded ledger values)</text>
<path d="M65 45 V205 H580" stroke="#444" fill="none"/>
<text x="5" y="55">{ymax:.4g}</text><text x="5" y="205">{ymin:.4g}</text>
<text x="65" y="225">{xmin:g}</text><text x="550" y="225">{xmax:g}</text>
<text x="240" y="250">optimizer update</text></g>
<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2"/>{dots}</svg>''')
    return True


def _audit():
    # Each limitation is a scientific/acceptance requirement still needing evidence.
    requirements = [
        ('0', 'Architecture and experiment provenance', ['configs/experiments/study.yaml', 'src/nano_audio/models/policy.py'],
         'Verify resolved revisions, weight hashes, vocabulary mapping and matched fresh initialization; source existence alone does not establish provenance.'),
        ('1', 'Hardware, assets and native inference', ['src/nano_audio/environment.py', 'src/nano_audio/assets.py', 'src/nano_audio/models/policy.py', 'src/nano_audio/models/mellow.py'],
         'Require per-family real native outputs, parity, backward checks and checkpoint prediction continuity; optional failed imports are not silently resolved.'),
        ('2', 'Data preparation and split controls', ['src/nano_audio/data/pipeline.py', 'src/nano_audio/data/slurp.py', 'src/nano_audio/data/synthetic.py', 'src/nano_audio/data/synthesis.py'],
         'Require actual human-source coverage, licenses, exclusions, sibling/split audits, TTS provenance and cache hashes; constructed-only data cannot establish the human-speech primary outcome.'),
        ('3', 'Published benchmark reproduction', ['src/nano_audio/evaluation/native.py'],
         'Native smoke output is not benchmark reproduction. Require pinned scorer, native protocol, sample coverage and actual released-checkpoint scores.'),
        ('4', 'Paired SFT/GRPO and fresh tracks', ['configs/experiments/study.yaml', 'src/nano_audio/training/core.py'],
         'A configured run matrix is not an executed comparison. Require shared parent, identical trainable set/pool, all stages, and three main seeds.'),
        ('5.6', 'Response-only SFT', ['src/nano_audio/training/core.py', 'tests/test_training.py'],
         'Offline fixtures do not establish speech grounding. Require real nonzero-LR updates, frozen encoder, EOS mapping, intended gradients and held-out free generation.'),
        ('5.7', 'GRPO objective and learning signal', ['src/nano_audio/training/core.py', 'tests/test_training.py'],
         'Require detached old/reference scores, frozen complete reference, exact rollout IDs and binary rewards. All-equal groups have zero reward advantage; loss or KL movement alone is not positive RL evidence.'),
        ('5.8', 'Local learning pilot', ['configs/profiles/local_pilot.yaml'],
         'Not established by smoke. Require eight-example overfit curve, 64 warm-up updates, 32 updates per continuation and held-out ID/OOD predictions.'),
        ('5.9', 'Evaluation and causal claims', ['src/nano_audio/evaluation/metrics.py', 'src/nano_audio/cli.py'],
         'Dev summaries are not final human-speech OOD estimates. Require utterance aggregation, paired bootstrap, three seeds, transcript/audio controls, noise, cascade and retention.'),
        ('5.10', 'Runnable orchestration', ['src/nano_audio/cli.py', 'pyproject.toml'],
         'Require completed stage artifacts with input hashes and resumable commands; a stage wrapper completion may still contain a partial or blocked scientific output.'),
        ('5.11', 'Resume and V100 portability', ['src/nano_audio/training/core.py', 'scripts/v100.slurm'],
         'Require real two-update continuity with positive LR, reference identity, RNG/sampler continuity and preemption evidence. A portable script is not an executed V100 job.'),
        ('5.12', 'Acceptance tests and research deliverables', ['tests/test_training.py', 'tests/test_reporting.py'],
         'Test files being present does not mean tests ran. Require recorded test outcomes and real integration evidence; full experiment acceptance remains unverified.'),
    ]
    return [dict(section=section, requirement=title,
                 status='source_present_unverified' if all((ROOT/p).is_file() for p in paths) else 'implementation_files_missing',
                 source_paths=paths, missing_paths=[p for p in paths if not (ROOT/p).is_file()],
                 acceptance_gap=gap) for section,title,paths,gap in requirements]


def _run(directory, profile, output, warnings):
    stage = _read(directory/'stage.json', warnings, {})
    rows = _jsonl(directory/'learning.jsonl', warnings)
    # Guard profile contamination and duplicate resumed rows; preserve the raw ledger.
    if stage.get('profile') not in (None, profile):
        warnings.append(f'{_relative(directory)}: stage profile mismatch; excluded')
        return None
    rows = [r for r in rows if r.get('profile') in (None, profile)]
    seen = set()
    for row in rows:
        step = row.get('step')
        if step in seen:
            warnings.append(f'{_relative(directory)}: duplicate ledger step {step}; totals may contain repeated work')
        seen.add(step)
    grpo = [r for r in rows if r.get('objective') == 'grpo']
    equal = [r['equal_reward_group_rate'] for r in grpo if _number(r.get('equal_reward_group_rate'))]
    all_zero_advantage = bool(grpo) and len(equal) == len(grpo) and all(x == 1 for x in equal)
    reward_values = [r['reward_mean'] for r in grpo if _number(r.get('reward_mean'))]
    signal = ('no_reward_advantage' if all_zero_advantage else
              'reward_variation_observed_not_improvement' if any(x < 1 for x in equal) else
              'unverified' if grpo else 'not_applicable')
    label = f'{directory.parent.name}/{directory.name}'
    curves = []
    for metric in ('loss', 'reward_mean', 'equal_reward_group_rate', 'kl', 'gradient_norm', 'elapsed_seconds'):
        dest = output/'curves'/directory.parent.name/directory.name/f'{metric}.svg'
        if _svg(dest, rows, metric):
            curves.append(str(dest.relative_to(output)))
    predictions = _read(directory/'dev_predictions.json', warnings, [])
    return dict(run=directory.parent.name, seed=directory.name, artifact_dir=_relative(directory),
                status=stage.get('status', 'ledger_only' if rows else 'unverified'),
                identity=stage.get('identity'), parent=stage.get('parent'), data_hash=stage.get('data_hash'),
                recorded_updates=len(rows), stage_updates=stage.get('updates'),
                fixed_probe=_read(directory/'probe.json',warnings,{}),
                checkpoint_resume_verified=stage.get('resume_verified'),
                objective=rows[-1].get('objective') if rows else None,
                evidence_class='smoke_only' if profile=='smoke' else 'training_observations_not_validated_comparison',
                rl_signal=signal, all_groups_zero_reward_advantage=all_zero_advantage,
                positive_rl_claim_supported=False,
                all_sampled_rewards_zero=bool(reward_values) and len(reward_values)==len(grpo) and all(x==0 for x in reward_values),
                equal_reward_group_rates=equal, reward_means=reward_values,
                dev_metrics=stage.get('dev_metrics'), dev_prediction_count=len(predictions) if isinstance(predictions,list) else None,
                elapsed_seconds=sum(r['elapsed_seconds'] for r in rows if _number(r.get('elapsed_seconds'))),
                peak_memory_bytes=max([r['peak_memory_bytes'] for r in rows if _number(r.get('peak_memory_bytes'))], default=None),
                last_ledger_record=rows[-1] if rows else None, curves=curves,
                checkpoints=_inventory(p for p in (directory/'checkpoint').rglob('*') if p.is_file()))


def build_report(profile)->dict:
    """Read existing artifacts and write reports/<profile>/research_report.{json,md}."""
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    output = ROOT/'artifacts/reports'/profile
    output.mkdir(parents=True, exist_ok=True)
    warnings = []
    root = ROOT/'artifacts'
    runs_root = root/'runs'/profile
    runs = []
    for directory in sorted(runs_root.glob('*/seed_*')):
        if directory.is_dir():
            result = _run(directory, profile, output, warnings)
            if result is not None:
                runs.append(result)
    environment = _read(root/'environment.json', warnings)
    evaluation = _read(output/'evaluation.json', warnings, {})
    primary = evaluation.get('human_speech_primary', {'status':'unavailable', 'reason':'No recorded human-speech OOD evaluation'})
    stages = _read(root/'stages'/f'{profile}.json', warnings, [])
    claims = _jsonl(root/'reproduction/claims.jsonl', warnings)
    provenance_paths = [root/'environment.json', root/'architecture.json', root/'experiment_manifest.json',
                        root/'assets/manifest.json', ROOT/'pyproject.toml', ROOT/'uv.lock', ROOT/'scripts/v100.slurm']
    for pattern in ('configs/**/*.yaml', '*requirements*.txt', '**/constraints*.txt', '**/*freeze*.txt'):
        provenance_paths.extend(p for p in ROOT.glob(pattern) if '.venv' not in p.parts and 'vendor' not in p.parts)
    commands = [f'nano-audio report --profile {profile}',
                f'nano-audio evaluate --profile {profile} --suite research',
                'PYTHONPATH=src .venv/bin/python -m pytest tests/test_training.py tests/test_reporting.py -q']
    for run in runs:
        seed = run['seed'].removeprefix('seed_')
        commands.append(f'nano-audio train --run {run["run"]} --profile {profile} --seed {seed} --device auto')
    commands += ['nano-audio run-study --profile local_pilot --device auto',
                 'nano-audio run-study --profile main --device cuda',
                 'sbatch scripts/v100.slurm']
    report = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(), profile=profile,
                  execution_scope='minimal_smoke_only_selected; larger profiles implemented but not executed by reporter',
                  evidence_class='smoke_only' if profile=='smoke' else 'artifact_observations',
                  experiment_complete=False, implementation_complete=False,
                  positive_rl_claim_supported=False,
                  conclusion=('Smoke infrastructure evidence only; no learning or GRPO superiority claim.' if profile=='smoke' else
                              'Artifacts alone do not establish a complete controlled learning comparison.'),
                  runs=runs, hardware=environment, provenance_files=_inventory(provenance_paths),
                  missing_provenance=[_relative(p) for p in provenance_paths[:8] if not p.is_file()],
                  smoke_evidence={name:_read(root/'smoke'/f'{name}.json', warnings) for name in ('text','audio','resume')},
                  human_speech_primary=primary, evaluation=evaluation, stages=stages,
                  reproduction_claims=claims, implementation_audit=_audit(),
                  mellow_native=_read(root/'reproduction/mellow_native.json',warnings),
                  mellow_precision=_read(root/'reproduction/mellow_precision.json',warnings),
                  grounding_controls=_read(output/'controls/results.json',warnings),
                  offline_test_log=(root/'offline-tests.log').read_text() if (root/'offline-tests.log').exists() else None,
                  integration_test_log=(root/'policy-integration-tests.log').read_text() if (root/'policy-integration-tests.log').exists() else None,
                  remaining_implementation=['Automatic SLURP real/synthetic assembly through prepare-data is not yet wired; conservative mapping and local-audio adapters exist.', 'Turnkey acquisition and official scoring for every paper benchmark remain incomplete; native adapters and claim ledger do not replace these.', 'Audio2Tool consistency audit is implemented; official evaluation and lowercase research-transfer bindings remain incomplete.', 'Compute-matched continuation orchestration, full three-seed paired summaries, and held-out-domain schema evaluation still need integration before main execution.'],
                  data_artifacts=_inventory(p for p in (root/'data'/profile).glob('*') if p.suffix in ('.json','.jsonl')),
                  commands=commands, warnings=warnings,
                  limitations=['Missing artifacts mean unverified, not successful.',
                               'Loss curves for SFT and GRPO have different meanings and are not directly comparable.',
                               'No policy learning claim when every GRPO group has identical rewards; KL alone can still change weights.',
                               'Source presence is an implementation inventory, not a complete semantic code audit or test pass.',
                               'Checkpoint files are inventoried without loading; file existence does not prove continuity.',
                               'Shared smoke artifacts are not evidence that every model family or requested profile ran.'])
    lines = ['# Research report', '', f'Profile: `{profile}`. Generated: {report["generated_at"]}.', '',
             report['conclusion'], '', 'Implementation completion: **unverified**. Full experiment completion: **not established**.', '',
             '## Measured artifacts', '', '| Run / seed | Stage status | Ledger updates | RL signal | Dev predictions |',
             '|---|---|---:|---|---:|']
    for run in runs:
        lines.append(f'| {run["run"]} / {run["seed"]} | {run["status"]} | {run["recorded_updates"]} | {run["rl_signal"]} | {run["dev_prediction_count"]} |')
        if run['all_groups_zero_reward_advantage']:
            lines.extend(['', f'**{run["run"]}/{run["seed"]}: every recorded GRPO group has zero reward advantage. No positive RL learning claim is supported.**', ''])
    if not runs:
        lines += ['', 'No training run artifacts found for this profile.']
    lines += ['', '## Fixed training-probe likelihood diagnostics', '', '| Run | Before response NLL | After response NLL |', '|---|---:|---:|']
    for run in runs:
        probe=run.get('fixed_probe',{})
        if 'before_response_nll' in probe:
            lines.append(f'| {run["run"]} | {probe["before_response_nll"]:.4f} | {probe["after_response_nll"]:.4f} |')
    lines += ['', 'These are teacher-forced training diagnostics, not free-generation speech-understanding scores.', '', 'Offline checks:', '', report.get('offline_test_log') or 'No recorded test log.', '', 'Real-model checks:', '', report.get('integration_test_log') or 'No recorded integration log.', '', '## Remaining integration work before main execution', '']
    lines += ['- '+item for item in report['remaining_implementation']]
    lines += ['', 'Human-speech primary outcome:', '', '```json', json.dumps(primary,indent=2), '```', '',
              'Hardware and software (recorded, not inferred):', '', '```json', json.dumps(environment,indent=2), '```', '',
              '## Implementation coverage against specification', '',
              'Presence of source files is not acceptance evidence. Gaps below remain unverified until their outputs and checks are recorded.', '']
    for item in report['implementation_audit']:
        lines += [f'- **{item["section"]}: {item["requirement"]}** — `{item["status"]}`. {item["acceptance_gap"]}']
        if item['missing_paths']:
            lines += ['  Missing: '+', '.join(f'`{p}`' for p in item['missing_paths'])+'.']
    lines += ['', '## Learning and resource curves', '']
    for run in runs:
        lines += [f'### {run["run"]} / {run["seed"]}', '',
                  f'Recorded training time: {run["elapsed_seconds"]:.3f} seconds. Peak allocated bytes: {run["peak_memory_bytes"]}.', '']
        for curve in run['curves']:
            lines += [f'![{Path(curve).stem}]({curve})', '']
        lines += ['Development metrics (not final test estimates):', '', '```json', json.dumps(run['dev_metrics'],indent=2), '```', '']
    lines += ['## Provenance, locks and checkpoints', '', '```json', json.dumps(dict(files=report['provenance_files'], missing=report['missing_provenance'], checkpoints={r['run']+'/'+r['seed']:r['checkpoints'] for r in runs}),indent=2), '```', '',
              '## Reproduction claims and stage blockers', '', '```json', json.dumps(dict(claims=claims,stages=stages),indent=2), '```', '',
              '## Commands', '', 'Commands below are runnable interfaces, not a record that they executed. Pilot, main and Slurm commands are for explicit later execution.', '',
              '```sh', *commands, '```', '', '## Limits and artifact warnings', '', *['- '+x for x in report['limitations']+warnings], '']
    write_json(output/'research_report.json', report)
    (output/'research_report.md').write_text('\n'.join(lines))
    return report
