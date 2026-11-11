# Samsone semi-pilot runner

Run a small, paired experiment with one published Samsone checkpoint and seed 17. Measure whether GRPO improves exact speech-to-tool accuracy more than continued SFT under the same prompt exposure. A positive held-out difference is evidence for this setup. A tie, regression, or flat development curve is also a result.

## What carries forward from the pilot

Keep the three old runs as synthetic-only diagnostics. The old `S_P1` is the parent of that comparison, and the two old continuation endpoints used unequal update counts. Start the mixed-data warm-up from the released Samsone checkpoint in the existing asset manifest. Freeze its new `P1` before either continuation. Both branches then start with the same policy file, the same training data, and the same sequence of prompts and tool order. GRPO also keeps a frozen copy of that policy as its reference.

Your pilot used **batch four**, so 1,024 SFT steps represented 4,096 sampled examples. Main's proposed 512 steps at batch eight also represent 4,096 draws. The pilot's 4/6 exact familiar-schema calls show SFT learned something, while the unfamiliar schemas remained at 0/6. One endpoint cannot establish a plateau. This runner measures development accuracy from step zero through the matched endpoint.

Start with Samsone because its audio path and checkpoints already work on your V100. Once the comparison is useful on held-out data, rerun it with the full audited dataset and paired seeds. Add Mellow as a second published-family comparison. The fresh track then starts from text-pretrained SmolLM2 and a pretrained audio encoder with a fresh connector, followed by text task SFT, connector alignment, and joint audio SFT before branching into continued SFT and GRPO. Those are separate experimental tracks. [Repository experiment guide](https://github.com/sleepydirt/sc4001_project/blob/4fdb0972b1569c5aa02d4b83f02854ac392e47ab/research_instructions.md#L328-L401)

## 1. Install the runner in the training checkout

Extract the supplied archive at the top level of the existing repository. It adds the `semi_pilot/` directory. Use the same checkout, `.venv`, assets and Kokoro cache as the working pilot. The launcher carries over the pilot's partition and module/Conda setup and runs an FP16 CUDA backward check. Change those cluster lines if your site differs.

From your Mac, substitute your actual SSH login and checkout path.

```bash
scp semi_pilot_runner.tar.gz user@server:~/sc4001_project/
```

On the server.

```bash
cd ~/sc4001_project
tar -xzf semi_pilot_runner.tar.gz
```

## 2. Pin, map and audit the dataset

Set a storage path visible to the login and compute nodes. Run downloads from a network-enabled node. Reuse an existing official SLURP checkout or extracted audio directory if you have one. The real-audio archive is about 3.9 GB compressed. Leave space for the extracted files and several GB for rolling model, optimizer and reference checkpoints.

```bash
DATA="$HOME/sc4001-data"
mkdir -p "$DATA"
git clone https://github.com/pswietojanski/slurp.git "$DATA/slurp"
git -C "$DATA/slurp" checkout --detach 8eb16545762be97ace75334109d73824217311f1

.venv/bin/python semi_pilot/data.py coverage --slurp-repo "$DATA/slurp"

curl -fL --retry 3 -o "$DATA/slurp_real.tar.gz" \
  https://zenodo.org/records/4274930/files/slurp_real.tar.gz
printf '9efc0f058ced47bf5131c7cb2cade513  %s\n' "$DATA/slurp_real.tar.gz" | md5sum -c -
tar -xzf "$DATA/slurp_real.tar.gz" -C "$DATA"
```

The mapper binds explicit kitchen and bedroom light references to the existing task catalog and accepts absolute percentages only when supported by the annotation. It counts ambiguous and unsupported examples, keeps the official partitions, excludes duplicate utterances across splits and checks selected WAVs for decoding, length, clipping and silence. Its metadata-only audit at the pinned revision found **35 train, 3 dev and 5 test** candidates, all light-power calls. The human subset is a narrow integration and generalization probe. Inspect the speaker-overlap counts before making any unseen-speaker claim. [Official SLURP annotations and licensing](https://github.com/pswietojanski/slurp/blob/master/README.md) · [SLURP audio and checksum](https://zenodo.org/records/4274930)

The default requests 12/3/4 SLURP groups and 64/12/12 newly constructed groups across train/dev/test. The runner samples **50/50 by source in every update**, independently of corpus size. Synthetic examples use the project’s eight household tools and contain direct, parameterized, correction and two-action requests. These overlap with capabilities Audio2Tool tests. The source data, tools and scorer are the project’s controlled task. Keep the public Audio2Tool benchmark for later external evaluation, after auditing its labels and catalog bindings. [Audio2Tool benchmark](https://audio2tool.github.io/) · [Project dataset roles](https://github.com/sleepydirt/sc4001_project/blob/4fdb0972b1569c5aa02d4b83f02854ac392e47ab/research_instructions.md#L178-L191)

Submit preparation as its own job.

```bash
EXP=semi-v1
sbatch semi_pilot/run.slurm prepare "$EXP" \
  --slurp-repo "$DATA/slurp" --slurp-audio "$DATA/slurp_real"
```

Wait for completion and inspect `artifacts/semi_pilot/$EXP/data/manifest.json`. It records the source counts, audio exclusions, split audit and frozen examples hash. Spot-check some training and dev audio against its transcript and target. If audio filtering makes a quota fail, inspect the log, then pass a smaller explicit `--real TRAIN DEV TEST` before the first successful freeze. Keep at least four real training groups for the diagnostic. The launcher uses the existing local-pilot Kokoro cache by default. If yours is elsewhere, pass `--reuse-tts-assets /absolute/path/to/tts_assets`.

## 3. Run the seed-17 development check

The mixed-data check is three small stages before the paired comparison. Wait for each job to succeed before submitting the next.

```bash
# Can the model memorize four real and four synthetic training commands?
sbatch semi_pilot/run.slurm overfit "$EXP" --cap 200

# After the diagnostic passes, train the shared initialization on all mixed training data.
sbatch semi_pilot/run.slurm warmup "$EXP" --updates 512 --batch-size 4 --eval-every 64

# Evaluate that warm-up and freeze the continuation plan.
sbatch semi_pilot/run.slurm gate "$EXP" --updates 1024 --batch-size 4 --eval-every 64
```

The overfit check requires at least 7/8 exact generations. The gate tests greedy familiar-schema exact accuracy on fixed development clips and samples four unconstrained completions per training probe under the GRPO settings. It records counts by source and requires at least 25% source-macro familiar-schema dev exact, at least four mixed-reward groups, and at least 10% mixed groups. These are engineering thresholds for this experiment. The log and `warmup/gate.json` show which condition passed or failed.

### Weights & Biases

The launcher defaults to `WANDB_PROJECT=team88_sc4001` and uses the W&B login already configured in your training environment. Set `WANDB_PROJECT` or `WANDB_ENTITY` before `sbatch` to select a different destination. Overfit and gate metrics are published when the stage returns. Warm-up, SFT, and GRPO publish metrics through the latest saved checkpoint while the job runs. Each Slurm job gets a separate W&B run grouped by experiment, including resumed jobs. The JSON ledgers remain the source for backfill.

From the repository root on a network-enabled node, backfill an already-completed overfit check using the experiment name you originally submitted.

```bash
WANDB_MODE=online .venv/bin/python semi_pilot/wandb_log.py backfill overfit "$EXP"
```

The command reads `overfit/learning.jsonl`, `overfit/curve.jsonl`, and `overfit/stage.json`. It uploads the saved loss and diagnostic scores with the original update numbers and uses no GPU. On compute nodes without W&B access, submit with `WANDB_MODE=offline sbatch ...` and sync the printed offline run directory from a network-enabled node with `.venv/bin/wandb sync <offline-run-directory>`. Use `WANDB_MODE=disabled` for local metrics only.

If the diagnostic fails, inspect prompt and output formatting, the gold bindings, and training audio first. If the gate fails, check which source is failing and whether the sampled groups are mostly all wrong or already all correct. All-equal groups supply zero reward advantage. For a weak initialization, retry the warm-up with a longer planned horizon in a new experiment. Keep the strict exact reward and the same initialization for both branches.

```bash
.venv/bin/python semi_pilot/run.py fork --from-experiment "$EXP" --experiment semi-v2
# Reuses the frozen dataset and completed diagnostic on the same code.
EXP=semi-v2
sbatch semi_pilot/run.slurm warmup "$EXP" --updates 1024 --batch-size 4 --eval-every 64
# After completion, submit gate again.
```

## 4. Launch the paired continuations and inspect the curve

Once the gate passes, these jobs can run separately. Use the same type of GPU and repeat the exact flags from the gate plan.

```bash
sbatch semi_pilot/run.slurm sft  "$EXP" --updates 1024 --batch-size 4 --eval-every 64
sbatch semi_pilot/run.slurm grpo "$EXP" --updates 1024 --batch-size 4 --eval-every 64
```

The launcher saves at evaluation boundaries and on Slurm's early `USR1` signal at an optimizer boundary. If a job reports exit 75 and a saved step, resubmit the **same command**. It resumes with the original schedule and data. The pilot GRPO log spent about 2.8 hours on 512 training updates, so a 1,024-update job may need a second allocation when evaluation and checkpoint overhead are included. Run one job per arm at a time.

After both jobs finish, run comparison from the login node.

```bash
.venv/bin/python semi_pilot/run.py compare --experiment "$EXP"
```

Copy `artifacts/semi_pilot/$EXP/analysis/` to your Mac and open `development.html`. It plots exact accuracy across the same sampled prompt counts. The CSVs break it down by source and familiar versus dev-only catalog, and include elapsed training-step time, including GRPO rollout and reference scoring. The GRPO CSV also gives mean binary reward and the fraction of groups with mixed rewards between evaluations. Equal prompt exposure and equal compute are different views. For a compute comparison, use overlapping measured times on the same hardware. Save a separately planned longer SFT curve later if the overlap is too small.

The report flags a development plateau when the last four evaluated points span at most one percentage point after 512 updates. This is a stability check. A flat zero still indicates failure, and a small set gives a coarse curve. If either arm is still improving and you want a longer comparison, plan a longer paired horizon from the same `P1`. The learning-rate schedule depends on that horizon, so rerun both arms in a new experiment with the retained initialization.

```bash
.venv/bin/python semi_pilot/run.py fork --from-experiment "$EXP" \
  --experiment semi-long --reuse-warmup
# Run gate with the longer plan, then submit both continuations with that plan.
```

## 5. Open the frozen test once the experiment is locked

After choosing the experiment from development results, evaluate its matched endpoint on the same held-out audio under familiar and unfamiliar catalogs. The runner records exact scores, paired differences and utterance-level bootstrap intervals by source. The human slice is tiny and single-seed, so treat any apparent advantage as a pilot finding.

```bash
sbatch semi_pilot/run.slurm test "$EXP"
```

After that job finishes, copy the entire `test-report` directory to your Mac.

```bash
scp -r user@server:~/sc4001_project/artifacts/semi_pilot/semi-v1/test-report .
open test-report/report.html
```

Change `semi-v1` to the experiment you actually tested. The report includes the expected and generated calls and plays the copied audio. Keep the final test out of warm-up, horizon, and threshold decisions. For the later main study, expand annotation-supported coverage, freeze the full data and run matrix, add paired seeds and a compute-matched view, then run the larger evaluation suite. [Server handoff](https://github.com/sleepydirt/sc4001_project/blob/4fdb0972b1569c5aa02d4b83f02854ac392e47ab/SERVER_HANDOFF.md#L34-L58)
