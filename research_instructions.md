# Nano audio learning dynamics — Codex execution specification

**Study:** Learning dynamics of supervised and reinforcement post-training for nano speech-to-tool models  
**Prepared:** 7 November 2026  
**Audience:** The Codex agent implementing and validating the research repository  
**Deliverable contract:** Implement the complete codebase in one development pass, then execute its local smoke and pilot profiles. This document is the specification; the commands below describe the interface to implement, not software that already exists.

## Purpose and motivation

Local assistants in smart devices, home hubs and capable embedded systems need to interpret speech and select actions with correct arguments. Broad world knowledge is less central than preserving device names, locations, values, corrections and the meaning of a supplied tool schema. Small audio-language models already exist. This study asks how their task competence changes with supervised fine-tuning (SFT) and subsequent Group Relative Policy Optimization (GRPO), especially when the tools presented at evaluation have unfamiliar schemas.

The intended contribution is a controlled empirical study of learning and generalization at nano scale. Better JSON formatting, better speech grounding and better schema generalization must be measured separately. The edge-device motivation provides context; this project does not require a mobile application or a new deployment runtime.

## Instructions to the implementing agent

- Implement all stages, commands, model interfaces, data transformations, evaluators and reports in this document. Do not stop after writing a plan, creating scaffolding, or finishing text-only inference.
- Work locally first. Detect the actual local hardware. A V100 server is a later execution target, not an assumed local device.
- Execute real local inference, a supervised update, a GRPO update and a small learning pilot. Mock tensors may test infrastructure, but must never be reported as evidence of speech understanding.
- Implement one fixed primary architecture. Do not add encoder, projector, model-size, LoRA or RL-algorithm sweeps. Published comparator models retain their native architectures; comparisons between them are reference comparisons, not architecture ablations.
- Interpret **from scratch** in Stage 4.3 as fresh audio alignment starting from text-pretrained SmolLM2 and a pretrained speech encoder. Do not randomly initialize or pretrain the language model on a large corpus.
- Keep native published-model reproduction separate from the research task interface. Reproduction uses original preprocessing and prompts; research experiments use the common task protocol.
- Generate one runnable repository. Avoid notebook-only implementations. Save configurations, dependency locks, manifests, predictions, logs and resumable checkpoints.
- Pin upstream revisions and checkpoint hashes at execution. Do not invent a commit, package version, published score or successful result.
- All ordinary tool calls in this study run against a deterministic local simulator. Do not control real devices or contact live tool endpoints.
- Write the entire implementation even when a download or optional official scorer is unavailable. Record the exact blocked dependency and continue independent implementation and tests. Never mark a blocked evaluation as completed.
- Default orchestration runs the local pilot. Implement the full experiment and V100 launch interfaces as explicit commands; do not submit remote jobs automatically.
- There is no project schedule in this specification. Stage gates describe scientific and implementation dependencies, not approval requests.

## Stage 0 — Initial setup and fixed research decisions

### 0.1 Questions and estimands

Answer these questions:

1. How do task accuracy, argument grounding and reference-transcript performance change during supervised alignment?
2. From the same task-SFT initialization, does GRPO outperform continued SFT on held-out speech and unfamiliar tool schemas?
3. Can a fresh audio alignment initialized from text-only SmolLM2 match or exceed post-trained published models on the same task?

The main causal comparison is **within the same model family and checkpoint initialization**: continued SFT versus GRPO. A fresh model and a published model have different inherited supervision, so their comparison measures complete initialization/training histories; it does not isolate the effect of an architecture or an optimizer.

Freeze the dataset and evaluation protocol before main training. Local development can repair bugs and adjust a documented pilot configuration; it must not select settings using final test answers.

### 0.2 Fixed model registry

| Registry ID | Initialization | Role |
|---|---|---|
| `smollm2_135m_base` | `HuggingFaceTB/SmolLM2-135M` | First text inference; source language weights for fresh alignment. |
| `samsone_134m_published` | Official server-side `Samsone134M` checkpoint | Primary published speech model; run first. |
| `mellow_v0_published` | Official Mellow `config=v0`, `model=v0` | Second published SmolLM2-based comparator, reproduced and post-trained within its own fixed family. |
| `samsone_134m_fresh` | Component-pretrained initialization with the **same architecture and vocabulary mapping** as `samsone_134m_published` | Fresh alignment and SFT→GRPO track. |

Fix the primary family to the released Samsone-134M structure: its Whisper-tiny encoder path, native nonlinear projector, audio pooling and SmolLM2-135M-derived text model. Inspect the pinned release configuration rather than rebuilding an approximately similar model. Official configuration currently identifies `openai/whisper-tiny`, projector dimensions 384→576 and a vocabulary-pruning map. Preserve these architectural choices. [R1–R3]

For the fresh initialization:

1. Load original text-pretrained SmolLM2-135M weights and original `openai/whisper-tiny` encoder weights.
2. Apply the **same official vocabulary map**, embedding/output-row selection, parameter tying and special-token conventions as the published Samsone checkpoint.
3. Randomly initialize the projector and any newly introduced modality/separator parameters.
4. Do not load a published audio-trained encoder, decoder or projector into this track.
5. Assert matching architecture fingerprints, tokenizer maps and tensor shapes against the published primary family. Report different weight provenance.

Samsone's vocabulary reduction means an untouched full-vocabulary 135M decoder is not automatically an exact architecture match. Resolve this explicitly; do not silently turn the fresh track into a different-size model. If the map cannot be reconstructed from the official code/assets, mark the matched fresh track blocked and document the missing material rather than substituting another architecture.

Mellow is a reference family with its native HTSAT-based audio path. It must not replace the primary speech encoder or become an encoder ablation. Do not instantiate additional model sizes. If a published comparison specifically uses Mellow `v0_s`, register that separately for reproducing that exact claim; never substitute its score for `v0` or silently change the training comparator. [R4–R5]

### 0.3 What remains fixed during research training

- Encoder architecture, projector architecture, audio token pooling and tokenizer are fixed within a family.
- Freeze the audio encoder during all **research** training tracks. Cache its outputs before the trainable projector. Published native reproduction keeps the original released inference behavior.
- During joint task SFT and GRPO, train the projector, modality/separator parameters, decoder and its embedding/output parameters, respecting weight tying. Use the identical parameter set for both branches. Explicitly override any upstream training freeze setting and log the resolved names.
- The primary research task is English, single-turn speech plus a tool catalog, producing a compact JSON object containing one or two calls.
- Gold transcripts are available for training/diagnostics, never inserted into a direct-audio inference prompt.
- Schema identities, schema renderings and final test partitions are fixed before main training.
- Main decoding is unconstrained greedy generation. Do not make grammar-constrained decoding a hidden source of improvement.
- Use GRPO as the only RL algorithm. There is no learned reward model or value-function network.

### 0.4 Freeze the experiment manifest

Write `artifacts/experiment_manifest.json` containing:

- model IDs, repository commits, source URLs, checkpoint SHA-256 values;
- tensor/parameter counts, trainable parameter names, tokenizer and vocabulary-map hashes;
- audio preprocessing, pooling, prompt serialization and maximum context sizes;
- data manifests and split hashes, task/metric versions, schema-family IDs;
- Python/package lock identifiers, hardware details, precision and random seeds;
- resolved training configurations and run-parent relationships.

The manifest must support reconstructing any prediction and identify whether a run is a smoke test, local pilot or main experiment.

**Exit condition:** decisions are encoded in configuration and validated by code, with no unresolved architecture choice.

## Stage 1 — Dependencies, downloads and first inference

### 1.1 Inspect before installation

Record the operating system, architecture, Python executables, available RAM/disk, CUDA/MPS availability, device properties, `torch`, `torchaudio`, `torchvision`, `transformers`, `tokenizers`, `datasets`, `accelerate` and optional `vllm` versions. Use import tests in addition to package metadata.

On CUDA, record driver information, `torch.version.cuda`, device compute capability, supported compiled architectures and BF16 support. Run a real tensor operation and backward pass. Do not infer support only from the CUDA version printed by `nvidia-smi`.

The user has previously reported a V100 32GB server with driver 580.173.02, a CUDA-13-capable driver and Torch 2.11.0+cu126. These are context, not facts about the current local environment. Preserve a working installation and test it before proposing replacements. Later V100 execution needs a Volta-capable CUDA 12.x build. [R16]

### 1.2 Isolate conflicting upstream environments

Use Python 3.12 for the main implementation unless inspection establishes a necessary compatible alternative. The current Samsone package requires Python ≥3.12 and pins Torch 2.9.1; Mellow's older requirements pin Torch 2.0.0 and Torchaudio 2.0.1. Their requirements cannot be blindly combined. [R3, R6]

Implement:

- a minimal core training environment with a deliberately selected Torch/Torchaudio pair;
- a separate upstream reproduction environment if a baseline needs incompatible dependencies;
- an optional isolated TTS environment if generation dependencies conflict;
- exact dependency constraints/locks per tested platform.

Inspect imports before installation. Samsone's distribution also lists mobile-export packages such as ExecuTorch; do not make Android/export tooling a prerequisite for this study. A minimal editable/no-dependencies installation is acceptable only after explicitly supplying necessary runtime imports and recording the compatibility deviation.

Do not use an unbounded `pip install --upgrade` to fix one import. Do not overwrite the user's system Python or working server environment. Native Torch generation is the reference path. An already-working vLLM installation can remain installed, but custom audio-prefix training must not depend on it.

### 1.3 Download and pin assets

Implement an idempotent download command and asset manifest for:

1. SmolLM2-135M base and tokenizer;
2. Whisper-tiny assets required by the primary family;
3. official Samsone-134M server weights, config and pruning assets;
4. Mellow `v0` weights/config through its official repository or model host;
5. selected dataset metadata/audio and TTS assets when their stages need them.

Resolve native release URLs from pinned upstream code; do not guess checkpoint filenames. Write checksums and license/provenance metadata. Support an explicit download root and an offline reuse mode. Download only the active profile's required audio initially, rather than all benchmarks for a text smoke test.

### 1.4 Achieve text-only SmolLM2 inference

Implement `nano_audio smoke-text`:

- load the base model and tokenizer on the detected device;
- use an appropriate plain completion prompt for the base checkpoint;
- generate at least three deterministic completions and save raw prompts, token IDs and outputs;
- test EOS/padding handling and save/load;
- run teacher-forced scoring on a known completion and one finite supervised backward pass;
- record peak memory and generation settings.

Coherent output is sufficient here; an unadapted base model need not already follow a tool schema reliably.

### 1.5 Establish audio-model interfaces

First run each official native inference example unchanged. Samsone documents `from samsone import Samsone134M`; Mellow documents `MellowWrapper`. Inspect their pinned signatures and device handling rather than assuming `AutoModelForCausalLM` directly loads the complete audio model. [R1, R4]

Then implement a common research interface with these capabilities:

```python
class AudioPolicy:
    def encode_audio(self, waveforms, sample_rates): ...
    def build_prefix(self, audio_features, prompts): ...
    def generate(self, batch, generation_config): ...
    def completion_logprobs(self, batch, completion_ids, completion_mask): ...
    def save_checkpoint(self, path, training_state): ...
    def load_checkpoint(self, path): ...
    def parameter_manifest(self): ...
```

The implementation must preserve gradients through the research projector and decoder. Inference wrappers that use `no_grad`, return detached embeddings, force CPU conversion, or discard generated IDs are insufficient as training interfaces. Use the underlying modules while keeping native inference as the parity reference.

For Mellow, inspect and preserve the official missing-second-audio convention for single-audio tasks. Do not duplicate the first recording into the second slot unless that is demonstrably the native protocol. Cache keys must include audio hash, resampling/preprocessing version, frozen encoder weight hash, pooling configuration and feature dtype; cache only frozen outputs before the trainable projector.

Validate that:

- decoder generation and teacher-forced scoring use the same prefix, positions and attention masks;
- generated text IDs do not include fictitious audio-token IDs;
- each family's padding, separators and tokenizer conventions survive batching;
- prompt/schema truncation cannot silently remove a candidate tool;
- `completion_logprobs` performs the causal one-position shift correctly;
- encoder gradients are absent and intended decoder/projector gradients exist;
- vocabulary pruning permits lossless encode/decode of all schema names, JSON syntax, numbers and allowed values;
- whole-model save/reload preserves an example prediction.

For task prompts, allow enough text context to include the complete catalog within the decoder's supported context. Any adjustment from a native wrapper's short question limit is a research prompt configuration, recorded separately from published reproduction. Do not change audio pooling or model layers to make the catalog fit.

**Exit condition:** real text inference and native audio inference run locally, differentiable task scoring works, and checkpoint reload is verified. Record failures per model without inventing outputs.

## Stage 2 — Data choices, acquisition and controlled generation

### 2.1 Dataset roles

| Source | Role | Handling |
|---|---|---|
| **SLURP** | Primary human-annotated speech/transcript/semantic data for task SFT and real-speech evaluation. | Keep official train/dev/test boundaries; map only examples whose annotations support an exact call. [R7] |
| **Controlled synthetic speech** | Explicit schemas, corrections, combinations and split-controlled schema generalization. | Generate from programs and templates, then synthesize audio; supervision comes from construction. |
| **Audio2Tool** | External spoken-tool-use evaluation, including parameter extraction and corrections. | Public configurations are labelled test; exclude from training and hyperparameter selection. [R8–R9] |
| **Native paper datasets** | Stage 3 claim reproduction and post-training retention checks. | Preserve official task formats; do not mix their test items into training. |

SLURP is the default because real speech and semantic annotations are available. A simpler intent dataset such as Fluent Speech Commands is a possible data replacement if access fails, but it is not an additional experiment to implement by default. A replacement changes the dataset manifest and requires an explicit report; it must not silently inherit SLURP claims.

Keep human annotations, programmatically constructed labels and any model-generated text labelled by provenance. Do not call a synthetic target a human gold-standard annotation. No paid LLM/TTS API is required.

### 2.2 Fixed task vocabulary and canonical operations

Use these eight simulator operations as the default task domain:

| Internal operation | Canonical arguments |
|---|---|
| `set_power` | `device`, `state` in `on/off` |
| `set_brightness` | `device`, integer `percent` in 0–100 |
| `set_volume` | `device`, integer `percent` in 0–100 |
| `set_temperature` | `device`, numeric `celsius` in an explicitly configured range |
| `start_timer` | `label`, positive integer `seconds` |
| `cancel_timer` | `label` |
| `set_playback` | `device`, `state` in `playing/paused/stopped` |
| `query_device_state` | `device` |

These are study-defined APIs, not a claim that SLURP uses these names. Give each device explicit capabilities. Enumerated names, unit ranges and argument types belong to the public schema. Derive supported SLURP mappings from actual annotations and publish a mapping table with examples and exclusion reasons. Do not infer missing labels from a language model.

Use deterministic, unit-tested parsing for supported spoken quantities. Preserve units. Skip ambiguous real examples with a counted reason; never manufacture an argument. Synthetic data supplies coverage for operations absent from the selected real subset.

The exposed tool/argument names vary by schema family; the internal canonical operation never appears as a hidden hint in the prompt. All task schemas and values use a round-trip-tested lowercase ASCII vocabulary compatible with the selected tokenizer. This is an explicit task restriction.

### 2.3 Synthetic data generation

Implement a local generator:

1. Sample a semantic request and its exact canonical call program.
2. Sample values, device, allowed state and an utterance template independently of the schema name.
3. Render a natural-language request from the semantic request.
4. Assign the entire related utterance group to a split before voice generation or schema expansion.
5. Synthesize speech with a pinned open TTS model, default `hexgrad/Kokoro-82M`. Discover available voice IDs from that pinned release; do not assume a voice file exists. [R14]
6. Save the source text, semantic program, random seeds, voice/model revisions, audio hash and duration.
7. Render task examples under the assigned schema families using the same underlying audio.

Use three request classes: **75% single actions, 15% self-corrections and 10% two-action instructions**. Correction targets reflect the final request, not both the retracted and corrected actions. Two-action examples specify whether order matters; include order-sensitive cases in which it does.

Use multiple template families and reserve phrasing patterns for held-out data. A paraphrase must preserve the complete semantic request. Start with deterministic templates; an LLM paraphrasing service is unnecessary. Balance operations, values and tool positions so common defaults cannot dominate rewards.

All benchmark-like speech used across the research model families must fit the common task window: **at most 8 seconds**. Regenerate synthetic speech or reject examples that exceed it. For real speech, count duration exclusions. Never cut off the part containing a correction or requested value to force a fit. Preserve source audio; produce separate deterministic model-specific resampling.

Quality control checks file decoding, finite samples, nonzero duration/RMS, clipping, audible completion and transcript/reference agreement on a fixed audit sample. Use a trusted ASR check as a diagnostic if available, not as a filter that removes difficult final-test examples. Keep quality thresholds independent of training outcomes. Do not replace failed speech synthesis with tones or silence carrying a semantic label.

### 2.4 Amounts and profiles

Counts below refer to independent utterance groups before schema expansion; report both group and recording counts.

| Profile | Real speech | Synthetic groups | Purpose |
|---|---|---|---|
| `smoke` | Up to 16 supported recordings | 16 train, 8 dev, 8 test | I/O, scorer, gradient and resume checks. |
| `local_pilot` | Up to 256 train, 64 dev, 128 test | 256 train, 64 dev, 128 test | Demonstrate that real learning and schema evaluation work locally. |
| `main` | Up to 5,000 train, 500 dev and 1,000 test, stratified over supported labels | 4,000 train, 500 dev, 1,000 test | Research comparison; use actual eligible counts and report coverage. |

For synthetic training, render two fixed training voices per group. Use a disjoint development voice and a disjoint test voice where the selected release supports enough voices. Thus the main synthetic target is about **9,500 recordings**: 8,000 training plus 500 development plus 1,000 test. Do not generate all recordings for a smoke run.

Use four training schema families, two development-only schema families, four name-renaming test families and four structural/domain-shift test families. Expanding one audio into several schemas does not create independent speech samples. Cache the waveform/features once.

If a real-data quota cannot be met without changing supported semantics, use all eligible items and report the shortfall. Do not fill missing real examples with synthetic examples while retaining a human-data label. Store train/dev/test seeds in configuration.

### 2.5 Split rules

- Preserve SLURP's official partitions. Audit speaker overlap before making unseen-speaker claims; do not assume it is absent.
- Group repeated recordings of an underlying utterance, normalized duplicate transcripts and generated paraphrase/voice/schema/counterfactual siblings. Resolve duplicates across source partitions by excluding affected groups from the research subsets with a recorded policy.
- Do not merge every example with the same intent or target call; that would remove necessary repeated labels across splits. Group by utterance provenance and generated sibling identity.
- Reserve 10% of eligible device/value combinations, using a fixed seed and a coverage-preserving assignment, for a **separate compositional test** while ensuring each constituent appears in training. Save the forbidden training tuples; do not select them after viewing model errors.
- Training must never access test schema definitions for optimization, final test transcripts for adaptation, or official benchmark answers for tuning.
- Pin final test generation rules and seeds before main training. Development schema families must be disjoint from final test families.
- Hash canonical manifests. Write an automated split audit and abort main training on forbidden overlap.

**Exit condition:** the profile's real data and synthetic data can be rebuilt deterministically; supported semantics, exclusions and leakage audits are saved.

## Stage 3 — Reproduce published baseline inference and evaluations

### 3.1 What reproduction means

Reproduce **released-checkpoint inference**, not the original large training run. Run untouched published weights before task SFT. Use original preprocessing, prompting, generation and scoring for published-score comparisons. The task protocol in Stage 5 is a different evaluation.

The goal is to check reproducibility and establish trustworthy baselines. Failure to match a score does not by itself establish that a claim is illegitimate. A different checkpoint, dataset revision, missing audio, prompt, preprocessing window or scorer can explain a difference.

### 3.2 Build a claim ledger before running comparisons

Read the selected paper versions and release documentation. Create `artifacts/reproduction/claims.jsonl` with one row per relevant inference-quality claim for the selected checkpoints:

```json
{
  "claim_id": "model_dataset_metric_unique_id",
  "model_id": "samsone_134m_published",
  "checkpoint_sha256": "resolved_at_execution",
  "source_url": "official_paper_or_release_url",
  "source_locator": "table_or_section_and_version",
  "benchmark_revision": "resolved_at_execution",
  "split": "official_split",
  "metric": "official_metric_name",
  "claimed_value": null,
  "expected_n": null,
  "evaluated_n": 0,
  "missing_audio_n": 0,
  "prompt_hash": null,
  "scorer_revision": null,
  "observed_value": null,
  "status": "unverified",
  "reason": null
}
```

Populate numeric claims from the actual pinned source before a full run. `null` is appropriate for an unverified result, not for a score that has been measured. Preserve percentages versus proportions. Do not copy earlier conversational scores without checking their release/version association.

### 3.3 Evaluation coverage

Implement native dataset/evaluator adapters for the selected papers' available inference-quality claims:

- **MMAU test-mini:** the complete labelled 1,000-example set for the full profile, with separate sound/music/speech scores and overall accuracy. The dataset has changed versions; retain its exact revision. [R10]
- **MMAU test/full and MMAU-Pro:** generate reproducible prediction artifacts and use official scoring where accessible. Hidden labels or an unavailable official service produce an explicit blocked score, not a substituted mini-set result. If scoring requires public leaderboard publication or unavailable credentials, export the submission file and report that external step pending; do not publish automatically.
- **ClothoAQA:** official split and answer normalization for binary/short-answer evaluation.
- **AudioCaps and Clotho captioning:** official references and the specific metrics claimed by each selected source, using pinned scorer implementations. Record missing audio and dependencies such as Java when required.
- **Mellow-specific ReasonAQA evaluation tasks:** register official entailment/difference/QA task files used by the selected release. Do not replace these with custom prompts while calling the score a reproduction.

Map the list above to the claims actually made for each selected checkpoint. Do not invent unsupported model-dataset claim pairs. Mobile-export/phone-speed claims are not part of this inference-quality reproduction stage.

Run each native repository example, a small labelled subset, then the complete available evaluation. The local pilot can use a fixed 16-item benchmark subset, but must label it `smoke_only`. Main reproduction reports need full available splits and coverage.

Mellow `v0`, `v0_s`, paper tables and later release tables may refer to different artifacts. Samsone's repository also exposes a changing results table. Reproduce the exact source/checkpoint pair or classify it as not directly comparable. [R1, R4–R5]

### 3.4 Reproduction output and gates

Save raw outputs, generated token IDs, answer parsing, per-item correctness and category aggregates. Report absolute score differences, coverage, preprocessing and decoding. An illustrative MMAU alert threshold is two percentage points when protocols and all 1,000 items match; report the actual difference and uncertainty rather than treating that threshold as proof of validity.

Use statuses `unverified`, `smoke_only`, `reproduced`, `differs`, `partial`, `not_comparable` or `blocked`. An unavailable asset must include its URL, missing dependency and a precise retry command.

Check model-wrapper correctness if native and common research inference disagree under equivalent inputs. Resolve tokenizer, audio ordering, normalization, padding, generation or scoring differences before attributing them to model quality.

Every relevant claim must receive either an executed result or an explicit limitation. The first native inference and labelled sanity checks are prerequisites for using a checkpoint in research training. Unavailable external scores need not prevent independent local development, but the reproduction report must remain incomplete for those claims.

Finally, run the **unadapted task baseline** for each published model on the common task development set, and save held-out task predictions after the protocol is frozen. This baseline is distinct from each model's native benchmark score.

**Exit condition:** native inference is verified; all selected claim evaluations have an honest recorded status; usable checkpoints have common-task zero-shot baselines.

## Stage 4 — Task data mix and three comparison tracks

### 4.0 Shared training information

For task SFT, use human-supported SLURP examples and programmatically verified synthetic examples from Stage 2. Start with a 50/50 real/synthetic sampling mixture when both sources are available. Balance operations within each source. If a source is unavailable, name the resulting condition differently; do not silently renormalize the main mixture.

Cycle only over **training** schema families. Randomize tool order, property order and allowed training renderings independently of the answer. Log utterance, audio and schema exposures separately.

The SFT/GRPO comparison after a shared initialization uses exactly the same eligible task-call pool. No ASR auxiliary objective is mixed into only one of the two post-training branches.

### 4.1 Track A — Further SFT of published speech models

For both `samsone_134m_published` and `mellow_v0_published`:

1. Preserve the released checkpoint as `P0`.
2. Apply a common task-SFT warm-up using the training mixture and output format. Save `P1` and its complete optimizer-free initialization package for branching.
3. Continue SFT from `P1` to produce `P_SFT`.
4. Evaluate `P0`, `P1`, `P_SFT` on the same locked research protocol and the native retention checks.

Use exact human annotations where supported; synthetic labels remain identified as constructed supervision. This track measures adaptation to the gold-labelled task, including gains that simply come from learning its output convention.

### 4.2 Track B — GRPO of the same published speech models

For each family, initialize GRPO from the **same `P1` weights** used for the continued-SFT control, producing `P_GRPO`. The shared warm-up avoids confounding the main learning comparison with inability to emit a tool call.

Use the same training pool, schema families, parameter set, tuning allowance and per-seed starting checkpoint. Focus evaluation on the OOD schema suites defined in Stage 5. Compare `P_GRPO - P_SFT`, not just `P_GRPO - P0`.

Keep a frozen reference policy containing the entire `P1` **projector plus decoder** and any trainable modality parameters. The frozen encoder's outputs can be shared, but the reference must not receive a prefix computed by the changing policy projector.

If correct samples are too rare, diagnose the training development set. Extend the **shared** task warm-up consistently for both arms and create a new initialization/version before main comparisons. Do not secretly give the RL arm extra SFT, change the final test, force correct rollout answers, or remove all difficult prompts.

### 4.3 Track C — Fresh alignment from text-only SmolLM2, then SFT→GRPO

Build `samsone_134m_fresh` as specified in Stage 0. Retain component pretraining but no inherited audio-language alignment.

Implement these stages:

1. **`F_TEXT`:** task-SFT the text decoder using training transcripts, catalogs and gold calls. Save a reference-transcript baseline.
2. **`F_ALIGN`:** initialize a new projector/separators. Freeze the speech encoder and temporarily freeze the text decoder. Train the connector on a fixed 50/50 mixture of transcript-generation and tool-call examples derived from the same training recordings.
3. **`F_AUDIO`:** unfreeze the decoder and jointly SFT the projector/decoder with 75% task calls and 25% transcript examples. Save the common aligned initialization.
4. Branch `F_AUDIO` into **`F_SFT`**, further supervised task-call training, and **`F_GRPO`**, GRPO on the same call-only training pool. This sequential SFT→GRPO branch is the required SFT+RL combination.

Both continuation branches use the same trainable modules and call-only pool. Save the trajectory through text, alignment and audio-SFT phases. Do not add interleaved RL/SFT variants or extra encoders.

Compare `F_GRPO - F_SFT` within the fresh track. Compare their final task scores with published-family results as complete-system comparisons, recording all in-project alignment compute and the published models' inherited data exposure. Claims that the fresh model beats another model require results on the same evaluation and an explicit description of the different histories.

### 4.4 Run matrix and controls

| Family | Untouched/component baseline | Shared SFT state | Continued SFT | GRPO |
|---|---|---|---|---|
| Published Samsone-134M | `S_P0` | `S_P1` | `S_P_SFT` | `S_P_GRPO` |
| Published Mellow v0 | `M_P0` | `M_P1` | `M_P_SFT` | `M_P_GRPO` |
| Fresh Samsone-compatible alignment | `F_TEXT`, `F_ALIGN` | `F_AUDIO` | `F_SFT` | `F_GRPO` |

Also implement:

- reference-transcript inference using the relevant text decoder;
- a **Whisper-tiny ASR → task-SFT SmolLM2** cascade with the same catalog and scorer;
- silence and mismatched-audio controls for the direct-audio policies.

Keep text-only inference on its text input modality. Do not describe a text-base checkpoint as an audio model before a connector has been trained.

### 4.5 Fair comparison

Use two complementary views:

1. **Exposure-matched:** same count of task prompts sampled from the same eligible pool after the common initialization. Report sampled completions and token counts separately; GRPO produces several completions per prompt.
2. **Compute-matched:** plot accuracy against measured training time/compute on the same hardware, including rollouts and reference scoring. Run the SFT control long enough to cover the GRPO run's compute range. Select a prespecified common budget from development runs, then select checkpoints using development performance within that budget.

Record shared warm-up costs separately and include fresh alignment in total in-project costs. Do not count unmeasured historical pretraining as zero or infer that two different GPUs have comparable elapsed-time budgets.

Main post-training seeds are `17`, `42` and `123`. A local pilot uses `17` only. Copy the same shared initialization into each paired branch; report that the seed variation is conditional on that initialization rather than three independent full pretraining runs.

**Exit condition:** all run configurations, branches, initialization checks and comparison ledgers exist and run under the local pilot profile.

## Stage 5 — Executable task, training and evaluation protocol

### 5.1 Canonical data record

Use versioned JSONL/Pydantic records. A research example contains:

```json
{
  "id": "unique_example",
  "utterance_group_id": "source_utterance_or_generated_sibling_group",
  "source": "slurp_or_constructed",
  "base_split": "train",
  "schema_family_id": "train_01",
  "schema_partition": "train",
  "audio_path": "relative_path.wav",
  "audio_sha256": "resolved_hash",
  "duration_seconds": 3.0,
  "speaker_id": "source_or_tts_voice_id",
  "transcript": "turn off the kitchen light",
  "task_kind": "single",
  "instruction": "Return the requested calls as JSON using the supplied tools.",
  "tools": [{"name": "switch_lamp", "description": "Set a lighting device's power", "parameters": {"type": "object", "properties": {"device": {"type": "string", "enum": ["kitchen_light"]}, "power": {"type": "string", "enum": ["on", "off"]}}, "required": ["device", "power"], "additionalProperties": false}}],
  "target": {"calls": [{"name": "switch_lamp", "arguments": {"device": "kitchen_light", "power": "off"}}]},
  "canonical_target": [{"operation": "set_power", "arguments": {"device": "kitchen_light", "state": "off"}}],
  "binding_id": "internal_binding_not_for_prompt",
  "provenance": {"dataset_revision": "resolved_revision", "generation_seed": null}
}
```

This illustrative record shows one tool for readability; actual research prompts use the complete predefined catalog of eight task operations. Gold, transcript, internal bindings, source paths and provenance are not automatically serialized into the audio model prompt.

Build a public-input whitelist. Transcripts are serialized only for the ASR/text diagnostic tasks. Add tests which intentionally add secret metadata fields and verify that model input does not change.

### 5.2 Prompt and output contract

Serialize a compact instruction and the full task tool catalog, with each tool's description, arguments, types, required fields, enum values and constraints. Append the model's native modality separator conventions and audio representation in a documented order.

Output exactly:

```json
{"calls":[{"name":"exposed_tool_name","arguments":{"exposed_argument":"value"}}]}
```

Allow one or two calls as required by the request. Tool-call order matters when the request specifies an order; dictionary key order and whitespace do not. A schema's descriptions must contain enough information to resolve its functions without a hidden canonical-operation ID.

Count the full serialized prompt with each model's tokenizer. The common research context target is at most 2,048 total positions, including audio/prompt and generation, subject to the actual supported context. Do not discard schema tails. If a catalog cannot fit, fail validation with its length and record the example; redesign compact serialization on development data rather than truncating final tests.

Use a maximum research completion length of 128 tokens for the main and pilot profiles. Construct the task so correct calls fit. Flag and count length-limited generations; do not omit them from evaluation.

### 5.3 OOD schema suites

Separate these test conditions:

| Suite | Changed at evaluation | Held fixed for paired evaluation |
|---|---|---|
| `id_schema` | New test utterances/audio under familiar schema families | Task operation support and training schema vocabulary |
| `ood_names` | Unseen tool names and argument-key names with equivalent descriptions | Same audio, operation and argument values |
| `ood_structure` | Unseen nesting/layout, enum encodings or explicitly described unit bindings | Same underlying request and canonical outcome |
| `ood_domain_schema` | A held-out device-domain catalog, e.g. cabin devices instead of household devices, with supplied descriptions | Comparable supported operation primitives; report changed device vocabulary |
| `composition` | Reserved operation/value/device combinations and two-action combinations | Each constituent is represented in training |
| `correction` | Final-value corrections and answer-changing value pairs | Same schema and controlled request family |

For `ood_names`, serialize the **same test recordings** under familiar and held-out name families. This isolates schema dependence from voice or speech differences. Structural changes require exact deterministic bindings, including unit conversion when used. Do not accept the model's old familiar name if that name is absent from the presented schema.

Keep a development analogue of each transformation in development-only schema families. Never train on final test bindings. All schema rendering and inverse canonicalization functions must have round-trip tests.

Renaming or restructuring familiar operations measures **schema-surface generalization**, not understanding of arbitrary unseen API semantics. Report that boundary explicitly. The held-out-domain and external tests provide additional evidence, with coverage and unsupported semantics stated.

### 5.4 Audio2Tool external evaluation

Keep Audio2Tool outside training and model selection. Implement both its original evaluation adapter and a clearly named research transfer subset. [R8–R9]

Before scoring, audit `tool_name`, expected calls, embedded `functions`, parameter labels and the tool registry for consistency. Do not silently repair labels using model predictions. Save canonicalization decisions and excluded IDs.

For the research subset, preregister a small fixed catalog per selected domain that fits all model contexts. Evaluate all eligible requests using that same catalog, not a per-item catalog containing only the gold tool. No transcript-based or answer-based retrieval may select the candidate functions. Report the number and kinds of examples outside this restricted catalog.

Use direct, parametric, multi-intent and correction subsets where supported. Other tiers remain part of the dataset description rather than extra research objectives. Restricting a catalog or reserializing schemas makes the result an adapted subset; keep it separate from native benchmark scores.

### 5.5 Strict parsing, semantic matching and simulator

Implement a deterministic scorer shared by SFT evaluation, GRPO rewards and the simulator:

1. Parse the entire output as the expected JSON object. Do not search arbitrary prose for a plausible final answer.
2. Reject duplicate JSON keys, conflicting fields, extra top-level content, unknown tools, extra calls, missing required arguments and unsupported types.
3. Validate against the **presented** schema before mapping to canonical operations.
4. Normalize only documented semantic equivalents, such as numeric spelling/units where the dataset contract permits them. A JSON boolean is not automatically an integer.
5. Compare the complete canonical argument structure and required call sequence. Do not reward a correct tool with incorrect arguments as a full success.
6. Execute valid calls in a sandboxed deterministic simulator to produce state/event traces for diagnostics and ordered composition checks. Extra side effects must not be hidden by an equal final state.

Primary exact semantic accuracy is the proportion of all evaluated examples with a completely correct valid call/program. Invalid and truncated responses remain in the denominator. Partial tool/argument metrics are reported separately.

### 5.6 SFT implementation

Use response-only cross-entropy. Prompt tokens, audio prefix positions and padding have labels `-100`. Include the target's EOS in the loss, and do not include target tokens in the conditioning prefix in a way that leaks the answer.

Implement gradient accumulation correctly, preserving effective example weights. Log teacher-forced loss on fixed development probes and free-generation task metrics. Do not compare an SFT training loss numerically with a GRPO loss as if they were the same quantity.

Starting defaults, finalized using development data and then frozen:

| Setting | Default |
|---|---|
| Optimizer | AdamW, weight decay 0.01, gradient norm clip 1.0 |
| Projector-only alignment LR | `1e-4` |
| Joint SFT decoder LR | `2e-5` |
| Joint SFT projector/modality LR | `1e-4` |
| Scheduler | 5% warm-up, then fixed documented decay |
| Main effective SFT batch | 8 examples; microbatch 1 if needed |
| Main published task warm-up | 256 optimizer updates |
| Main continued-SFT exposure run | 512 updates; extend separately to cover matched GRPO compute |
| Fresh stages | 256 text updates, 256 connector updates, 512 joint-audio updates as initial defaults |
| Main evaluation/checkpoint cadence | Every 32 updates and at completion |

These are proposed defaults, not published optimal hyperparameters. Record all updates, real/synthetic proportions, template/schema exposures, tokens and actual examples. Keep the tuning allowance equal for paired branches. Do not change hyperparameters after looking at final test scores.

### 5.7 GRPO implementation and verification

Prefer a small auditable Torch implementation around the verified native modules, or a pinned TRL integration that passes identical invariants. Do not assume that a generic text trainer automatically handles continuous audio prefixes, custom vocabulary maps or trainable projectors. Read the official GRPO implementation/documentation before coding. [R11–R13]

For prompt `x` containing audio and schema, sample `G=4` completions from the current rollout policy. Use temperature 1, no top-k/top-p truncation and no answer/JSON constraints in the main algorithm. Store the exact generated IDs, EOS/length status, completion masks and old-policy log probabilities.

The reward is:

```text
r(x, y) = 1 if the complete predicted call program is valid and semantically correct
          0 otherwise
```

There is no format bonus, chain-of-thought reward, learned reward model or partial-credit reward in the main study. The same reference labels are available to the SFT control; this is a comparison of learning objectives, not a claim that RL needs fewer labels.

Use the following explicitly versioned GRPO convention:

```text
A_i = (r_i - mean_group_reward) / (std_group_reward + 1e-6)
rho_it = exp(logp_current_it - logp_old_it)
d_it = logp_reference_it - logp_current_it
kl_it = exp(d_it) - d_it - 1

L_group = mean_i mean_valid_completion_tokens_t [
    -min(rho_it * A_i, clip(rho_it, 1-epsilon, 1+epsilon) * A_i)
    + beta * kl_it
]
```

Use population group standard deviation (`unbiased=False`), `epsilon=0.2`, `beta=0.01`, and one optimization pass over each freshly sampled rollout batch initially. Average groups equally. This states the normalization convention rather than relying on a changing library default.

Use decoder LR `1e-6` and projector/modality LR `5e-6` as pilot starting points. Main effective batch is 8 prompts, each with four completions; generate and score in microbatches to fit memory. Main exposure-matched continuation is 512 updates. Local pilot sizes are specified below.

Implementation invariants:

- Old-policy and reference log probabilities are detached; current-policy log probabilities remain differentiable through projector and decoder.
- The reference is the frozen common initialization, including its own projector and modality parameters.
- Log probabilities use the exact sampling/tokenization context. Store unmodified generated IDs, not re-tokenized normalized answers.
- Mask prompt/audio/padding tokens, include actual EOS, and treat length-capped incorrect outputs as failures. No reference target is appended to an RL rollout prompt.
- Disable dropout during rollout and policy/reference scoring so sampling and re-scoring are consistent. Evaluation mode must not disable gradients for the current-policy scoring pass.
- Groups with identical rewards have zero reward advantage; log their rate. The KL term can still contribute. Do not claim policy learning when all reward advantages are zero.
- Guard against NaN/Inf and log probability/KL explosions. Save a diagnostic batch rather than silently clamping a scientific failure away.
- Test microbatch accumulation against a whole-batch calculation on a small deterministic fixture.

If the task-SFT checkpoint rarely emits valid/correct calls, first diagnose schema conditioning, audio alignment, truncation and the scorer. Fix or extend the shared warm-up on training/development data and restart paired branches from a versioned new common initialization. A blocked RL-learning pilot must be reported, not disguised as a positive result.

### 5.8 Local smoke and learning pilot

Use profiles which run the same code paths as main training:

| Profile | Execution requirements |
|---|---|
| `smoke` | Real model loads; at least one real audio generation; one SFT optimizer update; one GRPO optimizer update; save/reload/resume. Small tensor fixtures can verify GRPO sign/normalization independently. |
| `local_pilot` | Primary published family first; overfit a fixed eight-example training subset; run 64 task warm-up updates and 32 updates in each continuation branch, effective 4 prompts/examples, GRPO group 4; evaluate actual held-out ID/OOD examples. Exercise the fresh initialization with the same staged code at reduced update counts. |
| `main` | Complete model/track matrix, locked data, three seeds, full available baseline evaluations and matched-budget analysis. |

For the eight-example overfit diagnostic, use a configurable cap, default 200 updates. Seek at least seven exact free-generation successes and no parser ambiguity, but record the measured curve if the model fails. A teacher-forced loss reduction alone is not proof of usable generation.

The local pilot's purpose is to establish that the idea and implementation produce a usable learning signal. It need not demonstrate a statistically significant GRPO advantage. Record sample counts, reward-bearing groups and real outputs. A single-seed tiny run cannot support a paper-level improvement claim.

On CPU/MPS, use FP32 as the initial correctness path. On CUDA, test FP16 autocast with a gradient scaler; enable BF16 only if hardware and kernels support it. V100 has no assumed BF16/FlashAttention path. Do not default to quantized training or Torch compilation. If local resources cannot complete a pilot, save precise resource measurements and runnable commands; do not label the pilot successful.

### 5.9 Evaluation metrics and learning dynamics

Implement:

- **Primary:** exact semantic accuracy on the human-speech SLURP-supported `ood_names` test subset, plus its paired difference from `id_schema` using the same audio. Average first over schema variants within utterance, so rendering many schemas does not inflate independent sample count. Report the synthetic suite separately; if the real-data subset is unavailable, label the primary human-speech outcome unavailable rather than substituting a synthetic headline.
- **Secondary:** exact accuracy on ID speech, structural/domain-shift schemas, composition and correction; valid-output rate; tool-selection accuracy; argument precision/recall/F1; exact joint-argument accuracy; ordered-program success.
- **Speech grounding:** reference-transcript versus audio scores for the same decoder, silence/mismatched audio controls and answer-changing value/correction pairs.
- **Practical baseline:** full Whisper-tiny ASR followed by task-SFT text SmolLM2, scored with the same catalogs and semantic evaluator. Count both decoding stages and their model weights.
- **Retention:** the reproduced native benchmark subset/full split as supported, before and after adaptation. Keep retention separate from primary tool metrics.
- **Dynamics:** semantic accuracy and fixed-probe target likelihood versus updates, sampled prompts, generated tokens and measured training compute; reward mean/variance, entropy, KL, gradient norm, invalid rate and all-equal-reward group rate.
- **Resources:** peak training/inference memory and stage-specific elapsed times. Device-independent claims must not be inferred from V100 speed.

Stratify task results by real versus synthetic speech, operation and request class. Reference-transcript performance is a diagnostic, not a guaranteed upper bound: removing audio also changes conditioning and can expose text-task forgetting. Avoid attributing the entire text/audio gap to the frozen encoder.

Use greedy generation for primary task evaluation with identical output conventions. A sampled diagnostic is allowed for rollout analysis but is not a replacement primary score.

For robustness, add reproducible noise at 15 and 5 dB SNR to a fixed held-out subset, saving the noise seed and scaling rule. Keep clean/noisy variants grouped; do not tune on final-test corruption results. Report this as an audio control, without changing the training objective.

Checkpoint selection uses a fixed development metric, default the equally weighted mean of development ID and development-schema-OOD exact accuracy, macro-averaged over real/synthetic sources when both are present. Final OOD schema names/transforms must not participate in selection. Evaluate final test predictions only after freezing settings; reopening tests after bug fixes requires a documented new evaluation version, not hidden retuning.

Report per-seed differences, mean and standard deviation across the three main seeds, and paired bootstrap intervals over utterance groups. Keep all audio/schema/voice/counterfactual siblings together. For multi-schema tests, bootstrap utterance-level averages. Keep dataset revision and evaluated coverage beside each score.

If only JSON validity improves, report a structured-output gain. If requested values and paired tests improve, report grounding gains. Schema renaming success alone is not proof of arbitrary novel-tool reasoning. Report negative results and uncertainty at this task/model/budget.

### 5.10 Repository and command interface

Implement a package named `nano_audio`, with these responsibilities:

| Path | Responsibility |
|---|---|
| `pyproject.toml` and tested lock/constraints files | Core entry point and reproducible environments |
| `configs/models/` | Pinned native/fresh model configurations |
| `configs/profiles/` | Smoke, local pilot and main limits |
| `configs/experiments/` | Full run matrix, optimizer groups, seeds and selection rules |
| `src/nano_audio/models/` | Published wrappers, fresh initialization and token/prefix adapters |
| `src/nano_audio/data/` | Downloads, source mappings, generation, synthesis, schemas, audits and cache |
| `src/nano_audio/training/` | SFT, GRPO, shared initialization, compute ledgers and resume |
| `src/nano_audio/evaluation/` | Native evaluators, strict semantic scorer, simulator and research suites |
| `src/nano_audio/reporting/` | Claim ledger, curves, tables and stage summaries |
| `tests/` | Offline unit tests and marked model integration tests |
| `scripts/` | Environment helpers and portable launch scripts |
| `artifacts/` | Versioned manifests, reports, predictions and checkpoints, excluded from source control as appropriate |

Expose the commands through a `nano-audio` console entry point after installing the new repository into its tested environment, for example with `uv pip install -e .`. Implement these commands with discoverable `--help` and structured errors:

```bash
nano-audio inspect-env --output artifacts/environment.json
nano-audio resolve-models --config configs/experiments/study.yaml
nano-audio download --profile smoke
nano-audio smoke-text --model smollm2_135m_base --device auto
nano-audio smoke-audio --model samsone_134m_published --device auto
nano-audio prepare-data --profile local_pilot
nano-audio audit-data --profile local_pilot
nano-audio reproduce --models samsone_134m_published mellow_v0_published --profile local_pilot
nano-audio train --run S_P1 --profile local_pilot --seed 17
nano-audio train --run S_P_SFT --profile local_pilot --seed 17
nano-audio train --run S_P_GRPO --profile local_pilot --seed 17
nano-audio evaluate --profile local_pilot --suite research
nano-audio run-study --profile local_pilot --device auto
nano-audio report --profile local_pilot
```

`run-study` is the resumable orchestrator for all requested stages and tracks. It starts with the primary family, then exercises the second published family and fresh track. It must not re-download or re-train completed matching artifacts. A stage JSON records inputs, outputs, hashes, status and exact next command.

Full execution uses the same interface:

```bash
nano-audio download --profile main
nano-audio prepare-data --profile main
nano-audio audit-data --profile main
nano-audio reproduce --models samsone_134m_published mellow_v0_published --profile main
nano-audio run-study --profile main --device cuda
nano-audio evaluate --profile main --suite all
nano-audio report --profile main
```

No command should require manually editing code between profiles. All output directories are run-specific. Handle published environments through explicit subprocess workers if their dependency versions cannot coexist; exchange versioned manifests/predictions rather than pickling live cross-environment objects.

### 5.11 Resume and V100 portability

Save model, optimizer, scheduler, scaler, RNG states, sampler/dataloader position, data/schema hashes, counters, precision, parameter manifest and reference-policy identity. A mid-rollout checkpoint must either preserve the rollout and old probabilities or explicitly discard the incomplete group before resuming. Verify continuity on a small deterministic run.

Provide a portable Slurm script for the user's later single-V100 allocation: one GPU, at most 20 CPUs, 64GB host RAM and six-hour jobs. Save periodically and on a handled preemption/termination signal. Use relocation-safe relative data paths and a configurable root. This is infrastructure portability, not a project schedule. Do not submit it during the local-first implementation request.

### 5.12 Required tests and acceptance evidence

Write meaningful tests for:

- strict JSON parsing, duplicate keys, booleans/numbers, wrong tool, missing/extra arguments, invalid enums, contradictory/extra calls and order-sensitive programs;
- schema binding round trips, renamed keys, nested arguments and permitted unit conversions;
- target/schema validity, split overlap, sibling grouping, prompt metadata isolation and full-catalog inclusion;
- vocabulary pruning, token round trips, special tokens, answer loss masks and causal scoring alignment;
- reward signs, detached old/reference paths, equal-reward groups, completion-only masks and microbatch equivalence;
- frozen encoder/reference parameters and nonzero intended policy gradients;
- checkpoint round trips/resume and deterministic cache invalidation;
- native-versus-common-interface inference parity where preprocessing/prompts match.

Run offline tests without downloading models. Mark real-model integration tests separately and execute them during local validation. No hand-written expected model answers count as measured predictions.

Deliver these generated research artifacts from the codebase:

1. environment/asset/architecture manifests and tested environment locks;
2. source and generated-data manifests with hashes, licenses, exclusions and split audits;
3. complete implemented source, configurations, tests and working CLI;
4. native example outputs, claim ledger, raw reproduction predictions and score/coverage tables;
5. real local SFT/GRPO smoke evidence, eight-example diagnostic and local-pilot predictions/curves;
6. fresh-initialization provenance and the full main run matrix;
7. learning/compute ledgers, checkpoints and exact main/V100 commands;
8. a research report distinguishing measured results, smoke-only evidence, failed checks and blocked/full experiments.

Implementation completion and experiment completion are separate statuses. All code can be complete while a hidden-label benchmark or a full three-seed run remains unexecuted. State that accurately. The research is not successful merely because training starts; it needs evaluated outputs and honest baselines.

## References and required reading

Read the native source/configuration before implementing its wrapper; read the reward and data/scoring code before training. Pin source revisions in the generated manifest. URLs below were checked while preparing this specification; an inaccessible asset is an execution issue to report, not a reason to invent its contents.

- **[R1] Samsone official code and server checkpoint API:** <https://github.com/SamsungLabs/samsone>. Inspect inference, model, vocabulary and training code; use the server checkpoint, not mobile-export weights.
- **[R2] Samsone paper:** <https://arxiv.org/abs/2609.21666>. Read architecture/initialization, supervised objective and benchmark protocol; do not mistake nano parameter count for a small original training budget.
- **[R3] Samsone configuration and dependencies:** <https://github.com/SamsungLabs/samsone/blob/main/configs/Samsone134M.yaml> and <https://github.com/SamsungLabs/samsone/blob/main/pyproject.toml>. Resolve vocabulary-map generation from the pinned repository.
- **[R4] Mellow official code and checkpoints:** <https://github.com/soham97/mellow> and <https://huggingface.co/soham97/mellow>. Read its native wrapper and linked training branch.
- **[R5] Mellow paper:** <https://arxiv.org/html/2503.08540>. Read experimental setup, evaluation tasks and initialization/adapter caveats.
- **[R6] Mellow requirements:** <https://github.com/soham97/mellow/blob/main/requirements.txt>. Treat these as an upstream environment to inspect, not a lock to install over a working modern Torch setup.
- **[R7] SLURP:** <https://github.com/pswietojanski/slurp> and <https://aclanthology.org/2020.emnlp-main.588/>. Inspect annotations, splits, official entity evaluation and separate audio/text licenses.
- **[R8] Audio2Tool paper/project:** <https://arxiv.org/abs/2604.22821> and <https://audio2tool.github.io/>. Read domain/tier definitions and the benchmark's actual scope.
- **[R9] Audio2Tool release/scoring:** <https://huggingface.co/datasets/RVtech/Audio2Tool> and <https://github.com/audio2tool/Audio2Tool>. Check the current public file layout rather than assuming an older README path is exact.
- **[R10] MMAU official evaluation:** <https://github.com/Sakshi113/MMAU>. Inspect dataset-version changes, labelled mini data and official full-test scoring.
- **[R11] GRPO documentation and source:** <https://huggingface.co/docs/trl/grpo_trainer>. Check objective normalization, masking and reference semantics against the explicitly selected convention in this document.
- **[R12] Audio RL precedent:** <https://arxiv.org/abs/2503.11197>. Read the SFT/RL comparison and its scope before extrapolating to nano models.
- **[R13] Audio-grounding caution:** <https://arxiv.org/abs/2505.09439>. Read the audio/text controls motivating paired grounding tests.
- **[R14] Local TTS:** <https://huggingface.co/hexgrad/Kokoro-82M> and <https://github.com/hexgrad/kokoro>. Inspect actual voices, dependencies, output sample rate and license.
- **[R15] Component checkpoints:** <https://huggingface.co/HuggingFaceTB/SmolLM2-135M> and <https://huggingface.co/openai/whisper-tiny>. Count the deployed speech encoder separately from the complete ASR model.
- **[R16] V100 build context:** <https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/>. Verify the actual installed build rather than inferring compatibility from the driver banner.

## Execution handoff

Start by inspecting the repository/workspace and Stage 0 decisions. Implement all stages and tests, run text and audio inference, generate only the local profile's data, run the real local training pilot, and produce the evidence/report artifacts above. Keep the architecture fixed. Use final test schemas only for final evaluation. Proceed through routine implementation choices autonomously, and report concrete external blockers with the completed independent work and exact commands to resume.

