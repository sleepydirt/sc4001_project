#!/usr/bin/env python3
"""Run nano-audio checkpoint inference and write an offline listening report."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote


DEFAULT_STEPS = {"S_P1": 64, "S_P_GRPO": 512, "S_P_SFT": 1024}
CATALOGS = {
    "train": "familiar",
    "dev": "dev",
    "names": "held-out names",
    "structure": "held-out structure",
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare nano-audio policies and export report.html with playable audio.",
        epilog="Copy the entire output directory to your computer and open report.html in a browser.",
    )
    parser.add_argument("--repo-root", type=Path, help="training checkout (default: find it from the current directory)")
    parser.add_argument("--output-dir", type=Path, help="new report directory (default: artifacts/pilot-reviews/<timestamp>)")
    parser.add_argument("--profile", default="local_pilot")
    parser.add_argument("--seed", type=int, default=17, help="training seed")
    parser.add_argument("--device", default="cuda", help="inference device, such as cuda, cuda:0 or cpu")
    parser.add_argument("--run", action="append", help="study run ID (repeatable, defaults to S_P1, S_P_GRPO and S_P_SFT)")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="RUN=FILE",
                        help="override a run's checkpoint/policy file (repeatable, trusted files only)")
    parser.add_argument("--expect-step", action="append", default=[], metavar="RUN=STEP",
                        help="override the expected saved step for a run (repeatable)")
    parser.add_argument("--allow-partial", action="store_true", help="include saves below their expected step")
    parser.add_argument("--split", choices=("test", "dev"), default="test")
    parser.add_argument("--catalog", action="append", choices=tuple(CATALOGS.values()),
                        help="catalog condition (repeatable, defaults to all conditions for the split)")
    parser.add_argument("--groups", type=positive_int, default=6, help="maximum held-out audio clips (default: 6)")
    parser.add_argument("--source", help="filter prepared examples by source, such as synthetic or slurp")
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=positive_int, default=128)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--audio", action="append", type=Path, help="custom WAV clip with a familiar catalog (repeatable)")
    inputs.add_argument("--custom-manifest", type=Path, help="JSON list of custom clips with optional labels")
    args = parser.parse_args(argv)

    args.run = args.run or list(DEFAULT_STEPS)
    args.catalog = args.catalog or (["dev"] if args.split == "dev" else [
        "familiar", "held-out names", "held-out structure",
    ])
    if len(set(args.run)) != len(args.run) or len(set(args.catalog)) != len(args.catalog):
        parser.error("run IDs and catalog conditions must be unique")
    if args.max_new_tokens > 128:
        parser.error("--max-new-tokens must be at most the training budget of 128")
    for name, option in (("checkpoint", "--checkpoint"), ("expect_step", "--expect-step")):
        values = {}
        for entry in getattr(args, name):
            run, sep, value = entry.partition("=")
            if not sep or not value or run not in args.run or run in values:
                parser.error(f"{option} requires one RUN=value per selected run")
            if name == "expect_step":
                if not value.isdecimal():
                    parser.error(f"{option} requires a nonnegative integer step")
                value = int(value)
            values[run] = value
        setattr(args, name, values)
    return args


def find_repo(explicit_root):
    candidates = [explicit_root.expanduser()] if explicit_root else [Path.cwd(), *Path.cwd().parents]
    for root in candidates:
        if (root / "src/nano_audio/cli.py").is_file():
            return root.resolve()
    raise FileNotFoundError("Training checkout not found. Run from the checkout or pass --repo-root.")


def catalog_condition(example):
    return CATALOGS[example["schema_family_id"].split("_", 1)[0]]


def select_examples(args, cli):
    if args.custom_manifest or args.audio:
        if args.custom_manifest:
            manifest = args.custom_manifest.expanduser().resolve()
            clips = json.loads(manifest.read_text(encoding="utf-8"))
            base = manifest.parent
        else:
            clips = [{"audio_path": path} for path in args.audio]
            base = Path.cwd()
        if not isinstance(clips, list) or not clips:
            raise ValueError("Provide a nonempty JSON list of custom clips.")
        families = cli.families()
        examples = []
        for index, clip in enumerate(clips, 1):
            family = families[clip.get("schema_family_id", "train_01")]
            gold = clip.get("canonical_target")
            clip_id = f"custom-{index:02d}"
            examples.append({
                "id": clip_id,
                "utterance_group_id": clip_id,
                "source": "custom",
                "split": "custom",
                "schema_family_id": family.family_id,
                "audio_path": str(base / Path(clip["audio_path"]).expanduser()),
                "tools": family.tools,
                "transcript": clip.get("transcript"),
                "canonical_target": gold,
                "target": family.encode(gold) if gold is not None else None,
                "order_matters": clip.get("order_matters", True),
            })
        return examples, None

    recordings = defaultdict(dict)
    rows = sorted(cli.records(args.profile, args.split), key=lambda row: (row["schema_family_id"], row["id"]))
    for row in rows:
        catalog = catalog_condition(row)
        if catalog in args.catalog and (args.source is None or row["source"] == args.source):
            key = (row["utterance_group_id"], row["recording_id"], row["audio_path"])
            recordings[key].setdefault(catalog, row)

    groups = {}
    for (group, _, _), rows in sorted(recordings.items()):
        if all(catalog in rows for catalog in args.catalog):
            groups.setdefault(group, [rows[catalog] for catalog in args.catalog])
    if not groups:
        raise ValueError(f"No complete audio/catalog pairs for split {args.split}, catalogs {args.catalog} and source {args.source}.")

    buckets = defaultdict(list)
    for group in groups.values():
        first = group[0]
        buckets[(first["source"], first.get("task_kind") or "")].append(group)
    rng = random.Random(args.sample_seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    # Alternate between sources and task types to show different behaviors in a small review.
    ordered = [group for batch in zip_longest(*(buckets[key] for key in sorted(buckets)))
               for group in batch if group is not None]
    return [row for group in ordered[:args.groups] for row in group], file_hash(cli.dataset_path(args.profile))


def copy_clip(source, bundle, clips):
    source = Path(source).resolve()
    if source not in clips:
        if not source.is_file():
            raise FileNotFoundError(f"Audio clip not found: {source}")
        suffix = source.suffix.lower()
        relative = f"audio/clip-{len(clips) + 1:03d}{suffix}"
        destination = bundle / relative
        shutil.copyfile(source, destination)
        clips[source] = {"path": relative, "sha256": file_hash(destination)}
    return clips[source]


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def evaluate_run(args, cli, study, run, examples, bundle, clips, feature_dir, data_hash):
    import torch

    config = study[run]
    model_id = config["model"]
    baseline = config["stage"] == "baseline"
    expected = args.expect_step.get(run, DEFAULT_STEPS.get(run) if args.profile == "local_pilot" else None)
    override = args.checkpoint.get(run)
    if baseline and (override is not None or expected not in (None, 0)):
        raise ValueError(f"{run} uses the published asset at step 0.")

    checkpoint = None
    if not baseline:
        checkpoint = Path(override).expanduser().resolve() if override else (
            cli.run_dir(args.profile, run, args.seed) / "checkpoint/policy"
        )
    info = {
        "run": run,
        "model": model_id,
        "requested_step": expected,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }
    if checkpoint and not checkpoint.is_file():
        return {**info, "status": "Missing policy file"}, []

    cli.seeded(args.sample_seed)
    policy = None
    try:
        policy = cli.load_policy(model_id, args.device, args.profile)
        if hasattr(policy, "feature_cache"):
            # Keep cache writes separate from any concurrent training job.
            policy.feature_cache = feature_dir
        step = 0
        policy_hash = None
        training_hash = None
        if checkpoint:
            # Samsone replaces this path atomically. Load and hash the same opened save.
            with checkpoint.open("rb") as handle:
                state = policy.load_checkpoint(handle)
                handle.seek(0)
                policy_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            step = state["step"]
            metadata = state["metadata"]
            saved_profile = state["config"]["profile"]
            if metadata["run"] != run or saved_profile != args.profile:
                raise ValueError(f"{checkpoint} belongs to {metadata['run']} / {saved_profile}.")
            if state["parameter_manifest"]["model_id"] != model_id:
                raise ValueError(f"{checkpoint} belongs to a different model family.")
            training_hash = metadata.get("data_hash")

        info.update(
            step=step, label=f"{run} ({step:,})", checkpoint_sha256=policy_hash,
            training_data_sha256=training_hash, device=str(policy.device),
            precision=getattr(policy, "research_precision", "fp16" if policy.device.type == "cuda" else "fp32"),
        )
        if expected is not None and step != expected:
            if not (args.allow_partial and step < expected):
                return {**info, "status": "Saved step differs from requested step"}, []
        if data_hash and training_hash and data_hash != training_hash:
            info["data_warning"] = "The prepared manifest differs from the training manifest. Verify the held-out split."

        predictions = []
        generation = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
        for example in examples:
            audio = example["report_audio"]
            item = {"audio_path": str(bundle / audio["path"]), "tools": example["tools"]}
            if "instruction" in example:
                item["instruction"] = example["instruction"]
            audio2 = None
            if model_id == "mellow_v0_published":
                _, second = policy.audio_paths(item)
                audio2 = copy_clip(second, bundle, clips)
                item["audio2"] = str(bundle / audio2["path"])

            output = policy.generate(item, generation)
            score = cli.scoring(example, output) if example.get("canonical_target") is not None else None
            predictions.append({
                "run": run, "label": info["label"], "step": step, "model": model_id,
                "id": example["id"], "utterance_group_id": example["utterance_group_id"],
                "source": example["source"], "split": example["split"], "task_kind": example.get("task_kind"),
                "catalog": catalog_condition(example), "schema_family_id": example["schema_family_id"],
                "audio": audio, "audio2": audio2, "transcript": example.get("transcript"),
                "expected": example.get("target"), "canonical_target": example.get("canonical_target"),
                "output": output, "score": score,
            })
        status = "Partial checkpoint" if expected is not None and step < expected else "Evaluated"
        return {**info, "status": status}, predictions
    finally:
        del policy
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()


def summarize(predictions):
    from nano_audio.evaluation.metrics import aggregate

    scored = defaultdict(list)
    for row in predictions:
        if row["score"] is not None:
            scored[(row["label"], row["source"], row["catalog"])].append({
                "utterance_group_id": row["utterance_group_id"], **row["score"],
            })
    return [
        {"run": run, "source": source, "catalog": catalog, **aggregate(rows)}
        for (run, source, catalog), rows in scored.items()
    ]


def outcome(row):
    from nano_audio.data.task import TaskError, strict_json

    score = row["score"]
    if row["output"]["length_capped"]:
        return "Hit token limit"
    if score is None:
        return "Unscored clip"
    if score["exact"]:
        return "Exact match"
    if not score["valid"]:
        try:
            strict_json(row["output"]["text"])
        except TaskError:
            return "Invalid JSON"
        return "JSON violates tool schema"
    return "Wrong arguments" if score["tool_accuracy"] else "Wrong tool or call order"


REPORT_STYLE = """
:root { color-scheme: light; font: 16px/1.5 system-ui, sans-serif; color: #182230; background: #f3f5f8; }
body { max-width: 92rem; margin: 0 auto; padding: clamp(1rem, 3vw, 3rem); }
h1 { font-size: clamp(1.8rem, 4vw, 2.65rem); margin: 0; }
h2, h3, p { margin: 0 0 .65rem; }
a { color: #1855a4; }
.muted { color: #47576c; }
.panel { margin: 1.4rem 0; padding: clamp(.8rem, 2vw, 1.5rem); background: #fff; border: 1px solid #d6dde7; border-radius: .8rem; }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; text-align: left; }
th, td { padding: .45rem .7rem; border-bottom: 1px solid #e2e7ee; vertical-align: top; }
.meter { display: flex; gap: .45rem; align-items: center; font-variant-numeric: tabular-nums; }
meter { width: 5rem; height: .85rem; }
.clip-header { position: sticky; top: 0; background: #fff; z-index: 1; padding: .7rem 0; border-bottom: 1px solid #d6dde7; }
audio { display: block; width: min(28rem, 100%); height: 2.65rem; }
.case { margin-top: 1rem; }
.answers { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 17rem), 1fr)); gap: .7rem; }
.answer { min-width: 0; padding: .75rem; background: #f4f7fb; border-radius: .45rem; }
.answer.expected { background: #eef5ff; }
.answer strong { display: block; }
pre { margin: .55rem 0 0; white-space: pre-wrap; overflow-wrap: anywhere; font-size: .86rem; }
nav a { display: inline-block; margin: 0 .8rem .35rem 0; }
"""


def audio_player(clip, label):
    path = escape(quote(clip["path"]), quote=True)
    return f'<audio controls preload="none" src="{path}" aria-label="{escape(label, quote=True)}"></audio>'


def render_report(bundle, info, predictions, summary):
    runs = "".join(
        "<tr>"
        f"<th scope='row'>{escape(run['run'])}</th>"
        f"<td>{run.get('step', '—')}</td><td>{run['requested_step'] if run['requested_step'] is not None else '—'}</td>"
        f"<td>{escape(run['status'])} {escape(run.get('data_warning', ''))}</td></tr>"
        for run in info["runs"]
    )
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Audio model comparison</title><style>{REPORT_STYLE}</style><body>",
        "<h1>Audio model comparison</h1>",
        f"<p class='muted'>{escape(info['profile'])} · {escape(info['split'])} · "
        f"Greedy decoding · {info['max_new_tokens']} token limit</p>",
        "<p>Play a clip and read each model’s raw tool call beside the expected answer.</p>",
        "<section class='panel'><h2>Checkpoints</h2><div class='scroll'><table>",
        "<thead><tr><th>Run</th><th>Saved step</th><th>Requested step</th><th>Status</th></tr></thead>",
        f"<tbody>{runs}</tbody></table></div></section>",
    ]

    if summary:
        parts.extend([
            "<section class='panel'><h2>Scores on the selected clips</h2>",
            "<p class='muted'>Validity checks the supplied schema. Exact match also checks the requested "
            "tool, arguments, ordering where applicable, and completion within the token limit. "
            "Differences between runs can reflect both the objective and the number of updates.</p>",
            "<div class='scroll'><table><thead><tr><th>Run</th><th>Source</th><th>Catalog</th><th>Clips</th>"
            "<th>Schema-valid</th><th>Exact match</th></tr></thead><tbody>",
        ])
        for row in summary:
            meters = "".join(
                f"<td><div class='meter'><meter min='0' max='1' value='{row[key]}' "
                f"aria-label='{label}'></meter><span>{row[key]:.0%}</span></div></td>"
                for key, label in (("valid", "Schema-valid"), ("exact", "Exact match"))
            )
            parts.append(
                f"<tr><th scope='row'>{escape(row['run'])}</th><td>{escape(row['source'])}</td>"
                f"<td>{escape(row['catalog'])}</td><td>{row['groups']}</td>{meters}</tr>"
            )
        parts.append("</tbody></table></div></section>")

    groups = defaultdict(dict)
    for row in predictions:
        groups[row["utterance_group_id"]].setdefault(row["id"], []).append(row)
    links = " ".join(f'<a href="#clip-{i}">Clip {i}</a>' for i in range(1, len(groups) + 1))
    parts.append(f"<nav aria-label='Jump to a clip'>{links}</nav>")
    for index, cases in enumerate(groups.values(), 1):
        first = next(iter(cases.values()))[0]
        detail = " · ".join(part for part in (first["source"], first["task_kind"]) if part)
        transcript = escape(first["transcript"] or "Reference transcript unavailable")
        parts.extend([
            f"<section class='panel' id='clip-{index}'><header class='clip-header'>",
            f"<h2>Clip {index} <small class='muted'>· {escape(detail)}</small></h2>",
            f"<p>{transcript}</p>{audio_player(first['audio'], f'Clip {index}')}</header>",
        ])
        for rows in cases.values():
            case = rows[0]
            heading = escape(f"{case['catalog']} · {case['schema_family_id']}")
            gold = "Unlabeled clip" if case["expected"] is None else json.dumps(case["expected"], indent=2, ensure_ascii=False)
            parts.append(
                f"<div class='case'><h3>{heading}</h3><div class='answers'>"
                f"<div class='answer expected'><strong>Expected</strong><pre>{escape(gold)}</pre></div>"
            )
            for row in rows:
                raw = escape(row["output"]["text"] or "(empty answer)")
                parts.append(
                    f"<div class='answer'><strong>{escape(row['label'])}</strong>"
                    f"<small>{escape(outcome(row))}</small><pre>{raw}</pre>"
                )
                if row["audio2"]:
                    parts.append(f"<small>Second audio input</small>{audio_player(row['audio2'], 'Second audio input')}")
                parts.append("</div>")
            parts.append("</div></div>")
        parts.append("</section>")
    parts.append("</body></html>")
    (bundle / "report.html").write_text("\n".join(parts), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    root = find_repo(args.repo_root)
    sys.path.insert(0, str(root / "src"))

    import torch
    import yaml
    from nano_audio import cli

    if cli.ROOT != root:
        raise RuntimeError("The imported nano-audio package belongs to a different checkout.")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run inside a GPU allocation or pass --device cpu.")

    study = yaml.safe_load((root / "configs/experiments/study.yaml").read_text())["runs"]
    unknown = set(args.run) - study.keys()
    if unknown:
        raise ValueError(f"Unknown study runs: {', '.join(sorted(unknown))}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else root / "artifacts/pilot-reviews" / timestamp
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    examples, data_hash = select_examples(args, cli)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".pilot-review-", dir=output_dir.parent) as staging:
        bundle = Path(staging) / "report"
        features = Path(staging) / "features"
        (bundle / "audio").mkdir(parents=True)
        features.mkdir()
        clips = {}
        for example in examples:
            example["report_audio"] = copy_clip(example["audio_path"], bundle, clips)
        print(f"{len(examples)} audio/catalog pairs from {len(clips)} clip(s).", flush=True)

        info = {
            "profile": args.profile, "split": examples[0]["split"], "training_seed": args.seed,
            "sample_seed": args.sample_seed, "evaluation_data_sha256": data_hash,
            "max_new_tokens": args.max_new_tokens, "decoding": "greedy", "runs": [],
        }
        predictions = []
        for run in args.run:
            print(f"Checking {run} ...", flush=True)
            status, rows = evaluate_run(args, cli, study, run, examples, bundle, clips, features, data_hash)
            info["runs"].append(status)
            predictions.extend(rows)
            print(f"  {status['status']} | saved step {status.get('step', '—')} | {len(rows)} generations", flush=True)
        if not predictions:
            raise RuntimeError("No predictions generated. Check the policy paths and requested steps above.")

        summary = summarize(predictions)
        if summary:
            with (bundle / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
        with (bundle / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (bundle / "comparison.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        render_report(bundle, info, predictions, summary)
        bundle.rename(output_dir)
    print(f"Report ready: {output_dir / 'report.html'}")
    print(f"Copy this directory to your computer: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"Error: {error}")

