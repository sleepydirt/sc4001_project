"""Run a frozen-data, paired Samsone SFT/GRPO development experiment."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import random
import shutil
import signal
import subprocess
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from uuid import uuid4

from data import ROOT, SPLITS, experiment_dir
from nano_audio.cli import load_policy, seeded
from nano_audio.data.task import schema_families
from nano_audio.evaluation.metrics import aggregate, paired_bootstrap, score_prediction
from nano_audio.utils import digest, read_json, sha, write_json

SOURCES = ("slurp", "synthetic")
SEED = 17
MODEL = "samsone_134m_published"
ARMS = ("sft", "grpo")
FAMILIES = {family.family_id: family for family in schema_families()}
GREEDY = {"max_new_tokens": 128, "do_sample": False}


def load_data(root):
    data = root / "data"
    manifest_path = data / "manifest.json"
    manifest = read_json(manifest_path)
    examples = data / "examples.jsonl"
    if (manifest["format"] != 1 or manifest["split_audit"]["status"] != "passed"
            or sha(examples) != manifest["examples_sha256"]):
        raise ValueError("Frozen data audit or examples hash does not match.")
    rows = {split: [] for split in SPLITS}
    checked = {}
    for line in examples.read_text().splitlines():
        row = json.loads(line)
        relative = row["audio_path"]
        audio = (data / relative).resolve()
        if not audio.is_relative_to(data.resolve()):
            raise ValueError(f"Audio path escapes the frozen dataset: {relative}")
        if audio not in checked:
            checked[audio] = sha(audio)
        if checked[audio] != row["audio_sha256"]:
            raise ValueError(f"Frozen audio hash does not match: {audio}")
        row["audio_path"] = str(audio)
        row["target_text"] = json.dumps(row["target"], separators=(",", ":"))
        rows[row["split"]].append(row)
    for split, items in rows.items():
        actual = {source: len({r["utterance_group_id"] for r in items if r["source"] == source})
                  for source in SOURCES}
        if any(actual[source] < 1 or actual[source] != manifest["counts"][source][split] for source in SOURCES):
            raise ValueError(f"Source counts differ from the frozen {split} manifest.")
    return rows, sha(manifest_path)


def score(example, output):
    return score_prediction(
        output["text"], example["canonical_target"], FAMILIES[example["schema_family_id"]],
        example["order_matters"], output["length_capped"],
    )


def training_pool(rows):
    result = {source: defaultdict(list) for source in SOURCES}
    for row in rows:
        result[row["source"]][row["utterance_group_id"]].append(row)
    return {source: [result[source][key] for key in sorted(result[source])] for source in SOURCES}


def batch_at(pool, step, size):
    # Separate RNG from model sampling. Both arms see identical prompts and tool order.
    rng = random.Random(f"semi-pilot-v1:{SEED}:{step}")
    batch = []
    for source in SOURCES:
        for _ in range(size // 2):
            example = dict(rng.choice(rng.choice(pool[source])))
            example["tools"] = list(example["tools"])
            rng.shuffle(example["tools"])
            batch.append(example)
    rng.shuffle(batch)
    return batch


def representative(rows, families):
    groups = defaultdict(dict)
    for row in sorted(rows, key=lambda x: (x["utterance_group_id"], x["recording_id"], x["id"])):
        if row["schema_family_id"] in families:
            groups[(row["utterance_group_id"], row["recording_id"])].setdefault(row["schema_family_id"], row)
    paired = {}
    for (group, _), variants in groups.items():
        if set(variants) == set(families):
            paired.setdefault(group, variants)
    expected = {row["utterance_group_id"] for row in rows}
    if set(paired) != expected:
        raise ValueError(f"Missing paired catalogs {families} for some utterance groups.")
    return [(label, group[family]) for group in paired.values() for family, label in families.items()]


def evaluate(policy, cases):
    rows = []
    for catalog, example in cases:
        item = {"audio_path": example["audio_path"], "tools": example["tools"]}
        output = policy.generate(item, GREEDY)
        rows.append({
            "id": example["id"], "utterance_group_id": example["utterance_group_id"],
            "source": example["source"], "catalog": catalog, "task_kind": example["task_kind"],
            "transcript": example["transcript"], "audio_path": example["audio_path"],
            "gold": example["target"], "output": output, **score(example, output),
        })
    cells = {}
    for source, catalog in sorted({(r["source"], r["catalog"]) for r in rows}):
        cells[f"{source}/{catalog}"] = aggregate([
            row for row in rows if row["source"] == source and row["catalog"] == catalog
        ])
    return {"macro_exact": mean(cell["exact"] for cell in cells.values()), "cells": cells, "predictions": rows}


@contextmanager
def stage_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is already using {directory}") from error
        yield


def policy_for(directory, device):
    policy = load_policy(MODEL, device)
    policy.feature_cache = directory / "features"
    policy.feature_cache.mkdir(parents=True, exist_ok=True)
    return policy


def code_identity():
    # Dirty repository changes are captured by hashes for the training paths used here.
    inputs = ("semi_pilot/run.py", "semi_pilot/data.py", "src/nano_audio/cli.py",
              "src/nano_audio/models/policy.py", "src/nano_audio/training/core.py",
              "src/nano_audio/data/task.py", "src/nano_audio/evaluation/metrics.py",
              "src/nano_audio/evaluation/simulator.py")
    asset = read_json(ROOT / "artifacts/assets/manifest.json")["models"][MODEL]["local_path"]
    revisions = {
        name: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        for name, path in (("project", ROOT), ("samsone", ROOT / "vendor/samsone"))
    }
    return digest({
        "source": {name: sha(ROOT / name) for name in inputs},
        "base_model": sha(Path(asset)),
        "revisions": revisions,
    })


def completed_stage(root, name, manifest_hash):
    directory = root / name
    state = read_json(directory / "stage.json")
    path = directory / state["policy"]
    if state["status"] != "completed" or state["data_sha256"] != manifest_hash or sha(path) != state["policy_sha256"]:
        raise ValueError(f"{name} completion or frozen checkpoint verification failed.")
    return state, path


def overfit(args, root, rows, manifest_hash):
    from nano_audio.training.diagnostics import overfit_eight

    directory = root / "overfit"
    with stage_lock(directory):
        identity_path = directory / "identity.json"
        identity = {"data_sha256": manifest_hash, "code_identity": code_identity(),
                    "cap": args.cap, "device": args.device}
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("Overfit settings changed. Use a new experiment.")
        write_json(identity_path, identity)
        done = directory / "stage.json"
        if done.exists():
            result = read_json(done)
        else:
            seeded(SEED)
            pool = training_pool(rows)
            if min(map(len, pool.values())) < 4:
                raise ValueError("The overfit check requires at least four training groups per source.")
            rng = random.Random(SEED)
            fixed = [group[0] for source in SOURCES for group in rng.sample(pool[source], 4)]
            policy = policy_for(directory, args.device)
            result = overfit_eight(policy, fixed, lambda ex, out: score(ex, out)["reward"],
                                   directory, cap=args.cap, seed=SEED)
        last = result["last"]
        print(f"Eight-example diagnostic: {last['exact_successes']}/{last['n']} exact at update {last['update']}")
        return 0 if result["criterion_reached"] else 2


def write_checkpoint(trainer, directory):
    pointer = directory / "resume.json"
    with TemporaryDirectory(prefix=".saving-", dir=directory) as temporary:
        path = Path(temporary) / "checkpoint"
        trainer.save_checkpoint(path)
        name = f"checkpoint-{trainer.step:06d}-{uuid4().hex[:8]}"
        path.rename(directory / name)
    write_json(pointer, {"directory": name, "step": trainer.step})
    for previous in directory.glob("checkpoint-??????-????????"):
        if previous.name != name and previous.is_dir() and not previous.is_symlink():
            shutil.rmtree(previous)
    return name


def truncate_ledger(path, step):
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A hard interruption can leave the final append incomplete.
            break
        if row["step"] <= step:
            kept.append(line)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(line + "\n" for line in kept))
    tmp.replace(path)


def train(args, root, rows, manifest_hash):
    import torch
    from nano_audio.training.core import Trainer, TrainingConfig

    directory = root / args.arm
    with stage_lock(directory):
        source_hash = code_identity()
        if args.arm == "warmup":
            overfit_stage = read_json(root / "overfit/stage.json")
            diagnostic = read_json(root / "overfit/identity.json")
            if (not overfit_stage["criterion_reached"] or diagnostic["data_sha256"] != manifest_hash
                    or diagnostic["code_identity"] != source_hash):
                raise RuntimeError("A passing eight-example diagnostic on this dataset and code is required.")
        parent, parent_path = (None, None)
        if args.arm in ARMS:
            parent, parent_path = completed_stage(root, "warmup", manifest_hash)
            gate = read_json(root / "warmup/gate.json")
            if (not gate["passed"] or gate["policy_sha256"] != parent["policy_sha256"]
                    or gate["data_sha256"] != manifest_hash or parent["code_identity"] != source_hash):
                raise ValueError("A passing gate on this exact data, code and warm-up policy is required.")
            # The sibling's configuration is fixed before launch by the passing gate.
            expected = gate["paired_plan"]
            actual = {"updates": args.updates, "batch_size": args.batch_size, "eval_every": args.eval_every}
            if actual != expected:
                raise ValueError(f"Continuation settings must match the gated paired plan: {expected}")

        seeded(SEED)
        precision = "fp16" if torch.device(args.device).type == "cuda" else "fp32"
        config = TrainingConfig.for_profile(
            "local_pilot", "grpo" if args.arm == "grpo" else "sft",
            total_updates=args.updates, effective_batch_size=args.batch_size,
            checkpoint_every=args.eval_every, precision=precision,
        )
        metadata = {
            "experiment": root.name, "arm": args.arm, "data_sha256": manifest_hash,
            "parent_sha256": parent["policy_sha256"] if parent else None,
            "code_identity": source_hash, "seed": SEED, "mixture": {"slurp": 0.5, "synthetic": 0.5},
            "hardware": torch.cuda.get_device_name(torch.device(args.device)) if precision == "fp16" else "cpu",
        }
        identity = digest({"config": asdict(config), "metadata": metadata, "torch": torch.__version__})
        stage_path = directory / "stage.json"
        if stage_path.exists():
            old = read_json(stage_path)
            if old["identity"] != identity:
                raise ValueError("This stage has different frozen settings. Use the original flags or a new experiment.")
            if old["status"] == "completed":
                completed_stage(root, args.arm, manifest_hash)
                print(f"{args.arm} already completed at update {old['step']}.")
                return 0
        stage = {**metadata, "identity": identity, "status": "running", "config": asdict(config)}
        write_json(stage_path, stage)

        stopping = False

        def on_stop(signum, frame):
            nonlocal stopping
            stopping = True

        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, on_stop)

        policy = policy_for(directory, args.device)
        if parent_path:
            policy.load_checkpoint(parent_path)
        ledger = directory / "learning.jsonl"
        trainer = Trainer(policy, config, reward_fn=lambda ex, out: score(ex, out)["reward"],
                          ledger_path=ledger, metadata={**metadata, "identity": identity})
        pool = training_pool(rows["train"])
        dev = representative(rows["dev"], {"train_01": "familiar", "dev_01": "schema_ood"})
        pointer_path = directory / "resume.json"
        if pointer_path.exists():
            pointer = read_json(pointer_path)
            trainer.load_checkpoint(directory / pointer["directory"])
            if trainer.step != pointer["step"] or trainer.sampler_position != trainer.step * args.batch_size:
                raise ValueError("Checkpoint counters disagree with the frozen sampling schedule.")
            checkpoint_name = pointer["directory"]
            truncate_ledger(ledger, trainer.step)
        else:
            truncate_ledger(ledger, 0)
            checkpoint_name = None

        def dev_snapshot():
            result = evaluate(policy, dev)
            write_json(directory / "dev" / f"{trainer.step:06d}.json", {"step": trainer.step, **result})
            print(json.dumps({"arm": args.arm, "step": trainer.step, "dev_macro_exact": result["macro_exact"]}), flush=True)

        if trainer.step == 0:
            dev_snapshot()
        while trainer.step < args.updates and not stopping:
            batch = batch_at(pool, trainer.step, args.batch_size)
            metrics = trainer.grpo_step(batch) if args.arm == "grpo" else trainer.sft_step(batch)
            print(json.dumps({**metrics, "arm": args.arm}), flush=True)
            if trainer.step % args.eval_every == 0 or trainer.step == args.updates:
                dev_snapshot()
                checkpoint_name = write_checkpoint(trainer, directory)
        if stopping and trainer.step < args.updates:
            if not pointer_path.exists() or read_json(pointer_path)["step"] != trainer.step:
                checkpoint_name = write_checkpoint(trainer, directory)
            write_json(stage_path, {**stage, "status": "interrupted", "step": trainer.step})
            print(f"Saved update {trainer.step}. Resubmit the same job with the same flags.", flush=True)
            return 75

        policy_path = directory / checkpoint_name / "policy"
        write_json(stage_path, {
            **stage, "status": "completed", "step": trainer.step, "policy": f"{checkpoint_name}/policy",
            "policy_sha256": sha(policy_path), "examples_seen": trainer.examples_seen,
        })
        print(f"{args.arm} completed at update {trainer.step}: {policy_path}", flush=True)
    return 0


def gate(args, root, rows, manifest_hash):
    with stage_lock(root / "gate-job"):
        parent, path = completed_stage(root, "warmup", manifest_hash)
        if parent["code_identity"] != code_identity():
            raise ValueError("Warm-up was created with different code or base assets.")
        plan = {"updates": args.updates, "batch_size": args.batch_size, "eval_every": args.eval_every}
        gate_path = root / "warmup/gate.json"
        if gate_path.exists():
            result = read_json(gate_path)
            if (result["paired_plan"] != plan or result["rollouts_per_source"] != args.rollouts_per_source
                    or result["policy_sha256"] != parent["policy_sha256"] or result["data_sha256"] != manifest_hash):
                raise ValueError("A gate is already recorded. Use its frozen plan or a new experiment.")
            print(f"Warm-up gate already recorded: {'passed' if result['passed'] else 'failed'}")
            return 0 if result["passed"] else 2

        seeded(SEED)
        policy = policy_for(root / "gate-job", args.device)
        policy.load_checkpoint(path)
        dev = evaluate(policy, representative(rows["dev"], {"train_01": "familiar", "dev_01": "schema_ood"}))
        pool = training_pool(rows["train"])
        rng = random.Random(SEED)
        sampled = {source: [] for source in SOURCES}
        for source in SOURCES:
            size = min(args.rollouts_per_source, len(pool[source]))
            for group in rng.sample(pool[source], size):
                item = dict(rng.choice(group))
                item["tools"] = list(item["tools"])
                rng.shuffle(item["tools"])
                outputs = [policy.generate({"audio_path": item["audio_path"], "tools": item["tools"]},
                                           {**GREEDY, "do_sample": True}) for _ in range(4)]
                sampled[source].append({
                    "id": item["id"], "outputs": outputs,
                    "rewards": [score(item, output)["reward"] for output in outputs],
                })
        mixed = {source: sum(len(set(item["rewards"])) > 1 for item in probes) for source, probes in sampled.items()}
        n = sum(map(len, sampled.values()))
        rate = sum(mixed.values()) / n
        familiar = mean(dev["cells"][f"{source}/familiar"]["exact"] for source in SOURCES)
        # Engineering gate fixed before running. These are development diagnostics.
        passed = familiar >= .25 and sum(mixed.values()) >= 4 and rate >= .10
        write_json(gate_path, {
            "passed": passed, "data_sha256": manifest_hash, "policy_sha256": parent["policy_sha256"],
            "paired_plan": plan, "rollouts_per_source": args.rollouts_per_source,
            "dev_familiar_macro_exact": familiar, "dev": dev, "mixed_groups": mixed,
            "n_groups": n, "mixed_rate": rate, "rollouts": sampled,
        })
        print(f"Gate {'passed' if passed else 'failed'}: dev familiar exact {familiar:.1%}, "
              f"mixed reward groups {sum(mixed.values())}/{n} ({rate:.1%}), by source {mixed}", flush=True)
    return 0 if passed else 2


def matched_stages(root, manifest_hash):
    parent, _ = completed_stage(root, "warmup", manifest_hash)
    stages = {arm: completed_stage(root, arm, manifest_hash) for arm in ARMS}
    left, right = (stages[arm][0] for arm in ARMS)
    for state in (left, right):
        if state["parent_sha256"] != parent["policy_sha256"]:
            raise ValueError("Continuations have a different shared initialization.")
    for key in ("step", "examples_seen", "code_identity", "hardware", "seed"):
        if left[key] != right[key]:
            raise ValueError(f"Continuation stages disagree on {key}.")
    for key in ("effective_batch_size", "checkpoint_every", "precision", "total_updates"):
        if left["config"][key] != right["config"][key]:
            raise ValueError(f"Continuation training settings disagree on {key}.")
    return stages


def compare(root, manifest_hash):
    stages = matched_stages(root, manifest_hash)
    directory = root / "analysis"
    with stage_lock(directory):
        curves = {}
        for arm, (state, _) in stages.items():
            times = [0.0]
            learning = []
            for line in (root / arm / "learning.jsonl").read_text().splitlines():
                row = json.loads(line)
                if row["step"] != len(times):
                    raise ValueError(f"Noncontiguous {arm} training ledger.")
                times.append(times[-1] + row["elapsed_seconds"])
                learning.append(row)
            if len(times) != state["step"] + 1:
                raise ValueError(f"Incomplete {arm} training ledger.")
            points = [read_json(path) for path in sorted((root / arm / "dev").glob("*.json"))]
            curves[arm] = [{
                "step": p["step"], "prompts": p["step"] * state["config"]["effective_batch_size"],
                "optimizer_seconds": times[p["step"]], "macro_exact": p["macro_exact"],
                **{f"{name}_exact": cell["exact"] for name, cell in p["cells"].items()},
            } for p in points if p["step"] <= state["step"]]
            if arm == "grpo":
                previous = 0
                for point in curves[arm]:
                    interval = learning[previous:point["step"]]
                    point["rollout_reward_mean"] = mean(row["reward_mean"] for row in interval) if interval else None
                    point["mixed_reward_group_rate"] = (
                        mean(1 - row["equal_reward_group_rate"] for row in interval) if interval else None
                    )
                    previous = point["step"]
        sft, grpo = (curves[arm] for arm in ARMS)
        if [p["step"] for p in sft] != [p["step"] for p in grpo]:
            raise ValueError("Continuation dev evaluations are not paired at the same steps.")
        paired = [{**{key: left[key] for key in ("step", "prompts")},
                   "sft_macro_exact": left["macro_exact"], "grpo_macro_exact": right["macro_exact"],
                   "grpo_minus_sft": right["macro_exact"] - left["macro_exact"],
                   "sft_optimizer_seconds": left["optimizer_seconds"],
                   "grpo_optimizer_seconds": right["optimizer_seconds"]}
                  for left, right in zip(sft, grpo)]
        # Stability over the last four evaluation points is an operational plateau signal.
        plateau = {arm: (len(points) >= 4 and points[-1]["step"] >= 512
                         and max(p["macro_exact"] for p in points[-4:]) - min(p["macro_exact"] for p in points[-4:]) <= .01)
                   for arm, points in curves.items()}
        for name, rows in {"paired": paired, **curves}.items():
            with (directory / f"{name}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        write_json(directory / "development.json", {
            "data_sha256": manifest_hash, "plateau": plateau, "final": paired[-1], "curves": curves,
        })
        plot_development(directory / "development.html", curves, paired, plateau)
        print(f"Final matched-step dev GRPO − SFT: {paired[-1]['grpo_minus_sft']:+.1%}. Plateau signal: {plateau}")
        print(f"Development report: {directory / 'development.html'}")


def plot_development(path, curves, paired, plateau):
    colors = {"sft": "#1e58a7", "grpo": "#9d3b15"}
    end = max(1, paired[-1]["step"])
    table = "".join(f"<tr><td>{p['step']}</td><td>{p['prompts']}</td><td>{p['sft_macro_exact']:.1%}</td>"
                    f"<td>{p['grpo_macro_exact']:.1%}</td><td>{p['grpo_minus_sft']:+.1%}</td></tr>" for p in paired)
    polylines = []
    for arm, points in curves.items():
        coords = " ".join(f"{60 + 540 * p['step'] / end:.1f},{232 - 200 * p['macro_exact']:.1f}" for p in points)
        polylines.append(f"<polyline points='{coords}' fill='none' stroke='{colors[arm]}' stroke-width='3'/>")
    grid = "".join(f"<text x='5' y='{237 - 50 * i}'>{i * 25}%</text>"
                   f"<path d='M60 {232 - 50 * i}H600' stroke='#ddd'/>" for i in range(5))
    path.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Samsone semi-pilot development</title>
<style>body{{font:16px system-ui;max-width:55rem;margin:2rem auto;padding:0 1rem}} td,th{{padding:.45rem;text-align:right}}</style>
<h1>Samsone semi-pilot development</h1><p>Exact accuracy averaged equally across the two sources and two catalog conditions.</p>
<p><b style="color:{colors['sft']}">SFT</b> · <b style="color:{colors['grpo']}">GRPO</b></p>
<svg viewBox="0 0 630 270" role="img" aria-label="Development exact accuracy by continuation update">
{grid}{''.join(polylines)}<text x="60" y="255">0 updates</text><text x="518" y="255">{end} updates</text></svg>
<p>Last four development points stable within one percentage point after at least 512 updates.
SFT {plateau['sft']}, GRPO {plateau['grpo']}.</p>
<table><tr><th>Updates</th><th>Prompts</th><th>SFT</th><th>GRPO</th><th>GRPO − SFT</th></tr>{table}</table>
<p>Per-source and per-catalog scores and measured optimizer time are in sft.csv and grpo.csv.</p>
</html>""", encoding="utf-8")


def evaluate_test(args, root, rows, manifest_hash):
    from html import escape

    stages = matched_stages(root, manifest_hash)
    current_code = code_identity()
    if any(state["code_identity"] != current_code for state, _ in stages.values()):
        raise ValueError("Use the original training code and assets for the frozen test.")
    if not (root / "analysis/development.json").is_file():
        raise ValueError("Create the matched development comparison before opening the frozen test.")
    directory = root / "test-report"
    with stage_lock(directory):
        if (directory / "result.json").exists():
            print(f"Frozen test was already evaluated: {directory}")
            return
        cases = representative(rows, {"train_01": "familiar", "names_01": "held_out_names",
                                      "structure_01": "held_out_structure"})
        predictions = {}
        for arm, (_, path) in stages.items():
            seeded(SEED)
            policy = policy_for(directory, args.device)
            policy.load_checkpoint(path)
            predictions[arm] = evaluate(policy, cases)
            del policy
        comparisons = {}
        for key in predictions["sft"]["cells"]:
            source, catalog = key.split("/")
            groups = {}
            for arm in ARMS:
                by_group = defaultdict(list)
                for row in predictions[arm]["predictions"]:
                    if row["source"] == source and row["catalog"] == catalog:
                        by_group[row["utterance_group_id"]].append(int(row["exact"]))
                groups[arm] = by_group
            comparisons[key] = paired_bootstrap(groups["grpo"], groups["sft"], seed=SEED)
        write_json(directory / "predictions.json", predictions)
        audio_dir = directory / "audio"
        audio_dir.mkdir(exist_ok=True)
        pairs = {}
        for arm, result in predictions.items():
            for row in result["predictions"]:
                pairs.setdefault(row["id"], {})[arm] = row
        cards, audio_files = [], {}
        for pair in pairs.values():
            row = pair["sft"]
            source_audio = row["audio_path"]
            if source_audio not in audio_files:
                filename = f"clip-{len(audio_files) + 1:03d}.wav"
                shutil.copyfile(source_audio, audio_dir / filename)
                audio_files[source_audio] = filename
            raw = [("Expected", json.dumps(row["gold"], indent=2))]
            raw.extend((arm.upper(), pair[arm]["output"]["text"]) for arm in ARMS)
            answers = "".join(f"<div><b>{label}</b><pre>{escape(text)}</pre></div>" for label, text in raw)
            heading = escape(f"{row['source']} · {row['catalog']}")
            cards.append(f"<section><h2>{heading}</h2><p>{escape(row['transcript'])}</p>"
                         f"<audio controls preload='none' src='audio/{audio_files[source_audio]}'></audio>"
                         f"<div class='answers'>{answers}</div></section>")
        metrics = "".join(f"<tr><td>{key}</td><td>{value['groups']}</td>"
                          f"<td>{predictions['sft']['cells'][key]['exact']:.1%}</td>"
                          f"<td>{predictions['grpo']['cells'][key]['exact']:.1%}</td>"
                          f"<td>{value['difference']:+.1%}</td></tr>" for key, value in comparisons.items())
        (directory / "report.html").write_text(
            "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Frozen test comparison</title><style>body{font:16px system-ui;max-width:80rem;margin:2rem auto;"
            "padding:0 1rem}td,th{padding:.4rem;text-align:left}section{border-top:1px solid #ddd;margin:1.5rem 0}"
            ".answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,18rem),1fr));gap:1rem}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Frozen test comparison</h1>"
            "<table><tr><th>Source / catalog</th><th>Groups</th><th>SFT exact</th><th>GRPO exact</th>"
            f"<th>GRPO − SFT</th></tr>{metrics}</table>{''.join(cards)}", encoding="utf-8",
        )
        write_json(directory / "result.json", {
            "data_sha256": manifest_hash, "policy_sha256": {arm: state["policy_sha256"] for arm, (state, _) in stages.items()},
            "scores": {arm: result["cells"] for arm, result in predictions.items()}, "paired_grpo_minus_sft": comparisons,
        })
        print(f"Frozen test report: {directory / 'report.html'}")


def fork_experiment(root, previous, reuse_warmup):
    _, manifest_hash = load_data(previous)
    if root.exists():
        raise FileExistsError(f"Experiment already exists: {root}")
    with TemporaryDirectory(prefix=".fork-", dir=root.parent) as temporary:
        copy = Path(temporary) / "experiment"
        shutil.copytree(previous / "data", copy / "data")
        diagnostic = previous / "overfit/stage.json"
        if diagnostic.is_file() and read_json(diagnostic)["criterion_reached"]:
            (copy / "overfit").mkdir()
            for filename in ("identity.json", "stage.json"):
                shutil.copy2(previous / "overfit" / filename, copy / "overfit" / filename)
        if reuse_warmup:
            state, policy = completed_stage(previous, "warmup", manifest_hash)
            destination = copy / "warmup" / state["policy"]
            destination.parent.mkdir(parents=True)
            shutil.copy2(policy, destination)
            write_json(copy / "warmup/stage.json", state)
        copy.rename(root)
    print(f"Created {root} with the same frozen data" + (" and P1 policy." if reuse_warmup else "."))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("overfit", "train", "gate", "compare", "test"):
        child = sub.add_parser(command)
        child.add_argument("--experiment", required=True)
        if command != "compare":
            child.add_argument("--device", default="cuda")
        if command == "overfit":
            child.add_argument("--cap", type=int, default=200)
        if command == "train":
            child.add_argument("--arm", choices=("warmup", *ARMS), required=True)
        if command in ("train", "gate"):
            child.add_argument("--updates", type=int, required=True, help="Planned updates, fixed across resubmissions")
            child.add_argument("--batch-size", type=int, default=4)
            child.add_argument("--eval-every", type=int, default=64)
        if command == "gate":
            child.add_argument("--rollouts-per-source", type=int, default=12)
    fork = sub.add_parser("fork", help="copy frozen data into a new experiment and optionally keep the same P1")
    fork.add_argument("--experiment", required=True)
    fork.add_argument("--from-experiment", required=True)
    fork.add_argument("--reuse-warmup", action="store_true")
    args = parser.parse_args(argv)
    if args.command in ("train", "gate"):
        if args.batch_size < 2 or args.batch_size % 2 or min(args.updates, args.eval_every) < 1:
            parser.error("Use an even batch of at least two, and positive update and evaluation counts.")
    if args.command == "gate" and args.rollouts_per_source < 1:
        parser.error("--rollouts-per-source must be positive")
    root = experiment_dir(args.experiment)
    if args.command == "fork":
        fork_experiment(root, experiment_dir(args.from_experiment), args.reuse_warmup)
        return 0
    rows, manifest_hash = load_data(root)
    if args.command == "overfit":
        return overfit(args, root, rows["train"], manifest_hash)
    if args.command == "train":
        return train(args, root, rows, manifest_hash)
    if args.command == "gate":
        return gate(args, root, rows, manifest_hash)
    if args.command == "compare":
        compare(root, manifest_hash)
    if args.command == "test":
        evaluate_test(args, root, rows["test"], manifest_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
