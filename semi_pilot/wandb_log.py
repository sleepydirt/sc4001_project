"""Publish semi-pilot scalar metrics from the existing experiment ledgers."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from threading import Event, Thread

from data import ROOT, experiment_dir

TRAINING_STAGES = ("warmup", "sft", "grpo")
TRACKED_STAGES = ("overfit", "gate", *TRAINING_STAGES)
DEFAULT_PROJECT = "team88_sc4001"


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def ledger_rows(path):
    if path.is_file():
        with path.open() as handle:
            for line in handle:
                # An active writer may still be appending the last line.
                if not line.endswith("\n"):
                    break
                yield json.loads(line)


def scalars(prefix, values):
    return {f"{prefix}/{key}": float(value) if isinstance(value, bool) else value
            for key, value in values.items() if isinstance(value, (int, float))}


def dev_metrics(snapshot):
    metrics = scalars("dev", snapshot)
    metrics.pop("dev/step", None)
    for cell, values in snapshot["cells"].items():
        metrics.update(scalars(f"dev/{cell}", values))
    return metrics


def saved_metrics(root, stage):
    if stage == "gate":
        result = read_json(root / "warmup/gate.json")
        if not result:
            return {}
        update = read_json(root / "warmup/stage.json")["step"]
        metrics = scalars("gate", result)
        metrics.update(scalars("gate/mixed_groups", result["mixed_groups"]))
        return {update: {**metrics, **dev_metrics(result["dev"])}}

    directory = root / stage
    state = read_json(directory / ("stage.json" if stage == "overfit" else "resume.json"))
    # The training pointer is published only after its checkpoint and evaluation are saved.
    limit = state.get("updates" if stage == "overfit" else "step")
    if limit is None:
        return {}
    points = defaultdict(dict)
    for row in ledger_rows(directory / "learning.jsonl"):
        step = row.pop("step")
        if step <= limit:
            points[step].update(scalars("train", row))
            for index, rate in enumerate(row.get("learning_rates", [])):
                points[step][f"train/learning_rate_group_{index}"] = rate
    if stage == "overfit":
        curve = {row["update"]: row for row in ledger_rows(directory / "curve.jsonl")}
        curve.setdefault(limit, state["last"])
        for update, row in curve.items():
            if update <= limit:
                points[update].update(scalars("diagnostic", {key: value for key, value in row.items() if key != "update"}))
                points[update]["diagnostic/exact"] = row["exact_successes"] / row["n"]
        points[limit]["diagnostic/criterion_reached"] = int(state["criterion_reached"])
    else:
        for path in (directory / "dev").glob("*.json"):
            snapshot = read_json(path)
            if snapshot["step"] <= limit:
                points[snapshot["step"]].update(dev_metrics(snapshot))
    return points


def start_run(root, stage, backfill):
    import wandb

    mode = os.environ.get("WANDB_MODE", "online")
    project = os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT
    job = os.environ.get("SLURM_JOB_ID", "local")
    directory = Path(os.environ.get("WANDB_DIR") or ROOT / "artifacts/wandb" / f"semi-{root.name}")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "backfill" if backfill else job
    tracker = wandb.init(
        project=project,
        name=f"{root.name}-{stage}-{suffix}",
        group=root.name,
        job_type=stage,
        dir=str(directory),
        mode=mode,
        save_code=False,
        config={"experiment": root.name, "stage": stage, "seed": 17, "backfill": backfill},
        settings=wandb.Settings(console="off", init_timeout=30),
    )
    tracker.define_metric("update")
    for prefix in ("train", "dev", "diagnostic", "gate"):
        tracker.define_metric(f"{prefix}/*", step_metric="update")
    print(f"W&B tracking {tracker.name} in {project} ({mode}).", flush=True)
    return tracker


def publish(tracker, points, sent):
    for update, metrics in sorted(points.items()):
        if update not in sent:
            tracker.log({"update": update, **metrics})
            sent.add(update)


def backfill_run(root, stage):
    if os.environ.get("WANDB_MODE") == "disabled":
        raise ValueError("Set WANDB_MODE=online or offline before backfilling.")
    points = saved_metrics(root, stage)
    if not points:
        raise FileNotFoundError(f"No saved {stage} metrics under {root}. Check the experiment name and stage files.")
    tracker = start_run(root, stage, backfill=True)
    with tracker:
        publish(tracker, points, set())
    print(f"Published {len(points)} saved updates for {stage}.", flush=True)


def run_job(root, stage, flags):
    from run import main as run_stage

    command = ["train", "--arm", stage] if stage in TRAINING_STAGES else [stage]
    args = [*command, "--experiment", root.name, *flags]
    if os.environ.get("WANDB_MODE") == "disabled":
        return run_stage(args)
    try:
        tracker = start_run(root, stage, backfill=False)
    except Exception as error:
        print(f"W&B initialization failed: {error}. Continuing with local metrics.", file=sys.stderr, flush=True)
        return run_stage(args)

    sent = set()
    stop = Event()

    def update():
        publish(tracker, saved_metrics(root, stage), sent)

    def watch():
        try:
            while not stop.wait(10):
                update()
        except Exception as error:
            print(f"W&B live logging paused: {error}", file=sys.stderr, flush=True)

    observer = None
    if stage in TRAINING_STAGES:
        observer = Thread(target=watch, name="wandb-ledger-watch", daemon=True)
        observer.start()
    exit_code = 1
    try:
        # Keep the stage in this process so Slurm still signals its original checkpoint handler.
        exit_code = run_stage(args)
        return exit_code
    finally:
        stop.set()
        if observer is not None:
            observer.join()
        try:
            update()
        except Exception as error:
            print(f"W&B metrics upload failed: {error}. Local ledgers are available for backfill.",
                  file=sys.stderr, flush=True)
        finally:
            try:
                tracker.finish(exit_code=exit_code)
            except Exception as error:
                print(f"W&B finalization failed: {error}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("job", "backfill"):
        child = sub.add_parser(command)
        child.add_argument("stage", choices=TRACKED_STAGES)
        child.add_argument("experiment")
        if command == "job":
            child.add_argument("flags", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = experiment_dir(args.experiment)
    if args.command == "backfill":
        try:
            backfill_run(root, args.stage)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        return 0
    return run_job(root, args.stage, args.flags)


if __name__ == "__main__":
    raise SystemExit(main())
