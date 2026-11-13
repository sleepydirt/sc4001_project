"""Check the saved GRPO resume, gate, and training fingerprint without writing to the experiment."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from unittest.mock import patch

import run


def find_earlier_revision(inputs, expected):
    revisions = subprocess.check_output(
        ["git", "-C", str(run.ROOT), "rev-list", "--all", "--reflog", "--max-count=256"],
        text=True,
    ).splitlines()
    for revision in revisions:
        if revision == inputs["revisions"]["project"]:
            continue
        candidate = {**inputs, "revisions": {**inputs["revisions"], "project": revision}}
        if run.digest(candidate) == expected:
            return revision
    return None


def inspect(root):
    data_path = root / "data/manifest.json"
    manifest = run.read_json(data_path)
    data_sha = run.sha(data_path)
    warmup = run.read_json(root / "warmup/stage.json")
    gate = run.read_json(root / "warmup/gate.json")
    grpo = run.read_json(root / "grpo/stage.json")
    resume = run.read_json(root / "grpo/resume.json")

    # Capture the inputs to the runner's fingerprint without duplicating its file list.
    with patch.object(run, "digest", side_effect=lambda inputs: inputs):
        inputs = run.code_identity()
    current_identity = run.digest(inputs)

    config = grpo["config"]
    saved_plan = {
        "updates": config["total_updates"],
        "batch_size": config["effective_batch_size"],
        "eval_every": config["checkpoint_every"],
    }
    policy_path = root / "warmup" / warmup["policy"]
    checkpoint_dir = root / "grpo" / resume["directory"]
    checks = {
        "gate_passed": gate["passed"] is True,
        "manifest_matches_gate_and_both_stages": (
            data_sha == gate["data_sha256"] == warmup["data_sha256"] == grpo["data_sha256"]
        ),
        "examples_match_manifest": run.sha(root / "data/examples.jsonl") == manifest["examples_sha256"],
        "warmup_policy_matches_gate_and_grpo_parent": (
            run.sha(policy_path) == warmup["policy_sha256"] == gate["policy_sha256"] == grpo["parent_sha256"]
        ),
        "warmup_code_matches_original_grpo_code": warmup["code_identity"] == grpo["code_identity"],
        "current_code_matches_warmup": current_identity == warmup["code_identity"],
        "current_code_matches_original_grpo": current_identity == grpo["code_identity"],
        "gate_plan_matches_saved_grpo_plan": gate["paired_plan"] == saved_plan,
        "resume_files_present": all((checkpoint_dir / filename).is_file()
                                    for filename in ("policy", "reference", "resume.pt")),
    }
    earlier_revision = None
    if checks["warmup_code_matches_original_grpo_code"] and not checks["current_code_matches_warmup"]:
        earlier_revision = find_earlier_revision(inputs, warmup["code_identity"])

    command = ["sbatch", "semi_pilot/run.slurm", "grpo", root.name,
               "--updates", str(saved_plan["updates"]), "--batch-size", str(saved_plan["batch_size"]),
               "--eval-every", str(saved_plan["eval_every"])]
    return {
        "checkout": str(run.ROOT),
        "nano_audio_root_override": os.environ.get("NANO_AUDIO_ROOT"),
        "experiment": str(root),
        "grpo_status": grpo["status"],
        "saved_update": resume["step"],
        "checkpoint": str(checkpoint_dir),
        "gated_plan": gate["paired_plan"],
        "saved_grpo_plan": saved_plan,
        "checks": checks,
        "current_revisions": inputs["revisions"],
        "matching_earlier_project_revision": earlier_revision,
        "resume_command_from_saved_plan": shlex.join(command),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="existing experiment name, e.g. semi-long")
    args = parser.parse_args()
    try:
        result = inspect(run.experiment_dir(args.experiment))
    except FileNotFoundError as error:
        parser.exit(1, f"Missing resume input: {error.filename}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

