"""Read the semi-pilot's saved GRPO diagnostics without loading models or touching test data."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_ledger(path):
    with path.open(encoding="utf-8") as handle:
        # A live writer can leave a partial last line during this read.
        return [json.loads(line) for line in handle if line.endswith("\n") and line.strip()]


def saved_step(directory):
    return read_json(directory / "resume.json").get("step", 0)


def dev_points(directory, through):
    result = {}
    for path in sorted((directory / "dev").glob("*.json")):
        if int(path.stem) <= through:
            row = read_json(path)
            result[row["step"]] = row
    return result


def show_gate(gate):
    if not gate:
        return
    print("\nWarm-up gate probe, by source (four completions per group)")
    for source, probes in gate["rollouts"].items():
        histogram = Counter(sum(probe["rewards"]) for probe in probes)
        mixed = sum(histogram[n] for n in (1, 2, 3))
        print(f"  {source:10} {histogram[0]:>2} all wrong, {mixed:>2} mixed, "
              f"{histogram[4]:>2} all correct")


def mixed_groups(row):
    return round(row["examples"] * (1 - row["equal_reward_group_rate"]))


def median_gradient(rows):
    return f"{median(row['gradient_norm'] for row in rows):.3g}" if rows else "—"


def show_training(rows, beta, window):
    if not rows:
        print("\nTraining metrics will appear after the first saved update.")
        return
    print("\nGRPO training through the saved checkpoint")
    print(f"{'Updates':>11} {'Reward':>8} {'Mixed groups':>13} {'Updates w/mix':>14} "
          f"{'Grad mix':>9} {'Grad equal':>10} {'Mean KL':>10}")
    for i in range(0, len(rows), window):
        block = rows[i:i + window]
        groups = sum(row["examples"] for row in block)
        reward = sum(row["reward_mean"] * row["examples"] for row in block) / groups
        mixed = [row for row in block if mixed_groups(row)]
        equal = [row for row in block if not mixed_groups(row)]
        rate = sum(map(mixed_groups, block)) / groups
        kl = sum(row["kl"] * row["examples"] for row in block) / groups
        print(f"{block[0]['step']:>5}–{block[-1]['step']:<5} {reward:>8.1%} {rate:>13.1%} "
              f"{len(mixed):>8}/{len(block):<5} {median_gradient(mixed):>9} "
              f"{median_gradient(equal):>10} {kl:>10.3g}")

    residual = max(abs(row["loss"] - beta * row["kl"]) for row in rows)
    seconds = sum(row["elapsed_seconds"] for row in rows)
    tokens = sum(row["tokens"] for row in rows)
    lr = rows[-1]["learning_rates"]
    print(f"  Max |loss − {beta:g} × KL|: {residual:.3g}")
    print(f"  Latest learning rates: {', '.join(f'{rate:.3g}' for rate in lr)}")
    if seconds:
        print(f"  Median update time: {median(row['elapsed_seconds'] for row in rows):.1f}s. "
              f"Rollout tokens per training-step second: {tokens / seconds:.1f}.")


def show_dev(grpo, sft):
    if not grpo:
        print("\nNo saved GRPO dev evaluations yet.")
        return
    print("\nFixed dev set (macro exact, matched saved updates)")
    print(f"{'Update':>6} {'GRPO':>9} {'SFT':>9}")
    for step, row in sorted(grpo.items()):
        other = f"{sft[step]['macro_exact']:.1%}" if step in sft else "—"
        print(f"{step:>6} {row['macro_exact']:>9.1%} {other:>9}")

    step = max(grpo)
    latest = grpo[step]
    baseline = grpo.get(0)
    print(f"\nGRPO at dev update {step} compared with shared P1 at update 0")
    for cell, result in sorted(latest["cells"].items()):
        old = baseline["cells"].get(cell) if baseline else None
        exact = f"{old['exact']:.1%} → " if old else ""
        valid = f"{old['valid']:.1%} → " if old else ""
        print(f"  {cell:22} n={result['groups']:<3} exact {exact}{result['exact']:.1%}, "
              f"valid {valid}{result['valid']:.1%}")

    if not baseline:
        return
    old = {row["id"]: row for row in baseline["predictions"]}
    new = {row["id"]: row for row in latest["predictions"]}
    if old.keys() != new.keys():
        print("  Output changes unavailable because the dev example IDs differ.")
        return
    changes = sum(old[key]["output"]["text"] != new[key]["output"]["text"] for key in old)
    gained = sum(not old[key]["exact"] and new[key]["exact"] for key in old)
    lost = sum(old[key]["exact"] and not new[key]["exact"] for key in old)
    print(f"  Changed answers: {changes}/{len(old)}. Became exact: {gained}. Lost exact: {lost}.")


def paired_dev(grpo_state, sft_dir):
    sft_state = read_json(sft_dir / "stage.json")
    if not sft_state:
        return {}
    keys = ("data_sha256", "parent_sha256", "code_identity", "hardware", "seed")
    settings = ("effective_batch_size", "checkpoint_every", "precision", "total_updates")
    if (any(grpo_state[key] != sft_state[key] for key in keys)
            or any(grpo_state["config"][key] != sft_state["config"][key] for key in settings)):
        print("SFT stage settings differ. Showing the GRPO dev curve alone.")
        return {}
    return dev_points(sft_dir, saved_step(sft_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path, help="e.g. artifacts/semi_pilot/semi-v1")
    parser.add_argument("--window", type=int, default=64, help="training updates per summary row")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")

    root = args.experiment_dir
    directory = root / "grpo"
    state = read_json(directory / "stage.json")
    if not state:
        parser.error(f"Missing GRPO stage file: {directory / 'stage.json'}")
    # Read the published checkpoint pointer first, then use only the corresponding ledger prefix.
    through = saved_step(directory)
    ledger = directory / "learning.jsonl"
    all_rows = read_ledger(ledger) if ledger.is_file() else []
    rows = [row for row in all_rows if row["step"] <= through]
    if [row["step"] for row in rows] != list(range(1, through + 1)):
        parser.error("The ledger has missing or duplicate updates through the saved checkpoint.")
    last = max((row["step"] for row in all_rows), default=0)
    print(f"GRPO {state['status']}. Saved update {through}, latest ledger update {last}.")
    show_gate(read_json(root / "warmup/gate.json"))
    show_training(rows, state["config"]["beta"], args.window)
    show_dev(dev_points(directory, through), paired_dev(state, root / "sft"))


if __name__ == "__main__":
    main()

