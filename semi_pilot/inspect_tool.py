"""Print the exact Samsone evaluation prompts and tool schemas for one recording."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CATALOGS = {
    "dev": {"train_01": "familiar", "dev_01": "schema_ood"},
    "test": {
        "train_01": "familiar",
        "names_01": "held_out_names",
        "structure_01": "held_out_structure",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path, help="e.g. artifacts/semi_pilot/semi-v1")
    parser.add_argument("--split", choices=CATALOGS, default="test")
    parser.add_argument("--example-id", help="ID from an evaluation snapshot or test predictions")
    args = parser.parse_args()

    from run import representative
    from nano_audio.models.policy import SamsonePolicy

    path = args.experiment_dir / "data/examples.jsonl"
    if not path.is_file():
        parser.error(f"Frozen examples not found at {path}")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    cases = representative([row for row in rows if row["split"] == args.split], CATALOGS[args.split])
    if not cases:
        parser.error(f"No paired {args.split} recordings in {path}")
    if args.example_id:
        chosen = next((row for _, row in cases if row["id"] == args.example_id), None)
        if chosen is None:
            parser.error(f"Example {args.example_id!r} is not among the selected {args.split} cases")
    else:
        chosen = cases[0][1]

    # The serializer uses the input record only. Construction would load the model.
    policy = SamsonePolicy.__new__(SamsonePolicy)
    output = []
    for catalog, row in cases:
        if row["utterance_group_id"] != chosen["utterance_group_id"]:
            continue
        item = {"audio_path": row["audio_path"], "tools": row["tools"]}
        output.append({
            "catalog": catalog,
            "schema_family_id": row["schema_family_id"],
            "example_id": row["id"],
            "audio_path": str(args.experiment_dir / "data" / row["audio_path"]),
            "transcript_for_review": row["transcript"],
            "expected_answer_for_review": row["target"],
            "tools": item["tools"],
            "prompt": policy.prompt(item),
        })
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

