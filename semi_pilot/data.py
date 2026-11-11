"""Prepare a small, versioned SLURP + constructed speech dataset."""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nano_audio.data.slurp import annotation_entities, parse_quantity
from nano_audio.data.synthetic import (
    audit_splits, composition_reserve, exclude_cross_partition_duplicates,
    expand_groups, generate_groups, normalized_text, tuples_for,
)
from nano_audio.data.task import SchemaFamily, TaskError, schema_families
from nano_audio.utils import digest, sha, write_json

SPLITS = {"train": "train", "dev": "devel", "test": "test"}
MAPPING_VERSION = "explicit-room-and-absolute-value-v1"
LIGHT_INTENTS = {
    "iot_hue_lighton": "on", "iot_hue_lightoff": "off",
    "iot_hue_lightdim": None, "iot_hue_lightup": None,
}
VOLUME_INTENTS = {"audio_volume_up", "audio_volume_down", "audio_volume_other"}
ROOM_LIGHTS = {
    "kitchen": "kitchen_light", "bedroom": "bedroom_light", "bed room": "bedroom_light",
    "my bedroom": "bedroom_light", "my bed room": "bedroom_light",
    "the kitchen": "kitchen_light", "the bedroom": "bedroom_light",
}
EXPLICIT_LIGHTS = {
    "kitchen light": "kitchen_light", "kitchen lights": "kitchen_light",
    "bedroom light": "bedroom_light", "bedroom lights": "bedroom_light",
    "bed room light": "bedroom_light", "bed room lights": "bedroom_light",
}
GENERIC_LIGHTS = {"light", "lights", "lamp", "lamps", "hue lights", "hue light"}
AUDIO_DEVICES = {"speaker": "speaker", "speakers": "speaker", "radio": "radio"}
UNSUPPORTED_LANGUAGE = re.compile(
    r"\b(?:all|every|except|unless|then|until|after|before|tomorrow|only|not|never|instead|if|when)\b"
    r"|\band\b|\bno\b"
)
CANONICAL = SchemaFamily("canonical", renamed=False)


def experiment_dir(name):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
        raise ValueError("Experiment name must use letters, numbers, hyphens or underscores.")
    return ROOT / "artifacts/semi_pilot" / name


def map_annotation(row):
    intent = row.get("intent")
    if intent not in LIGHT_INTENTS and intent not in VOLUME_INTENTS:
        return None, "unsupported_intent"
    if UNSUPPORTED_LANGUAGE.search(normalized_text(row["sentence"])):
        return None, "unsupported_qualifier_or_compound_request"
    try:
        entities = defaultdict(list)
        for kind, text in annotation_entities(row):
            entities[kind].append(normalized_text(text))
        allowed = {"house_place", "device_type", "change_amount"} if intent in LIGHT_INTENTS else {
            "device_type", "change_amount",
        }
        if entities.keys() - allowed or any(len(spans) != 1 for spans in entities.values()):
            return None, "unsupported_or_repeated_entity"

        room = next(iter(entities["house_place"]), None)
        device_text = next(iter(entities["device_type"]), None)
        if intent in LIGHT_INTENTS:
            room_device = ROOM_LIGHTS.get(room)
            device = EXPLICIT_LIGHTS.get(device_text) if device_text else room_device
            if device_text in GENERIC_LIGHTS:
                device = room_device
            if (room and room_device != device) or not device:
                return None, "missing_ambiguous_or_unsupported_light"
            state = LIGHT_INTENTS[intent]
            if state is not None:
                if entities["change_amount"]:
                    return None, "unexpected_power_quantity"
                operation, arguments = "set_power", {"device": device, "state": state}
            else:
                operation, arguments = "set_brightness", {"device": device}
        else:
            device = AUDIO_DEVICES.get(device_text)
            if not device:
                return None, "missing_or_unsupported_audio_device"
            operation, arguments = "set_volume", {"device": device}

        if operation in ("set_brightness", "set_volume"):
            amount = next(iter(entities["change_amount"]), "")
            if not amount.startswith("to ") or not amount.endswith(" percent"):
                return None, "missing_explicit_absolute_percent"
            arguments["percent"] = parse_quantity(amount.removeprefix("to "), "percent")
        program = [{"operation": operation, "arguments": arguments}]
        CANONICAL.encode(program)
        return program, None
    except (TaskError, TypeError, ValueError, OverflowError):
        return None, "invalid_annotation_or_quantity"


def read_candidates(annotation_dir, seed):
    paths = {split: annotation_dir / f"{stem}.jsonl" for split, stem in SPLITS.items()}
    forbidden = set(composition_reserve(seed))
    candidates, exclusions = [], {split: Counter() for split in SPLITS}
    seen = {split: Counter() for split in SPLITS}
    for split, path in paths.items():
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                seen[split]["annotation_rows"] += 1
                program, reason = map_annotation(row)
                if reason is None and split != "test" and forbidden.intersection(tuples_for(program)):
                    reason = "reserved_composition"
                if reason:
                    exclusions[split][reason] += 1
                    continue
                recordings = sorted({
                    item["file"] for item in row.get("recordings", [])
                    if item.get("status") == "correct" and item.get("file")
                })
                if not recordings:
                    exclusions[split]["no_recording_labelled_correct"] += 1
                    continue
                group_id = "slurp_" + str(row["slurp_id"])
                candidates.append({
                    "utterance_group_id": group_id, "source_utterance_id": str(row["slurp_id"]),
                    "split": split, "source": "slurp", "task_kind": "single",
                    "transcript": row["sentence"], "canonical_target": program, "order_matters": True,
                    "slurp_intent": row["intent"], "recording_names": recordings,
                })
    # Remove whole groups if their transcript, source ID or recording spans partitions.
    text_owners, file_owners, id_owners = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in candidates:
        split = row["split"]
        text_owners[normalized_text(row["transcript"])].add(split)
        id_owners[row["utterance_group_id"]].add(split)
        for filename in row["recording_names"]:
            file_owners[filename].add(split)
    kept, distinct = [], set()
    for row in candidates:
        split, group_id = row["split"], row["utterance_group_id"]
        if (len(text_owners[normalized_text(row["transcript"])]) > 1 or len(id_owners[group_id]) > 1
                or any(len(file_owners[filename]) > 1 for filename in row["recording_names"])):
            exclusions[split]["cross_partition_duplicate"] += 1
            continue
        text_key = (split, normalized_text(row["transcript"]))
        if text_key in distinct:
            exclusions[split]["repeated_transcript_in_partition"] += 1
            continue
        distinct.add(text_key)
        seen[split][row["canonical_target"][0]["operation"]] += 1
        kept.append(row)
    return kept, {
        "mapping_version": MAPPING_VERSION,
        "annotation_sha256": {split: sha(path) for split, path in paths.items()},
        "eligible_groups": dict(Counter(row["split"] for row in kept)),
        "counts": seen, "excluded_annotations": exclusions,
    }


def select_real(candidates, audio_root, output, counts, revision, seed, audit):
    import soundfile as sf
    from nano_audio.data.synthesis import audio_quality

    output.mkdir(parents=True)
    selected = []
    audio_exclusions = {split: Counter() for split in SPLITS}
    rng = random.Random(seed)
    for split, quota in counts.items():
        pool = [row for row in candidates if row["split"] == split]
        rng.shuffle(pool)
        # Alternate operations where supported while retaining deterministic selection.
        by_label = defaultdict(list)
        for row in pool:
            call = row["canonical_target"][0]
            key = (call["operation"], call["arguments"].get("state", ""))
            by_label[key].append(row)
        pool = [row for batch in zip_longest(*(by_label[key] for key in sorted(by_label)))
                for row in batch if row is not None]
        accepted = 0
        accepted_ids = set()
        for row in pool:
            if row["utterance_group_id"] in accepted_ids:
                continue
            for name in row["recording_names"]:
                original = audio_root / Path(name).name
                if not original.is_file():
                    audio_exclusions[split]["missing_recording"] += 1
                    continue
                try:
                    samples, rate = sf.read(original, dtype="float32", always_2d=True)
                    mono = samples.mean(axis=1)
                    audio_quality(mono, rate)
                except (RuntimeError, TaskError):
                    audio_exclusions[split]["invalid_or_overlong_audio"] += 1
                    continue
                recording_id = digest([row["utterance_group_id"], name])[:20]
                path = output / f"{recording_id}.wav"
                sf.write(path, mono, rate, subtype="PCM_16")
                quality = audio_quality(sf.read(path, dtype="float32")[0], rate)
                selected.append({
                    **{k: v for k, v in row.items() if k != "recording_names"},
                    "recording_id": recording_id, "audio_path": str(path),
                    "audio_sha256": sha(path), "sample_rate": rate, **quality,
                    "provenance": {
                        "label_source": "human_annotations", "dataset_revision": revision,
                        "mapping_version": MAPPING_VERSION, "source_audio": name,
                        "source_audio_sha256": sha(original),
                    },
                })
                accepted_ids.add(row["utterance_group_id"])
                accepted += 1
                break
            if accepted == quota:
                break
        if accepted != quota:
            raise ValueError(f"SLURP {split}: selected {accepted}/{quota} groups. "
                             f"Recording exclusions: {dict(audio_exclusions[split])}. "
                             "Check the extracted slurp_real directory or explicitly lower the quota.")
    audit["recording_exclusions_until_quotas_met"] = audio_exclusions
    return selected


def render_example(row, family):
    return {
        **row, "schema_family_id": family.family_id, "schema_partition": family.partition,
        "tools": family.tools, "target": family.encode(row["canonical_target"]),
        "id": digest([row["recording_id"], family.family_id])[:20],
    }


def real_examples(rows, families):
    for row in rows:
        for family in families:
            if family.partition != row["split"] and not (row["split"] != "train" and family.partition == "train"):
                continue
            yield render_example(row, family)


def prepare(args):
    from nano_audio.data.pipeline import acquire_kokoro, _ensure_g2p
    from nano_audio.data.synthesis import synthesize_groups

    root = experiment_dir(args.experiment)
    frozen = root / "data"
    if frozen.exists():
        raise FileExistsError(f"{frozen} is frozen. Use a new experiment name to change the dataset.")
    candidates, coverage = read_candidates(args.slurp_repo / "dataset/slurp", args.seed)
    revision = subprocess.check_output(
        ["git", "-C", str(args.slurp_repo), "rev-parse", "HEAD"], text=True,
    ).strip()
    real_counts = dict(zip(SPLITS, args.real))
    synth_counts = dict(zip(SPLITS, args.synthetic))
    print(json.dumps(coverage, indent=2), flush=True)
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=root) as temporary:
        stage = Path(temporary) / "data"
        real = select_real(candidates, args.slurp_audio.resolve(), stage / "slurp", real_counts, revision, args.seed, coverage)
        metadata_path = args.slurp_repo / "dataset/slurp/metadata.json"
        speakers = defaultdict(set)
        unknown_speakers = 0
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        for row in real:
            filename = row["provenance"]["source_audio"]
            recording = metadata.get(row["source_utterance_id"], {}).get("recordings", {}).get(filename, {})
            speaker = recording.get("usrid")
            if speaker:
                row["speaker_id"] = speaker
                speakers[speaker].add(row["split"])
            else:
                unknown_speakers += 1
        groups = generate_groups(seed=args.seed, counts=synth_counts)
        if args.reuse_tts_assets:
            assets = args.reuse_tts_assets.resolve()
        else:
            assets = ROOT / "artifacts/data/local_pilot/tts_assets"
        revision_tts, voices = acquire_kokoro(assets)
        _ensure_g2p(assets / "python")
        synthetic = synthesize_groups(groups, assets, stage / "synthetic", voices, revision_tts,
                                      device=args.device, seed=args.seed, speeds=(1.0, 1.15, 1.3, 1.5))
        families = schema_families()
        familiar = next(family for family in families if family.family_id == "train_01")
        synthetic_dev_familiar = [render_example(row, familiar) for row in synthetic if row["split"] == "dev"]
        rows = [*real_examples(real, families), *expand_groups(synthetic, families, voices), *synthetic_dev_familiar]
        rows, dropped = exclude_cross_partition_duplicates(rows)
        stats = defaultdict(set)
        for row in rows:
            stats[(row["source"], row["split"])].add(row["utterance_group_id"])
            row["audio_path"] = Path(row["audio_path"]).relative_to(stage).as_posix()
        for source, quotas in (("slurp", real_counts), ("synthetic", synth_counts)):
            for split, count in quotas.items():
                if len(stats[(source, split)]) != count:
                    raise ValueError(f"{source} {split} fell below quota after duplicate removal. "
                                     f"Dropped groups: {sorted({r['utterance_group_id'] for r in dropped})}")
        split_audit = audit_splits(rows, composition_reserve(args.seed))
        examples = stage / "examples.jsonl"
        with examples.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = {
            "format": 1, "data_seed": args.seed, "sources": ["slurp", "synthetic"],
            "counts": {source: {split: len(stats[(source, split)]) for split in SPLITS}
                       for source in ("slurp", "synthetic")},
            "source_mixture_per_update": {"slurp": 0.5, "synthetic": 0.5},
            "examples_sha256": sha(examples), "slurp_revision": revision,
            "kokoro_revision": revision_tts, "slurp_coverage": coverage,
            "split_audit": split_audit,
            "slurp_speakers": {"known": len(speakers), "unknown_recordings": unknown_speakers,
                               "cross_split": sum(len(splits) > 1 for splits in speakers.values())},
        }
        write_json(stage / "manifest.json", manifest)
        write_json(stage / "audit.json", split_audit)
        # The project synthesis cache includes the temporary absolute filenames.
        (stage / "synthetic/synthesis_manifest.json").unlink(missing_ok=True)
        stage.rename(frozen)
    print(f"Frozen data: {frozen}\nManifest SHA-256: {sha(frozen / 'manifest.json')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("coverage", "prepare"):
        child = sub.add_parser(name)
        child.add_argument("--slurp-repo", type=Path, required=True, help="Official SLURP Git checkout")
        child.add_argument("--seed", type=int, default=1701, help="Data seed")
        if name == "prepare":
            child.add_argument("--experiment", required=True)
            child.add_argument("--slurp-audio", type=Path, required=True, help="Extracted slurp_real directory")
            child.add_argument("--real", nargs=3, type=int, metavar=("TRAIN", "DEV", "TEST"), default=(12, 3, 4))
            child.add_argument("--synthetic", nargs=3, type=int, metavar=("TRAIN", "DEV", "TEST"), default=(64, 12, 12))
            child.add_argument("--device", default="cpu")
            child.add_argument("--reuse-tts-assets", type=Path)
    args = parser.parse_args()
    if args.command == "coverage":
        _, coverage = read_candidates(args.slurp_repo / "dataset/slurp", args.seed)
        print(json.dumps(coverage, indent=2))
    else:
        if min(*args.real, *args.synthetic) < 1:
            parser.error("All source/split quotas must be positive.")
        prepare(args)


if __name__ == "__main__":
    main()
