"""Native benchmark records, explicit scoring gates, and isolated inference worker.

No local heuristic is labelled an official scorer. Pass an audited scorer with
its pinned revision, or export predictions with an explicit blocked score.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

STATUSES = {"unverified", "smoke_only", "reproduced", "differs", "partial", "not_comparable", "blocked"}
SAMSONE_SOURCE = "https://github.com/SamsungLabs/samsone/blob/main/README.md"
MELLOW_SOURCE = "https://arxiv.org/abs/2503.08540"
BENCHMARKS = {
    "mmau_mini": {"url": "https://github.com/Sakshi113/MMAU", "split": "test-mini", "expected_n": 1000},
    "mmau_test": {"url": "https://github.com/Sakshi113/MMAU", "split": "test", "expected_n": None},
    "mmau_pro": {"url": "https://github.com/Sakshi113/MMAU", "split": "MMAU-Pro", "expected_n": None},
    "clothoaqa": {"url": "https://github.com/samuelg13/ClothoAQA", "split": "test", "expected_n": None},
    "audiocaps": {"url": "https://github.com/cdjkim/audiocaps", "split": "test", "expected_n": None},
    "clotho": {"url": "https://zenodo.org/records/4783391", "split": "evaluation", "expected_n": None},
    "reasonaqa": {"url": "https://zenodo.org/records/15036628", "split": "test", "expected_n": None},
}


@dataclass
class Claim:
    claim_id: str
    model_id: str
    source_url: str
    source_locator: str
    split: str
    metric: str
    claimed_value: float | None = None
    expected_n: int | None = None
    checkpoint_sha256: str | None = None
    benchmark_revision: str | None = None
    evaluated_n: int = 0
    missing_audio_n: int = 0
    prompt_hash: str | None = None
    scorer_revision: str | None = None
    observed_value: float | None = None
    status: str = "unverified"
    reason: str | None = None
    unit: str = "percent"
    category: str | None = None
    asset_url: str | None = None
    retry_command: str | None = None

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"Unknown reproduction status: {self.status}")


def claims_registry(source_root=None):
    """Read numeric Samsone claims from the actual local README, never memory.

    Mellow's row in that table does not identify v0 versus v0_s and therefore
    cannot substantiate a numeric claim for the requested v0 checkpoint.
    """
    rows = []
    if source_root is not None:
        source_root = Path(source_root)
        if (source_root / "samsone/README.md").is_file():
            source_root = source_root / "samsone"
        readme = source_root / "README.md"
        for line in readme.read_text().splitlines():
            if "| **Samsone-134M (ours)** |" not in line:
                continue
            cells = [v.strip().replace("**", "") for v in line.strip().strip("|").split("|")]
            columns = [("sound", "test-mini"), ("sound", "test"),
                       ("music", "test-mini"), ("music", "test"),
                       ("speech", "test-mini"), ("speech", "test"),
                       (None, "test-mini"), (None, "test"), (None, "MMAU-Pro")]
            if len(cells) != 11:
                raise ValueError("Upstream Samsone results table changed; inspect column mapping")
            for value, (category, split) in zip(cells[2:], columns):
                rows.append(Claim(
                    claim_id=f"samsone_134m_{split}_{category or 'overall'}_accuracy",
                    model_id="samsone_134m_published", source_url=SAMSONE_SOURCE,
                    source_locator="MMAU and MMAU-Pro results / Small Audio Language Models / Samsone-134M",
                    split=split, metric="accuracy", claimed_value=float(value), category=category,
                    expected_n=1000 if split == "test-mini" and category is None else None))
    rows.append(Claim("mellow_v0_source_checkpoint_association", "mellow_v0_published",
                      MELLOW_SOURCE, "Paper tables require pinned version and v0 checkpoint association",
                      "unresolved", "unresolved", status="not_comparable",
                      reason="Do not assign the Samsone README's unspecified Mellow row to v0 or substitute v0_s."))
    return rows


class NativeBenchmarkAdapter:
    """Adapt official records, retain exact prompts, and gate claimed scores.

    Records use id/prompt/audio/answer/category; ``adapt_mmau`` and
    ``adapt_reasonaqa`` handle their upstream field names. Other official task
    files can use ``adapt_record`` with an explicit field map, preserving raw.
    """
    def __init__(self, benchmark, *, revision, split=None, scorer=None, scorer_revision=None):
        if benchmark not in BENCHMARKS:
            raise ValueError(f"Unregistered benchmark: {benchmark}")
        if not revision:
            raise ValueError("An exact benchmark revision is required")
        self.benchmark = benchmark
        self.revision = revision
        self.split = split or BENCHMARKS[benchmark]["split"]
        self.scorer: Callable | None = scorer
        self.scorer_revision = scorer_revision

    @staticmethod
    def adapt_record(record, field_map):
        output = {key: record[source] for key, source in field_map.items() if source in record}
        output["raw"] = record
        if not output.get("id") or not output.get("prompt"):
            raise ValueError("Official record must provide stable id and native prompt")
        return output

    @staticmethod
    def adapt_mmau(record):
        # MMAU question/choices are kept separately: no unverified MCQ prompt
        # template or answer-letter heuristic is silently substituted.
        return {"id": str(record["id"]), "audio": record["audio_id"],
                "question": record["question"], "choices": record["choices"],
                "answer": record.get("answer"), "category": record.get("task"), "raw": record}

    @staticmethod
    def adapt_reasonaqa(record, *, record_id):
        return {"id": str(record_id), "audio1": record["filepath1"],
                "audio2": record.get("filepath2", ""), "prompt": record["input"],
                "answer": record.get("answer"), "category": record.get("subtype"), "raw": record}

    def evaluate(self, records, predict, claim, *, smoke_only=False,
                 protocol_verified=False, tolerance=2.0):
        """Scorer signature: scorer(prediction_records)->percent or metric dict.

        Each prediction preserves generated IDs when the policy supplies them.
        Hidden labels block scoring, including accidental partial-label sets.
        """
        records = list(records)
        ids = [str(item["id"]) for item in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate native benchmark IDs")
        predictions, missing = [], 0
        for item in records:
            try:
                result = predict(item)
                if isinstance(result, str):
                    result = {"text": result, "ids": None}
                if "text" not in result:
                    raise ValueError("Native prediction must contain text")
                predictions.append({"id": str(item["id"]), **result,
                    "answer": item.get("answer"), "category": item.get("category"),
                    "native_prompt": item.get("prompt"), "status": "predicted"})
            except FileNotFoundError as exc:
                missing += 1
                predictions.append({"id": str(item["id"]), "status": "missing_audio", "reason": str(exc)})
        selected = [p for p in predictions if p["status"] == "predicted"
                    and (claim.category is None or p.get("category") == claim.category)]
        result = replace(claim, evaluated_n=len(selected), missing_audio_n=missing,
                         benchmark_revision=self.revision, scorer_revision=self.scorer_revision,
                         observed_value=None)
        if claim.status == "not_comparable":
            return result, predictions
        reason = None
        if not selected:
            reason = "No evaluable native predictions"
        elif self.scorer is None or not self.scorer_revision:
            reason = "Pinned official scorer unavailable; predictions exported without a score"
        elif any(p["answer"] is None for p in selected):
            reason = "Official labels unavailable; export predictions for official scoring"
        if reason:
            result.status, result.reason = "blocked", reason
            result.asset_url = BENCHMARKS[self.benchmark]["url"]
            result.retry_command = f"nano-audio reproduce --models {claim.model_id} --profile main"
            return result, predictions
        measured = self.scorer(selected)
        value = measured[claim.metric] if isinstance(measured, dict) else measured
        import math
        result.observed_value = float(value)
        if not math.isfinite(result.observed_value):
            raise ValueError("Official scorer returned a nonfinite result")
        if smoke_only:
            result.status = "smoke_only"
        elif missing or result.expected_n is None or len(selected) != result.expected_n:
            result.status = "partial"
            result.reason = "Full expected coverage not established"
        elif (not protocol_verified or not result.checkpoint_sha256 or not result.prompt_hash
              or result.claimed_value is None or result.split != self.split):
            result.status = "not_comparable"
            result.reason = "Checkpoint, prompt, native protocol, split, or numeric claim not verified"
        else:
            result.status = "reproduced" if abs(result.observed_value - result.claimed_value) <= tolerance else "differs"
        return result, predictions


def write_claims(path, claims):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as handle:
        for claim in claims:
            handle.write(json.dumps(asdict(claim), allow_nan=False) + "\n")


def run_native_worker(request, *, python=sys.executable, timeout=300):
    """Subprocess isolation; training must use a differentiable in-process policy."""
    process = subprocess.run([str(python), "-m", "nano_audio.evaluation.native", "--worker"],
        input=json.dumps(request), text=True, capture_output=True, timeout=timeout)
    if process.returncode:
        raise RuntimeError(f"Native worker failed ({process.returncode}): {process.stderr[-4000:]}")
    # Upstream prints progress; worker writes its JSON on the last line.
    return json.loads(process.stdout.strip().splitlines()[-1])


def worker(request):
    model_id = request["model_id"]
    if model_id == "mellow_v0_published":
        from nano_audio.models.mellow import MellowPolicy
        policy = MellowPolicy.from_pretrained(
            vendor_root=request.get("vendor_root"), device=request.get("device", "cpu"),
            checkpoint_path=request.get("checkpoint_path"),
            asset_root=request.get("asset_root"), manifest_path=request.get("manifest_path"),
            second_audio_pool=request.get("second_audio_pool", ()), seed=request.get("seed", 17))
        return {"text": policy.native_generate(request["example"], **request.get("generation", {})),
                "ids": None, "ids_reason": "Official native wrapper returns decoded strings only",
                "model_id": model_id}
    if model_id == "samsone_134m_published":
        root = request.get("vendor_root")
        if root:
            sys.path.insert(0, str(Path(root) / "src"))
        from samsone import Samsone134M
        import torch
        dtype = getattr(torch, request.get("dtype", "float32"))
        model = Samsone134M(device=request.get("device", "cpu"), dtype=dtype,
                           checkpoint_path=request.get("checkpoint_path"))
        example = request["example"]
        return {"text": model(audio=example["audio"], prompt=example["prompt"],
                              **request.get("generation", {})), "ids": None,
                "ids_reason": "Official native wrapper returns decoded strings only", "model_id": model_id}
    raise ValueError(f"No official native worker for {model_id}")


def mellow_native_smoke(*, asset_root=None, device="auto", max_len=64):
    """One real native example; write exact result/failure with provenance."""
    import hashlib
    import importlib.metadata as metadata
    import time
    import traceback
    import random
    import torch
    from nano_audio.models.mellow import MellowPolicy
    root = Path(__file__).resolve().parents[3]
    asset_root = Path(asset_root or root / "artifacts/assets")
    output = root / "artifacts/reproduction/mellow_native.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [root / "vendor/mellow/resource/1.wav", root / "vendor/mellow/resource/2.wav"]
    packages = {}
    for name in ("torch", "torchaudio", "transformers", "importlib_resources", "torchlibrosa",
                 "librosa", "numba", "llvmlite", "numpy", "scipy", "soundfile"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result = {"status": "running", "profile": "smoke_only", "model_id": "mellow_v0_published",
              "source_manifest": str(asset_root / "manifest.json"), "seed": 17,
              "prompt": "caption the audio.", "generation": {"max_len": max_len, "top_p": 0.8, "temperature": 1.0},
              "packages": packages, "benchmark_score": None, "native_ids": None,
              "native_ids_reason": "Official wrapper returns strings only",
              "retry_command": "python -m nano_audio.evaluation.native --mellow-smoke --device " + device}
    output.write_text(json.dumps(result, indent=2) + "\n")
    started = time.time()
    try:
        manifest = json.loads((asset_root / "manifest.json").read_text())
        result["checkpoint_revision"] = manifest["models"]["mellow_v0_published"]["revision"]
        result["source_revision"] = manifest["sources"]["mellow"]["revision"]
        result["audio"] = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
        random.seed(17)
        torch.manual_seed(17)
        torch.set_num_threads(4)
        policy = MellowPolicy.from_pretrained(asset_root=asset_root, device=device, second_audio_pool=[str(paths[1])])
        result["device"] = str(policy.device)
        result["assets"] = policy.assets
        result["actual_output"] = policy.native_generate(
            {"audio1": str(paths[0]), "audio2": str(paths[1]), "prompt": result["prompt"]}, **result["generation"])
        result["parameter_manifest"] = policy.parameter_manifest()
        result["status"] = "smoke_only"
    except Exception as exc:
        result.update(status="blocked", exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
    finally:
        result["elapsed_seconds"] = time.time() - started
        result["torch_version_after"] = metadata.version("torch")
        output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mellow-smoke", action="store_true")
    parser.add_argument("--asset-root")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.mellow_smoke:
        result = mellow_native_smoke(asset_root=args.asset_root, device=args.device)
        print(json.dumps({key: result.get(key) for key in ("status", "actual_output", "exception_type", "exception", "elapsed_seconds")}))
        return
    if not args.worker:
        parser.error("Use --worker with a JSON request on stdin")
    print(json.dumps(worker(json.load(sys.stdin))))


if __name__ == "__main__":
    main()
