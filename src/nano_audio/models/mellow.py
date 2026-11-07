"""Differentiable adapter for the released Mellow v0 architecture.

Native inference remains available through ``native_generate``. Research prompts
are unpadded and untruncated; this deviation is deliberately separate from native.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

UPSTREAM = "https://github.com/soham97/mellow"
ASSETS = {
    "v0": "https://huggingface.co/soham97/mellow/resolve/main/v0.ckpt",
    "v0_s": "https://huggingface.co/soham97/mellow/resolve/main/v0_s.ckpt",
}
MISSING_AUDIO_SOURCE = UPSTREAM + "/blob/training/data/audiotext_eval_dataset.py"


def resolve_second_audio(first, second=None, *, candidates=(), seed=17):
    """Official dataset convention: random valid filepath1, NOT duplicated audio.

    Sorting removes upstream set-order nondeterminism. The chosen path must be
    recorded by the caller in its example manifest. No implicit fallback pool.
    """
    if second:
        return str(second)
    pool = sorted(set(str(p) for p in candidates if p))
    if not pool:
        raise ValueError("Mellow missing audio2 requires a declared distractor pool; " + MISSING_AUDIO_SOURCE)
    return random.Random(seed).choice(pool)


class MellowPolicy:
    model_id = "mellow_v0_published"

    def __init__(self, wrapper=None, *, device="auto", asset_root=None, vendor_root=None,
                 checkpoint_path=None, second_audio_pool=(), seed=17):
        if wrapper is None:
            loaded = type(self).from_pretrained(vendor_root=vendor_root, device=device,
                asset_root=asset_root, checkpoint_path=checkpoint_path,
                second_audio_pool=second_audio_pool, seed=seed)
            self.__dict__.update(loaded.__dict__)
            return
        self.wrapper = wrapper
        self.model: nn.Module = wrapper.model
        self.decoder = self.model.caption_decoder.lm
        self.tokenizer = wrapper.tokenizer
        self.second_audio_pool = tuple(second_audio_pool)
        self.seed = seed
        self.research_precision = "bf16" if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        self._feature_cache = {}
        self.model.audio_encoder.base.requires_grad_(False).eval()
        self.model.audio_encoder.projection.requires_grad_(True)
        self.decoder.requires_grad_(True)

    @classmethod
    def from_pretrained(cls, *, vendor_root=None, device="auto", checkpoint_path=None,
                        asset_root=None, manifest_path=None, second_audio_pool=(), seed=17):
        """Load the exact v0 architecture from manifest-pinned assets.

        Download metadata/hashes are written only to reproduction/mellow_assets.json.
        The shared manifest is read-only. Component loading uses local snapshots,
        bypassing the upstream wrapper's unpinned and unused config.json download.
        """
        project = Path(__file__).resolve().parents[3]
        root = Path(vendor_root or project / "vendor/mellow")
        source_manifest = Path(manifest_path or Path(asset_root or project / "artifacts/assets") / "manifest.json")
        manifest = json.loads(source_manifest.read_text())
        checkpoint_info = dict(manifest["models"][cls.model_id])
        component_info = dict(manifest["models"]["smollm2_135m_base"])
        if not checkpoint_info.get("revision") or not component_info.get("revision"):
            raise ValueError("Mellow and SmolLM2 require explicit manifest revisions")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from mellow.wrapper import MellowWrapper
        except ImportError as exc:
            raise RuntimeError("Mellow runtime unavailable: " + repr(exc) + "; requirements: "
                               + str(root / "requirements.txt")) from exc
        from huggingface_hub import hf_hub_download, snapshot_download
        if checkpoint_path is None:
            checkpoint_path = checkpoint_info.get("local_path")
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download(checkpoint_info["repo_id"], "v0.ckpt",
                                              revision=checkpoint_info["revision"])
        checkpoint_path = Path(checkpoint_path)
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        checkpoint_hash = digest.hexdigest()
        expected_hash = checkpoint_info.get("sha256")
        if isinstance(expected_hash, dict):
            expected_hash = expected_hash.get("v0.ckpt")
        if expected_hash and checkpoint_hash != expected_hash:
            raise ValueError("Mellow checkpoint SHA-256 differs from manifest")
        checkpoint_info.update(local_path=str(checkpoint_path), sha256=checkpoint_hash)
        component_path = component_info.get("local_path")
        if component_path is None:
            component_path = snapshot_download(component_info["repo_id"], revision=component_info["revision"],
                allow_patterns=["*.json", "*.safetensors", "*.txt"])
        component_path = Path(component_path)
        component_hashes = component_info.get("sha256", {})
        for filename, expected in component_hashes.items():
            path = component_path / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("SmolLM2 component hash mismatch: " + filename)
        if not component_hashes:
            component_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in component_path.iterdir() if path.is_file()}
        component_info.update(local_path=str(component_path), sha256=component_hashes)
        import subprocess
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        expected_source = manifest.get("sources", {}).get("mellow", {}).get("revision")
        if expected_source and revision != expected_source:
            raise ValueError("Mellow vendor checkout differs from manifest revision")
        assets = {"source_manifest": str(source_manifest), "sources": {"mellow": {"revision": revision, "url": UPSTREAM}},
                  "models": {cls.model_id: checkpoint_info, "smollm2_135m_base": component_info}}
        output = project / "artifacts/reproduction/mellow_assets.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(assets, indent=2) + "\n")
        target = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError("Native Mellow supports CPU or CUDA; no implicit device substitution")
        from mellow.model.model import get_model_class
        wrapper = MellowWrapper.__new__(MellowWrapper)
        wrapper.parent_path = root / "mellow"
        wrapper.config_path = str(wrapper.parent_path / "config/v0.yaml")
        wrapper.args = wrapper.read_config_as_args(wrapper.config_path)
        args = wrapper.args
        # Native DecoderModel recognizes SmolLM2 by substring in its path.
        if "smollm2" not in str(component_path).lower():
            raise ValueError("Pinned component path must identify SmolLM2 for upstream decoder dispatch")
        args.model["decoder"]["text_decoder"] = str(component_path)
        model_cls = get_model_class(args.model["model_type"])
        wrapper.model = model_cls(
            audioenc_name=args.model["encoder"]["audioenc_name"],
            d_in=args.model["encoder"]["out_emb"],
            text_decoder=args.model["decoder"]["text_decoder"],
            prefix_length=args.model["decoder"]["prefix_length"],
            d_out=args.model["encoder"]["d_proj"])
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state and all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        wrapper.model.load_state_dict(state, strict=True)
        from transformers import AutoTokenizer
        wrapper.tokenizer = AutoTokenizer.from_pretrained(component_path, local_files_only=True)
        wrapper.tokenizer.add_special_tokens({"pad_token": "!"})
        # Compatibility only: retain the original wrapper's exact max-length padding.
        # Modern Transformers removed the old pad_to_max_length keyword.
        original_encode_plus = wrapper.tokenizer.encode_plus
        def legacy_encode_plus(*args, **kwargs):
            legacy = kwargs.pop("pad_to_max_length", False)
            if legacy: kwargs.setdefault("padding", "max_length")
            return original_encode_plus(*args, **kwargs)
        wrapper.tokenizer.encode_plus = legacy_encode_plus
        assets["runtime_compatibility"] = {"tokenizer": "pad_to_max_length=True translated to padding=max_length; native max_length retained", "setuptools": "80.9.0 supplies pkg_resources needed by pinned librosa"}
        output.write_text(json.dumps(assets, indent=2) + "\n")
        wrapper.use_cuda = target.type == "cuda"
        wrapper.device = (target.index or 0) if wrapper.use_cuda else "cpu"
        wrapper.model.to(target).eval()
        policy = cls(wrapper, second_audio_pool=second_audio_pool, seed=seed)
        policy.assets = assets
        return policy

    @property
    def device(self):
        return next(self.model.parameters()).device

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().reshape(-1).tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def audio_paths(self, example):
        audio = example.get("audio", example.get("audio_path", example.get("filepath1")))
        if isinstance(audio, (list, tuple)):
            if not 1 <= len(audio) <= 2:
                raise ValueError("Mellow expects one or two clips")
            first, second = audio[0], audio[1] if len(audio) == 2 else None
        else:
            first, second = audio, None
        first = example.get("audio1", first)
        second = example.get("audio2", example.get("filepath2", second))
        if not first:
            raise ValueError("Mellow requires audio1")
        second = resolve_second_audio(first, second, candidates=self.second_audio_pool,
                                      seed=example.get("audio2_seed", self.seed))
        return str(first), second

    def encode_audio(self, example):
        """Return detached frozen outputs BEFORE trainable projection, per slot.

        Upstream wrapper repeats short clips and randomly crops long clips. RNG
        is isolated and seeded so cache keys can record this crop realization.
        """
        paths = self.audio_paths(example)
        rng = random.getstate()
        random.seed(example.get("audio_seed", self.seed))
        try:
            self.model.audio_encoder.base.eval()
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
                return tuple(self.model.audio_encoder.base(
                    self.wrapper.preprocess_audio([path], resample=True).squeeze(1)
                )["embedding"].detach() for path in paths)
        finally:
            random.setstate(rng)

    def prompt(self, example):
        if "_prompt" in example:
            return example["_prompt"]
        if "tools" in example:
            return (example.get("instruction", "return the requested calls as json using the supplied tools.")
                    + " tools:" + json.dumps(example["tools"], separators=(",", ":"), ensure_ascii=True)
                    + " answer: ")
        return example.get("prompt", example.get("input", ""))

    def cached_audio(self, example):
        paths = self.audio_paths(example)
        hashes = []
        for path in paths:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            hashes.append(digest.hexdigest())
        # Cache belongs to this exact frozen base instance; rebuilding/loading
        # replaces it, so features never cross encoder weight provenance.
        key = (tuple(hashes), example.get("audio_seed", self.seed),
               "upstream-repeat-randomcrop-32k-10s-frozen-encoder-fp32", str(next(self.model.parameters()).dtype))
        if key not in self._feature_cache:
            self._feature_cache[key] = self.encode_audio(example)
        return self._feature_cache[key]

    def build_prefix(self, example):
        prompt = self.prompt(example)
        if example.get("modality") == "text":
            prompt = "transcript: " + example["transcript"] + " " + prompt
            tokens = torch.tensor([self.encode(prompt)], device=self.device, dtype=torch.long)
            return self.decoder.get_input_embeddings()(tokens)
        features = example.get("audio_features")
        if features is None:
            features = self.cached_audio(example)
        if len(features) != 2:
            raise ValueError("Mellow requires frozen features for both audio slots")
        dtype = next(self.model.audio_encoder.projection.parameters()).dtype
        projected = [self.model.audio_encoder.projection(x.to(device=self.device, dtype=dtype))
                     for x in features]
        ids = torch.tensor([self.encode(prompt)], device=self.device, dtype=torch.long)
        return self.model.caption_decoder.generate_prefix_inference(
            *projected, {"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.research_precision == "bf16")

    def completion_logprobs(self, example, ids, mask=None):
        with self._autocast():
            return self._completion_logprobs_impl(example, ids, mask)

    def _completion_logprobs_impl(self, example, ids, mask=None):
        tokens = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(1, -1)
        if tokens.shape[1] == 0:
            return self.decoder.get_input_embeddings().weight.sum().reshape(1, 1)[:, :0]
        prefix = self.build_prefix(example)
        embeddings = torch.cat((prefix, self.decoder.get_input_embeddings()(tokens)), dim=1)
        self._check_context(embeddings.shape[1])
        logits = self.decoder(inputs_embeds=embeddings,
                              attention_mask=torch.ones(embeddings.shape[:2], device=self.device,
                                                        dtype=torch.long)).logits
        start = prefix.shape[1] - 1
        result = logits[:, start:start + tokens.shape[1]].float().log_softmax(-1).gather(
            -1, tokens.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            weights = torch.as_tensor(mask, device=self.device).reshape_as(result)
            result = result * weights
        return result

    def _check_context(self, length):
        limit = min(2048, getattr(self.decoder.config, "max_position_embeddings", 2048))
        if limit and length > limit:
            raise ValueError(f"Mellow context {length} exceeds {limit}; catalog truncation forbidden")

    def generate(self, example, config=None):
        config = config or {}
        get = config.get if isinstance(config, Mapping) else lambda key, default=None: getattr(config, key, default)
        limit = int(get("max_new_tokens", 100))
        if limit < 1:
            raise ValueError("max_new_tokens must be positive")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self._autocast():
                prefix = self.build_prefix(example)
                self._check_context(prefix.shape[1] + limit)
                kwargs = dict(max_new_tokens=limit, do_sample=bool(get("do_sample", False)),
                              eos_token_id=self.tokenizer.eos_token_id,
                              pad_token_id=self.tokenizer.pad_token_id)
                if kwargs["do_sample"]:
                    kwargs.update(temperature=float(get("temperature", 1.0)), top_p=float(get("top_p", 1.0)), top_k=int(get("top_k", 0)))
                ids = self.decoder.generate(inputs_embeds=prefix,
                    attention_mask=torch.ones(prefix.shape[:2], device=self.device, dtype=torch.long), **kwargs)[0].tolist()
        finally:
            self.model.train(was_training)
            self.model.audio_encoder.base.eval()
        eos = bool(ids and ids[-1] == self.tokenizer.eos_token_id)
        return {"ids": ids, "text": self.decode(ids), "eos": eos,
                "length_capped": len(ids) >= limit and not eos}

    def native_generate(self, example, *, max_len=300, top_p=0.8, temperature=1.0):
        first, second = self.audio_paths(example)
        rng = random.getstate()
        random.seed(example.get("audio_seed", self.seed))
        try:
            return self.wrapper.generate([[first, second, example["prompt"]]],
                max_len=max_len, top_p=top_p, temperature=temperature)[0]
        finally:
            random.setstate(rng)

    def parameter_manifest(self):
        parameters = list(self.model.named_parameters())
        shapes = {n: list(p.shape) for n, p in parameters}
        return {"model_id": self.model_id,
                "research_precision": self.research_precision,
                "frozen_encoder_precision": "fp32",
                "architecture_fingerprint": hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest(),
                "total_parameters": sum(p.numel() for _, p in parameters),
                "trainable_parameters": sum(p.numel() for _, p in parameters if p.requires_grad),
                "trainable_names": [n for n, p in parameters if p.requires_grad],
                "audio_encoder_frozen": not any(p.requires_grad for p in self.model.audio_encoder.base.parameters()),
                "vocabulary_map_hash": self._tokenizer_hash(),
                "tied_embeddings": self.decoder.get_input_embeddings().weight is self.decoder.get_output_embeddings().weight,
                "provenance": {"weights": "official Mellow v0", "encoder": "HTSAT",
                               "research_prompt": "unpadded, untruncated", "audio2": MISSING_AUDIO_SOURCE}}


    def save_checkpoint(self, path, training_state=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_id": self.model_id, "state_dict": self.model.state_dict(),
                    "training_state": training_state, "seed": self.seed,
                    "second_audio_pool": self.second_audio_pool,
                    "tokenizer_hash": self._tokenizer_hash()}, path)

    def _tokenizer_hash(self):
        return hashlib.sha256(json.dumps(self.tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()

    def load_checkpoint(self, path):
        data = torch.load(path, map_location=self.device, weights_only=True)
        if data["model_id"] != self.model_id or data["tokenizer_hash"] != self._tokenizer_hash():
            raise ValueError("Mellow checkpoint family/tokenizer mismatch")
        self.model.load_state_dict(data["state_dict"], strict=True)
        self._feature_cache.clear()
        self.seed = data["seed"]
        self.second_audio_pool = tuple(data["second_audio_pool"])
        return data.get("training_state")


MellowAudioPolicy = MellowPolicy
