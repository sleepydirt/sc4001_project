"""Differentiable adapter for the released Mellow v0 multimodal audio-text architecture.

Architecture Overview:
- Audio Encoder: Hierarchical Token-Semantic Audio Transformer (HTSAT) audio encoder.
  The base encoder (`model.audio_encoder.base`) extracts spectrogram features and is frozen.
  A linear projection layer (`model.audio_encoder.projection`) maps audio embeddings into the
  text decoder's embedding space and remains trainable.
- Text Decoder: SmolLM2-135M language model acting as the caption decoder (`model.caption_decoder.lm`).
- Dual-Audio Convention: Upstream Mellow v0 was trained on pairs of audio clips (audio1 and audio2).
  In single-speech tool calling tasks, a distractor audio clip from a designated pool is chosen
  deterministically via `resolve_second_audio`.

Inference & Training Modes:
- Research Mode: Constructs unpadded, untruncated prompt prefixes for exact teacher-forced
  log-probability calculation (`completion_logprobs`) and reward-guided policy optimization.
- Native Mode: Retains upstream generation pipeline (`native_generate`) with standard
  audio preprocessing, repetition, and cropping.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
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


def resolve_second_audio(
    first: str | Path,
    second: str | Path | None = None,
    *,
    candidates: tuple = (),
    seed: int = 17,
) -> str:
    """Resolve the second audio clip following official Mellow dataset conventions.
    
    Why two audio clips?
    Upstream Mellow v0 was pre-trained and evaluated on audio pairs (e.g. comparing or mixing
    two audio clips). In single-speech tool calling tasks, only one speech prompt is available.
    Rather than duplicating the speech clip (which alters the model's acoustic perception),
    the upstream dataset convention randomly samples a valid distractor audio from a pool.
    
    Why sorting?
    Python's `set` iteration order is salted differently across process invocations.
    Sorting candidate paths guarantees strict deterministic selection across different machines.
    
    Args:
        first: Path to primary audio clip.
        second: Optional explicit path to second audio clip.
        candidates: Candidate file paths for distractor audio sampling.
        seed: Random seed for deterministic distractor selection (default: 17).
        
    Returns:
        String path to the resolved second audio file.
        
    Raises:
        ValueError: If `second` is omitted and candidate pool is empty.
    """
    if second:
        return str(second)
    # Sort unique non-empty candidate paths to remove cross-platform hash seed variation
    pool = sorted(set(str(p) for p in candidates if p))
    if not pool:
        raise ValueError("Mellow missing audio2 requires a declared distractor pool; " + MISSING_AUDIO_SOURCE)
    # Deterministically select distractor audio using an isolated PRNG instance
    return random.Random(seed).choice(pool)


class MellowPolicy:
    """Multimodal policy adapter for the Mellow v0 audio-text model."""

    model_id = "mellow_v0_published"

    def __init__(
        self,
        wrapper=None,
        *,
        device="auto",
        asset_root=None,
        vendor_root=None,
        checkpoint_path=None,
        second_audio_pool=(),
        seed: int = 17,
    ):
        """Initialize the Mellow policy adapter.
        
        Args:
            wrapper: Optional pre-constructed MellowWrapper instance. If None,
                loads model via `from_pretrained`.
            device: Target device ('auto', 'cuda', 'cpu', etc.).
            asset_root: Root directory for assets and manifests.
            vendor_root: Path to vendor/mellow repository checkout.
            checkpoint_path: Path to v0.ckpt checkpoint file.
            second_audio_pool: Pool of candidate paths for distractor audio sampling.
            seed: Default random seed for distractor selection and audio preprocessing.
        """
        if wrapper is None:
            # Construct and load model from pretrained assets
            loaded = type(self).from_pretrained(
                vendor_root=vendor_root,
                device=device,
                asset_root=asset_root,
                checkpoint_path=checkpoint_path,
                second_audio_pool=second_audio_pool,
                seed=seed,
            )
            self.__dict__.update(loaded.__dict__)
            return

        self.wrapper = wrapper
        self.model: nn.Module = wrapper.model
        # The caption decoder wraps SmolLM2 as its language model
        self.decoder = self.model.caption_decoder.lm
        self.tokenizer = wrapper.tokenizer
        self.second_audio_pool = tuple(second_audio_pool)
        self.seed = seed
        # Use BF16 precision if supported by the CUDA device, otherwise FP32
        self.research_precision = (
            "bf16" if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
        )
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        self._feature_cache = {}

        # Parameter gradient setup:
        # - Freeze HTSAT base encoder (feature extractor).
        # - Enable gradients for audio projection layer and language model decoder.
        self.model.audio_encoder.base.requires_grad_(False).eval()
        self.model.audio_encoder.projection.requires_grad_(True)
        self.decoder.requires_grad_(True)

    @classmethod
    def from_pretrained(
        cls,
        *,
        vendor_root=None,
        device="auto",
        checkpoint_path=None,
        asset_root=None,
        manifest_path=None,
        second_audio_pool=(),
        seed: int = 17,
    ) -> MellowPolicy:
        """Load the exact v0 architecture from manifest-pinned assets.

        Verifies SHA-256 hashes of all checkpoints and component files.
        Download metadata and hashes are written to reproduction/mellow_assets.json.
        Component loading uses local snapshots, bypassing upstream unpinned config downloads.
        """
        project = Path(__file__).resolve().parents[3]
        root = Path(vendor_root or project / "vendor/mellow")
        source_manifest = Path(manifest_path or Path(asset_root or project / "artifacts/assets") / "manifest.json")
        manifest = json.loads(source_manifest.read_text())
        checkpoint_info = dict(manifest["models"][cls.model_id])
        component_info = dict(manifest["models"]["smollm2_135m_base"])

        # Enforce explicit revisions to guarantee reproducibility
        if not checkpoint_info.get("revision") or not component_info.get("revision"):
            raise ValueError("Mellow and SmolLM2 require explicit manifest revisions")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        try:
            from mellow.wrapper import MellowWrapper
        except ImportError as exc:
            raise RuntimeError(
                f"Mellow runtime unavailable: {exc!r}; requirements: {root / 'requirements.txt'}"
            ) from exc

        from huggingface_hub import hf_hub_download, snapshot_download

        # Step 1: Verify or download checkpoint file (v0.ckpt)
        if checkpoint_path is None:
            checkpoint_path = checkpoint_info.get("local_path")
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download(
                checkpoint_info["repo_id"], "v0.ckpt", revision=checkpoint_info["revision"]
            )
        checkpoint_path = Path(checkpoint_path)

        # Step 2: Compute SHA-256 hash of checkpoint and verify integrity against manifest
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

        # Step 3: Download and verify SmolLM2 text component snapshot
        component_path = component_info.get("local_path")
        if component_path is None:
            component_path = snapshot_download(
                component_info["repo_id"],
                revision=component_info["revision"],
                allow_patterns=["*.json", "*.safetensors", "*.txt"],
            )
        component_path = Path(component_path)
        component_hashes = component_info.get("sha256", {})
        for filename, expected in component_hashes.items():
            path = component_path / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("SmolLM2 component hash mismatch: " + filename)
        if not component_hashes:
            component_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in component_path.iterdir()
                if path.is_file()
            }
        component_info.update(local_path=str(component_path), sha256=component_hashes)

        # Step 4: Verify git revision of vendor repository
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        expected_source = manifest.get("sources", {}).get("mellow", {}).get("revision")
        if expected_source and revision != expected_source:
            raise ValueError("Mellow vendor checkout differs from manifest revision")

        # Record asset reproduction metadata
        assets = {
            "source_manifest": str(source_manifest),
            "sources": {"mellow": {"revision": revision, "url": UPSTREAM}},
            "models": {cls.model_id: checkpoint_info, "smollm2_135m_base": component_info},
        }
        output = project / "artifacts/reproduction/mellow_assets.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(assets, indent=2) + "\n")

        # Step 5: Configure target execution device
        target = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError("Native Mellow supports CPU or CUDA; no implicit device substitution")

        # Step 6: Instantiate wrapper and configure model arguments
        from mellow.model.model import get_model_class

        wrapper = MellowWrapper.__new__(MellowWrapper)
        wrapper.parent_path = root / "mellow"
        wrapper.config_path = str(wrapper.parent_path / "config/v0.yaml")
        wrapper.args = wrapper.read_config_as_args(wrapper.config_path)
        args = wrapper.args

        # Upstream Mellow dynamically inspects text_decoder string to dispatch model loading:
        # "smollm2" must be in the path string for proper decoder dispatch.
        if "smollm2" not in str(component_path).lower():
            raise ValueError("Pinned component path must identify SmolLM2 for upstream decoder dispatch")
        args.model["decoder"]["text_decoder"] = str(component_path)

        model_cls = get_model_class(args.model["model_type"])
        wrapper.model = model_cls(
            audioenc_name=args.model["encoder"]["audioenc_name"],
            d_in=args.model["encoder"]["out_emb"],
            text_decoder=args.model["decoder"]["text_decoder"],
            prefix_length=args.model["decoder"]["prefix_length"],
            d_out=args.model["encoder"]["d_proj"],
        )

        # Step 7: Load checkpoint weights, stripping DDP 'module.' prefix if present
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state and all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        wrapper.model.load_state_dict(state, strict=True)

        # Step 8: Configure tokenizer with compatibility shim for legacy pad_to_max_length
        from transformers import AutoTokenizer

        wrapper.tokenizer = AutoTokenizer.from_pretrained(component_path, local_files_only=True)
        wrapper.tokenizer.add_special_tokens({"pad_token": "!"})

        # Compatibility shim: upstream Mellow called `encode_plus(..., pad_to_max_length=True)`,
        # which was removed in modern Hugging Face Transformers. We translate it to `padding="max_length"`.
        original_encode_plus = wrapper.tokenizer.encode_plus

        def legacy_encode_plus(*args, **kwargs):
            legacy = kwargs.pop("pad_to_max_length", False)
            if legacy:
                kwargs.setdefault("padding", "max_length")
            return original_encode_plus(*args, **kwargs)

        wrapper.tokenizer.encode_plus = legacy_encode_plus

        assets["runtime_compatibility"] = {
            "tokenizer": "pad_to_max_length=True translated to padding=max_length; native max_length retained",
            "setuptools": "80.9.0 supplies pkg_resources needed by pinned librosa",
        }
        output.write_text(json.dumps(assets, indent=2) + "\n")

        wrapper.use_cuda = target.type == "cuda"
        wrapper.device = (target.index or 0) if wrapper.use_cuda else "cpu"
        wrapper.model.to(target).eval()

        policy = cls(wrapper, second_audio_pool=second_audio_pool, seed=seed)
        policy.assets = assets
        return policy

    @property
    def device(self) -> torch.device:
        """Return the device on which model parameters reside."""
        return next(self.model.parameters()).device

    def encode(self, text: str) -> list[int]:
        """Tokenize text into token IDs without special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Decode token IDs back to a text string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().reshape(-1).tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def audio_paths(self, example: dict) -> tuple[str, str]:
        """Extract primary audio path and resolve secondary distractor audio path.
        
        Args:
            example: Data sample dictionary containing audio path fields.
            
        Returns:
            Tuple of (audio1_path, audio2_path).
        """
        # Accommodate various naming conventions across datasets
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

        # Resolve second audio using candidate distractor pool
        second = resolve_second_audio(
            first,
            second,
            candidates=self.second_audio_pool,
            seed=example.get("audio2_seed", self.seed),
        )
        return str(first), second

    def encode_audio(self, example: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract detached frozen outputs from HTSAT base encoder BEFORE trainable projection.

        Upstream wrapper repeats short clips and randomly crops clips longer than 10s.
        The Python RNG is isolated and seeded so cache keys record this exact crop realization.
        
        Args:
            example: Data sample dictionary.
            
        Returns:
            Tuple of embeddings tensors for (audio1, audio2).
        """
        paths = self.audio_paths(example)
        # Capture and restore random state to avoid polluting global RNG during audio cropping
        rng = random.getstate()
        random.seed(example.get("audio_seed", self.seed))
        try:
            self.model.audio_encoder.base.eval()
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
                return tuple(
                    self.model.audio_encoder.base(
                        self.wrapper.preprocess_audio([path], resample=True).squeeze(1)
                    )["embedding"].detach()
                    for path in paths
                )
        finally:
            random.setstate(rng)

    def prompt(self, example: dict) -> str:
        """Format the canonical text prompt for an example.
        
        Serializes tool signatures as compact JSON and appends instruction.
        """
        if "_prompt" in example:
            return example["_prompt"]
        if "tools" in example:
            return (
                example.get("instruction", "return the requested calls as json using the supplied tools.")
                + " tools:"
                + json.dumps(example["tools"], separators=(",", ":"), ensure_ascii=True)
                + " answer: "
            )
        return example.get("prompt", example.get("input", ""))

    def cached_audio(self, example: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve pre-projector frozen audio features from in-memory cache.
        
        Cache key incorporates:
        - SHA-256 hashes of both audio files.
        - Audio random seed (used for deterministic audio cropping).
        - Preprocessing recipe ('upstream-repeat-randomcrop-32k-10s-frozen-encoder-fp32').
        - Model parameter dtype string.
        
        Args:
            example: Data sample dictionary.
            
        Returns:
            Tuple of frozen audio feature tensors for the two audio clips.
        """
        paths = self.audio_paths(example)
        hashes = []
        for path in paths:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            hashes.append(digest.hexdigest())

        # Cache key binds to exact frozen base instance and preprocessing parameters
        key = (
            tuple(hashes),
            example.get("audio_seed", self.seed),
            "upstream-repeat-randomcrop-32k-10s-frozen-encoder-fp32",
            str(next(self.model.parameters()).dtype),
        )
        if key not in self._feature_cache:
            self._feature_cache[key] = self.encode_audio(example)
        return self._feature_cache[key]

    def build_prefix(self, example: dict) -> torch.Tensor:
        """Construct multimodal prefix embeddings for SmolLM2 decoder conditioning.
        
        For text modality:
            Embeds 'transcript: <text> <prompt>' directly via decoder input embeddings.
        For audio modality:
            Projects frozen features from both audio slots through trainable linear projection,
            then invokes `caption_decoder.generate_prefix_inference` with prompt tokens.
            
        Args:
            example: Data sample dictionary.
            
        Returns:
            Prefix embeddings tensor of shape (1, prefix_length, hidden_dim).
        """
        prompt = self.prompt(example)
        if example.get("modality") == "text":
            prompt = "transcript: " + example["transcript"] + " " + prompt
            tokens = torch.tensor([self.encode(prompt)], device=self.device, dtype=torch.long)
            return self.decoder.get_input_embeddings()(tokens)

        # Retrieve cached frozen audio features (for both audio1 and audio2)
        features = example.get("audio_features")
        if features is None:
            features = self.cached_audio(example)
        if len(features) != 2:
            raise ValueError("Mellow requires frozen features for both audio slots")

        # Project frozen 768-dim HTSAT features through trainable projection layer to decoder dim
        dtype = next(self.model.audio_encoder.projection.parameters()).dtype
        projected = [
            self.model.audio_encoder.projection(x.to(device=self.device, dtype=dtype))
            for x in features
        ]
        ids = torch.tensor([self.encode(prompt)], device=self.device, dtype=torch.long)
        # Combine projected audio embeddings with prompt tokens via caption decoder
        return self.model.caption_decoder.generate_prefix_inference(
            *projected, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        )

    def _autocast(self):
        """Context manager for BFloat16 mixed-precision on CUDA."""
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.research_precision == "bf16")

    def completion_logprobs(
        self,
        example: dict,
        ids: list[int] | torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute teacher-forced log-probabilities of completion tokens under mixed precision."""
        with self._autocast():
            return self._completion_logprobs_impl(example, ids, mask)

    def _completion_logprobs_impl(
        self,
        example: dict,
        ids: list[int] | torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Internal implementation for teacher-forced log-probability scoring."""
        tokens = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(1, -1)
        if tokens.shape[1] == 0:
            return self.decoder.get_input_embeddings().weight.sum().reshape(1, 1)[:, :0]

        # Construct multimodal prefix and concatenate completion token embeddings
        prefix = self.build_prefix(example)
        embeddings = torch.cat((prefix, self.decoder.get_input_embeddings()(tokens)), dim=1)
        self._check_context(embeddings.shape[1])

        # Forward pass through SmolLM2 decoder
        logits = self.decoder(
            inputs_embeds=embeddings,
            attention_mask=torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long),
        ).logits

        # Causal offset alignment: first completion token is predicted by position [prefix_len - 1]
        start = prefix.shape[1] - 1
        result = (
            logits[:, start : start + tokens.shape[1]]
            .float()
            .log_softmax(-1)
            .gather(-1, tokens.unsqueeze(-1))
            .squeeze(-1)
        )

        # Apply loss mask weighting if provided
        if mask is not None:
            weights = torch.as_tensor(mask, device=self.device).reshape_as(result)
            result = result * weights
        return result

    def _check_context(self, length: int) -> None:
        """Verify that sequence length does not exceed maximum position embeddings (2048)."""
        limit = min(2048, getattr(self.decoder.config, "max_position_embeddings", 2048))
        if limit and length > limit:
            raise ValueError(f"Mellow context {length} exceeds {limit}; catalog truncation forbidden")

    def generate(self, example: dict, config: dict | None = None) -> dict:
        """Autoregressively generate completion tokens using SmolLM2 decoder.
        
        Args:
            example: Data sample dictionary.
            config: Optional generation parameters (max_new_tokens, do_sample, temperature, top_p, top_k).
            
        Returns:
            Dictionary with 'ids', 'text', 'eos', and 'length_capped' fields.
        """
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
                kwargs = dict(
                    max_new_tokens=limit,
                    do_sample=bool(get("do_sample", False)),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if kwargs["do_sample"]:
                    kwargs.update(
                        temperature=float(get("temperature", 1.0)),
                        top_p=float(get("top_p", 1.0)),
                        top_k=int(get("top_k", 0)),
                    )
                ids = self.decoder.generate(
                    inputs_embeds=prefix,
                    attention_mask=torch.ones(prefix.shape[:2], device=self.device, dtype=torch.long),
                    **kwargs,
                )[0].tolist()
        finally:
            self.model.train(was_training)
            self.model.audio_encoder.base.eval()

        eos = bool(ids and ids[-1] == self.tokenizer.eos_token_id)
        return {
            "ids": ids,
            "text": self.decode(ids),
            "eos": eos,
            "length_capped": len(ids) >= limit and not eos,
        }

    def native_generate(self, example: dict, *, max_len: int = 300, top_p: float = 0.8, temperature: float = 1.0):
        """Invoke upstream Mellow wrapper's native end-to-end generation routine."""
        first, second = self.audio_paths(example)
        rng = random.getstate()
        random.seed(example.get("audio_seed", self.seed))
        try:
            return self.wrapper.generate(
                [[first, second, example["prompt"]]],
                max_len=max_len,
                top_p=top_p,
                temperature=temperature,
            )[0]
        finally:
            random.setstate(rng)

    def parameter_manifest(self) -> dict:
        """Return a structured metadata manifest of parameter counts and provenance."""
        parameters = list(self.model.named_parameters())
        shapes = {n: list(p.shape) for n, p in parameters}
        return {
            "model_id": self.model_id,
            "research_precision": self.research_precision,
            "frozen_encoder_precision": "fp32",
            "architecture_fingerprint": hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest(),
            "total_parameters": sum(p.numel() for _, p in parameters),
            "trainable_parameters": sum(p.numel() for _, p in parameters if p.requires_grad),
            "trainable_names": [n for n, p in parameters if p.requires_grad],
            "audio_encoder_frozen": not any(p.requires_grad for p in self.model.audio_encoder.base.parameters()),
            "vocabulary_map_hash": self._tokenizer_hash(),
            "tied_embeddings": (
                self.decoder.get_input_embeddings().weight is self.decoder.get_output_embeddings().weight
            ),
            "provenance": {
                "weights": "official Mellow v0",
                "encoder": "HTSAT",
                "research_prompt": "unpadded, untruncated",
                "audio2": MISSING_AUDIO_SOURCE,
            },
        }

    def save_checkpoint(self, path: str | Path, training_state: dict | None = None) -> None:
        """Save model state dict, random seed, distractor pool, and tokenizer hash to checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_id": self.model_id,
                "state_dict": self.model.state_dict(),
                "training_state": training_state,
                "seed": self.seed,
                "second_audio_pool": self.second_audio_pool,
                "tokenizer_hash": self._tokenizer_hash(),
            },
            path,
        )

    def _tokenizer_hash(self) -> str:
        """Compute SHA-256 hash of tokenizer vocabulary for compatibility verification."""
        return hashlib.sha256(json.dumps(self.tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()

    def load_checkpoint(self, path: str | Path) -> dict | None:
        """Load checkpoint weights and verify model family and tokenizer compatibility.
        
        Args:
            path: Path to checkpoint file.
            
        Returns:
            The restored training state dictionary.
            
        Raises:
            ValueError: If model family or tokenizer hash differs from current instance.
        """
        data = torch.load(path, map_location=self.device, weights_only=True)
        if data["model_id"] != self.model_id or data["tokenizer_hash"] != self._tokenizer_hash():
            raise ValueError("Mellow checkpoint family/tokenizer mismatch")
        self.model.load_state_dict(data["state_dict"], strict=True)
        self._feature_cache.clear()
        self.seed = data["seed"]
        self.second_audio_pool = tuple(data["second_audio_pool"])
        return data.get("training_state")


# Backward compatibility alias
MellowAudioPolicy = MellowPolicy
