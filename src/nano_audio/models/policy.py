"""Differentiable adapter around the exact released Samsone-134M modules.

Architecture Overview:
- Audio Encoder: Pretrained Whisper-tiny encoder (frozen), operating on 16kHz mono audio
  with 30-second native padding and 50-token pooling.
- Projector: Trainable non-linear MLP mapping Whisper audio features (384-dim) to decoder
  embedding dimensions (576-dim).
- Text Decoder: SmolLM2-135M language model pruned using the official token reduction map.
  Input embeddings and LM head embeddings are tied.

Design Principles:
- Microbatch size of 1 is deliberate: it eliminates padding ambiguity and ensures identical
  token positioning between rollout generation and teacher-forced log-probability evaluation.
- Feature caching: Frozen Whisper encoder representations are cached to disk and memory
  *before* the trainable projector, avoiding redundant audio feature extraction across training epochs.
- Precision: Trainable master weights are kept in FP32; CUDA execution uses FP16 autocast.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

from ..utils import ARTIFACTS, ROOT, digest, read_json, sha, write_json


class SamsonePolicy:
    """Multimodal policy wrapper for the Samsone-134M speech-and-text tool calling model."""

    def __init__(
        self,
        model_id: str = "samsone_134m_published",
        device: str = "auto",
        asset_root: str | Path | None = None,
    ):
        """Initialize the Samsone policy adapter.
        
        Loads either the official published checkpoint or reconstructs a fresh
        architecture from base pretrained components for architectural verification.

        Args:
            model_id: Identifier for model weights ('samsone_134m_published' or 'samsone_134m_fresh').
            device: Target device ('auto', 'cuda', 'cpu', etc.).
            asset_root: Path to directory containing assets and manifest.json.
        """
        # Ensure Samsone vendor library is on Python path for submodule resolution
        sys.path.insert(0, str(ROOT / "vendor/samsone/src"))
        from samsone import Samsone134M
        from samsone.models.audio import WhisperAudioEncoder
        from samsone.models.text.lm import HuggingFaceLM

        self.model_id = model_id
        # Resolve target device: automatically detect CUDA availability if 'auto' is specified
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        )
        self.asset_root = Path(asset_root or ARTIFACTS / "assets")
        self.assets = read_json(self.asset_root / "manifest.json")
        checkpoint_path = self.assets["models"]["samsone_134m_published"]["local_path"]

        # Keep optimizer master weights FP32; CUDA operations use FP16 autocast.
        # max_new_tokens=128 matches the maximum expected tool calling JSON completion length.
        self.native = Samsone134M(
            device=self.device,
            dtype=torch.float32,
            checkpoint_path=checkpoint_path,
            max_new_tokens=128,
        )
        self.model = self.native.model
        self.tokenizer = self.model.text_model.tokenizer

        if model_id == "samsone_134m_fresh":
            # Architecture contract verification mode:
            # Reconstruct model from raw base components (Whisper-tiny + SmolLM2-135M)
            # using the official token reduction map, re-initialize projector weights,
            # and verify that the parameter shapes and token mapping match the release checkpoint.
            expected = self.fingerprint()
            from samsone.inference.models import _build_pruned_token_map

            # Build and persist token pruning map (mapping pruned 49k vocab to 128k base vocab)
            map_path = self.asset_root / "fresh_prune_map.json"
            write_json(map_path, _build_pruned_token_map(self.tokenizer))

            # Initialize base text decoder with token pruning applied
            self.model.text_model = HuggingFaceLM(
                self.assets["models"]["smollm2_135m_base"]["local_path"],
                prune_tokens_map_path=str(map_path),
                load_pretrained_weights=True,
            )
            # Initialize base audio encoder with pretrained Whisper-tiny weights
            self.model.audio_encoder.encoder = WhisperAudioEncoder(
                self.assets["models"]["whisper_tiny"]["local_path"],
                load_pretrained_weights=True,
            )
            # Randomly initialize trainable MLP projector and multimodal separator token
            self.model.projector.init_weights()
            with torch.no_grad():
                self.model.sep_token.normal_()
            self.model.to(self.device)

            # Guarantee that fresh construction exactly mirrors the official release architecture
            assert self.fingerprint() == expected, "Fresh architecture does not match release"
            self.provenance = {
                "decoder": "component text-pretrained SmolLM2 selected by official map",
                "encoder": "component pretrained whisper-tiny",
                "projector": "random",
                "separator": "random",
                "published_weights_retained": False,
            }
        else:
            self.provenance = {"all_weights": "official Samsone release checkpoint"}

        # Reference decoder model and vocabulary token mappings:
        # - mapping: maps pruned index (0..49151) -> original SmolLM2 tokenizer token ID
        # - inverse: maps original SmolLM2 token ID -> pruned index
        self.decoder = self.model.text_model.model
        self.mapping = self.model.text_model.tokens_map
        self.inverse = {v: k for k, v in self.mapping.items()}

        # Map EOS token to pruned index space and configure generation config
        self.eos_token_id = self.inverse[self.tokenizer.eos_token_id]
        self.decoder.generation_config.eos_token_id = self.eos_token_id
        self.decoder.generation_config.pad_token_id = self.eos_token_id

        # Parameter freeze setup:
        # - Audio encoder remains completely frozen during tool-calling post-training.
        # - MLP projector and text decoder parameters remain trainable.
        self.model.requires_grad_(True)
        self.model.audio_encoder.requires_grad_(False)
        self.model.eval()

        # Cache key components for pre-projector audio representations
        # Captures checkpoint/revision hashes to ensure feature cache validity
        self.encoder_hash = digest({
            n: sha(checkpoint_path) if model_id.endswith("published") else self.assets["models"]["whisper_tiny"]["revision"]
            for n in ["encoder"]
        })
        self.feature_cache = ARTIFACTS / "features"
        self.feature_cache.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}

    def fingerprint(self) -> str:
        """Compute an architectural fingerprint of parameter shapes, token mappings, and weight tying.
        
        Ensures exact structural compatibility between models by hashing:
        - Parameter name and shape dictionary.
        - Token pruning vocabulary map.
        - Embedding weight tying (input embeddings is output head embeddings).
        """
        return digest({
            "shapes": {k: list(v.shape) for k, v in self.model.state_dict().items()},
            "map": self.model.text_model.tokens_map,
            "tied": (
                self.model.text_model.model.get_input_embeddings().weight
                is self.model.text_model.model.get_output_embeddings().weight
            ),
        })

    def encode(self, text: str) -> list[int]:
        """Tokenize text and translate tokens to pruned vocabulary indices.
        
        Args:
            text: Input string to tokenize.
            
        Returns:
            List of token IDs in the pruned vocabulary space.
            
        Raises:
            ValueError: If a token is absent from the pruned vocabulary or fails round-trip decode.
        """
        # Step 1: Tokenize string with HuggingFace tokenizer without special tokens
        ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Step 2: Map each original token ID to its pruned vocabulary index
        try:
            result = [self.inverse[t] for t in ids]
        except KeyError as e:
            raise ValueError(f"Unavailable pruned token in {text!r}: {e}") from e

        # Step 3: Strict round-trip validation to verify tokenization fidelity
        if self.decode(result) != text:
            raise ValueError("Tokenizer round trip failed")
        return result

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Decode pruned vocabulary token IDs back to a text string.
        
        Args:
            ids: Pruned vocabulary token IDs.
            
        Returns:
            Decoded text string with special tokens skipped.
        """
        return self.tokenizer.decode(
            [self.mapping[int(t)] for t in ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _autocast(self):
        """Context manager for mixed precision autocasting on CUDA (FP16)."""
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda")

    def encode_audio(self, waveforms: str | Path | list[str | Path], sample_rates=None):
        """Encode raw audio waveforms using the native frozen Whisper encoder.
        
        Preserves exact 30s padding (480,000 samples at 16kHz) and released 50-token pooling
        before projection.
        
        Args:
            waveforms: Filepath or list of filepaths to audio clips.
            sample_rates: Unused, kept for API compatibility.
            
        Returns:
            Tuple of (features, mask) from the audio encoder.
        """
        paths = waveforms if isinstance(waveforms, list) else [waveforms]
        with torch.no_grad(), self._autocast():
            # Run upstream audio preprocessing: resamples to 16kHz mono, pads to 30s, extracts log-mel
            features = self.native._prepare_audio([Path(p) for p in paths])
            # Pass log-mel spectrogram through frozen Whisper encoder
            return self.model.audio_encoder(features)

    def cached_audio(self, path: str | Path):
        """Retrieve pre-projector audio representations from memory or disk cache.
        
        Feature caching uses a deterministic SHA-256 key over:
        - Audio file content hash (`sha(path)`).
        - Encoder weights identification (`self.encoder_hash`).
        - Audio preprocessing parameters ('native-16k-mono-480000-v1').
        - Pooling configuration ('official-fixed50').
        - Compute/storage precision ('float32-storage-fp16-compute').
        
        This prevents recomputing the expensive Whisper forward pass across training updates.
        
        Args:
            path: Path to the audio file.
            
        Returns:
            Tuple of (features, mask) tensors placed on `self.device`.
        """
        path = Path(path)
        key = digest({
            "audio": sha(path),
            "encoder": self.encoder_hash,
            "preprocessing": "native-16k-mono-480000-v1",
            "pooling": "official-fixed50",
            "dtype": "float32-storage-fp16-compute",
        })

        # Check in-memory RAM cache first
        if key not in self._memory_cache:
            file = self.feature_cache / (key + ".pt")
            # Check on-disk filesystem cache second
            if file.exists():
                features, mask = torch.load(file, map_location=self.device, weights_only=True)
            else:
                # Extract features using frozen Whisper encoder and persist to disk
                features, mask = self.encode_audio([path])
                features = features.float().detach()
                mask = mask.detach()
                torch.save((features.cpu(), mask.cpu()), file)
            self._memory_cache[key] = (features.to(self.device), mask.to(self.device))
        return self._memory_cache[key]

    def prompt(self, example: dict) -> str:
        """Format the canonical text prompt for an example.
        
        Strictly includes only public instruction and tools metadata.
        Never serializes ground-truth target calls, transcripts, or provenance.
        
        Format:
            "<instruction> tools:<compact_json_tools> answer: "
            
        Args:
            example: Data sample dictionary.
            
        Returns:
            Standardized lowercase prompt string.
        """
        if "_prompt" in example:
            return example["_prompt"]

        # Only public fields; never serialize target/transcript/provenance to avoid label leakage
        instruction = example.get("instruction", "return the requested calls as json using the supplied tools.")
        # separators=(',', ':') eliminates extraneous whitespace for consistent tokenization
        tools_str = json.dumps(example["tools"], separators=(",", ":"), ensure_ascii=True)
        return f"{instruction} tools:{tools_str} answer: ".lower()

    def build_prefix(self, example: dict) -> torch.Tensor:
        """Construct multimodal prefix embeddings for language model conditioning.
        
        Sequencing logic:
        - For audio modality:
            Prefix = [SEP token, projected audio features, SEP token, text prompt embeddings]
        - For text-only modality:
            Prefix = Embedding("transcript: <transcript> <prompt>")
            
        Args:
            example: Data sample dictionary containing modality, prompt, and audio paths.
            
        Returns:
            Prefix embeddings tensor of shape (1, prefix_length, hidden_dim).
            
        Raises:
            ValueError: If prefix length plus 128 max new tokens exceeds the context window.
        """
        prompt = self.prompt(example)

        if example.get("modality") == "text":
            # Text-only baseline: prepend transcript text directly to prompt
            prompt = "transcript: " + example["transcript"].lower().strip() + " " + prompt
            tokens = torch.tensor([self.encode(prompt)], device=self.device)
            result = self.decoder.get_input_embeddings()(tokens)
        else:
            # Audio-conditioned modality:
            tokens = torch.tensor([self.encode(prompt)], device=self.device)
            text = self.decoder.get_input_embeddings()(tokens)

            # Load cached frozen audio features (or silence control if requested)
            audio_path = example["silence_path"] if example.get("audio_control") == "silence" else example["audio_path"]
            features, mask = self.cached_audio(audio_path)

            # Project frozen audio features through trainable MLP projector
            audio, mask = self.model.projector(features, mask)
            # Filter valid (unpadded) audio token embeddings using the attention mask
            audio = audio[mask.bool()].unsqueeze(0)
            sep = self.model.sep_token.view(1, 1, -1)

            # Concatenate [SEP, audio, SEP, text] into unified prefix representation
            result = torch.cat([sep, audio, sep, text], dim=1)

        # Context boundary check: ensure prefix plus maximum completion tokens fits within 2048 positions
        max_pos = min(2048, self.decoder.config.max_position_embeddings)
        if result.shape[1] + 128 > max_pos:
            raise ValueError(f"Complete catalog exceeds context: {result.shape[1]}+128 > {max_pos}")
        return result

    def completion_logprobs(
        self,
        example: dict,
        completion_ids: list[int] | torch.Tensor,
        completion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute teacher-forced log-probabilities of completion tokens given prefix context.
        
        Causal Alignment Details:
        Given prefix length M and completion token IDs sequence length N:
          - Full sequence passed to decoder: inputs = [prefix, completion_embeddings] (length M + N).
          - In standard autoregressive decoders, the output hidden state at position k predicts the token
            at position k + 1.
          - Therefore, the first completion token (index 0) is predicted by logits at position M - 1.
          - The last completion token (index N - 1) is predicted by logits at position M + N - 2.
          - Slicing range: logits[:, M - 1 : M + N - 1, :].
        
        Args:
            example: Data sample dictionary.
            completion_ids: Sequence of completion token IDs (shape: (1, seq_len) or (seq_len,)).
            completion_mask: Optional mask tensor (unused in SamsonePolicy microbatch-1 path).
            
        Returns:
            Per-token log-probabilities tensor of shape (1, seq_len).
        """
        ids = torch.as_tensor(completion_ids, device=self.device, dtype=torch.long).reshape(1, -1)
        with self._autocast():
            prefix = self.build_prefix(example)
            emb = self.decoder.get_input_embeddings()(ids)
            inputs = torch.cat([prefix, emb], dim=1)

            # Forward pass through decoder language model
            out = self.decoder(
                inputs_embeds=inputs,
                attention_mask=torch.ones(inputs.shape[:2], device=self.device, dtype=torch.long),
                use_cache=False,
            ).logits

            # Slice causal prediction logits corresponding to target completion tokens
            logits = out[:, prefix.shape[1] - 1 : prefix.shape[1] + ids.shape[1] - 1, :].float()
            # Index the log probability for each target completion token using gather
            result = logits.log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        return result

    def generate(self, example: dict, generation_config: dict | None = None) -> dict:
        """Autoregressively generate completion tokens given multimodal prefix input.
        
        Args:
            example: Data sample dictionary.
            generation_config: Optional generation parameters:
                - 'max_new_tokens': Maximum generated tokens (default: 128).
                - 'do_sample': True for stochastic sampling (used in GRPO rollouts), False for greedy.
                - 'temperature', 'top_p', 'top_k': Sampling hyperparameters.
            
        Returns:
            Dictionary containing:
                - 'ids': List of generated token IDs (truncated at EOS if present).
                - 'text': Decoded string of generated tokens.
                - 'eos': Boolean indicating if an EOS token was generated.
                - 'length_capped': Boolean indicating if generation hit max_new_tokens without EOS.
        """
        config = {"max_new_tokens": 128, "do_sample": False, **(generation_config or {})}
        # In sampling mode (GRPO exploration), unconstrain top_k/top_p for full vocabulary coverage
        if config.get("do_sample"):
            config.update(top_k=0, top_p=1.0, temperature=1.0)

        self.model.eval()
        with torch.no_grad(), self._autocast():
            prefix = self.build_prefix(example)
            ids = self.decoder.generate(
                inputs_embeds=prefix,
                attention_mask=torch.ones(prefix.shape[:2], device=self.device, dtype=torch.long),
                eos_token_id=self.eos_token_id,
                pad_token_id=self.eos_token_id,
                **config,
            )[0].tolist()

        # Check for EOS token and truncate sequence at EOS position
        eos = self.eos_token_id in ids
        if eos:
            ids = ids[: ids.index(self.eos_token_id) + 1]

        return {
            "ids": ids,
            "text": self.decode(ids),
            "eos": eos,
            "length_capped": not eos and len(ids) >= config["max_new_tokens"],
        }

    def parameter_manifest(self) -> dict:
        """Return a structured metadata manifest of model architecture, parameter counts, and provenance."""
        return {
            "model_id": self.model_id,
            "architecture_fingerprint": self.fingerprint(),
            "vocabulary_map_hash": digest(self.mapping),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "trainable_names": [n for n, p in self.model.named_parameters() if p.requires_grad],
            "provenance": self.provenance,
            "audio_encoder_frozen": not any(p.requires_grad for p in self.model.audio_encoder.parameters()),
            "tied_embeddings": (
                self.decoder.get_input_embeddings().weight is self.decoder.get_output_embeddings().weight
            ),
        }

    def save_checkpoint(self, path: str | Path, training_state: dict | None = None) -> None:
        """Atomically persist policy weights, training state, and architecture manifest.
        
        Writes to a temporary file first before atomically replacing the destination
        file, preventing corruptions if the process or machine is interrupted.
        
        Args:
            path: Destination file path for checkpoint.
            training_state: Optional dictionary containing optimizer/scheduler/RNG states.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "training_state": training_state or {},
                "manifest": self.parameter_manifest(),
            },
            tmp,
        )
        tmp.replace(path)

    def load_checkpoint(self, path: str | Path) -> dict:
        """Load checkpoint weights and verify architectural fingerprint compatibility.
        
        Args:
            path: Checkpoint file path to restore.
            
        Returns:
            The restored training state dictionary.
            
        Raises:
            ValueError: If the checkpoint architecture fingerprint differs from current policy.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # Verify architecture fingerprint matches current model to prevent loading incompatible weights
        if checkpoint["manifest"]["architecture_fingerprint"] != self.fingerprint():
            raise ValueError("Checkpoint architecture mismatch")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        return checkpoint["training_state"]
