"""Offline contract tests, never presented as real speech-model evidence."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from nano_audio.evaluation.native import Claim, NativeBenchmarkAdapter, claims_registry
from nano_audio.models.mellow import MellowPolicy, resolve_second_audio


class NativeTests(unittest.TestCase):
    def claim(self, **kwargs):
        return Claim("claim", "samsone_134m_published", "https://example.org", "table", "test-mini",
                     "accuracy", claimed_value=75, expected_n=2, **kwargs)

    def test_missing_second_requires_pool_and_seeded_selection(self):
        with self.assertRaisesRegex(ValueError, "distractor pool"):
            resolve_second_audio("first.wav")
        self.assertEqual(resolve_second_audio("first.wav", candidates=["z", "a"], seed=9),
                         resolve_second_audio("first.wav", candidates=["a", "z"], seed=9))
        self.assertEqual(resolve_second_audio("first.wav", "explicit.wav"), "explicit.wav")

    def test_hidden_labels_and_missing_scorer_block(self):
        records = [{"id": "a", "prompt": "native", "answer": None}]
        adapter = NativeBenchmarkAdapter("mmau_mini", revision="pinned", scorer=lambda rows: 100,
                                         scorer_revision="official-sha")
        result, predictions = adapter.evaluate(records, lambda item: "a", self.claim())
        self.assertEqual(result.status, "blocked")
        self.assertIsNone(result.observed_value)
        self.assertEqual(predictions[0]["native_prompt"], "native")
        records[0]["answer"] = "a"
        adapter.scorer = None
        self.assertEqual(adapter.evaluate(records, lambda item: "a", self.claim())[0].status, "blocked")

    def test_coverage_and_smoke_cannot_be_reproduction(self):
        adapter = NativeBenchmarkAdapter("mmau_mini", revision="pinned", scorer=lambda rows: 75,
                                         scorer_revision="official-sha")
        records = [{"id": "a", "answer": "a"}]
        result, _ = adapter.evaluate(records, lambda item: "a", self.claim())
        self.assertEqual(result.status, "partial")
        result, _ = adapter.evaluate(records, lambda item: "a", self.claim(), smoke_only=True)
        self.assertEqual(result.status, "smoke_only")
        records.append({"id": "b", "answer": "b"})
        claim = self.claim(checkpoint_sha256="hash", prompt_hash="prompt")
        result, _ = adapter.evaluate(records, lambda item: "a", claim, protocol_verified=True)
        self.assertEqual(result.status, "reproduced")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            adapter.evaluate(records * 2, lambda item: "a", claim)

    def test_unresolved_mellow_claim(self):
        claim = claims_registry()[-1]
        self.assertEqual(claim.status, "not_comparable")
        self.assertIsNone(claim.claimed_value)

    def test_official_fields_preserved(self):
        record = {"id": "x", "audio_id": "x.wav", "question": "Q", "choices": ["A", "B"],
                  "answer": "A", "task": "speech"}
        out = NativeBenchmarkAdapter.adapt_mmau(record)
        self.assertIs(out["raw"], record)
        self.assertNotIn("prompt", out)  # Rendering must be verified upstream.
        qa = NativeBenchmarkAdapter.adapt_reasonaqa(
            {"filepath1": "a.wav", "filepath2": "", "input": "caption", "answer": "sound"}, record_id="r")
        self.assertEqual(qa["audio2"], "")

    def test_causal_scoring_preserves_projector_and_decoder_gradients(self):
        # Small independent causal decoder validates alignment, not speech quality.
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(7, 4)
                self.head = nn.Linear(4, 7)
                self.config = SimpleNamespace(max_position_embeddings=100)
            def get_input_embeddings(self):
                return self.embedding
            def forward(self, inputs_embeds, attention_mask):
                return SimpleNamespace(logits=self.head(inputs_embeds.cumsum(1)))
        class Caption(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm = Decoder()
            def generate_prefix_inference(self, a, b, texts):
                return torch.cat((a, b, self.lm.embedding(texts["input_ids"])), 1)
        class Tokenizer:
            def encode(self, text, add_special_tokens=True):
                return [1, 2]
        model = nn.Module()
        model.audio_encoder = nn.Module()
        model.audio_encoder.base = nn.Linear(3, 3)
        model.audio_encoder.projection = nn.Linear(3, 4)
        model.caption_decoder = Caption()
        policy = MellowPolicy(SimpleNamespace(model=model, tokenizer=Tokenizer()))
        example = {"prompt": "q", "audio_features": (torch.ones(1, 1, 3), torch.ones(1, 1, 3))}
        ids = [3, 4]
        actual = policy.completion_logprobs(example, ids)
        prefix = policy.build_prefix(example)
        embeddings = torch.cat((prefix, policy.decoder.embedding(torch.tensor([ids]))), 1)
        logits = policy.decoder(inputs_embeds=embeddings, attention_mask=None).logits
        expected = torch.stack([logits[0, prefix.shape[1]-1+i].log_softmax(0)[token]
                                for i, token in enumerate(ids)]).reshape(1, 2)
        torch.testing.assert_close(actual, expected)
        (-actual.mean()).backward()
        self.assertGreater(model.audio_encoder.projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(policy.decoder.head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.audio_encoder.base.parameters()))
        masked = policy.completion_logprobs(example, ids, [1, 0])
        self.assertEqual(masked[0, 1].item(), 0)


if __name__ == "__main__":
    unittest.main()
