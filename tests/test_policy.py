"""Offline unit tests and causal-position contract fixtures for policy adapters.

These tests verify:
1. Exact causal token indexing: Log-probabilities for completion tokens align with the
   corresponding antecedent hidden states, and gradients flow back through the prefix.
2. Prompt isolation: Non-public fields (target, transcript, binding_id) are strictly excluded
   from generated prompt strings to avoid label leakage.
3. Architecture contracts: Round-trip tokenization consistency and parameter fingerprint
   matching between published and freshly initialized models.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from nano_audio.models.policy import SamsonePolicy


class DummyCausalDecoder(torch.nn.Module):
    """Minimal mock causal decoder for verifying completion log-probability indexing.
    
    Why cumulative summation?
    In a standard transformer decoder with causal attention masking, the hidden state at
    position t depends on all preceding tokens 0..t. We simulate this causal dependency
    using `inputs_embeds.cumsum(dim=1)`:
      - Position 0 depends only on token 0.
      - Position t depends on the sum of tokens 0..t.
    Applying a linear projection head to these cumulative states produces logits whose
    gradient at position t depends on all preceding prefix and completion tokens.
    """

    def __init__(self, vocab_size: int = 7, hidden_dim: int = 4):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, vocab_size)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        """Return the embedding layer for consistency with Hugging Face PreTrainedModel API."""
        return self.emb

    def forward(self, inputs_embeds: torch.Tensor, **kwargs) -> SimpleNamespace:
        """Forward pass applying cumulative sum and linear projection head."""
        return SimpleNamespace(logits=self.head(inputs_embeds.cumsum(dim=1)))


def test_causal_completion_alignment():
    """Verify that completion log-probabilities are indexed at the correct causal positions.
    
    Autoregressive Causal Math:
    Given:
      - Prefix length M = 3 (tokens at indices 0, 1, 2)
      - Completion length N = 2 (tokens at indices 3, 4 with IDs [2, 4])
      - Full sequence length = M + N = 5
      
    Logit-to-Token Alignment:
      - Logits at position 0 predict token 1
      - Logits at position 1 predict token 2
      - Logits at position 2 (last prefix position, index M - 1) predict completion token 0 (index 3)
      - Logits at position 3 predict completion token 1 (index 4)
      - Logits at position 4 predict completion token 2 (beyond sequence)
      
    Therefore, the logits predicting the completion sequence [2, 4] must be sliced at:
      logits[:, M - 1 : M + N - 1] = logits[:, 2 : 4]
      
    Gradient Backpropagation:
      Gradients computed from completion log-probabilities must flow backward through the
      decoder and into the multimodal prefix tensor (`prefix.grad`), confirming that the
      audio projector and prefix receive training loss signals.
    """
    # Instantiate policy without executing full __init__ (avoids downloading weights)
    policy = SamsonePolicy.__new__(SamsonePolicy)
    policy.device = torch.device("cpu")
    policy.decoder = DummyCausalDecoder()

    # Create dummy prefix embeddings with requires_grad=True to test gradient flow
    prefix_len = 3
    hidden_dim = 4
    prefix = torch.randn(1, prefix_len, hidden_dim, requires_grad=True)
    policy.build_prefix = lambda example: prefix

    # Completion sequence of 2 tokens (token IDs 2 and 4)
    completion_ids = torch.tensor([[2, 4]])

    # 1. Compute completion log-probabilities via the policy adapter method
    actual_logprobs = policy.completion_logprobs({}, completion_ids)

    # 2. Compute reference ground-truth values via full sequence forward pass
    # Concatenate prefix embeddings and completion token embeddings
    inputs = torch.cat([prefix, policy.decoder.emb(completion_ids)], dim=1)
    full_logits = policy.decoder(inputs_embeds=inputs).logits

    # Slice logits at indices [prefix_len - 1 : prefix_len + len(ids) - 1] = [2 : 4]
    expected_logits = full_logits[:, prefix_len - 1 : prefix_len + completion_ids.shape[1] - 1]
    # Gather target token log-probabilities along vocabulary dimension
    expected_logprobs = expected_logits.log_softmax(-1).gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)

    # Verify exact match between adapter implementation and ground truth
    torch.testing.assert_close(actual_logprobs, expected_logprobs)

    # 3. Verify autograd backpropagation: loss gradients must reach prefix embeddings
    loss = -actual_logprobs.mean()
    loss.backward()
    assert prefix.grad is not None, "Prefix tensor did not receive gradients"
    assert prefix.grad.abs().sum() > 0, "Prefix gradients are all zero"


def test_public_prompt_whitelist():
    """Verify that private, transcript, and ground-truth target fields are stripped from prompts.
    
    Data Leakage Prevention:
    When constructing prompts during training and evaluation, the policy must only format
    publicly visible metadata:
      - User instruction (e.g. "return the requested calls as json...")
      - Available tool specifications (names, parameters, descriptions)
      
    Private fields such as:
      - 'target': The ground-truth tool call JSON
      - 'transcript': The transcribed text of the spoken audio command
      - 'binding_id': Internal dataset identifier
    MUST NEVER be serialized into the prompt string.
    """
    policy = SamsonePolicy.__new__(SamsonePolicy)
    public_record = {"tools": [{"name": "some_tool", "description": "Set A Device"}]}

    # Prompt constructed from clean public record
    clean_prompt = policy.prompt(public_record)

    # Prompt constructed from record polluted with private ground-truth fields
    polluted_record = {
        **public_record,
        "target": "SECRET_CALL",
        "transcript": "SECRET_TRANSCRIPT",
        "binding_id": "SECRET_ID",
    }
    isolated_prompt = policy.prompt(polluted_record)

    # Assert outputs are identical, proving private fields are strictly ignored
    assert clean_prompt == isolated_prompt, "Private fields leaked into prompt formatting"
    # Assert prompt is consistently lowercased
    assert clean_prompt.islower(), "Prompt is not normalized to lowercase"
    # Assert no secret strings exist in the prompt
    assert "secret" not in isolated_prompt


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("NANO_AUDIO_INTEGRATION") != "1", reason="Requires pinned downloaded models")
def test_real_policy_roundtrip_and_fresh_architecture():
    """Integration test verifying full SamsonePolicy contracts with actual weights.
    
    Only runs when NANO_AUDIO_INTEGRATION=1 is set in the environment.
    
    Verifies:
    1. Architectural parity: Published checkpoint and freshly assembled architecture
       (Whisper-tiny + SmolLM2 base with pruned token map) share identical fingerprints.
    2. Tokenizer fidelity: Tokenizer encode-decode round trip succeeds losslessly on
       structured JSON tool calling syntax without unmapped token errors.
    3. Parameter tying: Input embedding weights and LM output head weights are identical
       tensors (weight tying).
    """
    published = SamsonePolicy(device="cpu")
    fresh = SamsonePolicy("samsone_134m_fresh", device="cpu")

    # 1. Verify architectural fingerprint match
    assert published.fingerprint() == fresh.fingerprint(), "Architecture fingerprint mismatch"

    # 2. Verify tokenizer roundtrip fidelity on sample tool call syntax
    tool_call_json = '{"calls":[{"name":"train_01_action_0","arguments":{"field_0":"kitchen_light","field_1":"off"}}]}'
    encoded_ids = published.encode(tool_call_json)
    decoded_text = published.decode(encoded_ids)
    assert decoded_text == tool_call_json, "Tokenizer roundtrip altered JSON call string"

    # 3. Verify tied input and output LM head embeddings
    input_weights = published.model.text_model.model.get_input_embeddings().weight
    output_weights = published.decoder.get_output_embeddings().weight
    assert input_weights is output_weights, "Input and output embeddings are not tied"
