"""Logit post-processing and token sampling.

All functions operate on a ``(batch, vocab)`` float tensor and are pure: they
never mutate their input. Sampling draws from an explicit ``torch.Generator``
so that every run is reproducible from a seed.
"""

from __future__ import annotations

import torch

from .config import SamplingConfig

NEG_INF = float("-inf")


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0; use greedy sampling instead")
    if temperature == 1.0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep the ``top_k`` highest logits per row, mask the rest to -inf."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, NEG_INF)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest prefix whose mass exceeds ``top_p``.

    The token that crosses the threshold is kept, so the surviving mass is
    always >= ``top_p`` and at least one token always survives.
    """
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if top_p == 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    # Drop tokens whose *preceding* cumulative mass already reached top_p.
    remove_sorted = (cumulative - probs) >= top_p
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, NEG_INF)


def process_logits(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """Apply temperature, then top-k, then top-p, in that order."""
    out = apply_temperature(logits, cfg.temperature)
    if cfg.top_k is not None:
        out = apply_top_k(out, cfg.top_k)
    if cfg.top_p is not None:
        out = apply_top_p(out, cfg.top_p)
    return out


def probs_from_logits(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """Return the exact distribution a sampler would draw from.

    Greedy is expressed as a one-hot distribution so that speculative decoding
    can use one code path for both greedy and stochastic requests.
    """
    if cfg.greedy:
        out = torch.zeros_like(logits)
        out.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return out
    return torch.softmax(process_logits(logits, cfg), dim=-1)


def sample_from_probs(
    probs: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Multinomial draw over the last dimension, returning ``(batch,)`` ids."""
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def sample(
    logits: torch.Tensor,
    cfg: SamplingConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token id per row of ``logits``."""
    if cfg.greedy:
        return logits.argmax(dim=-1)
    return sample_from_probs(probs_from_logits(logits, cfg), generator)
