"""Shared fixtures. Everything here is deliberately tiny so CI stays fast."""

from __future__ import annotations

import pytest
import torch

from microserve.cache import PagedKVCache
from microserve.config import CacheConfig, ModelConfig
from microserve.model import Transformer

VOCAB = 32


@pytest.fixture(autouse=True)
def _single_thread() -> None:
    """Keep tests deterministic and cheap regardless of the host CPU."""
    torch.set_num_threads(1)
    torch.manual_seed(0)


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB, n_layers=2, d_model=32, n_heads=4, max_seq_len=128
    )


@pytest.fixture
def draft_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB, n_layers=1, d_model=16, n_heads=2, max_seq_len=128
    )


@pytest.fixture
def cache_cfg() -> CacheConfig:
    return CacheConfig(block_size=8, num_blocks=32)


def make_model(cfg: ModelConfig, seed: int = 0, amplify: float = 1.0) -> Transformer:
    """Build a model with reproducible weights.

    ``amplify`` scales the residual output projections. Freshly initialised
    models are near-degenerate (every one of them predicts "repeat the last
    token"), which makes agreement tests vacuous; amplifying gives two models
    that genuinely disagree.
    """
    torch.manual_seed(seed)
    model = Transformer(cfg).eval()
    if amplify != 1.0:
        with torch.no_grad():
            for block in model.blocks:
                block.attn.o_proj.weight *= amplify
                block.ffn.down_proj.weight *= amplify
    return model


@pytest.fixture
def model(model_cfg: ModelConfig) -> Transformer:
    return make_model(model_cfg, seed=0, amplify=20.0)


@pytest.fixture
def draft(draft_cfg: ModelConfig) -> Transformer:
    return make_model(draft_cfg, seed=1, amplify=20.0)


@pytest.fixture
def cache(model_cfg: ModelConfig, cache_cfg: CacheConfig) -> PagedKVCache:
    return PagedKVCache(model_cfg, cache_cfg)


def dense_greedy(model: Transformer, prompt: list[int], n: int) -> list[int]:
    """Greedy generation through the dense reference path."""
    past = None
    current = torch.tensor(prompt)[None]
    out: list[int] = []
    for _ in range(n):
        with torch.no_grad():
            logits, past = model(current, past_kv=past, return_cache=True)
        token = int(logits[0, -1].argmax())
        out.append(token)
        current = torch.tensor([[token]])
    return out
