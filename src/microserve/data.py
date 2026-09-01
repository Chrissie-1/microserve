"""Corpus loading and batching for TinyShakespeare.

Shared by ``scripts/train.py`` and the benchmark harness, which draws its
prompts from the validation split so that prompts are in-distribution but
unseen during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .tokenizer import CharTokenizer

DEFAULT_CORPUS = Path("data/tinyshakespeare.txt")


@dataclass
class Corpus:
    """An encoded corpus split into train and validation halves."""

    tokenizer: CharTokenizer
    train: torch.Tensor
    val: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def split(self, name: str) -> torch.Tensor:
        if name not in {"train", "val"}:
            raise ValueError(f"unknown split {name!r}")
        return self.train if name == "train" else self.val


def load_corpus(
    path: str | Path = DEFAULT_CORPUS,
    val_fraction: float = 0.1,
    tokenizer: CharTokenizer | None = None,
) -> Corpus:
    """Read ``path``, build (or reuse) a tokenizer, and split off a tail for validation."""
    text = Path(path).read_text(encoding="utf-8")
    tok = tokenizer or CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.int16)
    cut = int(len(ids) * (1.0 - val_fraction))
    return Corpus(tokenizer=tok, train=ids[:cut], val=ids[cut:])


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` random windows; targets are inputs shifted by one."""
    if len(data) <= seq_len + 1:
        raise ValueError("corpus split is shorter than one training window")
    starts = torch.randint(
        len(data) - seq_len - 1, (batch_size,), generator=generator
    )
    x = torch.stack([data[s : s + seq_len] for s in starts]).to(torch.int64)
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts]).to(torch.int64)
    return x.to(device), y.to(device)


def sample_prompts(
    corpus: Corpus,
    count: int,
    lengths: list[int],
    generator: torch.Generator,
) -> list[list[int]]:
    """Draw ``count`` prompts of the given lengths from the validation split."""
    data = corpus.val
    prompts: list[list[int]] = []
    for i in range(count):
        length = max(1, lengths[i])
        start = int(
            torch.randint(len(data) - length - 1, (1,), generator=generator).item()
        )
        prompts.append(data[start : start + length].to(torch.int64).tolist())
    return prompts
