"""Turn a set of (sequence, new tokens) pairs into a :class:`PagedBatch`.

Both the ordinary decode path and the speculative path need the same thing:
reserve cache slots for some new tokens, record where they went, and describe
the result as one flat batch. That logic lives here so the two paths cannot
drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .cache import PagedKVCache
from .model import PagedBatch


def build_paged_batch(
    cache: PagedKVCache,
    seq_ids: Sequence[int],
    token_lists: Sequence[Sequence[int]],
    start_positions: Sequence[int],
    device: str | torch.device = "cpu",
) -> PagedBatch:
    """Allocate slots for ``token_lists`` and flatten them into one batch.

    ``start_positions[i]`` is the absolute position of the first new token of
    sequence ``i``; it must equal the number of tokens already cached for that
    sequence, which is what makes the gathered key index equal the logical
    position (see :meth:`PagedKVCache.gather`).
    """
    if not (len(seq_ids) == len(token_lists) == len(start_positions)):
        raise ValueError("seq_ids, token_lists and start_positions must align")

    ids: list[int] = []
    positions: list[int] = []
    slots: list[int] = []
    q_lens: list[int] = []

    for seq_id, tokens, start in zip(seq_ids, token_lists, start_positions, strict=True):
        if not tokens:
            raise ValueError(f"sequence {seq_id} contributes no tokens")
        cached = cache.num_tokens(seq_id)
        if cached != start:
            raise ValueError(
                f"sequence {seq_id}: cache holds {cached} tokens but the batch "
                f"claims to start at position {start}"
            )
        slots.extend(cache.append(seq_id, len(tokens)))
        ids.extend(tokens)
        positions.extend(range(start, start + len(tokens)))
        q_lens.append(len(tokens))

    block_tables, context_lens = cache.build_block_tables(list(seq_ids))
    return PagedBatch(
        input_ids=torch.tensor(ids, dtype=torch.long, device=device),
        positions=torch.tensor(positions, dtype=torch.long, device=device),
        slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
        block_tables=block_tables,
        context_lens=context_lens,
        q_lens=torch.tensor(q_lens, dtype=torch.long, device=device),
    )


def last_token_rows(q_lens: torch.Tensor) -> torch.Tensor:
    """Row indices of each sequence's final query token in the flat output."""
    return torch.cumsum(q_lens, 0) - 1
