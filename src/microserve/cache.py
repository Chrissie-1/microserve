"""Paged KV cache: block allocator, physical storage, and per-sequence tables.

The cache is a flat pool of ``num_blocks * block_size`` slots. A sequence owns
an ordered list of block ids (its *block table*); logical token position ``p``
of that sequence lives at physical slot::

    block_table[p // block_size] * block_size + (p % block_size)

Blocks are reference counted so that a forked sequence can share the prefix of
its parent. Writes to a shared block trigger copy-on-write.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import CacheConfig, ModelConfig

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class OutOfCacheMemory(RuntimeError):
    """Raised when the block pool cannot satisfy an allocation."""


class BlockAllocator:
    """Reference-counted free list over ``num_blocks`` physical blocks."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))
        self._refcount: list[int] = [0] * num_blocks

    # -- queries ----------------------------------------------------------
    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def utilization(self) -> float:
        return self.num_used / self.num_blocks

    def refcount(self, block: int) -> int:
        return self._refcount[block]

    # -- mutation ---------------------------------------------------------
    def allocate(self) -> int:
        if not self._free:
            raise OutOfCacheMemory("block pool exhausted")
        block = self._free.pop()
        self._refcount[block] = 1
        return block

    def allocate_many(self, n: int) -> list[int]:
        if n > len(self._free):
            raise OutOfCacheMemory(f"need {n} blocks, {len(self._free)} free")
        return [self.allocate() for _ in range(n)]

    def incref(self, block: int) -> None:
        if self._refcount[block] == 0:
            raise ValueError(f"cannot incref free block {block}")
        self._refcount[block] += 1

    def free(self, block: int) -> None:
        """Drop one reference; the block returns to the pool at zero refs."""
        if self._refcount[block] == 0:
            raise ValueError(f"double free of block {block}")
        self._refcount[block] -= 1
        if self._refcount[block] == 0:
            self._free.append(block)

    def check_invariants(self) -> None:
        """Assert internal consistency; used by tests and debug builds."""
        if len(set(self._free)) != len(self._free):
            raise AssertionError("duplicate block in free list")
        for block in self._free:
            if self._refcount[block] != 0:
                raise AssertionError(f"free block {block} has references")
        held = [b for b in range(self.num_blocks) if self._refcount[b] > 0]
        if len(held) + len(self._free) != self.num_blocks:
            raise AssertionError("blocks leaked: neither free nor referenced")


@dataclass
class BlockTable:
    """Logical to physical mapping for one sequence."""

    blocks: list[int] = field(default_factory=list)
    num_tokens: int = 0

    def capacity(self, block_size: int) -> int:
        return len(self.blocks) * block_size

    def slot(self, position: int, block_size: int) -> int:
        return self.blocks[position // block_size] * block_size + position % block_size


class PagedKVCache:
    """Physical key/value storage plus the per-sequence tables that index it."""

    def __init__(
        self,
        model: ModelConfig,
        cache: CacheConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model_cfg = model
        self.cfg = cache
        self.block_size = cache.block_size
        self.dtype = _DTYPES[cache.dtype]
        self.device = torch.device(device)
        self.allocator = BlockAllocator(cache.num_blocks)
        self.tables: dict[int, BlockTable] = {}

        shape = (cache.num_slots, model.n_heads, model.d_head)
        # One flat tensor per layer for K and V. Flat storage makes both the
        # scatter (write) and the gather (read) a single index op.
        self.k_cache = [
            torch.zeros(shape, dtype=self.dtype, device=self.device)
            for _ in range(model.n_layers)
        ]
        self.v_cache = [
            torch.zeros(shape, dtype=self.dtype, device=self.device)
            for _ in range(model.n_layers)
        ]

    # -- capacity accounting ---------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free

    @property
    def utilization(self) -> float:
        return self.allocator.utilization

    def blocks_needed(self, seq_id: int, num_new_tokens: int) -> int:
        """Extra blocks required to append ``num_new_tokens`` to ``seq_id``."""
        table = self.tables.get(seq_id, BlockTable())
        total = table.num_tokens + num_new_tokens
        needed = (total + self.block_size - 1) // self.block_size
        return max(0, needed - len(table.blocks))

    def can_append(self, seq_id: int, num_new_tokens: int) -> bool:
        return self.blocks_needed(seq_id, num_new_tokens) <= self.allocator.num_free

    # -- sequence lifecycle ----------------------------------------------
    def append(self, seq_id: int, num_new_tokens: int) -> list[int]:
        """Reserve slots for ``num_new_tokens`` and return their physical ids."""
        table = self.tables.setdefault(seq_id, BlockTable())
        for block in self.allocator.allocate_many(
            self.blocks_needed(seq_id, num_new_tokens)
        ):
            table.blocks.append(block)
        start = table.num_tokens
        table.num_tokens += num_new_tokens
        return [
            table.slot(pos, self.block_size)
            for pos in range(start, start + num_new_tokens)
        ]

    def free(self, seq_id: int) -> None:
        table = self.tables.pop(seq_id, None)
        if table is None:
            return
        for block in table.blocks:
            self.allocator.free(block)

    def fork(self, parent_id: int, child_id: int) -> None:
        """Share the blocks of ``parent_id`` with a new sequence (copy-on-write)."""
        if child_id in self.tables:
            raise ValueError(f"sequence {child_id} already exists")
        parent = self.tables[parent_id]
        for block in parent.blocks:
            self.allocator.incref(block)
        self.tables[child_id] = BlockTable(list(parent.blocks), parent.num_tokens)

    def unshare_last_block(self, seq_id: int) -> None:
        """Copy the tail block if another sequence still references it."""
        table = self.tables[seq_id]
        if not table.blocks:
            return
        old = table.blocks[-1]
        if self.allocator.refcount(old) == 1:
            return
        new = self.allocator.allocate()
        lo, hi = old * self.block_size, (old + 1) * self.block_size
        nlo = new * self.block_size
        for layer in range(self.model_cfg.n_layers):
            self.k_cache[layer][nlo : nlo + self.block_size] = self.k_cache[layer][lo:hi]
            self.v_cache[layer][nlo : nlo + self.block_size] = self.v_cache[layer][lo:hi]
        table.blocks[-1] = new
        self.allocator.free(old)

    def num_tokens(self, seq_id: int) -> int:
        table = self.tables.get(seq_id)
        return 0 if table is None else table.num_tokens

    def truncate(self, seq_id: int, num_tokens: int) -> None:
        """Shrink a sequence to ``num_tokens``, releasing blocks past the end.

        Used by speculative decoding to roll back rejected draft tokens.
        """
        table = self.tables[seq_id]
        if num_tokens > table.num_tokens:
            raise ValueError("truncate cannot grow a sequence")
        keep = (num_tokens + self.block_size - 1) // self.block_size
        for block in table.blocks[keep:]:
            self.allocator.free(block)
        del table.blocks[keep:]
        table.num_tokens = num_tokens

    # -- tensor plumbing --------------------------------------------------
    def write(
        self,
        layer: int,
        slots: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Scatter ``(T, H, D)`` keys/values into the flat pool at ``slots``."""
        self.k_cache[layer].index_copy_(0, slots, keys.to(self.dtype))
        self.v_cache[layer].index_copy_(0, slots, values.to(self.dtype))

    def build_block_tables(
        self, seq_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded ``(B, max_blocks)`` block ids and ``(B,)`` context lengths."""
        tables = [self.tables[s] for s in seq_ids]
        max_blocks = max(max((len(t.blocks) for t in tables), default=1), 1)
        padded = torch.zeros(
            (len(tables), max_blocks), dtype=torch.long, device=self.device
        )
        for row, table in enumerate(tables):
            if table.blocks:
                padded[row, : len(table.blocks)] = torch.tensor(
                    table.blocks, dtype=torch.long, device=self.device
                )
        lens = torch.tensor(
            [t.num_tokens for t in tables], dtype=torch.long, device=self.device
        )
        return padded, lens

    def gather(
        self, layer: int, block_tables: torch.Tensor, max_context: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialise contiguous ``(B, max_context, H, D)`` K and V views.

        This is the honest CPU stand-in for a fused paged-attention kernel: we
        pay a real copy here, where CUDA would walk the block table inside the
        attention kernel and never materialise the gathered tensor.
        """
        bsz, _ = block_tables.shape
        offsets = torch.arange(self.block_size, device=block_tables.device)
        slots = (block_tables[:, :, None] * self.block_size + offsets).reshape(bsz, -1)
        slots = slots[:, :max_context]
        flat = slots.reshape(-1)
        keys = self.k_cache[layer].index_select(0, flat)
        values = self.v_cache[layer].index_select(0, flat)
        shape = (bsz, slots.size(1), self.model_cfg.n_heads, self.model_cfg.d_head)
        return keys.view(shape), values.view(shape)

    def reset(self) -> None:
        """Drop every sequence and return all blocks to the pool."""
        for seq_id in list(self.tables):
            self.free(seq_id)
