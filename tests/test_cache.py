"""Block allocator and paged cache invariants."""

from __future__ import annotations

import pytest
import torch

from microserve.cache import BlockAllocator, OutOfCacheMemory, PagedKVCache
from microserve.config import CacheConfig, ModelConfig


class TestBlockAllocator:
    def test_allocate_and_free_round_trips(self) -> None:
        alloc = BlockAllocator(4)
        assert alloc.num_free == 4
        blocks = alloc.allocate_many(3)
        assert len(set(blocks)) == 3
        assert alloc.num_free == 1
        assert alloc.num_used == 3
        for block in blocks:
            alloc.free(block)
        assert alloc.num_free == 4
        alloc.check_invariants()

    def test_exhaustion_raises(self) -> None:
        alloc = BlockAllocator(2)
        alloc.allocate_many(2)
        with pytest.raises(OutOfCacheMemory):
            alloc.allocate()
        with pytest.raises(OutOfCacheMemory):
            alloc.allocate_many(1)

    def test_double_free_raises(self) -> None:
        alloc = BlockAllocator(2)
        block = alloc.allocate()
        alloc.free(block)
        with pytest.raises(ValueError, match="double free"):
            alloc.free(block)

    def test_refcounting_defers_release(self) -> None:
        alloc = BlockAllocator(2)
        block = alloc.allocate()
        alloc.incref(block)
        assert alloc.refcount(block) == 2
        alloc.free(block)
        # Still held by the second reference.
        assert alloc.num_free == 1
        assert alloc.refcount(block) == 1
        alloc.free(block)
        assert alloc.num_free == 2
        alloc.check_invariants()

    def test_incref_on_free_block_raises(self) -> None:
        alloc = BlockAllocator(1)
        with pytest.raises(ValueError, match="cannot incref"):
            alloc.incref(0)

    def test_zero_blocks_rejected(self) -> None:
        with pytest.raises(ValueError):
            BlockAllocator(0)

    def test_utilization_tracks_usage(self) -> None:
        alloc = BlockAllocator(4)
        assert alloc.utilization == 0.0
        alloc.allocate_many(2)
        assert alloc.utilization == 0.5
        alloc.allocate_many(2)
        assert alloc.utilization == 1.0


class TestPagedKVCache:
    def test_slots_are_contiguous_within_a_block(self, cache: PagedKVCache) -> None:
        slots = cache.append(0, cache.block_size)
        assert slots == list(range(slots[0], slots[0] + cache.block_size))

    def test_append_allocates_only_what_is_needed(self, cache: PagedKVCache) -> None:
        before = cache.num_free_blocks
        cache.append(0, 1)
        assert cache.num_free_blocks == before - 1
        # The rest of that block is free capacity, no new block required.
        cache.append(0, cache.block_size - 1)
        assert cache.num_free_blocks == before - 1
        cache.append(0, 1)
        assert cache.num_free_blocks == before - 2

    def test_blocks_needed_matches_actual_allocation(self, cache: PagedKVCache) -> None:
        for count in (1, 5, 8, 9, 17):
            predicted = cache.blocks_needed(0, count)
            before = cache.num_free_blocks
            cache.append(0, count)
            assert before - cache.num_free_blocks == predicted

    def test_free_returns_every_block(self, cache: PagedKVCache) -> None:
        total = cache.num_free_blocks
        cache.append(1, 40)
        cache.append(2, 7)
        cache.free(1)
        cache.free(2)
        assert cache.num_free_blocks == total
        cache.allocator.check_invariants()

    def test_free_of_unknown_sequence_is_a_no_op(self, cache: PagedKVCache) -> None:
        total = cache.num_free_blocks
        cache.free(999)
        assert cache.num_free_blocks == total

    def test_can_append_reports_exhaustion(self, model_cfg: ModelConfig) -> None:
        cache = PagedKVCache(model_cfg, CacheConfig(block_size=4, num_blocks=2))
        assert cache.can_append(0, 8)
        assert not cache.can_append(0, 9)

    def test_write_then_gather_round_trips(self, cache: PagedKVCache) -> None:
        n = 20
        slots = torch.tensor(cache.append(0, n))
        shape = (n, cache.model_cfg.n_heads, cache.model_cfg.d_head)
        keys = torch.randn(shape)
        values = torch.randn(shape)
        cache.write(0, slots, keys, values)

        block_tables, lens = cache.build_block_tables([0])
        assert int(lens[0]) == n
        got_k, got_v = cache.gather(0, block_tables, n)
        torch.testing.assert_close(got_k[0, :n], keys)
        torch.testing.assert_close(got_v[0, :n], values)

    def test_gather_is_per_layer(self, cache: PagedKVCache) -> None:
        slots = torch.tensor(cache.append(0, 4))
        shape = (4, cache.model_cfg.n_heads, cache.model_cfg.d_head)
        cache.write(0, slots, torch.ones(shape), torch.ones(shape))
        block_tables, _ = cache.build_block_tables([0])
        layer0, _ = cache.gather(0, block_tables, 4)
        layer1, _ = cache.gather(1, block_tables, 4)
        assert layer0[0, :4].eq(1).all()
        assert layer1[0, :4].eq(0).all()

    def test_truncate_releases_tail_blocks(self, cache: PagedKVCache) -> None:
        total = cache.num_free_blocks
        cache.append(0, 24)  # 3 blocks of 8
        assert cache.num_free_blocks == total - 3
        cache.truncate(0, 9)  # needs 2 blocks
        assert cache.num_free_blocks == total - 2
        assert cache.num_tokens(0) == 9
        cache.allocator.check_invariants()

    def test_truncate_cannot_grow(self, cache: PagedKVCache) -> None:
        cache.append(0, 4)
        with pytest.raises(ValueError, match="cannot grow"):
            cache.truncate(0, 5)

    def test_append_after_truncate_reuses_the_partial_block(
        self, cache: PagedKVCache
    ) -> None:
        cache.append(0, 24)
        cache.truncate(0, 9)
        slots = cache.append(0, 3)
        assert cache.num_tokens(0) == 12
        assert len(slots) == 3

    def test_fork_shares_blocks(self, cache: PagedKVCache) -> None:
        cache.append(0, 16)
        free_before = cache.num_free_blocks
        cache.fork(0, 1)
        # Sharing is free: no new blocks.
        assert cache.num_free_blocks == free_before
        assert cache.tables[0].blocks == cache.tables[1].blocks
        for block in cache.tables[0].blocks:
            assert cache.allocator.refcount(block) == 2

    def test_fork_then_free_parent_keeps_child_alive(self, cache: PagedKVCache) -> None:
        cache.append(0, 16)
        cache.fork(0, 1)
        blocks = list(cache.tables[1].blocks)
        cache.free(0)
        for block in blocks:
            assert cache.allocator.refcount(block) == 1
        cache.free(1)
        cache.allocator.check_invariants()

    def test_fork_onto_existing_sequence_raises(self, cache: PagedKVCache) -> None:
        cache.append(0, 4)
        cache.append(1, 4)
        with pytest.raises(ValueError, match="already exists"):
            cache.fork(0, 1)

    def test_copy_on_write_unshares_the_tail(self, cache: PagedKVCache) -> None:
        n = 12
        slots = torch.tensor(cache.append(0, n))
        shape = (n, cache.model_cfg.n_heads, cache.model_cfg.d_head)
        keys = torch.randn(shape)
        cache.write(0, slots, keys, keys)
        cache.fork(0, 1)

        old_tail = cache.tables[1].blocks[-1]
        cache.unshare_last_block(1)
        new_tail = cache.tables[1].blocks[-1]
        assert new_tail != old_tail
        assert cache.allocator.refcount(old_tail) == 1
        # The copy must preserve the data that was already there.
        block_tables, _ = cache.build_block_tables([1])
        got_k, _ = cache.gather(0, block_tables, n)
        torch.testing.assert_close(got_k[0, :n], keys)

    def test_unshare_is_a_no_op_when_unique(self, cache: PagedKVCache) -> None:
        cache.append(0, 12)
        tail = cache.tables[0].blocks[-1]
        free_before = cache.num_free_blocks
        cache.unshare_last_block(0)
        assert cache.tables[0].blocks[-1] == tail
        assert cache.num_free_blocks == free_before

    def test_reset_releases_everything(self, cache: PagedKVCache) -> None:
        total = cache.num_free_blocks
        for seq in range(4):
            cache.append(seq, 10)
        cache.reset()
        assert cache.num_free_blocks == total
        assert cache.tables == {}
        cache.allocator.check_invariants()

    def test_bytes_for_matches_tensor_sizes(self, model_cfg: ModelConfig) -> None:
        cfg = CacheConfig(block_size=8, num_blocks=16)
        cache = PagedKVCache(model_cfg, cfg)
        actual = sum(t.numel() * t.element_size() for t in cache.k_cache + cache.v_cache)
        assert cfg.bytes_for(model_cfg) == actual
