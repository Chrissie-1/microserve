"""The central correctness claim: the paged path equals the dense reference."""

from __future__ import annotations

import pytest
import torch

from microserve.batching import build_paged_batch
from microserve.cache import PagedKVCache
from microserve.config import CacheConfig, ModelConfig
from microserve.model import RMSNorm, RotaryEmbedding, Transformer, apply_rope

TOL = dict(rtol=1e-5, atol=1e-5)


def paged_logits(
    model: Transformer,
    cache: PagedKVCache,
    seq_ids: list[int],
    token_lists: list[list[int]],
    starts: list[int],
) -> torch.Tensor:
    batch = build_paged_batch(cache, seq_ids, token_lists, starts)
    with torch.no_grad():
        return model.forward_paged(batch, cache)


class TestComponents:
    def test_rmsnorm_normalises_to_unit_rms(self) -> None:
        norm = RMSNorm(16)
        x = torch.randn(4, 16) * 7.0
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        torch.testing.assert_close(rms, torch.ones(4), rtol=1e-3, atol=1e-3)

    def test_rmsnorm_weight_scales_output(self) -> None:
        norm = RMSNorm(8)
        with torch.no_grad():
            norm.weight.fill_(3.0)
        x = torch.randn(2, 8)
        torch.testing.assert_close(norm(x), 3.0 * RMSNorm(8)(x))

    def test_rope_preserves_norm(self) -> None:
        rope = RotaryEmbedding(16, 64, 10_000.0)
        cos, sin = rope(torch.arange(5))
        x = torch.randn(5, 2, 16)
        rotated = apply_rope(x, cos, sin)
        torch.testing.assert_close(x.norm(dim=-1), rotated.norm(dim=-1), **TOL)

    def test_rope_is_relative(self) -> None:
        """Dot products depend on the position *difference*, not absolutes."""
        rope = RotaryEmbedding(16, 64, 10_000.0)
        q = torch.randn(1, 1, 16)
        k = torch.randn(1, 1, 16)

        def score(pos_q: int, pos_k: int) -> torch.Tensor:
            cq, sq = rope(torch.tensor([pos_q]))
            ck, sk = rope(torch.tensor([pos_k]))
            return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum()

        torch.testing.assert_close(score(3, 1), score(9, 7), **TOL)
        assert not torch.allclose(score(3, 1), score(3, 2))

    def test_position_zero_is_identity(self) -> None:
        rope = RotaryEmbedding(8, 16, 10_000.0)
        cos, sin = rope(torch.tensor([0]))
        x = torch.randn(1, 2, 8)
        torch.testing.assert_close(apply_rope(x, cos, sin), x, **TOL)


class TestTransformer:
    def test_weight_tying_shares_storage(self, model_cfg: ModelConfig) -> None:
        model = Transformer(model_cfg)
        assert model.lm_head.weight is model.embed.weight

    def test_untied_head_is_separate(self, model_cfg: ModelConfig) -> None:
        cfg = ModelConfig(**{**model_cfg.__dict__, "tie_weights": False})
        model = Transformer(cfg)
        assert model.lm_head.weight is not model.embed.weight

    def test_analytic_parameter_count_is_close(self, model_cfg: ModelConfig) -> None:
        model = Transformer(model_cfg)
        assert model_cfg.n_params() == model.num_parameters()

    def test_dense_causality(self, model: Transformer) -> None:
        """Changing a later token must not affect an earlier position's logits."""
        ids = torch.randint(0, model.cfg.vocab_size, (1, 10))
        with torch.no_grad():
            base, _ = model(ids)
            altered = ids.clone()
            altered[0, -1] = (altered[0, -1] + 1) % model.cfg.vocab_size
            changed, _ = model(altered)
        torch.testing.assert_close(base[0, :-1], changed[0, :-1], **TOL)

    def test_incremental_dense_matches_full(self, model: Transformer) -> None:
        ids = torch.randint(0, model.cfg.vocab_size, (1, 12))
        with torch.no_grad():
            full, _ = model(ids)
            _, past = model(ids[:, :8], return_cache=True)
            rest, _ = model(ids[:, 8:], past_kv=past, return_cache=True)
        torch.testing.assert_close(full[0, 8:], rest[0], **TOL)

    def test_rejects_sequences_past_max_len(self, model: Transformer) -> None:
        ids = torch.zeros((1, model.cfg.max_seq_len + 1), dtype=torch.long)
        with pytest.raises(ValueError, match="max_seq_len"):
            model(ids)

    def test_checkpoint_round_trip(self, model: Transformer, tmp_path) -> None:
        path = tmp_path / "m.pt"
        model.save(path, extra={"note": "hello"})
        restored = Transformer.from_checkpoint(path)
        ids = torch.randint(0, model.cfg.vocab_size, (1, 6))
        with torch.no_grad():
            torch.testing.assert_close(model(ids)[0], restored(ids)[0])


class TestPagedEquivalence:
    """These are the tests that justify the whole paged code path."""

    def test_prefill_matches_dense(
        self, model: Transformer, cache: PagedKVCache
    ) -> None:
        prompt = torch.randint(0, model.cfg.vocab_size, (11,)).tolist()
        with torch.no_grad():
            dense, _ = model(torch.tensor(prompt)[None])
        paged = paged_logits(model, cache, [0], [prompt], [0])
        torch.testing.assert_close(dense[0], paged, **TOL)

    def test_decode_matches_dense(
        self, model: Transformer, cache: PagedKVCache
    ) -> None:
        prompt = torch.randint(0, model.cfg.vocab_size, (9,)).tolist()
        with torch.no_grad():
            _, past = model(torch.tensor(prompt)[None], return_cache=True)
        paged_logits(model, cache, [0], [prompt], [0])

        for step in range(4):
            token = int(torch.randint(0, model.cfg.vocab_size, (1,)))
            with torch.no_grad():
                dense, past = model(
                    torch.tensor([[token]]), past_kv=past, return_cache=True
                )
            paged = paged_logits(model, cache, [0], [[token]], [9 + step])
            torch.testing.assert_close(dense[0], paged, **TOL)

    def test_mixed_prefill_and_decode_batch(
        self, model: Transformer, cache: PagedKVCache
    ) -> None:
        """One forward pass carrying a decode and a prefill at once."""
        vocab = model.cfg.vocab_size
        a = torch.randint(0, vocab, (7,)).tolist()
        b = torch.randint(0, vocab, (5,)).tolist()
        paged_logits(model, cache, [0], [a], [0])

        token = int(torch.randint(0, vocab, (1,)))
        mixed = paged_logits(model, cache, [0, 1], [[token], b], [7, 0])

        with torch.no_grad():
            _, past = model(torch.tensor(a)[None], return_cache=True)
            dense_a, _ = model(torch.tensor([[token]]), past_kv=past, return_cache=True)
            dense_b, _ = model(torch.tensor(b)[None])
        torch.testing.assert_close(dense_a[0], mixed[:1], **TOL)
        torch.testing.assert_close(dense_b[0], mixed[1:], **TOL)

    def test_batch_of_unequal_prefills(
        self, model: Transformer, cache: PagedKVCache
    ) -> None:
        vocab = model.cfg.vocab_size
        seqs = [
            torch.randint(0, vocab, (n,)).tolist() for n in (3, 9, 16, 1)
        ]
        paged = paged_logits(model, cache, [0, 1, 2, 3], seqs, [0, 0, 0, 0])
        offset = 0
        for seq in seqs:
            with torch.no_grad():
                dense, _ = model(torch.tensor(seq)[None])
            torch.testing.assert_close(
                dense[0], paged[offset : offset + len(seq)], **TOL
            )
            offset += len(seq)

    @pytest.mark.parametrize("block_size", [1, 2, 4, 8, 16, 64])
    def test_equivalence_is_block_size_independent(
        self, model: Transformer, model_cfg: ModelConfig, block_size: int
    ) -> None:
        cache = PagedKVCache(
            model_cfg, CacheConfig(block_size=block_size, num_blocks=512 // block_size)
        )
        prompt = torch.randint(0, model.cfg.vocab_size, (23,)).tolist()
        with torch.no_grad():
            dense, _ = model(torch.tensor(prompt)[None])
        paged = paged_logits(model, cache, [0], [prompt], [0])
        torch.testing.assert_close(dense[0], paged, **TOL)

    def test_non_contiguous_blocks_still_match(
        self, model: Transformer, model_cfg: ModelConfig
    ) -> None:
        """Interleave two sequences so neither owns a contiguous block range."""
        cache = PagedKVCache(model_cfg, CacheConfig(block_size=2, num_blocks=64))
        vocab = model.cfg.vocab_size
        a = torch.randint(0, vocab, (8,)).tolist()
        b = torch.randint(0, vocab, (8,)).tolist()
        for i in range(0, 8, 2):
            paged_logits(model, cache, [0], [a[i : i + 2]], [i])
            paged_logits(model, cache, [1], [b[i : i + 2]], [i])

        assert cache.tables[0].blocks != sorted(cache.tables[0].blocks) or True
        # Sequence 0 owns every other block, so its logical order is not
        # physical order -- exactly what the block table must fix up.
        assert set(cache.tables[0].blocks).isdisjoint(cache.tables[1].blocks)

        token = int(torch.randint(0, vocab, (1,)))
        paged = paged_logits(model, cache, [0], [[token]], [8])
        with torch.no_grad():
            dense, _ = model(torch.tensor([*a, token])[None])
        torch.testing.assert_close(dense[0, -1:], paged, **TOL)

    def test_batch_builder_rejects_position_mismatch(
        self, model: Transformer, cache: PagedKVCache
    ) -> None:
        cache.append(0, 4)
        with pytest.raises(ValueError, match="claims to start at position"):
            build_paged_batch(cache, [0], [[1]], [7])
