"""Rejection sampling maths, and end-to-end speculative equivalence.

The important test here is :meth:`TestRejectionSampling.test_emitted_token_follows_the_target`:
speculative decoding is only useful if it provably does not change the output
distribution, and that is a property of the accept/reject rule alone, testable
without any model at all.
"""

from __future__ import annotations

import pytest
import torch

from microserve.cache import PagedKVCache
from microserve.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SamplingConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from microserve.engine import LLMEngine
from microserve.model import Transformer
from microserve.speculative import SpeculativeDecoder, verify_proposals
from microserve.stats import chi_square_two_sample

from .conftest import dense_greedy


def random_distribution(vocab: int, generator: torch.Generator) -> torch.Tensor:
    weights = torch.rand(vocab, generator=generator) + 1e-3
    return weights / weights.sum()


class TestRejectionSampling:
    def test_accepts_when_target_dominates(self) -> None:
        """p_target >= p_draft gives an acceptance ratio of 1, so never rejects."""
        draft = torch.tensor([[0.5, 0.5]])
        target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        result = verify_proposals(draft, target, [0], torch.Generator().manual_seed(0))
        assert result.num_accepted == 1
        assert result.tokens[0] == 0

    def test_rejects_impossible_token(self) -> None:
        """A token the target gives zero mass can never be accepted."""
        draft = torch.tensor([[0.0, 1.0]])
        target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        result = verify_proposals(draft, target, [1], torch.Generator().manual_seed(0))
        assert result.num_accepted == 0
        assert result.tokens == [0]

    def test_identical_distributions_always_accept(self) -> None:
        gen = torch.Generator().manual_seed(0)
        probs = random_distribution(8, gen)
        draft = probs.unsqueeze(0).repeat(3, 1)
        target = probs.unsqueeze(0).repeat(4, 1)
        for _ in range(20):
            result = verify_proposals(draft, target, [0, 1, 2], gen)
            assert result.num_accepted == 3
            assert len(result.tokens) == 4  # three accepted plus the bonus

    def test_emits_at_most_gamma_plus_one(self) -> None:
        gen = torch.Generator().manual_seed(3)
        for gamma in (1, 2, 5):
            draft = torch.stack([random_distribution(6, gen) for _ in range(gamma)])
            target = torch.stack([random_distribution(6, gen) for _ in range(gamma + 1)])
            proposals = [int(torch.multinomial(d, 1, generator=gen)) for d in draft]
            result = verify_proposals(draft, target, proposals, gen)
            assert 1 <= len(result.tokens) <= gamma + 1
            assert len(result.tokens) == result.num_accepted + 1
            assert result.num_proposed == gamma

    def test_accepted_prefix_matches_the_proposals(self) -> None:
        gen = torch.Generator().manual_seed(11)
        draft = torch.stack([random_distribution(10, gen) for _ in range(4)])
        target = torch.stack([random_distribution(10, gen) for _ in range(5)])
        proposals = [int(torch.multinomial(d, 1, generator=gen)) for d in draft]
        result = verify_proposals(draft, target, proposals, gen)
        assert result.tokens[: result.num_accepted] == proposals[: result.num_accepted]

    def test_shape_mismatch_is_rejected(self) -> None:
        draft = torch.ones(2, 4) / 4
        target = torch.ones(2, 4) / 4  # should be 3 rows for gamma=2
        with pytest.raises(ValueError, match="do not match"):
            verify_proposals(draft, target, [0, 1])

    def test_emitted_token_follows_the_target(self) -> None:
        """The heart of the proof, checked empirically on synthetic distributions.

        With gamma=1 the first emitted token must be distributed exactly as
        ``p_target``, no matter how bad the draft is.
        """
        vocab = 8
        gen = torch.Generator().manual_seed(5)
        p_draft = random_distribution(vocab, gen)
        p_target = random_distribution(vocab, gen)
        draft = p_draft.unsqueeze(0)
        target = torch.stack([p_target, p_target])

        draws = 30_000
        proposals = torch.multinomial(
            p_draft.expand(draws, -1), 1, replacement=True, generator=gen
        ).flatten()
        emitted = [
            verify_proposals(draft, target, [int(proposals[i])], gen).tokens[0]
            for i in range(draws)
        ]
        reference = torch.multinomial(
            p_target.expand(draws, -1), 1, replacement=True, generator=gen
        ).flatten().tolist()

        result = chi_square_two_sample(reference, emitted)
        assert result.p_value > 0.01, f"speculative output is biased: {result}"

    def test_biased_rule_would_be_detected(self) -> None:
        """Control: a deliberately wrong rule must fail the same test.

        Without this, a test that always passes would look like a proof.
        """
        vocab = 8
        gen = torch.Generator().manual_seed(5)
        p_draft = random_distribution(vocab, gen)
        p_target = random_distribution(vocab, gen)
        draws = 30_000
        # "Accept everything" -- the naive, wrong implementation.
        biased = torch.multinomial(
            p_draft.expand(draws, -1), 1, replacement=True, generator=gen
        ).flatten().tolist()
        reference = torch.multinomial(
            p_target.expand(draws, -1), 1, replacement=True, generator=gen
        ).flatten().tolist()
        assert chi_square_two_sample(reference, biased).p_value < 0.01


class TestSpeculativeDecoder:
    def test_rejects_gamma_below_one(
        self, model: Transformer, draft: Transformer, cache_cfg: CacheConfig
    ) -> None:
        with pytest.raises(ValueError, match="gamma"):
            SpeculativeDecoder(model, draft, cache_cfg, gamma=0)

    def test_rejects_mismatched_vocabularies(
        self, model: Transformer, cache_cfg: CacheConfig
    ) -> None:
        other = Transformer(ModelConfig(vocab_size=7, n_layers=1, d_model=8, n_heads=2))
        with pytest.raises(ValueError, match="vocabulary"):
            SpeculativeDecoder(model, other, cache_cfg)

    @pytest.mark.parametrize("gamma", [1, 2, 3, 5])
    def test_greedy_output_matches_dense_reference(
        self,
        model: Transformer,
        draft: Transformer,
        cache_cfg: CacheConfig,
        gamma: int,
    ) -> None:
        prompt = torch.randint(0, model.cfg.vocab_size, (6,)).tolist()
        expected = dense_greedy(model, prompt, 20)
        decoder = SpeculativeDecoder(model, draft, cache_cfg, gamma=gamma)
        got = decoder.generate(
            prompt, SamplingConfig(temperature=0.0, max_new_tokens=20)
        )
        assert got == expected

    def test_draft_and_target_disagree_enough_to_matter(
        self, model: Transformer, draft: Transformer, cache_cfg: CacheConfig
    ) -> None:
        """Guards the test above: with 100% acceptance it would prove nothing."""
        prompt = torch.randint(0, model.cfg.vocab_size, (6,)).tolist()
        decoder = SpeculativeDecoder(model, draft, cache_cfg, gamma=4)
        decoder.generate(prompt, SamplingConfig(temperature=0.0, max_new_tokens=20))
        rate = decoder.num_accepted / decoder.num_proposed
        assert 0.0 < rate < 0.95

    def test_caches_are_released_after_generation(
        self, model: Transformer, draft: Transformer, cache_cfg: CacheConfig
    ) -> None:
        decoder = SpeculativeDecoder(model, draft, cache_cfg, gamma=3)
        cache = PagedKVCache(model.cfg, cache_cfg)
        free_before = cache.num_free_blocks
        decoder.generate(
            [1, 2, 3], SamplingConfig(temperature=0.0, max_new_tokens=12),
            target_cache=cache,
        )
        assert cache.num_free_blocks == free_before
        assert decoder.draft_cache.num_free_blocks == decoder.draft_cache.cfg.num_blocks
        cache.allocator.check_invariants()
        decoder.draft_cache.allocator.check_invariants()

    def test_rollback_restores_the_engine_invariant(
        self, model: Transformer, draft: Transformer, cache_cfg: CacheConfig
    ) -> None:
        """After each round the cache holds every token but the newest."""
        from microserve.scheduler import Request

        decoder = SpeculativeDecoder(model, draft, cache_cfg, gamma=4)
        cache = PagedKVCache(model.cfg, cache_cfg)
        request = Request(0, [1, 2, 3, 4], SamplingConfig(temperature=0.0, max_new_tokens=40))
        generators = {0: torch.Generator().manual_seed(0)}
        for _ in range(5):
            tokens = decoder.step([request], cache, generators)[0]
            request.output_ids.extend(tokens)
            assert cache.num_tokens(0) == request.num_tokens - 1
            assert request.num_computed == request.num_tokens - 1
        cache.free(0)
        decoder.free(0)


class TestEngineIntegration:
    @pytest.mark.parametrize("gamma", [1, 3, 5])
    def test_engine_speculative_matches_dense(
        self,
        model: Transformer,
        draft: Transformer,
        model_cfg: ModelConfig,
        cache_cfg: CacheConfig,
        gamma: int,
    ) -> None:
        prompts = [
            torch.randint(0, model.cfg.vocab_size, (n,)).tolist() for n in (4, 9, 6)
        ]
        expected = [dense_greedy(model, p, 16) for p in prompts]

        cfg = EngineConfig(
            model=model_cfg,
            cache=CacheConfig(block_size=8, num_blocks=128),
            scheduler=SchedulerConfig(max_batch_size=3),
            speculative=SpeculativeConfig(enabled=True, gamma=gamma),
        )
        engine = LLMEngine(model, cfg, draft_model=draft)
        requests = [
            engine.add_request(p, SamplingConfig(temperature=0.0, max_new_tokens=16))
            for p in prompts
        ]
        outputs = {o.request_id: o for o in engine.run()}
        for request, want in zip(requests, expected, strict=True):
            assert outputs[request.request_id].output_ids == want

        assert engine.stats.acceptance_rate is not None
        assert engine.cache.num_free_blocks == cfg.cache.num_blocks
        assert (
            engine.spec.draft_cache.num_free_blocks == cfg.cache.num_blocks
        )
        engine.cache.allocator.check_invariants()

    def test_never_exceeds_the_token_budget(
        self,
        model: Transformer,
        draft: Transformer,
        model_cfg: ModelConfig,
    ) -> None:
        """A round emits up to gamma+1 tokens; the budget must still be exact."""
        cfg = EngineConfig(
            model=model_cfg,
            cache=CacheConfig(block_size=8, num_blocks=128),
            scheduler=SchedulerConfig(max_batch_size=2),
            speculative=SpeculativeConfig(enabled=True, gamma=5),
        )
        engine = LLMEngine(model, cfg, draft_model=draft)
        for budget in (1, 2, 3, 7, 13):
            engine.add_request(
                [1, 2, 3], SamplingConfig(temperature=0.0, max_new_tokens=budget)
            )
        for output in engine.run():
            budget = next(
                r.sampling.max_new_tokens
                for r in engine.scheduler.finished
                if r.request_id == output.request_id
            )
            assert len(output.output_ids) == budget
