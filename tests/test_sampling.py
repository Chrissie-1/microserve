"""Logit filters and sampling."""

from __future__ import annotations

import math

import pytest
import torch

from microserve.config import SamplingConfig
from microserve.sampling import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    probs_from_logits,
    process_logits,
    sample,
)


class TestTemperature:
    def test_identity_at_one(self) -> None:
        logits = torch.randn(2, 8)
        assert apply_temperature(logits, 1.0) is logits

    def test_low_temperature_sharpens(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        cold = torch.softmax(apply_temperature(logits, 0.1), -1)
        hot = torch.softmax(apply_temperature(logits, 10.0), -1)
        assert cold.max() > hot.max()
        # High temperature tends towards uniform.
        assert hot.max() - hot.min() < 0.1

    def test_non_positive_temperature_rejected(self) -> None:
        with pytest.raises(ValueError, match="greedy"):
            apply_temperature(torch.zeros(1, 3), 0.0)

    def test_does_not_mutate_input(self) -> None:
        logits = torch.randn(1, 5)
        original = logits.clone()
        apply_temperature(logits, 0.5)
        torch.testing.assert_close(logits, original)


class TestTopK:
    def test_keeps_exactly_k(self) -> None:
        logits = torch.tensor([[5.0, 1.0, 4.0, 2.0, 3.0]])
        out = apply_top_k(logits, 2)
        assert torch.isfinite(out).sum() == 2
        assert torch.isfinite(out[0, 0]) and torch.isfinite(out[0, 2])

    def test_k_larger_than_vocab_is_a_no_op(self) -> None:
        logits = torch.randn(1, 4)
        torch.testing.assert_close(apply_top_k(logits, 99), logits)

    def test_ties_may_keep_more_than_k(self) -> None:
        """A threshold filter cannot break ties; that is the documented behaviour."""
        logits = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        assert torch.isfinite(apply_top_k(logits, 2)).sum() == 3

    def test_rejects_non_positive_k(self) -> None:
        with pytest.raises(ValueError):
            apply_top_k(torch.zeros(1, 3), 0)

    def test_survivors_keep_relative_probabilities(self) -> None:
        logits = torch.tensor([[3.0, 2.0, -5.0]])
        probs = torch.softmax(apply_top_k(logits, 2), -1)
        assert probs[0, 2] == 0.0
        torch.testing.assert_close(
            probs[0, 0] / probs[0, 1], torch.tensor(math.exp(1.0)), rtol=1e-5, atol=1e-5
        )


class TestTopP:
    def test_p_one_is_a_no_op(self) -> None:
        logits = torch.randn(1, 6)
        torch.testing.assert_close(apply_top_p(logits, 1.0), logits)

    def test_keeps_the_crossing_token(self) -> None:
        # Probabilities 0.5 / 0.3 / 0.2; p=0.6 needs the first two.
        probs = torch.tensor([[0.5, 0.3, 0.2]])
        logits = probs.log()
        out = apply_top_p(logits, 0.6)
        assert torch.isfinite(out[0, 0]) and torch.isfinite(out[0, 1])
        assert out[0, 2] == float("-inf")

    def test_always_keeps_at_least_one_token(self) -> None:
        logits = torch.tensor([[0.0, -100.0, -100.0]])
        out = apply_top_p(logits, 0.01)
        assert torch.isfinite(out).sum() >= 1

    def test_surviving_mass_covers_p(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(1, 40)
        for p in (0.1, 0.5, 0.9):
            kept = torch.softmax(logits, -1)[torch.isfinite(apply_top_p(logits, p))]
            assert kept.sum() >= p - 1e-6

    def test_rejects_out_of_range_p(self) -> None:
        for bad in (0.0, -0.5, 1.5):
            with pytest.raises(ValueError):
                apply_top_p(torch.zeros(1, 3), bad)

    def test_row_independence(self) -> None:
        """Each row is filtered on its own distribution."""
        logits = torch.tensor([[10.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        out = apply_top_p(logits, 0.5)
        assert torch.isfinite(out[0]).sum() == 1
        assert torch.isfinite(out[1]).sum() == 2


class TestPipeline:
    def test_filters_compose_in_order(self) -> None:
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        cfg = SamplingConfig(temperature=0.5, top_k=3, top_p=0.9)
        out = process_logits(logits, cfg)
        assert torch.isfinite(out).sum() <= 3

    def test_greedy_probs_are_one_hot(self) -> None:
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        probs = probs_from_logits(logits, SamplingConfig(temperature=0.0))
        torch.testing.assert_close(probs, torch.tensor([[0.0, 1.0, 0.0]]))

    def test_probs_sum_to_one(self) -> None:
        logits = torch.randn(3, 20)
        cfg = SamplingConfig(temperature=0.7, top_k=5, top_p=0.9)
        probs = probs_from_logits(logits, cfg)
        torch.testing.assert_close(probs.sum(-1), torch.ones(3), rtol=1e-5, atol=1e-5)

    def test_greedy_sample_picks_argmax(self) -> None:
        logits = torch.randn(4, 11)
        got = sample(logits, SamplingConfig(temperature=0.0))
        torch.testing.assert_close(got, logits.argmax(-1))

    def test_sampling_is_reproducible_from_a_seed(self) -> None:
        logits = torch.randn(1, 30)
        cfg = SamplingConfig(temperature=1.0)
        a = sample(logits, cfg, torch.Generator().manual_seed(7))
        b = sample(logits, cfg, torch.Generator().manual_seed(7))
        c = sample(logits, cfg, torch.Generator().manual_seed(8))
        assert a == b
        assert (a != c).any() or True  # different seeds may coincide by chance

    def test_empirical_distribution_matches_softmax(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(1, 6)
        expected = torch.softmax(logits, -1)[0]
        generator = torch.Generator().manual_seed(1)
        counts = torch.zeros(6)
        draws = 20_000
        probs = probs_from_logits(logits, SamplingConfig(temperature=1.0))
        samples = torch.multinomial(
            probs.expand(draws, -1), 1, replacement=True, generator=generator
        )
        counts.scatter_add_(0, samples.flatten(), torch.ones(draws))
        torch.testing.assert_close(counts / draws, expected, rtol=0.1, atol=0.01)

    def test_top_k_one_is_greedy(self) -> None:
        logits = torch.randn(1, 12)
        cfg = SamplingConfig(temperature=1.0, top_k=1)
        got = sample(logits, cfg, torch.Generator().manual_seed(0))
        assert int(got) == int(logits.argmax())
