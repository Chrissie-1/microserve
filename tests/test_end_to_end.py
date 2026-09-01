"""Whole-engine behaviour: correctness, determinism, and resource hygiene."""

from __future__ import annotations

import pytest
import torch

from microserve.cache import OutOfCacheMemory
from microserve.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SamplingConfig,
    SchedulerConfig,
)
from microserve.engine import LLMEngine
from microserve.model import Transformer
from microserve.tokenizer import CharTokenizer

from .conftest import dense_greedy


def build_engine(
    model: Transformer,
    model_cfg: ModelConfig,
    *,
    block_size: int = 8,
    num_blocks: int = 64,
    batching: str = "continuous",
    policy: str = "fcfs",
    max_batch_size: int = 4,
    tokenizer: CharTokenizer | None = None,
) -> LLMEngine:
    cfg = EngineConfig(
        model=model_cfg,
        cache=CacheConfig(block_size=block_size, num_blocks=num_blocks),
        scheduler=SchedulerConfig(
            batching=batching, policy=policy, max_batch_size=max_batch_size
        ),
    )
    return LLMEngine(model, cfg, tokenizer=tokenizer)


@pytest.fixture
def prompts(model: Transformer) -> list[list[int]]:
    torch.manual_seed(3)
    return [
        torch.randint(0, model.cfg.vocab_size, (n,)).tolist() for n in (5, 17, 3, 12)
    ]


class TestGreedyEquivalence:
    """The engine must reproduce the dense reference under every configuration."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"block_size": 1},
            {"block_size": 32},
            {"batching": "static"},
            {"policy": "sjf"},
            {"policy": "lifo"},
            {"max_batch_size": 1},
            {"num_blocks": 12},  # tight enough to force preemption
        ],
        ids=[
            "default", "block1", "block32", "static",
            "sjf", "lifo", "batch1", "tight-cache",
        ],
    )
    def test_matches_dense_reference(
        self,
        model: Transformer,
        model_cfg: ModelConfig,
        prompts: list[list[int]],
        kwargs: dict[str, object],
    ) -> None:
        budgets = [11, 6, 18, 4]
        expected = [dense_greedy(model, p, n) for p, n in zip(prompts, budgets, strict=True)]
        engine = build_engine(model, model_cfg, **kwargs)  # type: ignore[arg-type]
        requests = [
            engine.add_request(p, SamplingConfig(temperature=0.0, max_new_tokens=n))
            for p, n in zip(prompts, budgets, strict=True)
        ]
        outputs = {o.request_id: o for o in engine.run()}
        assert len(outputs) == len(prompts)
        for request, want in zip(requests, expected, strict=True):
            assert outputs[request.request_id].output_ids == want

    def test_preemption_actually_happens_in_the_tight_case(
        self, model: Transformer, model_cfg: ModelConfig, prompts: list[list[int]]
    ) -> None:
        """Guards the parametrised case above: without this it proves nothing."""
        engine = build_engine(model, model_cfg, num_blocks=12)
        for p in prompts:
            engine.add_request(p, SamplingConfig(temperature=0.0, max_new_tokens=18))
        engine.run()
        assert engine.stats.num_preemptions > 0


class TestDeterminism:
    def test_same_seed_gives_the_same_tokens(
        self, model: Transformer, model_cfg: ModelConfig, prompts: list[list[int]]
    ) -> None:
        def run(seed: int) -> list[list[int]]:
            engine = build_engine(model, model_cfg)
            for p in prompts:
                engine.add_request(
                    p, SamplingConfig(temperature=1.0, max_new_tokens=12, seed=seed)
                )
            outputs = sorted(engine.run(), key=lambda o: o.request_id)
            return [o.output_ids for o in outputs]

        assert run(99) == run(99)

    def test_different_seeds_diverge(
        self, model: Transformer, model_cfg: ModelConfig, prompts: list[list[int]]
    ) -> None:
        def run(seed: int) -> list[list[int]]:
            engine = build_engine(model, model_cfg)
            for p in prompts:
                engine.add_request(
                    p, SamplingConfig(temperature=1.0, max_new_tokens=16, seed=seed)
                )
            return [o.output_ids for o in sorted(engine.run(), key=lambda o: o.request_id)]

        assert run(1) != run(2)

    def test_output_is_independent_of_batch_composition(
        self, model: Transformer, model_cfg: ModelConfig, prompts: list[list[int]]
    ) -> None:
        """A request must not be affected by who it shares a batch with."""
        alone = build_engine(model, model_cfg, max_batch_size=1)
        alone.add_request(prompts[1], SamplingConfig(temperature=0.0, max_new_tokens=15))
        solo = alone.run()[0].output_ids

        crowded = build_engine(model, model_cfg, max_batch_size=8)
        target = crowded.add_request(
            prompts[1], SamplingConfig(temperature=0.0, max_new_tokens=15)
        )
        for p in prompts:
            crowded.add_request(p, SamplingConfig(temperature=0.0, max_new_tokens=15))
        outputs = {o.request_id: o for o in crowded.run()}
        assert outputs[target.request_id].output_ids == solo


class TestResourceHygiene:
    def test_all_blocks_are_returned(
        self, model: Transformer, model_cfg: ModelConfig, prompts: list[list[int]]
    ) -> None:
        engine = build_engine(model, model_cfg)
        for p in prompts:
            engine.add_request(p, SamplingConfig(temperature=0.0, max_new_tokens=9))
        engine.run()
        assert engine.cache.num_free_blocks == engine.config.cache.num_blocks
        assert engine.cache.tables == {}
        engine.cache.allocator.check_invariants()

    def test_reset_clears_state(
        self, model: Transformer, model_cfg: ModelConfig, prompts: list[list[int]]
    ) -> None:
        engine = build_engine(model, model_cfg)
        engine.add_request(prompts[0], SamplingConfig(max_new_tokens=4))
        engine.run()
        engine.reset()
        assert engine.stats.num_steps == 0
        assert not engine.has_unfinished
        assert engine.cache.num_free_blocks == engine.config.cache.num_blocks

    def test_exact_token_budgets(self, model: Transformer, model_cfg: ModelConfig) -> None:
        engine = build_engine(model, model_cfg)
        for budget in (1, 2, 5, 9):
            engine.add_request([1, 2, 3], SamplingConfig(max_new_tokens=budget))
        outputs = sorted(engine.run(), key=lambda o: o.request_id)
        assert [len(o.output_ids) for o in outputs] == [1, 2, 5, 9]

    def test_stop_token_truncates(self, model: Transformer, model_cfg: ModelConfig) -> None:
        engine = build_engine(model, model_cfg)
        reference = dense_greedy(model, [1, 2, 3], 20)
        stop = reference[4]
        engine.add_request(
            [1, 2, 3],
            SamplingConfig(temperature=0.0, max_new_tokens=20, stop_token=stop),
        )
        output = engine.run()[0]
        assert output.output_ids[-1] == stop
        assert len(output.output_ids) == reference.index(stop) + 1


class TestValidation:
    def test_empty_prompt_rejected(self, model: Transformer, model_cfg: ModelConfig) -> None:
        engine = build_engine(model, model_cfg)
        with pytest.raises(ValueError, match="at least one token"):
            engine.add_request([])

    def test_request_longer_than_context_rejected(
        self, model: Transformer, model_cfg: ModelConfig
    ) -> None:
        engine = build_engine(model, model_cfg, num_blocks=512)
        with pytest.raises(ValueError, match="max_seq_len"):
            engine.add_request([1] * 100, SamplingConfig(max_new_tokens=100))

    def test_request_larger_than_the_cache_rejected(
        self, model: Transformer, model_cfg: ModelConfig
    ) -> None:
        engine = build_engine(model, model_cfg, block_size=8, num_blocks=2)
        with pytest.raises(ValueError, match="could never be served"):
            engine.add_request([1] * 8, SamplingConfig(max_new_tokens=16))

    def test_text_prompt_needs_a_tokenizer(
        self, model: Transformer, model_cfg: ModelConfig
    ) -> None:
        engine = build_engine(model, model_cfg)
        with pytest.raises(ValueError, match="tokenizer"):
            engine.add_request("hello")

    def test_text_round_trip_with_tokenizer(
        self, model: Transformer, model_cfg: ModelConfig
    ) -> None:
        tokenizer = CharTokenizer([chr(ord("a") + i) for i in range(model.cfg.vocab_size)])
        engine = build_engine(model, model_cfg, tokenizer=tokenizer)
        output = engine.generate(["abc"], SamplingConfig(max_new_tokens=6))[0]
        assert output.text is not None
        assert len(output.text) == 6
        assert tokenizer.encode(output.text) == output.output_ids


class TestStats:
    def test_counters_add_up(
        self, model: Transformer, model_cfg: ModelConfig, prompts: list[list[int]]
    ) -> None:
        engine = build_engine(model, model_cfg)
        for p in prompts:
            engine.add_request(p, SamplingConfig(temperature=0.0, max_new_tokens=7))
        outputs = engine.run()
        assert engine.stats.num_generated_tokens == sum(
            len(o.output_ids) for o in outputs
        )
        assert engine.stats.num_prefill_tokens >= sum(len(p) for p in prompts)
        assert 0.0 < engine.stats.mean_utilization <= 1.0
        assert engine.stats.mean_batch_size >= 1.0
        assert engine.stats.acceptance_rate is None

    def test_timing_metrics_are_ordered(
        self, model: Transformer, model_cfg: ModelConfig
    ) -> None:
        engine = build_engine(model, model_cfg)
        engine.add_request([1, 2, 3], SamplingConfig(max_new_tokens=8))
        output = engine.run()[0]
        assert output.ttft is not None and output.ttft >= 0
        assert output.latency is not None and output.latency >= output.ttft
        assert output.tpot is not None and output.tpot >= 0

    def test_step_on_empty_engine_is_a_no_op(
        self, model: Transformer, model_cfg: ModelConfig
    ) -> None:
        engine = build_engine(model, model_cfg)
        assert engine.step() == []
        assert engine.stats.num_steps == 0


def test_oversubscribed_cache_raises_clearly(
    model: Transformer, model_cfg: ModelConfig
) -> None:
    """If admission control is bypassed the failure must be legible."""
    engine = build_engine(model, model_cfg, block_size=4, num_blocks=4)
    request = engine.add_request([1] * 8, SamplingConfig(max_new_tokens=8))
    engine.cache.append(999, 16)  # steal every block behind the scheduler's back
    with pytest.raises(OutOfCacheMemory, match=r"raise cache\.num_blocks"):
        engine.build_batch([request])
