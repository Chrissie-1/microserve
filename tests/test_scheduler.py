"""Queueing policy, admission control, and preemption under memory pressure."""

from __future__ import annotations

import pytest

from microserve.cache import PagedKVCache
from microserve.config import CacheConfig, ModelConfig, SamplingConfig, SchedulerConfig
from microserve.scheduler import Request, RequestStatus, Scheduler


def make_request(rid: int, prompt: int = 4, arrival: float = 0.0, budget: int = 8) -> Request:
    return Request(
        request_id=rid,
        prompt_ids=list(range(prompt)),
        sampling=SamplingConfig(max_new_tokens=budget),
        arrival_time=arrival,
    )


@pytest.fixture
def scheduler(model_cfg: ModelConfig) -> Scheduler:
    cache = PagedKVCache(model_cfg, CacheConfig(block_size=4, num_blocks=16))
    return Scheduler(SchedulerConfig(max_batch_size=4), cache)


class TestRequest:
    def test_prefill_then_decode_transition(self) -> None:
        request = make_request(0, prompt=5)
        assert request.is_prefill and request.num_uncomputed == 5
        request.num_computed = 5
        request.output_ids.append(9)
        assert not request.is_prefill and request.num_uncomputed == 1

    def test_stops_at_token_budget(self) -> None:
        request = make_request(0, budget=3)
        request.output_ids.extend([1, 2])
        assert not request.should_stop()
        request.output_ids.append(3)
        assert request.should_stop()

    def test_stop_token_ends_generation(self) -> None:
        request = Request(0, [1, 2], SamplingConfig(max_new_tokens=99, stop_token=7))
        request.output_ids.append(5)
        assert not request.should_stop()
        request.output_ids.append(7)
        assert request.should_stop()

    def test_recompute_keeps_generated_tokens(self) -> None:
        request = make_request(0, prompt=4)
        request.num_computed = 6
        request.output_ids.extend([11, 12])
        request.reset_for_recompute()
        assert request.num_computed == 0
        assert request.output_ids == [11, 12]
        assert request.num_uncomputed == 6
        assert request.num_preemptions == 1

    def test_metrics_are_none_until_available(self) -> None:
        request = make_request(0)
        assert request.ttft() is None and request.tpot() is None
        request.first_token_time = 1.0
        request.arrival_time = 0.5
        assert request.ttft() == pytest.approx(0.5)
        assert request.tpot() is None  # not finished yet
        request.finish_time = 3.0
        request.output_ids.extend([1, 2, 3])
        assert request.tpot() == pytest.approx(1.0)
        assert request.latency() == pytest.approx(2.5)


class TestPolicies:
    @pytest.mark.parametrize(
        "policy,expected",
        [("fcfs", [0, 1, 2]), ("lifo", [2, 1, 0]), ("sjf", [1, 2, 0])],
    )
    def test_ordering(
        self, model_cfg: ModelConfig, policy: str, expected: list[int]
    ) -> None:
        cache = PagedKVCache(model_cfg, CacheConfig(block_size=4, num_blocks=64))
        scheduler = Scheduler(SchedulerConfig(policy=policy), cache)
        scheduler.add(make_request(0, prompt=20, arrival=0.0, budget=20))
        scheduler.add(make_request(1, prompt=2, arrival=1.0, budget=2))
        scheduler.add(make_request(2, prompt=4, arrival=2.0, budget=4))
        order = [r.request_id for r in sorted(scheduler.waiting, key=scheduler.priority)]
        assert order == expected

    def test_unknown_policy_rejected(self) -> None:
        with pytest.raises(ValueError, match="scheduling policy"):
            SchedulerConfig(policy="random")

    def test_unknown_batching_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="batching mode"):
            SchedulerConfig(batching="dynamic")

    def test_swap_preemption_is_not_implemented(self) -> None:
        with pytest.raises(ValueError, match="recompute"):
            SchedulerConfig(preemption_mode="swap")

    def test_priority_breaks_ties_on_id(self, scheduler: Scheduler) -> None:
        a = make_request(5, arrival=1.0)
        b = make_request(2, arrival=1.0)
        assert scheduler.priority(b) < scheduler.priority(a)


class TestAdmission:
    def test_respects_max_batch_size(self, scheduler: Scheduler) -> None:
        for i in range(10):
            scheduler.add(make_request(i, prompt=2))
        plan = scheduler.schedule(0.0)
        assert len(plan.scheduled) == 4
        assert len(scheduler.waiting) == 6

    def test_does_not_over_commit_blocks(self, model_cfg: ModelConfig) -> None:
        """Admission must debit blocks as it admits, not compare against a stale free count."""
        cache = PagedKVCache(model_cfg, CacheConfig(block_size=4, num_blocks=6))
        scheduler = Scheduler(SchedulerConfig(max_batch_size=8, watermark=0.0), cache)
        for i in range(5):
            scheduler.add(make_request(i, prompt=8))  # 2 blocks each
        plan = scheduler.schedule(0.0)
        needed = sum(
            cache.blocks_needed(r.request_id, r.num_uncomputed) for r in plan.scheduled
        )
        assert needed <= cache.num_free_blocks

    def test_admits_over_token_budget_when_alone(self, model_cfg: ModelConfig) -> None:
        cache = PagedKVCache(model_cfg, CacheConfig(block_size=8, num_blocks=64))
        scheduler = Scheduler(
            SchedulerConfig(max_batched_tokens=4, max_batch_size=4), cache
        )
        scheduler.add(make_request(0, prompt=50))
        plan = scheduler.schedule(0.0)
        # A prompt bigger than the whole budget still has to run, or it starves.
        assert [r.request_id for r in plan.scheduled] == [0]

    def test_schedule_time_recorded_once(self, scheduler: Scheduler) -> None:
        scheduler.add(make_request(0))
        scheduler.schedule(5.0)
        assert scheduler.running[0].schedule_time == 5.0
        scheduler.running[0].num_computed = scheduler.running[0].num_tokens
        scheduler.running[0].output_ids.append(1)
        scheduler.schedule(9.0)
        assert scheduler.running[0].schedule_time == 5.0

    def test_empty_queue_produces_empty_plan(self, scheduler: Scheduler) -> None:
        assert scheduler.schedule(0.0).is_empty


class TestPreemption:
    def _advance(self, scheduler: Scheduler, request: Request) -> None:
        """Pretend the engine ran a step for this request."""
        n = request.num_uncomputed
        scheduler.cache.append(request.request_id, n)
        request.num_computed += n
        request.output_ids.append(0)

    def test_preempts_under_memory_pressure(self, model_cfg: ModelConfig) -> None:
        cache = PagedKVCache(model_cfg, CacheConfig(block_size=4, num_blocks=8))
        scheduler = Scheduler(
            SchedulerConfig(max_batch_size=4, watermark=0.0), cache
        )
        for i in range(4):
            scheduler.add(make_request(i, prompt=6, arrival=float(i), budget=40))

        preempted_any = False
        for _ in range(12):
            plan = scheduler.schedule(0.0)
            preempted_any |= bool(plan.preempted)
            for request in list(plan.scheduled):
                self._advance(scheduler, request)
        assert preempted_any
        assert scheduler.total_preemptions > 0
        cache.allocator.check_invariants()

    def test_preemption_frees_the_victims_blocks(self, model_cfg: ModelConfig) -> None:
        cache = PagedKVCache(model_cfg, CacheConfig(block_size=4, num_blocks=8))
        scheduler = Scheduler(SchedulerConfig(max_batch_size=4, watermark=0.0), cache)
        for i in range(4):
            scheduler.add(make_request(i, prompt=6, arrival=float(i), budget=40))

        for _ in range(12):
            plan = scheduler.schedule(0.0)
            for victim in plan.preempted:
                assert victim.request_id not in cache.tables
                assert victim.status is RequestStatus.PREEMPTED
                assert victim in scheduler.waiting
            for request in list(plan.scheduled):
                self._advance(scheduler, request)

    def test_lowest_priority_is_evicted_first(self, model_cfg: ModelConfig) -> None:
        cache = PagedKVCache(model_cfg, CacheConfig(block_size=4, num_blocks=6))
        scheduler = Scheduler(SchedulerConfig(max_batch_size=4, watermark=0.0), cache)
        for i in range(3):
            scheduler.add(make_request(i, prompt=7, arrival=float(i), budget=40))
        for _ in range(6):
            plan = scheduler.schedule(0.0)
            if plan.preempted:
                # FCFS evicts the latest arrival first.
                ids = [r.request_id for r in plan.preempted]
                assert max(ids) == ids[0]
                return
            for request in list(plan.scheduled):
                self._advance(scheduler, request)

    def test_single_request_is_never_preempted(self, model_cfg: ModelConfig) -> None:
        cache = PagedKVCache(model_cfg, CacheConfig(block_size=4, num_blocks=4))
        scheduler = Scheduler(SchedulerConfig(max_batch_size=1, watermark=0.0), cache)
        scheduler.add(make_request(0, prompt=8, budget=8))
        for _ in range(4):
            plan = scheduler.schedule(0.0)
            assert not plan.preempted
            for request in list(plan.scheduled):
                self._advance(scheduler, request)


class TestStaticBatching:
    def test_static_holds_the_gate_until_the_batch_drains(
        self, model_cfg: ModelConfig
    ) -> None:
        cache = PagedKVCache(model_cfg, CacheConfig(block_size=8, num_blocks=64))
        scheduler = Scheduler(
            SchedulerConfig(batching="static", max_batch_size=2), cache
        )
        for i in range(4):
            scheduler.add(make_request(i, prompt=4, arrival=float(i)))

        first = scheduler.schedule(0.0)
        assert len(first.scheduled) == 2
        admitted = {r.request_id for r in first.scheduled}

        # A later step must not pull in a new request while the batch is alive.
        for request in first.scheduled:
            request.num_computed = request.num_tokens
            request.output_ids.append(0)
            cache.append(request.request_id, request.num_tokens)
        second = scheduler.schedule(1.0)
        assert {r.request_id for r in second.scheduled} == admitted

        for request in list(scheduler.running):
            scheduler.finish(request, 2.0)
        third = scheduler.schedule(3.0)
        assert {r.request_id for r in third.scheduled} == {2, 3}

    def test_continuous_admits_immediately(self, model_cfg: ModelConfig) -> None:
        cache = PagedKVCache(model_cfg, CacheConfig(block_size=8, num_blocks=64))
        scheduler = Scheduler(SchedulerConfig(max_batch_size=4), cache)
        scheduler.add(make_request(0, prompt=4))
        scheduler.schedule(0.0)
        scheduler.running[0].num_computed = scheduler.running[0].num_tokens
        scheduler.running[0].output_ids.append(1)
        cache.append(0, 4)

        scheduler.add(make_request(1, prompt=4, arrival=1.0))
        plan = scheduler.schedule(1.0)
        assert {r.request_id for r in plan.scheduled} == {0, 1}


class TestCompletion:
    def test_finish_releases_cache_and_dequeues(self, scheduler: Scheduler) -> None:
        scheduler.add(make_request(0, prompt=4))
        scheduler.schedule(0.0)
        scheduler.cache.append(0, 4)
        free_before = scheduler.cache.num_free_blocks
        scheduler.finish(scheduler.running[0], 1.0)
        assert scheduler.cache.num_free_blocks > free_before
        assert scheduler.num_unfinished == 0
        assert scheduler.finished[0].status is RequestStatus.FINISHED
        assert scheduler.finished[0].finish_time == 1.0
