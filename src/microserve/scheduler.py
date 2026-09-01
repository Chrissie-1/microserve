"""Request bookkeeping and admission control.

The scheduler decides, each engine step, which sequences run and which get
evicted. It owns no tensors: it manipulates :class:`Request` objects and asks
the :class:`~microserve.cache.PagedKVCache` whether an allocation would fit.

Two batching modes:

``continuous``
    Requests join and leave the running set at every step. A finished sequence
    frees its blocks immediately and a waiting one takes its place.

``static``
    A batch is formed once and runs until *every* member finishes; only then is
    the next batch admitted. This is the baseline continuous batching is
    measured against, and it is only different when output lengths vary.

Preemption is by recomputation: the victim releases all of its blocks and
returns to the waiting queue with its generated tokens intact, so it will be
re-prefilled on ``prompt + generated`` when it is admitted again.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field

from .cache import PagedKVCache
from .config import SamplingConfig, SchedulerConfig


class RequestStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class Request:
    """One generation request and everything measured about it."""

    request_id: int
    prompt_ids: list[int]
    sampling: SamplingConfig
    arrival_time: float = 0.0

    output_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    # Number of leading tokens whose KV is resident in the cache.
    num_computed: int = 0

    # Timing, all in engine-clock seconds.
    schedule_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None

    num_preemptions: int = 0
    num_draft_proposed: int = 0
    num_draft_accepted: int = 0

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def num_uncomputed(self) -> int:
        return self.num_tokens - self.num_computed

    @property
    def is_prefill(self) -> bool:
        """True when the next pass must process more than one token."""
        return self.num_uncomputed > 1

    @property
    def finished(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.ABORTED)

    def should_stop(self) -> bool:
        if len(self.output_ids) >= self.sampling.max_new_tokens:
            return True
        stop = self.sampling.stop_token
        return stop is not None and bool(self.output_ids) and self.output_ids[-1] == stop

    def reset_for_recompute(self) -> None:
        """Forget cache residency; generated tokens are kept and re-prefilled."""
        self.num_computed = 0
        self.status = RequestStatus.PREEMPTED
        self.num_preemptions += 1

    # -- derived metrics --------------------------------------------------
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    def tpot(self) -> float | None:
        """Mean seconds per output token after the first."""
        if self.finish_time is None or self.first_token_time is None:
            return None
        extra = len(self.output_ids) - 1
        if extra <= 0:
            return None
        return (self.finish_time - self.first_token_time) / extra

    def latency(self) -> float | None:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time


@dataclass
class SchedulerOutput:
    """The plan for a single engine step."""

    scheduled: list[Request] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled

    @property
    def num_tokens(self) -> int:
        return sum(r.num_uncomputed for r in self.scheduled)


class Scheduler:
    """Priority queueing, admission control, and preemption."""

    def __init__(self, cfg: SchedulerConfig, cache: PagedKVCache) -> None:
        self.cfg = cfg
        self.cache = cache
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self._ids = itertools.count()
        self.total_preemptions = 0
        # Speculative decoding writes gamma + 1 tokens per sequence per step,
        # so admission has to reserve that much rather than a single slot.
        self.tokens_per_step = 1

    # -- queue management -------------------------------------------------
    def add(self, request: Request) -> Request:
        request.status = RequestStatus.WAITING
        self.waiting.append(request)
        return request

    def next_request_id(self) -> int:
        return next(self._ids)

    @property
    def num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running)

    def priority(self, request: Request) -> tuple[float, int]:
        """Lower sorts first. Ties break on request id for determinism."""
        if self.cfg.policy == "fcfs":
            key = request.arrival_time
        elif self.cfg.policy == "lifo":
            key = -request.arrival_time
        else:  # sjf: an oracle estimate of total work, prompt + budget
            key = float(len(request.prompt_ids) + request.sampling.max_new_tokens)
        return (key, request.request_id)

    # -- the step plan ----------------------------------------------------
    def schedule(self, now: float) -> SchedulerOutput:
        out = SchedulerOutput()
        # Static batching holds the gate shut until the whole batch drains.
        if self.cfg.batching == "static" and self.running:
            self._ensure_room(self.running, out, now)
            out.scheduled = list(self.running)
            return out

        self._ensure_room(self.running, out, now)
        out.scheduled = list(self.running)
        self._admit(out, now)
        return out

    def _ensure_room(
        self, running: list[Request], out: SchedulerOutput, now: float
    ) -> None:
        """Preempt from the tail until every running request can take a step."""
        del now  # preemption is not time dependent; kept for signature symmetry
        while running:
            needed = sum(
                self.cache.blocks_needed(
                    r.request_id, max(self.tokens_per_step, r.num_uncomputed)
                )
                for r in running
            )
            if needed <= self.cache.num_free_blocks:
                return
            if len(running) == 1:
                # Nothing left to evict: the single survivor gets the cache.
                return
            victim = max(running, key=self.priority)
            running.remove(victim)
            self.cache.free(victim.request_id)
            victim.reset_for_recompute()
            self.waiting.append(victim)
            out.preempted.append(victim)
            self.total_preemptions += 1

    def _admit(self, out: SchedulerOutput, now: float) -> None:
        """Pull waiting requests into the running set while budgets allow.

        Blocks are debited from a local tally as requests are admitted: the
        cache is not actually allocated until the engine builds the batch, so
        the scheduler has to do the arithmetic itself or it will over-commit.
        """
        token_budget = self.cfg.max_batched_tokens - out.num_tokens
        reserve = int(self.cache.allocator.num_blocks * self.cfg.watermark)
        free_blocks = self.cache.num_free_blocks - sum(
            self.cache.blocks_needed(
                r.request_id, max(self.tokens_per_step, r.num_uncomputed)
            )
            for r in out.scheduled
        )

        for request in sorted(self.waiting, key=self.priority):
            if len(self.running) >= self.cfg.max_batch_size:
                break
            need = request.num_uncomputed
            # A prompt larger than the whole budget still has to run alone,
            # otherwise it would never be admitted at all.
            if need > token_budget and out.scheduled:
                continue
            blocks = self.cache.blocks_needed(
                request.request_id, max(need, self.tokens_per_step)
            )
            headroom = free_blocks - (reserve if out.scheduled else 0)
            if blocks > headroom:
                continue

            self.waiting.remove(request)
            request.status = RequestStatus.RUNNING
            if request.schedule_time is None:
                request.schedule_time = now
            self.running.append(request)
            out.scheduled.append(request)
            token_budget -= need
            free_blocks -= blocks

    # -- completion -------------------------------------------------------
    def finish(self, request: Request, now: float) -> None:
        request.status = RequestStatus.FINISHED
        request.finish_time = now
        if request in self.running:
            self.running.remove(request)
        self.cache.free(request.request_id)
        self.finished.append(request)
