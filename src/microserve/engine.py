"""The serving loop: scheduler + paged cache + model, one ``step()`` at a time.

Each step builds a single flattened :class:`~microserve.model.PagedBatch` from
whatever the scheduler chose -- prefills and decodes together -- runs one paged
forward pass, samples one token per sequence, and retires anything that hit a
stop condition.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from .cache import OutOfCacheMemory, PagedKVCache
from .config import EngineConfig, SamplingConfig
from .model import PagedBatch, Transformer
from .sampling import sample
from .scheduler import Request, RequestStatus, Scheduler
from .tokenizer import CharTokenizer


@dataclass
class RequestOutput:
    """What the caller gets back when a request finishes."""

    request_id: int
    prompt_ids: list[int]
    output_ids: list[int]
    text: str | None = None
    ttft: float | None = None
    tpot: float | None = None
    latency: float | None = None
    num_preemptions: int = 0
    acceptance_rate: float | None = None


@dataclass
class EngineStats:
    """Counters accumulated over the lifetime of the engine."""

    num_steps: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_generated_tokens: int = 0
    num_preemptions: int = 0
    num_draft_proposed: int = 0
    num_draft_accepted: int = 0
    utilization_samples: list[float] = field(default_factory=list)
    batch_size_samples: list[int] = field(default_factory=list)

    @property
    def mean_utilization(self) -> float:
        s = self.utilization_samples
        return sum(s) / len(s) if s else 0.0

    @property
    def peak_utilization(self) -> float:
        return max(self.utilization_samples, default=0.0)

    @property
    def mean_batch_size(self) -> float:
        s = self.batch_size_samples
        return sum(s) / len(s) if s else 0.0

    @property
    def acceptance_rate(self) -> float | None:
        if not self.num_draft_proposed:
            return None
        return self.num_draft_accepted / self.num_draft_proposed


class LLMEngine:
    """Continuous-batching inference engine over a paged KV cache."""

    def __init__(
        self,
        model: Transformer,
        config: EngineConfig | None = None,
        tokenizer: CharTokenizer | None = None,
        draft_model: Transformer | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config or EngineConfig(model=model.cfg)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.clock = clock
        self.cache = PagedKVCache(model.cfg, self.config.cache, self.config.device)
        self.scheduler = Scheduler(self.config.scheduler, self.cache)
        self.stats = EngineStats()
        self._generators: dict[int, torch.Generator] = {}

        self.spec = None
        if draft_model is not None and self.config.speculative.enabled:
            from .speculative import SpeculativeDecoder

            self.spec = SpeculativeDecoder(
                target=self.model,
                draft=draft_model,
                cache_cfg=self.config.cache,
                gamma=self.config.speculative.gamma,
                device=self.config.device,
            )
            self.scheduler.tokens_per_step = self.config.speculative.gamma + 1

    # -- request lifecycle ------------------------------------------------
    def add_request(
        self,
        prompt_ids: Sequence[int] | str,
        sampling: SamplingConfig | None = None,
        request_id: int | None = None,
        arrival_time: float | None = None,
    ) -> Request:
        """Enqueue a request. Accepts token ids, or text if a tokenizer is set."""
        if isinstance(prompt_ids, str):
            if self.tokenizer is None:
                raise ValueError("text prompts need a tokenizer")
            ids = self.tokenizer.encode(prompt_ids)
        else:
            ids = list(prompt_ids)
        if not ids:
            raise ValueError("prompt must contain at least one token")

        sampling = sampling or SamplingConfig()
        rid = self.scheduler.next_request_id() if request_id is None else request_id
        request = Request(
            request_id=rid,
            prompt_ids=ids,
            sampling=sampling,
            arrival_time=self.clock() if arrival_time is None else arrival_time,
        )
        if request.num_tokens + sampling.max_new_tokens > self.model.cfg.max_seq_len:
            raise ValueError("prompt plus max_new_tokens exceeds max_seq_len")
        worst_case = request.num_tokens + sampling.max_new_tokens
        block_size = self.cache.block_size
        if -(-worst_case // block_size) > self.cache.allocator.num_blocks:
            raise ValueError(
                f"request needs up to {worst_case} tokens of KV cache but the pool "
                f"holds only {self.config.cache.num_slots}; it could never be served"
            )

        seed = sampling.seed if sampling.seed is not None else self.config.seed + rid
        generator = torch.Generator(device=self.config.device)
        generator.manual_seed(seed)
        self._generators[rid] = generator
        return self.scheduler.add(request)

    @property
    def has_unfinished(self) -> bool:
        return self.scheduler.num_unfinished > 0

    # -- one iteration ----------------------------------------------------
    def step(self) -> list[RequestOutput]:
        """Run one scheduling + forward iteration. Returns finished requests."""
        now = self.clock()
        plan = self.scheduler.schedule(now)
        self.stats.num_preemptions = self.scheduler.total_preemptions
        if plan.is_empty:
            return []

        if self.spec is not None:
            for request in plan.preempted:
                self.spec.free(request.request_id)

        gamma = self.config.speculative.gamma
        use_spec = (
            self.spec is not None
            and all(not r.is_prefill for r in plan.scheduled)
            and all(
                self.cache.can_append(r.request_id, gamma + 1) for r in plan.scheduled
            )
        )
        if use_spec:
            assert self.spec is not None
            new_tokens = self.spec.step(plan.scheduled, self.cache, self._generators)
            self.stats.num_draft_proposed = self.spec.num_proposed
            self.stats.num_draft_accepted = self.spec.num_accepted
        else:
            new_tokens = self._forward_step(plan.scheduled)

        now = self.clock()
        finished: list[RequestOutput] = []
        for request, tokens in zip(plan.scheduled, new_tokens, strict=True):
            for token in tokens:
                if request.should_stop():
                    break
                request.output_ids.append(token)
                self.stats.num_generated_tokens += 1
                if request.first_token_time is None:
                    request.first_token_time = now
            if request.should_stop():
                self.scheduler.finish(request, now)
                if self.spec is not None:
                    self.spec.free(request.request_id)
                finished.append(self.make_output(request))

        self.stats.num_steps += 1
        self.stats.utilization_samples.append(self.cache.utilization)
        self.stats.batch_size_samples.append(len(plan.scheduled))
        return finished

    def _forward_step(self, scheduled: list[Request]) -> list[list[int]]:
        """Standard (non-speculative) batched forward: one token per sequence."""
        batch = self.build_batch(scheduled)
        with torch.no_grad():
            logits = self.model.forward_paged(batch, self.cache)

        # The logits that matter are the ones at each sequence's final token.
        last = torch.cumsum(batch.q_lens, 0) - 1
        tokens: list[list[int]] = []
        for row, request in zip(last.tolist(), scheduled, strict=True):
            token = sample(
                logits[row : row + 1],
                request.sampling,
                self._generators[request.request_id],
            )
            tokens.append([int(token.item())])
        return tokens

    def build_batch(self, scheduled: list[Request]) -> PagedBatch:
        """Allocate cache slots for the step and flatten it into a PagedBatch."""
        ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        q_lens: list[int] = []

        for request in scheduled:
            n_new = request.num_uncomputed
            if n_new <= 0:
                raise RuntimeError(f"request {request.request_id} has nothing to do")
            if not self.cache.can_append(request.request_id, n_new):
                raise OutOfCacheMemory(
                    f"request {request.request_id} needs "
                    f"{self.cache.blocks_needed(request.request_id, n_new)} blocks, "
                    f"{self.cache.num_free_blocks} free; raise cache.num_blocks"
                )
            start = request.num_computed
            ids.extend(request.all_ids[start : start + n_new])
            positions.extend(range(start, start + n_new))
            slots.extend(self.cache.append(request.request_id, n_new))
            q_lens.append(n_new)

            if request.is_prefill:
                self.stats.num_prefill_tokens += n_new
            else:
                self.stats.num_decode_tokens += n_new
            request.num_computed += n_new
            request.status = RequestStatus.RUNNING

        device = self.config.device
        seq_ids = [r.request_id for r in scheduled]
        block_tables, context_lens = self.cache.build_block_tables(seq_ids)
        return PagedBatch(
            input_ids=torch.tensor(ids, dtype=torch.long, device=device),
            positions=torch.tensor(positions, dtype=torch.long, device=device),
            slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
            block_tables=block_tables,
            context_lens=context_lens,
            q_lens=torch.tensor(q_lens, dtype=torch.long, device=device),
        )

    # -- results ----------------------------------------------------------
    def make_output(self, request: Request) -> RequestOutput:
        accepted = None
        if request.num_draft_proposed:
            accepted = request.num_draft_accepted / request.num_draft_proposed
        return RequestOutput(
            request_id=request.request_id,
            prompt_ids=request.prompt_ids,
            output_ids=request.output_ids,
            text=(
                self.tokenizer.decode(request.output_ids)
                if self.tokenizer is not None
                else None
            ),
            ttft=request.ttft(),
            tpot=request.tpot(),
            latency=request.latency(),
            num_preemptions=request.num_preemptions,
            acceptance_rate=accepted,
        )

    # -- offline convenience ----------------------------------------------
    def run(self, max_steps: int | None = None) -> list[RequestOutput]:
        """Drive the loop until every enqueued request finishes."""
        outputs: list[RequestOutput] = []
        steps = 0
        while self.has_unfinished:
            outputs.extend(self.step())
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        return outputs

    def generate(
        self,
        prompts: Sequence[str] | Sequence[Sequence[int]],
        sampling: SamplingConfig | None = None,
    ) -> list[RequestOutput]:
        """Offline batch generation; results come back in prompt order."""
        ids = [self.add_request(p, sampling).request_id for p in prompts]
        outputs = {o.request_id: o for o in self.run()}
        return [outputs[i] for i in ids]

    def reset(self) -> None:
        """Clear all state; the model weights are untouched."""
        self.cache.reset()
        self.scheduler = Scheduler(self.config.scheduler, self.cache)
        if self.spec is not None:
            self.spec.reset()
            self.scheduler.tokens_per_step = self.config.speculative.gamma + 1
        self.stats = EngineStats()
        self._generators.clear()
