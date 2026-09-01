"""Speculative decoding: draft-then-verify with exact rejection sampling.

The draft model proposes ``gamma`` tokens autoregressively; the target model
scores all of them in a single forward pass; a rejection-sampling test decides
how many to keep. The test is the one from Leviathan et al. (2023), *Fast
Inference from Transformers via Speculative Decoding*:

    accept q ~ p_draft  with probability  min(1, p_target(q) / p_draft(q))
    on rejection, resample from  normalise( max(0, p_target - p_draft) )

The point of that residual is that the tokens emitted are distributed exactly
as ``p_target`` -- speculation buys latency, not a different distribution. See
``tests/test_speculative.py`` for the numerical proof, and ``scripts/verify_speculative.py``
for the end-to-end statistical check.

Because the target scores ``gamma + 1`` positions per pass, a step emits
between 1 and ``gamma + 1`` tokens per sequence: one bonus token is always
available from the position after the last accepted draft.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .batching import build_paged_batch
from .cache import PagedKVCache
from .config import CacheConfig, SamplingConfig
from .model import Transformer
from .sampling import probs_from_logits, sample_from_probs
from .scheduler import Request


@dataclass
class VerificationResult:
    """Outcome of verifying one sequence's proposals."""

    tokens: list[int]  # accepted drafts followed by exactly one extra token
    num_accepted: int
    num_proposed: int


def verify_proposals(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    proposals: list[int],
    generator: torch.Generator | None = None,
) -> VerificationResult:
    """Run the accept/reject test for one sequence.

    ``draft_probs``  -- ``(gamma, vocab)``, the draft distribution each proposal
    was drawn from.
    ``target_probs`` -- ``(gamma + 1, vocab)``, the target distribution at the
    same positions plus one extra for the bonus token.
    """
    gamma = len(proposals)
    if draft_probs.shape[0] != gamma or target_probs.shape[0] != gamma + 1:
        raise ValueError("probability tensors do not match the proposal length")

    accepted: list[int] = []
    for i, token in enumerate(proposals):
        p_target = target_probs[i, token]
        p_draft = draft_probs[i, token]
        # A token the draft could not have produced cannot appear here, so
        # p_draft > 0 whenever this token was actually proposed.
        ratio = torch.clamp(p_target / p_draft.clamp_min(1e-10), max=1.0)
        roll = torch.rand((), generator=generator, device=p_target.device)
        if roll < ratio:
            accepted.append(int(token))
            continue
        # Rejected: resample from the positive part of (p_target - p_draft).
        residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
        total = residual.sum()
        if total <= 1e-9:
            # The two distributions agree here; falling back to the target is
            # both correct and the only numerically sane option.
            residual = target_probs[i]
            total = residual.sum()
        extra = int(sample_from_probs((residual / total).unsqueeze(0), generator).item())
        return VerificationResult([*accepted, extra], len(accepted), gamma)

    # Every proposal survived, so the target's own next-token distribution at
    # position gamma is free: that is the bonus token.
    bonus = int(sample_from_probs(target_probs[gamma].unsqueeze(0), generator).item())
    return VerificationResult([*accepted, bonus], gamma, gamma)


class SpeculativeDecoder:
    """Batched draft-then-verify over a set of decoding requests."""

    def __init__(
        self,
        target: Transformer,
        draft: Transformer,
        cache_cfg: CacheConfig,
        gamma: int = 4,
        device: str | torch.device = "cpu",
    ) -> None:
        if gamma < 1:
            raise ValueError("gamma must be at least 1")
        if target.cfg.vocab_size != draft.cfg.vocab_size:
            raise ValueError("draft and target must share a vocabulary")
        self.target = target.eval()
        self.draft = draft.eval()
        self.gamma = gamma
        self.device = device
        # The draft keeps its own pool: its KV entries have a different shape
        # and a different lifetime from the target's.
        self.draft_cache = PagedKVCache(draft.cfg, cache_cfg, device)
        self.num_proposed = 0
        self.num_accepted = 0

    # -- draft phase ------------------------------------------------------
    def _propose(
        self,
        requests: list[Request],
        generators: dict[int, torch.Generator],
    ) -> tuple[list[list[int]], list[torch.Tensor]]:
        """Roll the draft model forward ``gamma`` times for every request."""
        proposals: list[list[int]] = [[] for _ in requests]
        probs: list[list[torch.Tensor]] = [[] for _ in requests]

        # Catch the draft cache up to whatever the target has already consumed.
        pending: list[list[int]] = []
        starts: list[int] = []
        for request in requests:
            cached = self.draft_cache.num_tokens(request.request_id)
            pending.append(request.all_ids[cached:])
            starts.append(cached)

        for _ in range(self.gamma):
            batch = build_paged_batch(
                self.draft_cache,
                [r.request_id for r in requests],
                pending,
                starts,
                self.device,
            )
            with torch.no_grad():
                logits = self.draft.forward_paged(batch, self.draft_cache)
            rows = (torch.cumsum(batch.q_lens, 0) - 1).tolist()

            pending, starts = [], []
            for i, (request, row) in enumerate(zip(requests, rows, strict=True)):
                dist = probs_from_logits(logits[row : row + 1], request.sampling)
                token = int(
                    sample_from_probs(dist, generators[request.request_id]).item()
                )
                proposals[i].append(token)
                probs[i].append(dist.squeeze(0))
                pending.append([token])
                starts.append(self.draft_cache.num_tokens(request.request_id))

        return proposals, [torch.stack(p) for p in probs]

    # -- verify phase -----------------------------------------------------
    def step(
        self,
        requests: list[Request],
        target_cache: PagedKVCache,
        generators: dict[int, torch.Generator],
    ) -> list[list[int]]:
        """Advance every request by one speculative round.

        Returns the tokens accepted per request. The caller appends them to the
        request and is responsible for stop conditions.
        """
        proposals, draft_probs = self._propose(requests, generators)

        # One target pass over [last uncached token] + [gamma proposals].
        token_lists: list[list[int]] = []
        starts: list[int] = []
        for request, proposal in zip(requests, proposals, strict=True):
            cached = target_cache.num_tokens(request.request_id)
            token_lists.append(request.all_ids[cached:] + proposal)
            starts.append(cached)

        batch = build_paged_batch(
            target_cache,
            [r.request_id for r in requests],
            token_lists,
            starts,
            self.device,
        )
        with torch.no_grad():
            logits = self.target.forward_paged(batch, target_cache)

        ends = torch.cumsum(batch.q_lens, 0).tolist()
        results: list[list[int]] = []
        for i, request in enumerate(requests):
            end = ends[i]
            # The last gamma+1 rows score: the token after the real prefix, and
            # the token after each accepted proposal.
            window = logits[end - (self.gamma + 1) : end]
            target_probs = probs_from_logits(window, request.sampling)
            outcome = verify_proposals(
                draft_probs[i],
                target_probs,
                proposals[i],
                generators[request.request_id],
            )

            self._rollback(request, target_cache, outcome.num_accepted)
            request.num_draft_proposed += outcome.num_proposed
            request.num_draft_accepted += outcome.num_accepted
            self.num_proposed += outcome.num_proposed
            self.num_accepted += outcome.num_accepted
            results.append(outcome.tokens)

        return results

    def _rollback(
        self, request: Request, target_cache: PagedKVCache, num_accepted: int
    ) -> None:
        """Drop KV entries for rejected drafts and restore the engine invariant.

        After a round the cache must hold exactly the tokens the request really
        has, minus the newest one, which the next pass will compute.
        """
        seq_id = request.request_id
        keep = request.num_tokens + num_accepted
        target_cache.truncate(seq_id, keep)
        request.num_computed = keep
        draft_len = self.draft_cache.num_tokens(seq_id)
        self.draft_cache.truncate(seq_id, min(draft_len, keep))

    def free(self, seq_id: int) -> None:
        self.draft_cache.free(seq_id)

    def reset(self) -> None:
        self.draft_cache.reset()
        self.num_proposed = 0
        self.num_accepted = 0

    # -- offline reference ------------------------------------------------
    def generate(
        self,
        prompt_ids: list[int],
        sampling: SamplingConfig,
        seq_id: int = 0,
        generator: torch.Generator | None = None,
        target_cache: PagedKVCache | None = None,
    ) -> list[int]:
        """Single-sequence speculative generation, for tests and benchmarks."""
        cache = target_cache or PagedKVCache(
            self.target.cfg, self.draft_cache.cfg, self.device
        )
        gen = generator or torch.Generator(device=self.device).manual_seed(0)
        request = Request(request_id=seq_id, prompt_ids=list(prompt_ids), sampling=sampling)
        generators = {seq_id: gen}
        try:
            while len(request.output_ids) < sampling.max_new_tokens:
                for token in self.step([request], cache, generators)[0]:
                    if request.should_stop():
                        break
                    request.output_ids.append(token)
            return request.output_ids
        finally:
            cache.free(seq_id)
            self.draft_cache.free(seq_id)
