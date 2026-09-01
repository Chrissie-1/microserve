"""Prove that speculative decoding changes speed, not the output distribution.

Three checks, all written to ``artifacts/spec_stats.json``:

1. **Greedy equivalence** -- at temperature 0 the speculative output must be
   token-for-token identical to the dense reference implementation, for every
   gamma. This is a hard assertion, not a statistic.
2. **Distributional equivalence** -- at temperature 1 the first token emitted by
   the speculative sampler is compared, by a chi-square test of homogeneity,
   against exact draws from the target's own next-token distribution.
3. **Acceptance rate and speedup** -- swept over gamma, measured on the same
   prompts, against the ordinary (non-speculative) engine.

Usage::

    python scripts/verify_speculative.py --trials 3000 --gammas 1 2 4 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microserve.cache import PagedKVCache
from microserve.config import (
    CacheConfig,
    EngineConfig,
    SamplingConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from microserve.data import load_corpus, sample_prompts
from microserve.engine import LLMEngine
from microserve.model import Transformer
from microserve.sampling import probs_from_logits
from microserve.scheduler import Request
from microserve.speculative import SpeculativeDecoder
from microserve.stats import chi_square_two_sample
from microserve.tokenizer import CharTokenizer


def dense_greedy(model: Transformer, prompt: list[int], n: int) -> list[int]:
    """Ground-truth greedy generation through the dense reference path."""
    past = None
    current = torch.tensor(prompt)[None]
    out: list[int] = []
    for _ in range(n):
        with torch.no_grad():
            logits, past = model(current, past_kv=past, return_cache=True)
        token = int(logits[0, -1].argmax())
        out.append(token)
        current = torch.tensor([[token]])
    return out


def check_greedy(
    target: Transformer,
    draft: Transformer,
    prompts: list[list[int]],
    gammas: list[int],
    cache_cfg: CacheConfig,
    n_tokens: int,
) -> list[dict[str, object]]:
    results = []
    for gamma in gammas:
        decoder = SpeculativeDecoder(target, draft, cache_cfg, gamma=gamma)
        matches = 0
        for i, prompt in enumerate(prompts):
            decoder.reset()
            reference = dense_greedy(target, prompt, n_tokens)
            got = decoder.generate(
                prompt, SamplingConfig(temperature=0.0, max_new_tokens=n_tokens), seq_id=i
            )
            matches += int(got == reference)
        rate = decoder.num_accepted / max(1, decoder.num_proposed)
        results.append(
            {
                "gamma": gamma,
                "prompts": len(prompts),
                "exact_matches": matches,
                "all_match": matches == len(prompts),
                "acceptance_rate": rate,
            }
        )
        status = "OK " if matches == len(prompts) else "FAIL"
        print(
            f"  [{status}] gamma={gamma}: {matches}/{len(prompts)} exact, "
            f"acceptance={rate:.3f}"
        )
    return results


def first_token_distribution_test(
    target: Transformer,
    draft: Transformer,
    prompt: list[int],
    gamma: int,
    trials: int,
    cache_cfg: CacheConfig,
    temperature: float,
    seed: int,
) -> dict[str, object]:
    """Compare speculative first tokens against exact target draws."""
    sampling = SamplingConfig(temperature=temperature, max_new_tokens=1)

    # Ground truth: the target's own next-token distribution, sampled directly.
    with torch.no_grad():
        logits, _ = target(torch.tensor(prompt)[None])
    probs = probs_from_logits(logits[0, -1:], sampling)
    truth_gen = torch.Generator().manual_seed(seed)
    baseline = torch.multinomial(
        probs.expand(trials, -1), num_samples=1, replacement=True, generator=truth_gen
    ).squeeze(-1).tolist()

    decoder = SpeculativeDecoder(target, draft, cache_cfg, gamma=gamma)
    observed: list[int] = []
    for trial in range(trials):
        cache = PagedKVCache(target.cfg, cache_cfg)
        request = Request(request_id=0, prompt_ids=list(prompt), sampling=sampling)
        generator = torch.Generator().manual_seed(seed + 10_000 + trial)
        tokens = decoder.step([request], cache, {0: generator})[0]
        observed.append(tokens[0])
        cache.free(0)
        decoder.free(0)

    result = chi_square_two_sample(baseline, observed)
    print(
        f"  gamma={gamma} T={temperature}: {result} -> "
        f"{'indistinguishable' if result.p_value > 0.05 else 'DIFFERENT'}"
    )
    return {
        "gamma": gamma,
        "temperature": temperature,
        "trials": trials,
        "chi2": result.statistic,
        "dof": result.dof,
        "p_value": result.p_value,
        "categories": result.num_categories,
        "acceptance_rate": decoder.num_accepted / max(1, decoder.num_proposed),
        "passes": result.p_value > 0.05,
    }


def measure_speedup(
    target: Transformer,
    draft: Transformer,
    prompts: list[list[int]],
    gammas: list[int],
    cache_cfg: CacheConfig,
    n_tokens: int,
    temperature: float,
) -> list[dict[str, object]]:
    """Wall-clock comparison against the ordinary engine, batch size 1."""
    sampling = SamplingConfig(temperature=temperature, max_new_tokens=n_tokens)
    scheduler_cfg = SchedulerConfig(max_batch_size=1)

    def run(gamma: int | None) -> tuple[float, int, float | None]:
        cfg = EngineConfig(
            model=target.cfg,
            cache=cache_cfg,
            scheduler=scheduler_cfg,
            speculative=SpeculativeConfig(enabled=gamma is not None, gamma=gamma or 1),
        )
        engine = LLMEngine(target, cfg, draft_model=draft if gamma else None)
        for prompt in prompts:
            engine.add_request(prompt, sampling)
        start = time.perf_counter()
        engine.run()
        elapsed = time.perf_counter() - start
        return elapsed, engine.stats.num_generated_tokens, engine.stats.acceptance_rate

    base_time, base_tokens, _ = run(None)
    print(
        f"  baseline: {base_time:.2f}s for {base_tokens} tokens "
        f"({base_tokens / base_time:.1f} tok/s)"
    )
    rows: list[dict[str, object]] = [
        {
            "gamma": 0,
            "wall_clock_s": base_time,
            "tokens": base_tokens,
            "tokens_per_s": base_tokens / base_time,
            "speedup": 1.0,
            "acceptance_rate": None,
        }
    ]
    for gamma in gammas:
        elapsed, tokens, acceptance = run(gamma)
        rows.append(
            {
                "gamma": gamma,
                "wall_clock_s": elapsed,
                "tokens": tokens,
                "tokens_per_s": tokens / elapsed,
                "speedup": base_time / elapsed,
                "acceptance_rate": acceptance,
            }
        )
        print(
            f"  gamma={gamma}: {elapsed:.2f}s ({tokens / elapsed:.1f} tok/s) "
            f"speedup={base_time / elapsed:.2f}x acceptance={acceptance:.3f}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--data", default="data/tinyshakespeare.txt")
    parser.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 4, 7])
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--greedy-prompts", type=int, default=8)
    parser.add_argument("--greedy-tokens", type=int, default=48)
    parser.add_argument("--speed-prompts", type=int, default=4)
    parser.add_argument("--speed-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20240)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    art = Path(args.artifacts)

    target = Transformer.from_checkpoint(art / "target.pt")
    draft = Transformer.from_checkpoint(art / "draft.pt")
    tokenizer = CharTokenizer.from_file(art / "tokenizer.json")
    corpus = load_corpus(args.data, tokenizer=tokenizer)
    cache_cfg = CacheConfig(block_size=16, num_blocks=256)

    gen = torch.Generator().manual_seed(args.seed)
    prompts = sample_prompts(
        corpus, args.greedy_prompts, [32] * args.greedy_prompts, gen
    )
    speed_prompts = prompts[: args.speed_prompts]

    print(f"target {target.num_parameters():,} params | draft {draft.num_parameters():,} params")
    print("\n1. greedy equivalence vs dense reference")
    greedy = check_greedy(
        target, draft, prompts, args.gammas, cache_cfg, args.greedy_tokens
    )

    print(f"\n2. distributional equivalence (chi-square, {args.trials} trials)")
    distribution = [
        first_token_distribution_test(
            target, draft, prompts[0], gamma, args.trials, cache_cfg,
            args.temperature, args.seed,
        )
        for gamma in args.gammas
    ]

    print("\n3. acceptance rate and wall-clock speedup")
    speed = measure_speedup(
        target, draft, speed_prompts, args.gammas, cache_cfg,
        args.speed_tokens, args.temperature,
    )

    payload = {
        "seed": args.seed,
        "threads": args.threads,
        "target_params": target.num_parameters(),
        "draft_params": draft.num_parameters(),
        "greedy_equivalence": greedy,
        "distribution_test": distribution,
        "speedup": speed,
    }
    out = art / "spec_stats.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if not all(r["all_match"] for r in greedy):
        raise SystemExit("greedy equivalence FAILED")
    if not all(r["passes"] for r in distribution):
        raise SystemExit("distributional equivalence FAILED")


if __name__ == "__main__":
    main()
