"""Workload generator and benchmark harness.

Requests arrive as a Poisson process, prompt lengths are lognormal, and output
lengths are **variable** -- that last part matters: with a constant output
length static and continuous batching are the same algorithm, and the headline
result would be an artefact of the workload rather than the scheduler.

Sweeps (each writes one JSON file into ``artifacts/benchmark/``):

``batching``   continuous vs static, across arrival rates
``policy``     FCFS vs SJF vs LIFO, across arrival rates
``blocksize``  KV block size vs internal fragmentation
``batchsize``  the latency/throughput frontier
``capacity``   cache pressure vs preemption count
``gamma``      speculative proposal length (needs a draft checkpoint)

Usage::

    python scripts/benchmark.py --sweeps batching policy --requests 48
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microserve.config import (
    CacheConfig,
    EngineConfig,
    SamplingConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from microserve.data import load_corpus, sample_prompts
from microserve.engine import LLMEngine, RequestOutput
from microserve.model import Transformer
from microserve.stats import percentile
from microserve.tokenizer import CharTokenizer


@dataclass
class WorkloadSpec:
    """Everything that defines a synthetic workload."""

    num_requests: int = 48
    arrival_rate: float = 8.0  # requests per second
    prompt_mean: float = 100.0
    prompt_std: float = 50.0
    prompt_min: int = 8
    prompt_max: int = 256
    output_min: int = 1
    output_max: int = 50
    temperature: float = 0.8
    seed: int = 4242


@dataclass
class WorkloadItem:
    request_id: int
    prompt: list[int]
    arrival: float  # seconds after the run starts
    max_new_tokens: int


def build_workload(spec: WorkloadSpec, corpus_tokens: object) -> list[WorkloadItem]:
    """Draw arrival times, prompt lengths, and output budgets."""
    from microserve.data import Corpus

    assert isinstance(corpus_tokens, Corpus)
    gen = torch.Generator().manual_seed(spec.seed)

    # Lognormal parameters chosen so the *linear* mean and std match the spec.
    variance = math.log(1.0 + (spec.prompt_std / spec.prompt_mean) ** 2)
    mu = math.log(spec.prompt_mean) - variance / 2.0
    raw = torch.exp(torch.randn(spec.num_requests, generator=gen) * math.sqrt(variance) + mu)
    lengths = raw.clamp(spec.prompt_min, spec.prompt_max).to(torch.int64).tolist()

    # Poisson process == exponential inter-arrival gaps.
    gaps = -torch.log1p(-torch.rand(spec.num_requests, generator=gen)) / spec.arrival_rate
    arrivals = torch.cumsum(gaps, 0).tolist()

    span = spec.output_max - spec.output_min + 1
    outputs = (
        torch.randint(0, span, (spec.num_requests,), generator=gen) + spec.output_min
    ).tolist()

    prompts = sample_prompts(corpus_tokens, spec.num_requests, lengths, gen)
    return [
        WorkloadItem(i, prompts[i], arrivals[i], int(outputs[i]))
        for i in range(spec.num_requests)
    ]


@dataclass
class RunResult:
    """Metrics for one engine configuration on one workload."""

    label: str
    params: dict[str, object] = field(default_factory=dict)
    num_requests: int = 0
    makespan_s: float = 0.0
    generated_tokens: int = 0
    prefill_tokens: int = 0
    throughput_tok_s: float = 0.0
    request_throughput_s: float = 0.0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    tpot_mean: float = 0.0
    tpot_p95: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    mean_batch_size: float = 0.0
    mean_utilization: float = 0.0
    peak_utilization: float = 0.0
    preemptions: int = 0
    num_steps: int = 0
    acceptance_rate: float | None = None


def summarise(
    label: str,
    params: dict[str, object],
    outputs: list[RequestOutput],
    engine: LLMEngine,
    makespan: float,
) -> RunResult:
    ttfts = [o.ttft for o in outputs if o.ttft is not None]
    tpots = [o.tpot for o in outputs if o.tpot is not None]
    latencies = [o.latency for o in outputs if o.latency is not None]
    generated = sum(len(o.output_ids) for o in outputs)
    return RunResult(
        label=label,
        params=params,
        num_requests=len(outputs),
        makespan_s=makespan,
        generated_tokens=generated,
        prefill_tokens=engine.stats.num_prefill_tokens,
        throughput_tok_s=generated / makespan if makespan else 0.0,
        request_throughput_s=len(outputs) / makespan if makespan else 0.0,
        ttft_p50=percentile(ttfts, 50) if ttfts else 0.0,
        ttft_p95=percentile(ttfts, 95) if ttfts else 0.0,
        ttft_p99=percentile(ttfts, 99) if ttfts else 0.0,
        tpot_mean=sum(tpots) / len(tpots) if tpots else 0.0,
        tpot_p95=percentile(tpots, 95) if tpots else 0.0,
        latency_p50=percentile(latencies, 50) if latencies else 0.0,
        latency_p95=percentile(latencies, 95) if latencies else 0.0,
        latency_p99=percentile(latencies, 99) if latencies else 0.0,
        mean_batch_size=engine.stats.mean_batch_size,
        mean_utilization=engine.stats.mean_utilization,
        peak_utilization=engine.stats.peak_utilization,
        preemptions=engine.stats.num_preemptions,
        num_steps=engine.stats.num_steps,
        acceptance_rate=engine.stats.acceptance_rate,
    )


def run_workload(
    engine: LLMEngine,
    workload: list[WorkloadItem],
    spec: WorkloadSpec,
) -> tuple[list[RequestOutput], float]:
    """Drive the engine in real time, releasing requests at their arrival."""
    outputs: list[RequestOutput] = []
    pending = sorted(workload, key=lambda item: item.arrival)
    index = 0
    start = time.perf_counter()

    while index < len(pending) or engine.has_unfinished:
        now = time.perf_counter()
        while index < len(pending) and start + pending[index].arrival <= now:
            item = pending[index]
            engine.add_request(
                item.prompt,
                SamplingConfig(
                    temperature=spec.temperature,
                    max_new_tokens=item.max_new_tokens,
                    seed=spec.seed + item.request_id,
                ),
                arrival_time=start + item.arrival,
            )
            index += 1

        if engine.has_unfinished:
            outputs.extend(engine.step())
        elif index < len(pending):
            # Idle: nothing queued yet. Wait for the next arrival rather than
            # spinning, so the measured idle time is real.
            time.sleep(max(0.0, start + pending[index].arrival - time.perf_counter()))

    return outputs, time.perf_counter() - start


def make_engine(
    target: Transformer,
    draft: Transformer | None,
    cache_cfg: CacheConfig,
    sched_cfg: SchedulerConfig,
    gamma: int | None = None,
) -> LLMEngine:
    cfg = EngineConfig(
        model=target.cfg,
        cache=cache_cfg,
        scheduler=sched_cfg,
        speculative=SpeculativeConfig(enabled=gamma is not None, gamma=gamma or 1),
    )
    return LLMEngine(target, cfg, draft_model=draft if gamma is not None else None)


# -- individual sweeps ----------------------------------------------------
def sweep_batching(ctx: Context) -> list[RunResult]:
    rows = []
    for rate in ctx.rates:
        spec = ctx.spec_at(rate)
        workload = build_workload(spec, ctx.corpus)
        for mode in ("continuous", "static"):
            engine = make_engine(
                ctx.target, None, ctx.cache_cfg,
                SchedulerConfig(batching=mode, max_batch_size=ctx.max_batch_size),
            )
            outputs, makespan = run_workload(engine, workload, spec)
            rows.append(
                summarise(f"{mode}@{rate}", {"batching": mode, "arrival_rate": rate},
                          outputs, engine, makespan)
            )
            ctx.log(rows[-1])
    return rows


def sweep_policy(ctx: Context) -> list[RunResult]:
    rows = []
    for rate in ctx.rates:
        spec = ctx.spec_at(rate)
        workload = build_workload(spec, ctx.corpus)
        for policy in ("fcfs", "sjf", "lifo"):
            engine = make_engine(
                ctx.target, None, ctx.cache_cfg,
                SchedulerConfig(policy=policy, max_batch_size=ctx.max_batch_size),
            )
            outputs, makespan = run_workload(engine, workload, spec)
            rows.append(
                summarise(f"{policy}@{rate}", {"policy": policy, "arrival_rate": rate},
                          outputs, engine, makespan)
            )
            ctx.log(rows[-1])
    return rows


def sweep_blocksize(ctx: Context) -> list[RunResult]:
    rows = []
    spec = ctx.spec_at(ctx.rates[len(ctx.rates) // 2])
    workload = build_workload(spec, ctx.corpus)
    total_slots = ctx.cache_cfg.num_slots
    for block_size in (4, 8, 16, 32, 64):
        cache_cfg = CacheConfig(block_size=block_size, num_blocks=total_slots // block_size)
        engine = make_engine(
            ctx.target, None, cache_cfg,
            SchedulerConfig(max_batch_size=ctx.max_batch_size),
        )
        outputs, makespan = run_workload(engine, workload, spec)
        result = summarise(
            f"block{block_size}",
            {"block_size": block_size, "num_blocks": cache_cfg.num_blocks},
            outputs, engine, makespan,
        )
        # Internal fragmentation: slots held minus tokens actually stored.
        wasted = sum(
            (-(-len(o.prompt_ids + o.output_ids) // block_size)) * block_size
            - len(o.prompt_ids + o.output_ids)
            for o in outputs
        )
        stored = sum(len(o.prompt_ids + o.output_ids) for o in outputs)
        result.params["fragmentation"] = wasted / (stored + wasted)
        rows.append(result)
        ctx.log(result)
    return rows


def sweep_batchsize(ctx: Context) -> list[RunResult]:
    rows = []
    spec = ctx.spec_at(max(ctx.rates))
    workload = build_workload(spec, ctx.corpus)
    for batch in (1, 2, 4, 8, 16, 32):
        engine = make_engine(
            ctx.target, None, ctx.cache_cfg, SchedulerConfig(max_batch_size=batch)
        )
        outputs, makespan = run_workload(engine, workload, spec)
        rows.append(
            summarise(f"batch{batch}", {"max_batch_size": batch}, outputs, engine, makespan)
        )
        ctx.log(rows[-1])
    return rows


def sweep_capacity(ctx: Context) -> list[RunResult]:
    rows = []
    spec = ctx.spec_at(max(ctx.rates))
    workload = build_workload(spec, ctx.corpus)
    for num_blocks in (48, 64, 96, 128, 256, 512):
        cache_cfg = CacheConfig(block_size=16, num_blocks=num_blocks)
        engine = make_engine(
            ctx.target, None, cache_cfg,
            SchedulerConfig(max_batch_size=ctx.max_batch_size),
        )
        outputs, makespan = run_workload(engine, workload, spec)
        rows.append(
            summarise(
                f"blocks{num_blocks}",
                {"num_blocks": num_blocks, "slots": cache_cfg.num_slots},
                outputs, engine, makespan,
            )
        )
        ctx.log(rows[-1])
    return rows


def sweep_gamma(ctx: Context) -> list[RunResult]:
    if ctx.draft is None:
        print("  skipped: no draft checkpoint")
        return []
    rows = []
    spec = ctx.spec_at(ctx.rates[0])
    workload = build_workload(spec, ctx.corpus)
    for gamma in (None, 1, 2, 3, 5, 7):
        engine = make_engine(
            ctx.target, ctx.draft, ctx.cache_cfg,
            SchedulerConfig(max_batch_size=ctx.max_batch_size), gamma,
        )
        outputs, makespan = run_workload(engine, workload, spec)
        rows.append(
            summarise(
                f"gamma{gamma or 0}", {"gamma": gamma or 0}, outputs, engine, makespan
            )
        )
        ctx.log(rows[-1])
    return rows


SWEEPS = {
    "batching": sweep_batching,
    "policy": sweep_policy,
    "blocksize": sweep_blocksize,
    "batchsize": sweep_batchsize,
    "capacity": sweep_capacity,
    "gamma": sweep_gamma,
}


@dataclass
class Context:
    target: Transformer
    draft: Transformer | None
    corpus: object
    cache_cfg: CacheConfig
    base_spec: WorkloadSpec
    rates: list[float]
    max_batch_size: int

    def spec_at(self, rate: float) -> WorkloadSpec:
        spec = WorkloadSpec(**asdict(self.base_spec))
        spec.arrival_rate = rate
        return spec

    def log(self, result: RunResult) -> None:
        print(
            f"  {result.label:22s} tput={result.throughput_tok_s:7.1f} tok/s "
            f"ttft_p95={result.ttft_p95:6.3f}s lat_p95={result.latency_p95:6.3f}s "
            f"bs={result.mean_batch_size:5.2f} util={result.mean_utilization:.2f} "
            f"preempt={result.preemptions}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--data", default="data/tinyshakespeare.txt")
    parser.add_argument("--out", default="artifacts/benchmark")
    parser.add_argument("--sweeps", nargs="+", choices=sorted(SWEEPS), default=sorted(SWEEPS))
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--rates", type=float, nargs="+", default=[2.0, 8.0, 32.0])
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    art = Path(args.artifacts)
    target = Transformer.from_checkpoint(art / "target.pt")
    draft_path = art / "draft.pt"
    draft = Transformer.from_checkpoint(draft_path) if draft_path.exists() else None
    tokenizer = CharTokenizer.from_file(art / "tokenizer.json")
    corpus = load_corpus(args.data, tokenizer=tokenizer)

    ctx = Context(
        target=target,
        draft=draft,
        corpus=corpus,
        cache_cfg=CacheConfig(block_size=args.block_size, num_blocks=args.num_blocks),
        base_spec=WorkloadSpec(num_requests=args.requests, seed=args.seed),
        rates=list(args.rates),
        max_batch_size=args.max_batch_size,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"model {target.num_parameters():,} params | cache "
        f"{ctx.cache_cfg.num_slots} slots "
        f"({ctx.cache_cfg.bytes_for(target.cfg) / 1e6:.1f} MB) | "
        f"{args.requests} requests | threads={args.threads}"
    )

    for name in args.sweeps:
        print(f"\n== sweep: {name} ==")
        started = time.perf_counter()
        rows = SWEEPS[name](ctx)
        if not rows:
            continue
        payload = {
            "sweep": name,
            "seed": args.seed,
            "threads": args.threads,
            "workload": asdict(ctx.base_spec),
            "cache": asdict(ctx.cache_cfg),
            "max_batch_size": args.max_batch_size,
            "target_params": target.num_parameters(),
            "elapsed_s": time.perf_counter() - started,
            "results": [asdict(r) for r in rows],
        }
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
