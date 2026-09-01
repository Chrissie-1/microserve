# MicroServe

[![CI](https://github.com/Chrissie-1/microserve/actions/workflows/ci.yml/badge.svg)](https://github.com/Chrissie-1/microserve/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A small LLM inference engine, written to be read: **paged KV cache**,
**continuous batching**, **preemption**, and **speculative decoding** — each one
implemented against a dense reference implementation that proves it did not
change the answer.

Everything runs on one CPU core. The models are trained from scratch on
TinyShakespeare in about half an hour, so the whole project reproduces
end-to-end from `make all` with no GPU and no downloads beyond PyPI.

```
                    ┌──────────────┐
   add_request ───► │  Scheduler   │  FCFS / SJF / LIFO, admission control,
                    │              │  preempt-by-recompute under pressure
                    └──────┬───────┘
                           │ scheduled requests (prefills + decodes together)
                           ▼
                    ┌──────────────┐        ┌────────────────────────┐
                    │  PagedBatch  │◄──────►│    PagedKVCache        │
                    │  (flattened) │ slots  │  block allocator,      │
                    └──────┬───────┘        │  refcounts, COW        │
                           │                └────────────────────────┘
                           ▼                            ▲
                    ┌──────────────┐                    │ gather / write
                    │ Transformer  │────────────────────┘
                    │ forward_paged│   RMSNorm · RoPE · SwiGLU · tied weights
                    └──────┬───────┘
                           │ logits
                           ▼
                    ┌──────────────┐        ┌────────────────────────┐
                    │   Sampling   │◄──────►│ SpeculativeDecoder     │
                    │ T / top-k/p  │        │ draft → verify → roll  │
                    └──────┬───────┘        │ back rejected KV       │
                           │                └────────────────────────┘
                           ▼
                    RequestOutput  (TTFT, TPOT, latency, acceptance rate)
```

## Quickstart

```bash
make setup          # venv + CPU-only torch + the package
make data           # fetch TinyShakespeare
make train          # target (~25 min) and draft (~7 min), one CPU core
make verify         # prove speculative decoding preserves the distribution
make benchmark      # every sweep
make report         # plots + RESULTS.md
make check          # ruff + mypy --strict + 181 tests
```

Or use it as a library:

```python
from microserve.config import CacheConfig, EngineConfig, SamplingConfig
from microserve.engine import LLMEngine
from microserve.model import Transformer
from microserve.tokenizer import CharTokenizer

model = Transformer.from_checkpoint("artifacts/target.pt")
engine = LLMEngine(
    model,
    EngineConfig(model=model.cfg, cache=CacheConfig(block_size=16, num_blocks=256)),
    tokenizer=CharTokenizer.from_file("artifacts/tokenizer.json"),
)
print(engine.generate(["ROMEO:"], SamplingConfig(temperature=0.8, max_new_tokens=200))[0].text)
```

```
ROMEO:
Away and myself that beats them know this maids
Which with many itself for party-date, who were the
addle, and you were slain protector restorments;
```

Requests can also be added while the engine is running — that is what
continuous batching means:

```python
engine.add_request(prompt_ids, SamplingConfig(max_new_tokens=64))
while engine.has_unfinished:
    for finished in engine.step():
        print(finished.request_id, finished.ttft, finished.text)
```

## What is actually verified

The paged path is not asserted to be correct, it is *tested* to be:

| Claim | How it is checked |
|---|---|
| Paged attention == dense attention | Prefill, decode, mixed prefill+decode batches, unequal lengths, block sizes 1–64, deliberately non-contiguous block tables |
| Preemption does not change output | Greedy runs with a cache tight enough to force recompute preemptions still match the dense reference token for token; a guard test asserts preemptions actually occurred |
| Batch composition does not change output | A request run alone and run in a crowded batch produces identical tokens |
| No block leaks | Every test asserts the allocator's invariants and a full pool at teardown |
| Speculative decoding preserves the target distribution | Model-free chi-square on 30k emissions **plus a negative control** that the naive rule fails; greedy equivalence at γ ∈ {1,2,3,5}; end-to-end chi-square at 5,000 trials |
| Determinism | Same seed → same tokens; different seeds diverge |

181 tests, `ruff` clean, `mypy --strict` clean, on Python 3.11 / 3.12 / 3.13.

## Headline results

Single CPU thread, 812K-parameter target, 64 requests, lognormal prompt lengths
(mean 100 tokens), **variable** output lengths (uniform 1–50 — with a constant
output length, static and continuous batching are the same algorithm and the
comparison would be meaningless). Full tables and plots in
[RESULTS.md](RESULTS.md); reasoning in [DESIGN.md](DESIGN.md).

**Continuous batching cuts time-to-first-token by 4–5.5× at low load — and
loses throughput at high load.**

| arrival rate | TTFT p95 continuous | TTFT p95 static | throughput continuous | throughput static |
|---|---|---|---|---|
| 4 req/s | **12 ms** | 48 ms | 90 tok/s | 90 tok/s |
| 16 req/s | **19 ms** | 105 ms | 354 tok/s | 355 tok/s |
| 64 req/s | 755 ms | **482 ms** | 783 tok/s | **973 tok/s** |
| 256 req/s | 1686 ms | **1116 ms** | 739 tok/s | **1078 tok/s** |

The second half is the opposite of the usual result, and it is real rather than
a bug: our gather materialises `B × max_context` slots of K/V, so a
heterogeneous batch pays for its longest member — and on CPU a 3 MB model is
compute-bound, so batching does not amortise weight loads the way it does on a
GPU. Continuous batching delivers the concurrency it promises (mean batch 12.2
vs 6.3); the hardware effect that normally turns concurrency into throughput is
simply absent at this scale. [DESIGN.md §6](DESIGN.md#6-what-the-measurements-showed)
works through it.

**Shortest-job-first is a large, free win under load.**

| arrival rate | FCFS TTFT p95 | SJF TTFT p95 | LIFO TTFT p95 |
|---|---|---|---|
| 64 req/s | 887 ms | **546 ms** | 760 ms |
| 256 req/s | 3714 ms | **2702 ms** | 3835 ms |

**Paging bounds fragmentation, and preemption degrades gracefully.**

| block size | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| internal fragmentation | 1.1% | 2.7% | 5.8% | 11.3% | 21.8% |

| cache blocks | 48 | 64 | 96 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| mean utilisation | 0.89 | 0.85 | 0.79 | 0.68 | 0.34 | 0.17 |
| preemptions | 32 | 22 | 24 | 2 | 0 | 0 |

**Speculative decoding is exactly correct and barely faster here.**

| γ | 1 | 2 | 4 | 7 |
|---|---|---|---|---|
| acceptance rate | 0.74 | 0.72 | 0.59 | 0.37 |
| wall-clock speedup | 1.02× | **1.06×** | 0.95× | 0.70× |
| chi-square p-value | 0.43 | 0.14 | 0.26 | 0.39 |

Every p-value is well above 0.05: the speculative sampler is statistically
indistinguishable from sampling the target directly, which is the property the
rejection rule exists to guarantee. The speed story is honest and unflattering
— an 812K-parameter target on CPU is dominated by per-step dispatch overhead,
so `γ` draft steps cost nearly as much as the target steps they replace.

## Limitations

Read these before trusting a number:

- **The gather is not a fused kernel.** It materialises K/V into a dense
  tensor, where CUDA would walk the block table inside the attention kernel.
  This is the biggest gap from a production engine, and it is what makes the
  throughput comparison above come out the way it does.
- CPU, float32, single thread. No CUDA, quantisation, or flash-attention.
- No chunked prefill — one long prompt produces one expensive step.
- Prefix sharing (`fork` + copy-on-write) is implemented and tested but nothing
  automatically detects shared prefixes yet.
- Preemption is recompute-only; swap is explicitly rejected, not stubbed.
- SJF uses each request's declared `max_new_tokens`, which a real server does
  not know — it is an oracle upper bound.
- Character-level tokenizer, ~1M-parameter models.
- Benchmarks are wall-clock on a shared desktop. Trends across a sweep are
  meaningful; single-point differences under ~10% are noise.

## Repository layout

```
src/microserve/
  config.py       every tunable, as dataclasses
  tokenizer.py    character-level tokenizer
  model.py        transformer; forward (dense reference) + forward_paged
  cache.py        block allocator, paged KV storage, refcounts, COW
  batching.py     slot reservation → PagedBatch (shared by both decode paths)
  scheduler.py    queueing policy, admission control, preemption
  engine.py       the serving loop
  sampling.py     temperature / top-k / top-p, greedy as a one-hot distribution
  speculative.py  draft-then-verify with exact rejection sampling
  stats.py        chi-square test (so tests need no SciPy)
  data.py         corpus loading and batching
scripts/
  train.py                 train the target / draft models
  verify_speculative.py    greedy + distributional equivalence, speedup sweep
  benchmark.py             workload generator and sweeps
  report.py                plots + RESULTS.md
tests/                     181 tests
artifacts/                 checkpoints, loss curves, benchmark JSON, plots
```

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving
  with PagedAttention* (vLLM), 2023 — paged KV cache and continuous batching.
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*,
  2023 — the accept/reject rule implemented in `speculative.py`.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
- Shazeer, *GLU Variants Improve Transformer*, 2020 — SwiGLU.
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019.
- Karpathy's char-rnn TinyShakespeare corpus.

## License

MIT — see [LICENSE](LICENSE).
