# Design notes

Why MicroServe is built the way it is, and what the measurements actually
showed — including the places where the textbook answer did not reproduce.

---

## 1. Two forward paths, one of which is the oracle

Every serving optimisation in this repository is a change to *how* keys and
values are stored and scheduled, never to the arithmetic. That makes a
reference implementation the single most valuable thing in the codebase.

`Transformer.forward` is the dense path: ordinary `(B, T)` batches, a KV cache
that is just a list of tensors, no paging. It is what training uses, and it is
slow and obviously correct.

`Transformer.forward_paged` is the production path. It reads and writes a
`PagedKVCache` and consumes a *flattened* batch, so one forward pass can carry
a 200-token prefill and six single-token decodes at once.

The contract between them is asserted, not assumed. `tests/test_model.py`
checks the two paths agree on prefill, on decode, on mixed batches, on
unequal-length batches, at block sizes 1 / 2 / 4 / 8 / 16 / 64, and when a
sequence's blocks are deliberately non-contiguous in the pool. For a single
sequence the two paths are bit-identical; in mixed batches they differ by
~1e-7, which is float32 matmul reassociation, not a logic difference.

This pays off constantly. Every bug found while building the scheduler and the
speculative decoder was caught as "the paged path stopped matching the dense
path", which is a far more useful signal than "the generated text looks worse".

## 2. Paged KV cache

The cache is a flat pool of `num_blocks × block_size` slots. A sequence owns an
ordered list of block ids — its *block table* — and logical position `p` lives
at physical slot:

```
block_table[p // block_size] * block_size + (p % block_size)
```

Two consequences make the whole system work:

- **No contiguity requirement.** A sequence can grow one token at a time
  without ever being copied or moved, so admitting and evicting requests costs
  a list append and a refcount decrement.
- **Internal fragmentation is bounded by `block_size - 1` tokens per
  sequence**, instead of by the difference between a sequence's reserved
  maximum length and its actual length. That bound is what the block-size sweep
  measures directly.

Blocks are reference counted, which makes `fork()` (share a prefix with a new
sequence) free and gives copy-on-write on the tail block. The refcounting and
COW machinery is implemented and tested; it is *not* yet wired to a prefix tree
that would deduplicate shared system prompts automatically — see Limitations.

### The gather is the honest compromise

On a GPU, paged attention is a fused kernel: the block table is walked *inside*
the attention kernel and the gathered K/V never exists as a tensor. There is no
way to write that in pure PyTorch on CPU.

So `PagedKVCache.gather` does the straightforward thing — it materialises a
contiguous `(B, max_context, H, D)` copy of K and V, and attention runs on
that. It is correct, it is honest, and it costs real memory traffic
proportional to `B × max_context`, whereas the useful work is proportional to
`sum(context_lengths)`. Section 6 shows this is not a footnote: it is the
reason one of the headline results comes out backwards.

## 3. Scheduler

The scheduler owns no tensors. It manipulates `Request` objects and asks the
cache whether an allocation would fit. Three decisions were worth making
carefully:

**Admission must debit blocks as it admits.** The first version compared each
candidate against `cache.num_free_blocks` and admitted several requests that
each individually fit but collectively did not; the failure surfaced one step
later as an out-of-memory error deep in `build_batch`. The scheduler now keeps
a running tally. `tests/test_scheduler.py::test_does_not_over_commit_blocks`
locks the behaviour in.

**Preemption is by recomputation.** When the pool cannot satisfy every running
sequence, the lowest-priority victim releases *all* of its blocks and returns
to the waiting queue with its generated tokens intact — so when it is
readmitted it re-prefills on `prompt + generated so far`. Recompute costs one
prefill of work; the alternative (swapping KV to host memory) costs a copy in
each direction and, on a CPU-only engine where "host memory" is the only
memory, would be meaningless. `SchedulerConfig(preemption_mode="swap")` raises
rather than silently doing something else.

Crucially, preemption is *output-preserving*: the greedy equivalence tests run
with a cache tight enough to force preemptions and still demand token-for-token
identity with the dense reference.

**A single running request is never preempted**, because there is nobody to
preempt it in favour of. If a request cannot fit in an empty cache it is
rejected at `add_request` time with an explicit error, rather than being
admitted and failing later.

## 4. Continuous vs static batching

Static batching forms a batch and runs it until *every* member finishes.
Continuous batching lets sequences join and leave at every step.

The two are only distinguishable when output lengths vary — with a constant
output length, every member of a static batch finishes on the same step and the
two algorithms are literally the same schedule. The workload generator
therefore draws output lengths uniformly from 1..50. This is the single most
important detail in the benchmark harness; getting it wrong would produce a
headline number that measures nothing.

## 5. Speculative decoding

The draft proposes `gamma` tokens; the target scores all `gamma + 1` positions
in one pass; each proposal is accepted with probability
`min(1, p_target(q) / p_draft(q))`, and on the first rejection the token is
resampled from the normalised positive part of `p_target - p_draft`. If every
proposal survives, the target's distribution at the last position yields a free
bonus token.

The residual is the whole point: it is what makes the emitted token distributed
*exactly* as `p_target`, no matter how bad the draft is. A naive
"accept everything" implementation would be faster and wrong, and the wrongness
would be invisible in any eyeball test of the output text.

So it is tested three ways:

1. **Unit, model-free.** With `gamma=1` on synthetic distributions, 30,000
   emissions are compared against direct draws from `p_target` by a chi-square
   test of homogeneity. Paired with a **negative control** that runs the same
   test against the naive accept-everything rule and asserts it *fails* — a
   test that cannot fail proves nothing.
2. **Greedy, end-to-end.** At temperature 0, speculative output must equal the
   dense reference token for token, at every gamma. A guard test asserts the
   acceptance rate is strictly between 0 and 0.95, because at 100% acceptance
   the equivalence test would be vacuous. (Freshly initialised transformers are
   degenerate — they all predict "repeat the last token" and agree perfectly —
   which is exactly the trap that guard exists to catch.)
3. **Distributional, end-to-end.** `scripts/verify_speculative.py` samples the
   first emitted token 5,000 times at temperature 1 and chi-square tests it
   against exact draws from the target's own next-token distribution.

Rolling back is where the implementation bugs live. After a round the target
cache must hold exactly `prompt + generated - 1` tokens: the rejected drafts'
KV entries are released by `PagedKVCache.truncate`, and the newest token is
deliberately left uncomputed so the ordinary decode invariant still holds. The
draft keeps its own pool, truncated to the same accepted length.

## 6. What the measurements showed

Full numbers and plots: [RESULTS.md](RESULTS.md). Single CPU thread, 812K-param
target, 64 requests, lognormal prompts (mean 100 tokens), uniform 1..50 output
lengths.

### Continuous batching improved latency and *lost* throughput

| arrival rate | TTFT p95, continuous | TTFT p95, static | throughput, continuous | throughput, static |
|---|---|---|---|---|
| 4 req/s | **12 ms** | 48 ms | 90 tok/s | 90 tok/s |
| 16 req/s | **19 ms** | 105 ms | 354 tok/s | 355 tok/s |
| 64 req/s | 755 ms | **482 ms** | 783 tok/s | **973 tok/s** |
| 256 req/s | 1686 ms | **1116 ms** | 739 tok/s | **1078 tok/s** |

The low-load half is the expected result, and a large effect: continuous
batching cuts TTFT p95 by 4–5.5× because an arriving request joins the very
next step instead of waiting for the current batch to drain.

The high-load half is backwards from the usual vLLM-style result, and it is not
a bug — it is section 2's compromise showing up as a number. Continuous
batching does exactly what it advertises: mean concurrency rises from 6.3 to
12.2 sequences. But on this engine a bigger batch is not cheaper per token:

- The gather materialises `B × max_context` slots of K/V. In a heterogeneous
  batch `max_context` is set by the longest member, so a batch of twelve
  sequences where one is long pays twelve times the long one's cost. Static
  batching's smaller, more homogeneous batches waste far less.
- On a GPU, batched decode wins because it amortises reading the weights out of
  HBM across the batch — decode is memory-bandwidth-bound. Here the whole model
  is 3 MB and lives in cache, so decode is compute-bound and batching buys
  nothing to offset the padding waste.

The mechanism that makes continuous batching a throughput win on real hardware
is absent at this scale, and the mechanism that makes it a latency win is not.
Reporting only the latency half would have been the more flattering choice and
the less honest one.

### The other sweeps behaved

- **Scheduling policy.** SJF beats FCFS decisively under load — TTFT p95 546 ms
  vs 887 ms at 64 req/s, 2.70 s vs 3.71 s at 256 req/s — the classic result that
  short jobs should not queue behind long ones. LIFO tracks FCFS on p95: it
  helps whoever arrived most recently at the direct expense of everyone else,
  which p95 is designed to notice. Note SJF here is an *oracle*: it uses each
  request's declared `max_new_tokens`, which a real server does not know.
- **Block size.** Internal fragmentation rises monotonically from 1.1% at
  `block_size=4` to 21.8% at `block_size=64` — the `block_size - 1` bound,
  measured. Small blocks mean longer block tables and more gather indices, so
  the choice is a real trade; 16 is a reasonable default.
- **Cache capacity.** Shrinking the pool from 512 to 48 blocks drives mean
  utilisation from 0.17 to 0.89 and preemptions from 0 to 32, with output still
  identical to the reference throughout. Preemption is a graceful-degradation
  mechanism and it degrades gracefully.
- **Batch size.** Throughput peaks at `max_batch_size=4` (1064 tok/s) and falls
  away on both sides — same padding effect as above.
- **Speculative decoding.** Acceptance falls from 0.74 at `gamma=1` to 0.37 at
  `gamma=7`. Wall-clock speedup peaks at a thin **1.06× at `gamma=2`** and drops
  below break-even from `gamma=4`. The draft has 8× fewer parameters but is
  nowhere near 8× cheaper per step: at this size, per-step Python and PyTorch
  dispatch overhead dominates the matmuls, so `gamma` draft steps cost far more
  than the theory's `gamma / 8` target steps. Speculative decoding needs a
  target model expensive enough that the draft is genuinely cheap in
  comparison; an 812K-parameter model on CPU is not that.

## 7. Limitations

Stated plainly, because several of them shape the results above:

- **The gather is not a fused kernel.** `PagedKVCache.gather` materialises K/V
  into a dense `(B, max_context, H, D)` tensor. This is the single biggest gap
  between MicroServe and a production engine, and section 6 quantifies what it
  costs.
- **CPU, float32, single thread.** No CUDA, no quantisation, no flash-attention.
- **No chunked prefill.** A prompt is prefilled in one pass, so a long prompt
  produces one expensive step that delays every decode sharing that batch.
- **Prefix sharing is implemented but not automatic.** `fork()` and
  copy-on-write work and are tested, but nothing yet detects that two requests
  share a prefix and forks them.
- **Preemption is recompute-only.** Swapping to host memory is rejected rather
  than stubbed.
- **SJF is an oracle**, as noted above.
- **Character-level tokenizer and tiny models.** A 65-token vocabulary keeps the
  embedding from dominating a 1M-parameter model; it is not a serious tokenizer.
- **Benchmarks are wall-clock on a shared desktop.** The driver runs in real
  time with real `sleep`, so absolute numbers carry scheduling noise — visible
  as non-monotonic throughput in the capacity and block-size sweeps. Trends
  across a sweep are meaningful; single-point differences under ~10% are not.

## 8. What I would do next

1. Chunked prefill, so a long prompt cannot stall a batch of decodes.
2. A prefix-radix tree over block tables, to make the existing COW machinery
   pay for itself on shared system prompts.
3. Bucket sequences by context length when forming a batch, which would
   directly attack the `max_context` padding waste and probably flip the
   throughput result in section 6.
4. A draft model chosen for *step cost* rather than parameter count — or
   Medusa-style multi-head drafting that reuses the target's own trunk and
   removes the separate draft forward pass entirely.
