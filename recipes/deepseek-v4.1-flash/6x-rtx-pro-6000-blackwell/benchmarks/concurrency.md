# Concurrent DeepSeek V4.1 Flash on six Blackwell GPUs

The parallel profile keeps the original checkpoint, TP2 × PP3 partition
`8,12,20`, FP8 KV allocation of 1.8 GiB per worker, 1,536-token prefill chunks,
B12X kernels and FlashInfer PCIe IPC. It raises the active-request limit to 24
and captures decode graphs for 1, 2, 4, 8, 16 and 24 requests. The six selected
96 GB cards all use PCIe x16. CPU weight and Engram offload stay disabled.

The roughly **1,164,421-token KV capacity is shared**, while **1,048,576 is the
per-request input-plus-output limit**. Increasing the active-request limit does
not multiply the cache. The original [single profile](../profile.json) remains
the launcher's default; select [the parallel profile](../profile-parallel.json)
with `--profile parallel`.

[Numeric evidence](concurrency-2026-09-12.json) includes every staged and final
benchmark, per-stream timings, request/cache counter deltas and OMP checks.
The underlying runtime/profile was tested on the GPUs; the portable Docker
wrapper was rebuilt and CPU checked separately. It was not itself launched on
the six-GPU machine.

## Workspace change

The earlier patch bounded the indexer workspace only when `max_num_seqs=1`.
Allowing a second request restored the upstream allocation for 40 maximum-length
contexts. The new patch keeps one maximum-length gathered-key buffer and uses
the existing splitter to partition multiple requests and query rows.

The splitter must receive the **physical capacity after compression**. V4.1
allocates `max_model_len // compress_ratio` entries, so the ratio-2 metadata
must not schedule twice as many gathered keys. Tests execute the actual runtime
workspace function, metadata argument and chunk splitter for mixed lengths,
nonzero request offsets, ratios 1/2, and 128/512 MiB logits limits. The old
runtime fails these regressions; the new runtime passes them.

The runtime was then started and tested at limits 2, 4, 8, 16 and 24. Every stage
passed concurrent marker checks. Other model types retain the upstream workspace
policy. These patches target the pinned preview, not arbitrary vLLM releases.

## Short-input throughput

All rows below use the final 24-slot profile, exactly 1,024 input and 1,024 forced
output tokens per request, and temperature zero. Aggregate rate includes the
group's prefill and decode time. Per-stream decode rate excludes that stream's
initial wait, but can include pauses for other clients' prefills.

| Active streams | Aggregate output tokens/s | Per-stream decode tokens/s, median per run |
| ---: | ---: | ---: |
| 1 | 101.3 | 103.9 |
| 2 | 185.0 | 94.9 |
| 4 | 237.3 | 61.0 |
| 8 | 285.1–355.8 | 36.6–46.0 |
| 16 | 488.3 | 31.6 |
| 24 | 491.9–754.4 | 21.1–33.2 |

The ranges retain the observed variation rather than selecting the fastest
randomized run. One run was taken at 1/2/4/16, three at 8 and five at 24. The
latter groups include controlled repeats with fixture seed 42: eight streams
produced 353.6 and 355.8 tokens/s; 24 streams produced 754.4, 753.1 and 753.2.
Those seeded runs use identical model-visible inputs and fresh cache salts,
with zero prefix hits. Response hashes still differ: identical inputs and
zero temperature do not establish bitwise deterministic batched generation.
The precise cause of the wider randomized-run variation is not isolated.
No thermal slowdown was active when the GPU clocks were checked.

The earlier staged profiles have their own graph limits and randomized inputs.
Their rates are retained in the JSON but are not directly substituted for the
final profile's measurements. Use the [benchmark protocol](../../../../docs/benchmarking.md)
and repeat your own workload before treating a peak as a sustained rate.

## Context and prefix sharing

Each concurrent request below generates 1,024 output tokens. All listed clients
were observed running simultaneously. Every request returned its own marker;
all cases had **zero preemptions**, exact generation/request counters and no
unrelated traffic in their measurement windows.

| Clients × input tokens | Prefix | Aggregate output tokens/s | Median first token | Peak KV usage |
| --- | --- | ---: | ---: | ---: |
| 24 × 32,768 | Independent, fresh | 234.7 | 36.69 s | 74.8% |
| 8 × 131,072 | Independent, fresh | 68.2 | 52.46 s | 91.6% |
| 4 × 262,144 | Independent, fresh | 36.1 | 61.01 s | 90.6% |
| 24 × 131,072 | Prewarmed shared prefix | 489.1 | 1.72 s | 17.6% |
| 24 × 262,144 | Prewarmed shared prefix | 509.3 | 3.47 s | 28.0% |

Fresh long prompts spend substantial time in prefill. Their low end-to-end
output rate is not a steady decode rate. A common prewarmed prefix both avoids
that work and shares physical KV blocks; the shared rows do not demonstrate
capacity for 24 independent 128K or 256K histories. Waiting counts can briefly
reach clients-minus-one during a burst, even when every client later runs at once.

The exact single-request boundary test passed with **1,048,448 input + 128 output
tokens in 171.99 seconds**, recovering all three separated codes. That is close
to the prior single-request profile's 170.08 seconds, but the test has only one
repetition and checks synthetic record recall rather than general model quality.
Arithmetic, separate reasoning, streaming and an automatic tool round trip pass.

## Conversation soak and OMP

<!-- SOAK_RESULTS -->
The conversation soak passed after **61.23 minutes** with **2,505 completed requests** across **835 complete three-turn conversations**. All 24 clients were observed running together. There were **zero request or correctness errors, zero server preemptions and zero worker restarts**.

Actual input sizes across the soak requests ranged from **7,382 to 225,803 tokens**; these varied with conversation history and are separate from the exact-length context benchmarks above. The server's peak KV occupancy was **82.9%** and peak waiting count was **20**. The ordinary endpoint stayed available during this test, so server-wide counters can include other traffic. The JSON reports the driver's own request/token totals separately.

The minimum sampled free VRAM through the final profile's qualification and soak was **407 MiB**. Sampling at five-second intervals does not establish instantaneous minimum headroom.
<!-- END_SOAK_RESULTS -->

The soak uses the [conversation driver](../../../../tools/soak_conversations.py):
changing eligible client counts, fresh conversations, reused prefixes within
conversations, a checked tool call/result, longer answers and a recall follow-up.
Tool schemas remain present across turns. The follow-up does not repeat the
expected report code in the user message. This is API-level agent traffic with
synthetic tool results; generated code is never executed.

Separately, OMP **18.1.17** passed two turns in one real RPC session. The first
turn used 21,587 input tokens with no cache hits. The second used 21,610 input
tokens and reused **21,504** according to the server. OMP first-token times were
1.719 and 0.257 seconds. Both windows contain exactly one completed request and
six generated tokens. OMP's own `cacheRead` field remains zero because the
endpoint omits the corresponding usage detail; server counters establish reuse.

## Memory and operational limits

The 24-slot functional qualification sampled a minimum **409 MiB free VRAM** at
five-second intervals. Sampling can miss lower transient headroom. Two allocator
mapping retries occurred during weight preparation; there were no further
mapping warnings, serving errors or worker restarts through the functional
qualification. The pre-promotion soak is reported separately above.

Keep the fixed 1.8 GiB KV budget and 1M per-request limit together. Increasing
`gpu-memory-utilization` alone has no effect on an explicit KV-byte allocation.
Do not copy GLM's DCP configuration: the pinned V4.1 indexer rejects DCP above
one when compression exceeds one. The ratio-2 caches here therefore require
DCP1. No additional CPU/SSD offload was introduced to enable concurrency; the
server, loading, compilation and filesystem cache still require host RAM.
