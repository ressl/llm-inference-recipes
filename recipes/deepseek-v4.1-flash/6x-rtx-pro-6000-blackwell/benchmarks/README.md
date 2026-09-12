# Measured results — 2026-09-12

These measurements qualify the underlying six-GPU runtime and configuration.
They precede extraction into the portable public package. The public Docker
build and CPU checks are separate validation; they are not a new GPU benchmark.

Hardware: six RTX PRO 6000 Blackwell 96 GB cards, all selected links x16, one
NUMA node, four 600 W and two 350 W power limits. Driver 615.71.09 and Linux
7.0.0-31-generic. Model revision and dependency pins are in the
[recipe](../README.md). No CPU weight or Engram offload; one request at a time.

## What made the difference

| Experiment | 1K decode before → after | 32K | 128K | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Eager → corrected CUDA graphs | 15.48 → 60.72 | 15.49 → 60.73 | 15.18 → 60.65 | About 4×; largest single improvement |
| NCCL P2P disabled → PHB | 59.51 → 60.11 | 59.31 → 59.86 | 59.05 → 59.74 | About 1% |
| CUTLASS → Marlin dense kernels | 60.09 → 94.46 | 59.89 → 94.37 | 59.70 → 93.83 | About 57% |
| Fresh Marlin baseline → final combined profile | 94.59 → 104.86 | 94.06 → 103.37 | 93.77 → 103.84 | About 10–11% further client decode improvement |

Rates are output tokens/s during decode, excluding prefill. Each comparison
has its own baseline; gains must not be added. Marlin was an intermediate
configuration, while the final profile uses B12X for both linears and experts.
The early rows are historical summaries; the final tuning round includes
[per-run numeric evidence](results-2026-09-12.json).

## Final tuning round

| Variant | 1K tokens/s | 32K tokens/s | 128K tokens/s | 32K TTFT | 128K TTFT | Exact 1M test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh Marlin + NCCL, prefill 1024 | 94.59 | 94.06 | 93.77 | 2.689 s | 11.622 s | 187.10 s¹ |
| Marlin + FlashInfer PCIe IPC, prefill 1024 | 97.51 | 96.94 | 96.91 | 2.735 s | 11.779 s | 186.06 s |
| B12X linears + NCCL, prefill 1024 | 94.99 | 94.41 | 95.00 | 2.596 s | 11.196 s | 180.31 s |
| B12X linears + PCIe IPC, prefill 1536 | 99.31 | 99.56 | 98.29 | 2.447 s | 10.382 s | 175.38 s |
| B12X experts + Marlin linears + NCCL, prefill 1024 | 102.52 | 101.92 | 102.28 | 2.752 s | 11.677 s | 184.66 s |
| **B12X linears + experts + PCIe IPC, prefill 1536** | **104.86** | **103.37** | **103.84** | **2.383 s** | **9.848 s** | **170.08 s** |

¹ Baseline 1M timing is from its preceding acceptance run, not a repeat during
the fresh short-context baseline. Rows without B12X experts retain DeepGEMM
experts. All use the original checkpoint. The last two rows include the bounded
zero-sign normalization fix in the B12X adapter.

The final configuration shortened 128K TTFT by 15.3%. Its exact-context request
used 1,048,448 input and 128 forced output tokens and recovered three synthetic
records. Completion time fell from the earlier 187.10 s to 170.08 s (9.1%).

## Short prompts and measurement uncertainty

At 1K, median TTFT increased from 0.166 s baseline to 0.313 s in the final main
run. A focused three-run recheck after the context test measured 0.239 s and
103.08 decode tokens/s. The 512-token request still completed faster overall:
about 5.20 s versus 5.57 s baseline. The final setting favors decode and longer
prompts; it does not improve every latency measure.

The [protocol](../../../../docs/benchmarking.md) uses 512 forced output tokens,
unique prefixes, concurrency 1, three 1K runs, two 32K runs and one 128K run per
variant. Warmup is excluded. All 36 primary request windows and three focused
recheck windows had one completed request, 512 generated tokens and zero prefix
hits. Server/client decode rates agreed within 2.3%; all paired values are
included in the JSON. For the final main run, server decode rates were roughly
101.5–103.1 tokens/s, slightly below client estimates of 103–105 tokens/s.
These few repetitions are not confidence intervals or a multi-client study.

## Correctness and operational limits

Arithmetic, separate reasoning, automatic tool round trips and streaming passed
on the final runtime. The full-context fixture checks exact usage and three
record lookups; it does not establish general reasoning or coding quality.
B12X dense kernels use FP8 activations, compared with BF16 for Marlin. A local
matrix check against dequantized FP32 produced relative RMSE around 0.0267 for
B12X and 0.00166 for Marlin. That microbenchmark is not a model-level accuracy
evaluation and does not justify assuming numerical equivalence.

The final qualification had zero worker restarts, CUDA/NCCL errors or host GPU
Xid events. Two allocator mapping retries recovered during startup; there were
none while serving the acceptance and benchmark requests. Five-second sampling
observed a minimum **375 MiB** free VRAM. It can miss a lower instantaneous
minimum. The fixed profile is tightly packed and must be requalified after
changes that affect allocations.

## Experiments that were not promoted

- **Legacy custom all-reduce:** graph/IPC registration failed with expandable
  allocation. An experimental staging fix passed correctness but measured
  59.41/59.30/59.21 tokens/s versus 60.09/59.89/59.70. It provided no speed gain;
  the final recipe keeps the legacy path disabled.
- **Prefill 2048 on an earlier backend:** the 1M fixture passed in 174.36 s, but
  only 3 MiB remained free and there were 415 recovered allocator mapping
  failures. This was rejected for memory pressure, not reported as a fatal OOM.
- **Layer partition 8,11,21:** violated KV-sharing boundaries. Use the qualified
  8,12,20 partition.
- **Unbounded B12X zero-sign normalization:** a whole-tensor temporary caused a
  startup allocation failure. Chunking was a prerequisite for the final expert
  backend, not an independently measured decode speedup.

Changing the final prefill budget, concurrency, KV allocation or kernel is a new
experiment. Keep the original comparison data and rerun correctness, memory
observation and the exact-context test before treating a faster result as usable.
