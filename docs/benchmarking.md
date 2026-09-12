# Benchmarking

Measure an otherwise idle, healthy server with one client and one request at a
time. Fix the checkpoint, GPU order, power limits, runtime, context, kernels,
allocator and prefill budget. Run baseline and candidate with the same protocol.

The [vLLM benchmark tool](../tools/benchmark_vllm.py) uses the historical fixture:
a unique prefix, repeated reference text and a code-generation instruction.
The server tokenizer applies the chat template with thinking disabled; the tool
then submits exactly the requested number of token IDs to `/v1/completions`.
It preserves the final 128 prompt tokens while trimming the reference text.

| Case | Input tokens | Forced output tokens | Repetitions |
| --- | ---: | ---: | ---: |
| Warmup, excluded | 256 | 64 | 1 |
| 1K | 1,024 | 512 | 3 |
| 32K | 32,768 | 512 | 2 |
| 128K | 131,072 | 512 | 1 |

Temperature is 1.0, top-p is 0.95, and `ignore_eos=true` forces the output length.
Generated code is a workload, not a quality evaluation, and is never executed.
The tool checks the stream terminator, finish reason and exact usage. It stores
per-run values and before/after server metrics in a new output directory.

TTFT is client time from request start to first nonempty content. Decode rate is
`(output_tokens - 1) / (last_content_time - first_content_time)`. End-to-end output
rate includes prefill and is reported separately. Stream chunks can contain
multiple tokens, so compare client rate with the server decode-time counter.

Each metric window must contain exactly one completed request, the expected
generation-token delta and zero prefix-cache hits. The tool filters by served
model and fails on missing counters, cached requests, counter resets or extra
traffic. Run it after startup warmup has completed. Metrics must come from the
same vLLM process serving the requests; a load-balanced endpoint is unsuitable.

```sh
python3 tools/benchmark_vllm.py \
  --url http://127.0.0.1:30000 \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --output-dir results/my-run
```

The tokenizer and metrics API used here are vLLM-specific. Other models can use
the tool if they support these fields and context lengths, but only the first
recipe has been GPU qualified. The adapted portable tool has CPU regression
tests; its historical predecessor produced the published measurements.

Report all repetitions and use medians for repeated cases. One 128K run and one
full-context check are limited evidence; repeat them for decisions needing
confidence intervals. The exact-context test verifies a boundary and three
synthetic records, not general long-context reasoning quality. Record startup
retries separately from serving errors and sample memory throughout the run.
Do not add percentage gains from experiments with different baselines.
