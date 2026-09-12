# Benchmarking

For the serial baseline, measure an otherwise idle, healthy server with one
client and one request at a time. Fix the checkpoint, GPU order, power limits, runtime, context, kernels,
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

## Concurrent streams

[benchmark_concurrency.py](../tools/benchmark_concurrency.py) starts a specified
number of clients together and checks a unique marker at the beginning of every
response. It uses exact tokenized input lengths, temperature zero and forced
output lengths. Its fixture differs from the historical serial benchmark above;
compare concurrency levels using the same tool and output length.

```sh
python3 tools/benchmark_concurrency.py \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --concurrency 4 --prompt-tokens 32768 --output-tokens 1024 \
  --output-dir results/concurrent-4-32k
```

By default each prompt has a fresh prefix, and any measured prefix reuse fails
the run. Add `--shared-prefix` to prewarm a common prefix before the timed group.
Report that case separately: sharing a prefix reduces both prefill work and
physical cache occupancy. It does not establish that the same number of wholly
independent long histories fits.

The tool samples actual running and waiting request counts and KV occupancy
every 250 ms. It checks completed-request and generated-token counters against
the entire group and records preemptions. All clients must use the same process,
and unrelated requests invalidate a measurement. A long fresh prefill can delay
new streams until earlier streams have finished: report observed running counts,
not merely the number of HTTP clients. Sampling can miss shorter peaks.

Aggregate output rate includes the entire group's prefill and decode wall time.
Per-stream decode rate can include pauses caused by other clients' prefills.
It is not the same as dividing aggregate throughput by configured concurrency.

For controlled repeats, set `--fixture-seed 42`. This reproduces model-visible
prompts and request markers while a fresh, unreported vLLM `cache_salt` prevents
reuse across timed groups. The endpoint must support that request field. Response
hashes help distinguish output/workload variation from runtime variation; seeded
inputs and temperature zero do not guarantee identical batched floating-point
results. Keep the seed, concurrency and token lengths the same when comparing
repeats.

## Conversation soak

[soak_conversations.py](../tools/soak_conversations.py) repeatedly exercises a
three-request conversation: an automatic tool call, a synthetic tool result
followed by a longer answer, and a checked follow-up. Request-specific markers
check that tool arguments and conversation state stay with the right client.
Generated code is never executed. This is API-level agent traffic, not a run of
an external agent harness.

```sh
python3 tools/soak_conversations.py \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --clients 24 --seconds 3600 --output-dir results/conversation-soak
```

Client eligibility changes every 45 seconds, producing quiet and busy periods
and dynamic batch transitions. Existing conversations finish before workers
pause. Histories use small, medium and long synthetic reference texts; exact API
token counts are retained per request, rather than inferred from text size.
Each conversation reuses its prefix, while a new conversation gets a fresh one.
Outputs end naturally or at the 512-token limit.

The soak stores request timings/usage, correctness failures and one-second server
samples. It requires the requested elapsed duration and completed conversations.
Unlike the isolated throughput benchmark, it can run alongside ordinary traffic:
its own request counts are separate from process-wide counters. Also retain pod
restart counts, GPU memory samples and server/driver errors independently. A
short successful burst cannot replace a sustained run, and an hour cannot prove
indefinite stability.
