# Native FP4 cache: two full 1M contexts

The September 16, 2026 target-only configuration passed **two independent
1,048,576-token contexts** and **four independent 524,288-token contexts** on six
RTX PRO 6000 Blackwell 96 GB cards with x16 PCIe links. Each context includes
input and generated output. DSpark was disabled throughout these tests.
[Machine-readable measurements](fp4-cache-2026-09-16.json) accompany this report.

## What changed

The native checkpoint, GPU-resident Engram, TP2/PP3 partition `8,12,20`, index
selection, compressor and model projections remain in use. A narrow adapter
uses B12X's existing SM120/SM121 DeepSeek V4.1 cache writers and compressed
sparse MLA kernels, pinned to
[`38ae4b6`](https://github.com/local-inference-lab/b12x/tree/38ae4b6cc1cde0f04d6ac45b7367004c12d1c53e).
It does not adopt the separate serving fork's approximate CED prefill.

| Record | Previous bytes | New bytes / format |
| --- | ---: | --- |
| Main latent | 584 | 288 / NVFP4 |
| Sliding-window latent | 584 | 528 / block-scaled FP8 |
| Indexer key | 132 | 132 / FP8 |

Main records use 512 E2M1 values and 32 E4M3 scales, one per 16 values.
SWA uses 512 E4M3 values and 16 UE8M0 scales, one per 32 values.
The RoPE dimensions are included. Main logical blocks stay at 64 tokens;
compression ratios 2/1 use 32/64 physical entries. Indexer pages remain 64 entries.

Three integration details are essential:

- Clear inherited `page_size_padded` when replacing the main cache spec.
  Merely changing the record width retains legacy padding and loses the gain.
- Keep circular state buffers in separate cache groups when pipeline stages
  project the global layer groups into their local pools. Otherwise small
  state groups can widen every block in the middle stage.
- Borrow shared workspace and use one split per prefill row. Reserving the
  decode kernel's 64 splits for all 1,536 prefill rows wastes several GiB.

The original loader remains in use. `VLLM_PLUGINS` is empty because this B12X
revision's optional loader plugin expects a different vLLM fork.

## Capacity and measured concurrency

The cache budget remains **1,932,735,283 bytes (1.8 GiB) per worker**.
The installed planner and real startup report **2,101,450 token equivalents**,
up from 1,164,421 (**+80.47%**). There are 57,195 blocks, versus 29,491 before;
the local worker stride is 33,792 bytes. The conservative estimate includes
6,144 tokens in flight.

| Independent sessions | Input + output per session | Decode overlap across all sessions | Maximum completion time |
| --- | --- | ---: | ---: |
| 2 × 128K | 129,024 + 2,048 | 29.05 s | 67.3 s |
| 2 × 1M | 1,046,528 + 2,048 | 23.35 s | 466.2 s |
| 4 × 512K | 522,240 + 2,048 | 31.34 s | 533.8 s |

Each prompt starts with a different random identity to prevent meaningful
prefix sharing. Three distinct codes near 10%, 50% and 90% must be recalled
in order. Exact token usage, completed streams and overlapping decode are
checked. All cases passed with zero recorded preemptions. The two 1M streams
produced their first visible tokens after 198.8 and 440.3 seconds.

The test forces generation past EOS to occupy the entire output budget.
It demonstrates capacity and synthetic recall, not normal-answer throughput,
broad accuracy parity, or a long-duration reliability result. FP4 is lossy;
these checks are not a substitute for task-specific quality evaluation.

The 24-request scheduler limit shares the pool. It does not provide 24 full
1M contexts, and four full 1M contexts do not fit this configuration. Long
prefill still takes minutes and slows streams that are already decoding.

During 2×1M, the middle TP pair reached sampled minima of **475/395 MiB free
VRAM**, while the first pair retained about 13.4 GiB and the last pair about
19.2 GiB each. The middle stage remains the memory constraint. Retain the
fixed cache budget; unused memory on another stage is not directly fungible.

## Functional and kernel qualification

- Forty upstream B12X V4.1 GPU tests passed, including cache encoding,
  large mapped offsets and graph replay.
- Six local geometry cases passed with 32 query heads, main pages 32/64,
  SWA-only layers, 24-row decode, 1,536-row prefill and strided storage.
  RoPE ratios 1/2 and changed-input graph replay were checked.
- Actual adapter cache-spec and installed planner regressions passed, alongside
  existing PP, indexer, vision and packed-zero checks.
- Generation, reasoning, tool round trips, streaming, OCR, shapes and image
  history passed. The configured image limit is 16, including history;
  limit-plus-one rejection passed. This does not establish quality for every
  possible 16-image request.
- Tests with 8 and 24 clients submitting short mixed text/image requests passed.
  These are submission tests, not proof that all 24 decoded at the same instant.

The full-model measurements used the deployed six-GPU runtime. This repository
ports the same adapter and patch logic with portable names and packaging.
A fresh six-GPU full-model launch of this public Docker wrapper is not claimed.
The public Docker build passed, its installed planner reproduced 57,195 blocks
and 2,101,450 tokens, and all 13 portable CPU tests passed. The entrypoint also
rejected an FP4 image/profile mismatch before GPU initialization.

## Build and use the FP4 profile

From the repository root, after preparing the model and GPUs as described in
the [recipe](../README.md):

```sh
export RECIPE=recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell
docker build --platform linux/amd64 --build-arg ENABLE_FP4_CACHE=1 \
  -t llm-inference-recipes/deepseek-v41:2026-09-16-fp4 "$RECIPE"
python3 "$RECIPE/launch.py" --profile fp4 --gpus "$GPUS" \
  --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" --dry-run
python3 "$RECIPE/launch.py" --profile fp4 --gpus "$GPUS" \
  --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR"
```

Use the `fp4` image and profile together. The entrypoint rejects a mismatch.
The selector flag still says `--kv-cache-dtype fp8` because the upstream
metadata path requires it; the adapter explicitly defines the physical
NVFP4/FP8 layouts. Do not change that flag to an unsupported `fp4` value.

For CPU planner checks inside the built image:

```sh
docker run --rm --platform linux/amd64 --env VLLM_PLUGINS= \
  --entrypoint python3 llm-inference-recipes/deepseek-v41:2026-09-16-fp4 \
  /opt/recipe/tests/test_fp4_planner.py
```

On an isolated matching GPU environment, run
[`qualify_fp4_cache.py`](../tests/qualify_fp4_cache.py) before full-model tests.
Once the server is healthy:

```sh
python3 "$RECIPE/tests/acceptance.py" --url http://127.0.0.1:30000 \
  --output /tmp/fp4-api.json
# Requires Pillow in your client environment.
python3 "$RECIPE/tests/acceptance_vision.py" --image-limit 16 --concurrency 24 \
  --output /tmp/fp4-vision.json
python3 "$RECIPE/tests/acceptance_concurrent_context.py" --sessions 2 \
  --context 1048576 --output /tmp/fp4-2x1m.json
python3 "$RECIPE/tests/acceptance_concurrent_context.py" --sessions 4 \
  --context 524288 --output /tmp/fp4-4x512k.json
```

Expect these long-context tests to occupy the serving GPUs for several minutes.
For rollback, stop/remove the FP4 container and launch the separately built
legacy `vision` or `parallel` profile. Cache contents are incompatible and
must be recreated on restart; model weights need not be downloaded again.

## DSpark status

DSpark was disabled on September 16 after the operator reported intermittent
problems suspected to be related to it. No controlled A/B reproduction or
DSpark-specific root cause has been established in the published evidence.
The successful September 14 benchmarks remain historical measurements;
they do not establish long-term reliability.

The current FP4 adapter explicitly rejects speculative decoding. Combining
this cache path with DSpark is **not qualified**. Use the target-only FP4
profile for the configuration measured here. See the
[DSpark status note](https://github.com/ressl/vllm/blob/dspark-pipeline-parallel/docs/features/speculative_decoding/dspark_pipeline_parallel.md#status-update-september-16-2026)
for the earlier fork's scope and unresolved stability report.
