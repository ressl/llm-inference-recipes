# Native vision qualification — 2026-09-12

DeepSeek V4.1 Flash supports image input in its original checkpoint. The earlier
`single` and `parallel` profiles deliberately selected language-only operation.
The new `vision` profile keeps TP2/PP3, partition `8,12,20`, B12X, decode graphs,
24 scheduler slots and the same fixed KV allocation, and enables up to four
images per request, including images retained in conversation history.

## Memory fix

The pinned upstream wrapper constructed the vision tower and aligner on every
pipeline stage. Its V2 runner only encodes images on the first stage. The recipe
now uses vLLM's existing missing-layer/weight-skip mechanism on later stages.
The first stage retains the encoder with tensor-parallel weight sharding
(`--mm-encoder-tp-mode weights`). The CPU regression exercises the actual
constructor and weight loader for first/later stages and image limits zero/four.
It fails on the unpatched wrapper and passes on the patched runtime.

This keeps the **1,164,421-token shared KV pool** and **1,048,576-token request
limit**, without CPU weight offload or an additional GPU. Images consume context
and processing time. The four-image cap is a serving limit, not a claim about
all capabilities of the model. A fifth image returns HTTP 400.

## Observed results

[Machine-readable results](vision-2026-09-12.json) record the exact patch/profile
hashes. All six GPUs had 600 W power limits for this run.

| Check | Result |
| --- | --- |
| Single image: shapes, colors, printed code | Correct; 0.86 s request time |
| Follow-up about the preceding image | Correct |
| Four images, portrait/landscape, ordered codes | All four correct; 1.22 s request time |
| Five-image limit | HTTP 400 as expected |
| 24 simultaneously submitted short requests | 12 distinct image-code reads and 12 arithmetic requests correct |
| Text arithmetic, reasoning, tools and streaming | Passed |
| Exact 1M boundary | 1,048,448 input + 128 output tokens; all three records recovered; 131.12 s |
| Existing five runtime regressions + vision allocation regression | Passed |

Request times include processing and generation; they are not decode tokens/s
or controlled comparisons with the earlier text-only benchmarks. Short mixed
requests do not prove that 24 encoders execute simultaneously or qualify every
mixture of long image conversations. The earlier one-hour soak and long-history
concurrency measurements remain measurements of the text-only profile.

A post-qualification GPU snapshot had **433 MiB minimum free VRAM** on the
selected cards; this is not a sampled peak. Two allocator mapping retries were
recovered during startup, and none appeared in the subsequent qualification
logs. Memory headroom remains small. Changing batch sizes, image processing,
runtime or topology requires new qualification.

## Run the profile and checks

Build the current recipe image as described in the [recipe](../README.md), then
add `--profile vision` to the launcher command. Keep only one profile running on
the selected six GPUs. Install Pillow in the environment used for image checks:

```sh
python3 -m pip install Pillow
python3 "$RECIPE/tests/acceptance_vision.py" \
  --url http://127.0.0.1:30000 --concurrency 24 \
  --output results/acceptance/vision.json
```

Clients must advertise this model as accepting text and images. For Open WebUI,
enable the workspace model's Vision capability while preserving its access
grants and desired defaults. Use OpenAI-compatible `image_url` content blocks
with PNG/JPEG data URLs or accessible image URLs. The checkpoint's native
processor controls resizing and its image-token budget; this profile does not
add a separate OCR service or external vision model.
