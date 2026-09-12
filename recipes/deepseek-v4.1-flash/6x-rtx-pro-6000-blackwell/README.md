# DeepSeek V4.1 Flash on six RTX PRO 6000 Blackwell GPUs

Three profiles for six 96 GB cards with PCIe x16 links: `single` allows one
active text request, `parallel` allows up to **24 active text requests**, and
`vision` adds native image input with the same 24-request limit. All keep
a **1,048,576-token per-request limit** and share the same fixed KV allocation.
Weights and Engram remain on the GPUs; CPU weight offload is disabled.

The parallel profile passes **24 independent 32K**, **eight independent 128K**
and **four independent 256K** input contexts, with 1,024 output tokens per stream.
These are separate measured cases, not a guarantee for every mixture. See the
[concurrency results](benchmarks/concurrency.md) and the
[earlier single-request tuning](benchmarks/README.md).

Status: the earlier text-only runtime/profile passed GPU qualification and a one-hour
conversation soak on 2026-09-12. This
portable Docker packaging was subsequently rebuilt and CPU checked. A fresh
six-GPU run of the public launcher is not claimed. Its entrypoint, mounts,
user permissions and readiness handling replace the original deployment wrapper.

## Requirements

- Linux x86-64, Docker Engine and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Six available **RTX PRO 6000 Blackwell 96 GB** GPUs, each reporting a current
  x16 link. Select three suitable two-GPU TP pairs on one NUMA node, with working
  PCIe peer access. Review the [P2P guide](../../../docs/pcie-p2p.md).
- The measured system used driver **615.71.09** and kernel **7.0.0-31-generic**.
  Original qualification used four 600 W and two 350 W GPU power limits;
  a separate [600 W follow-up](benchmarks/power-600w.md) records the later change.
  Other drivers/topologies need validation.
- The checkpoint's 48 safetensors files total **510,296,708,312 bytes**. Engram's
  203,073,077,576 bytes are included in that total. Plan at least 650 GB of space
  for the model and writable caches, plus separate Docker build/image storage.
  Peak temporary download/build usage depends on the local cache setup.
- Host RAM is still needed for loading, the server, compilation and filesystem
  cache. The launcher sets a **96 GiB container memory limit** and 16 GiB shared
  memory; these are configured limits, not a measured minimum RAM requirement.
  Zero CPU offload does not mean zero host RAM use.
- Python 3.12 on the host for the launcher and API checks. Install the Hugging
  Face `hf` CLI separately for downloading the checkpoint.

This profile has very little VRAM headroom. Do not share these six cards with
other workloads. GPU indices are not accepted: supply full UUIDs in TP-pair order.

## Download and build

Run from the repository root. Choose two separate paths on your own machine:

```sh
export MODEL_DIR=/absolute/path/to/deepseek-v41
export CACHE_DIR=/absolute/path/to/deepseek-v41-cache
export RECIPE=recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell

hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --local-dir "$MODEL_DIR"

docker build --platform linux/amd64 \
  -t llm-inference-recipes/deepseek-v41:2026-09-12-vision "$RECIPE"
```

Download/build require network access; the serving profile uses offline Hugging
Face/Transformers mode. The launcher checks basic checkpoint files but does not
hash all 510 GB or establish that a reused directory matches the pinned revision.
Use a dedicated complete download of the revision above. Model files must be
readable and the cache writable by your current user. Avoid running the launcher
as root if you want its user-ID mapping to remain unprivileged.

## Select GPUs and start

```sh
nvidia-smi --query-gpu=uuid,name,pci.bus_id,pcie.link.width.current,memory.total --format=csv
nvidia-smi topo -m
```

Set `GPUS` to six real full UUIDs, comma-separated: first TP pair, second TP
pair, third TP pair. For example, the *shape* of the value is
`GPU-<uuid1>,GPU-<uuid2>,GPU-<uuid3>,GPU-<uuid4>,GPU-<uuid5>,GPU-<uuid6>`; replace
every placeholder with an identifier from your inventory.

```sh
python3 "$RECIPE/launch.py" --gpus "$GPUS" \
  --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" --dry-run

python3 "$RECIPE/launch.py" --gpus "$GPUS" \
  --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR"

docker logs --follow deepseek-v41-recipe
```

The entrypoint checks the exact six-device allocation, model name, memory size
and current x16 link widths before loading weights. It does not prove NUMA
placement or P2P correctness; check those separately. The endpoint is published
at `http://127.0.0.1:30000`, and the model directory is mounted read-only.

Wait until this reports `healthy` before using the API:

```sh
docker inspect --format '{{.State.Health.Status}}' deepseek-v41-recipe
```

The first successful health check sends 1K and 32K warmup prompts, then records
a local completion marker. Later checks only call `/health`. Cold startup can
take several minutes or longer because of model loading and kernel compilation;
the health check allows a 30-minute startup grace period. Docker health status
does not prevent clients from contacting the API early.

```sh
curl --fail http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"What is 17 times 19?"}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

To stop and remove this container later, keeping downloaded weights and cache:

```sh
docker stop deepseek-v41-recipe
docker rm deepseek-v41-recipe
```

## Fixed runtime and settings

The complete contracts are [profile.json](profile.json) for `single`,
[profile-parallel.json](profile-parallel.json) for `parallel`, and
[profile-vision.json](profile-vision.json) for `vision`. The launcher defaults
to `single`; add `--profile parallel` to the launch or dry-run command to select
24 active requests, or `--profile vision` for text plus up to four images per
request including history. See [vision qualification](benchmarks/vision.md).
Run one profile at a time on the same six GPUs.

| Setting | Value / reason |
| --- | --- |
| Checkpoint | [dba1be0a40aa45a94ad051997016db3960a90277](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277), original MXFP4 expert / FP8 dense weights |
| Public vLLM base | `vllm/vllm-openai@sha256:4f3c8bcf6328305b8cb6f61146dbfa125b0b04ae98eef3d091019ec30f564a9b` |
| Installed vLLM / PyTorch | `0.1.dev20904+g179dd0fa9` / `2.13.0+cu130` |
| FlashInfer | [c05407ceffb7d9ce111a73553a8e37ad752adb62](https://github.com/flashinfer-ai/flashinfer/tree/c05407ceffb7d9ce111a73553a8e37ad752adb62), built as 0.7.0 |
| B12X | 1.3.0, wheel SHA-256 `c97d88635521a7fdd4f67717c1835a017d0dcea49d49cc7aa5b4299f290d19e3` |
| Parallelism | TP2 × PP3; layer partition `8,12,20` |
| Concurrency / context | `single`: 1 active sequence; `parallel`: up to 24; 1,048,576 input + output tokens per request, sharing one cache pool |
| Weight / Engram offload | `--cpu-offload-gb 0`; Engram `cpu_offload=false` |
| KV / indexer budget | FP8 KV, fixed 1,932,735,283 bytes per worker; indexer logits 128 MiB |
| Prefill | 1,536 batched tokens |
| Graphs | `FULL_DECODE_ONLY`, compilation mode 0; `single`: size 1; `parallel`: sizes 1, 2, 4, 8, 16, 24 |
| Linear / expert kernels | B12X / B12X |
| Collectives | NCCL P2P allowed through PHB; FlashInfer PCIe IPC; legacy custom all-reduce disabled |
| Allocator | `expandable_segments:True` |
| Autotuning | General FlashInfer autotune disabled; PCIe IPC's TP-local tuning remains enabled |

Six-way TP does not divide the model's 64 attention heads. TP2/PP3 fits the
tested cards, and `8,12,20` respects the model's KV-sharing boundaries. The
alternative `8,11,21` was invalid. These are model-specific constraints.
The fixed KV-byte budget is intentional; increasing memory utilization alone
does not enlarge that explicit allocation.

The checkpoint and Engram are loaded from SSD at startup; normal inference does
not use CPU weight or SSD KV offload in this profile. SSD also holds compilation
caches. Keep the distinction between storage and runtime offload explicit.

## Patches and validation

[apply_runtime_patches.py](patches/apply_runtime_patches.py) refuses unexpected
upstream file hashes. It implements these compatibility and memory fixes:

- Vision tower and aligner allocated only on the first pipeline stage; later
  stages skip their weights. Language-only profiles remain supported.
- Current token counts on pipeline-parallel ordinary and CUDA graph paths.
- Empty KV-cache group handling.
- SM120 sparse-attention/indexer page-size compatibility, including FlashInfer's
  missing secondary-size-32 dispatch and separate physical indexer pages.
- Indexer workspace bounded to one full-context request's gathered keys.
  Multiple prefills are partitioned into request/query chunks; the splitter
  respects the smaller physical buffer of compressed caches. Other models
  retain upstream sizing. This bound controls temporary workspace, not total
  persistent KV capacity.
- Chunked packed MXFP4 zero-sign normalization in vLLM's B12X adapter. It avoids
  a whole-tensor temporary during model loading. Full-size GPU validation used
  a 2,264,924,160-byte packed tensor with 58,982,400 bytes of peak extra allocation.

These patches target the exact base, not arbitrary current vLLM versions.
See [upstream attribution](../../../THIRD_PARTY_NOTICES.md).

Run CPU regressions **inside the built runtime**, one script at a time:

```sh
for test in test_pp_cache test_pp_graph_tokens test_indexer_workspace test_indexer_multi_request test_b12x_zero_signs test_pp_vision_tower; do
  docker run --rm --entrypoint python3 \
    -e XDG_CACHE_HOME=/tmp/cache -e HF_HOME=/tmp/huggingface \
    llm-inference-recipes/deepseek-v41:2026-09-12-vision \
    "/opt/recipe/tests/$test.py" || exit 1
done
```

After startup is healthy, run API qualification from the repository root:

```sh
mkdir -p results/acceptance
python3 "$RECIPE/tests/acceptance.py" --output results/acceptance/basic.json
python3 "$RECIPE/tests/acceptance_context.py" \
  --context 65536 --output results/acceptance/64k.json
python3 "$RECIPE/tests/acceptance_context.py" \
  --context 1048576 --output results/acceptance/1m.json
python3 tools/benchmark_vllm.py \
  --model deepseek-ai/DeepSeek-V4.1-Flash --output-dir results/benchmark-1
```

The original German arithmetic, reasoning, tool and synthetic-record prompts
are retained to reproduce the qualification workload. The 1M test forces 128
output tokens and verifies all three records and exact API usage; it is not a
general language-quality evaluation. It uses nearly all of the shared KV pool for
roughly three minutes on the measured system; competing requests can wait or
be preempted when their combined cache demand is too large.

The [PCIe graph replay probe](benchmarks/qualify_pcie_graph.py) and
[dense-kernel microbenchmark](benchmarks/benchmark_mxfp8.py) are separate opt-in
GPU experiments. Run them only with their required idle GPUs and the pinned
runtime. Changing a backend can change numerical behavior: B12X's dense path
uses FP8 activations, while the compared Marlin path used BF16 activations.

## Known limits

- The final full-context run sampled as little as **375 MiB free VRAM**. It had
  two recovered allocator mapping retries during startup and none during
  serving. Sampling every five seconds can miss lower instantaneous headroom.
- A 2,048-token prefill experiment reached 3 MiB free and 415 recovered mapping
  failures. It was rejected despite passing a context test. Keep 1,536 here.
- Short-prompt TTFT increased: the final main run measured 0.313 s at 1K versus
  0.166 s baseline. A focused recheck measured 0.239 s. Long-prompt TTFT improved.
- Concurrency consumes a shared cache budget. Allowing 24 active requests does
  not provide 24 independent 1M contexts. The vision profile has separate
  [bounded qualification](benchmarks/vision.md); speculative decoding and other
  GPU counts remain unqualified.
- A changed driver, power cap, PCIe placement, dependency or kernel can change
  both speed and memory behavior. Re-run correctness and full-context checks
  before accepting new benchmark results.
