# LLM inference recipes

Reproducible inference configurations with measured tradeoffs: exact model and
runtime revisions, GPU layouts, patches, launch commands and benchmark evidence.
Recipes are organized by model and hardware so more models and serving backends
can be added without changing the scope of the project.

## Recipes

| Model | Hardware | Backend | Measured result |
| --- | --- | --- | --- |
| [DeepSeek V4.1 Flash](recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell/README.md) | 6 × RTX PRO 6000 Blackwell 96 GB, PCIe x16 | Patched, pinned vLLM; TP2 × PP3 | Native FP4 cache: 2 × full 1M or 4 × 512K contexts with overlapping decode; 1M per-request limit, shared cache pool |

The September 16 [FP4 profile and measurements](recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell/benchmarks/fp4-cache.md)
increase cache capacity from **1.16M to 2.10M tokens (+80%)** at the same
1.8 GiB per-worker budget. Two independent full 1M contexts and four independent
512K contexts passed recall and overlapping-decode tests. Context sizes include
input and output. The profile retains native vision with a 16-image request
limit and runs **without DSpark**. Public build/CPU checks are separate from
the deployed runtime's six-GPU full-model qualification.

Weights and Engram stay on the GPUs. The earlier profiles and measurements
remain available: [concurrency](recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell/benchmarks/concurrency.md),
[native vision](recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell/benchmarks/vision.md)
and [single-request tuning](recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell/benchmarks/README.md).
Results are from one six-GPU system, not a throughput or quality guarantee.

An optional [DSpark pipeline-parallel fork](https://github.com/ressl/vllm/blob/dspark-pipeline-parallel/docs/features/speculative_decoding/dspark_pipeline_parallel.md)
passed the documented tests on September 14. With 1K input and 1K output, it
measured 286 decode tokens/s for one stream and 726 aggregate output tokens/s
at 24 requests. On September 16 it was disabled after the operator reported
suspected intermittent issues. A DSpark-specific root cause has **not** been
established. Those speed measurements remain historical, not a long-term
stability guarantee. The FP4 adapter rejects speculation; FP4 plus DSpark
has not been qualified.

## Start here

1. Open the model recipe and check its hardware, memory and topology requirements.
2. Download its pinned checkpoint, build its runtime, and supply your own paths
   and GPU UUIDs to the launcher.
3. Run the acceptance checks, then repeat the benchmark on an otherwise idle
   server. Keep your results with the exact configuration you used.

The repository ships source and instructions. Model weights and prebuilt GPU
runtime images are not included. Public CI runs CPU checks; GPU qualification
is a separate, explicit operation described by each recipe.

## Shared guides

- [PCIe topology and P2P](docs/pcie-p2p.md)
- [CUDA graphs](docs/cuda-graphs.md)
- [Benchmark protocol](docs/benchmarking.md)
- [Adding a recipe](CONTRIBUTING.md) and [recipe template](docs/recipe-template.md)
- [Security reporting](SECURITY.md)

## Local checks

Python 3.12, with no third-party Python dependencies:

```sh
python3 -m unittest discover -s tests -v
python3 ci/check_repository.py
```

Runtime-specific tests import the pinned vLLM/PyTorch environment and live next
to the recipe. They are intentionally separate from these portable CPU checks.

## License and attribution

Original contributions are licensed under [AGPL-3.0-only](LICENSE).
Copied upstream source fragments retain their original licenses, as detailed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Runtime dependencies and model
weights have their own licenses. This is an independent community project.
