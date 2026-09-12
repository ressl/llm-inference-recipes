# LLM inference recipes

Reproducible inference configurations with measured tradeoffs: exact model and
runtime revisions, GPU layouts, patches, launch commands and benchmark evidence.
Recipes are organized by model and hardware so more models and serving backends
can be added without changing the scope of the project.

## Recipes

| Model | Hardware | Backend | Measured result |
| --- | --- | --- | --- |
| [DeepSeek V4.1 Flash](recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell/README.md) | 6 × RTX PRO 6000 Blackwell 96 GB, PCIe x16 | Patched, pinned vLLM; TP2 × PP3 | 24 independent 32K requests active together; approximately 104 decode tokens/s for one stream; 1M per-request context limit with a shared KV pool |

The first recipe keeps weights and Engram on the GPUs. Its parallel profile also
passes eight independent 128K histories and four independent 256K histories.
A single full-context check used 1,048,448 input tokens plus 128 output tokens
and completed in 172 seconds with the 24-request limit enabled.
These are measurements from one six-GPU system, not a throughput or quality
guarantee. Read the [concurrency results and limits](recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell/benchmarks/concurrency.md)
before choosing the profile. A separate [native vision profile](recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell/benchmarks/vision.md)
now supports up to four images per request while retaining the same KV pool
and 1M context limit.

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
