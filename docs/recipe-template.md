# Recipe title

Status: proposal / CPU checked / GPU qualified. Qualification date:

## Hardware and scope

Model ID and revision; GPU count/model/VRAM; link widths and topology; driver;
host RAM and storage; backend; supported modalities; concurrency and total
context limit. Distinguish measured usage from configured limits.

## Reproduce

Pinned download, build, device selection, launch, readiness and stop commands.
Use configurable paths and identifiers. Document offline operation and any
required GPU, CPU or SSD offload explicitly.

## Why these settings

Explain partitioning, kernels, cache/workspace budgets, collectives and patches.
Attribute upstream code, record hashes and include regression checks.

## Validate

CPU/runtime tests, basic generation, reasoning/tool tests where applicable,
context-boundary checks and an isolated benchmark invocation.

## Evidence and limitations

Baseline and candidate configuration; per-run results and medians; TTFT and
decode rate; cache hits, memory pressure and errors; quality checks and their
limits. Describe rejected settings and separate historical GPU qualification
from packaging or CI checks performed later.
