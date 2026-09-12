# LLM inference recipes

Approved scope: a model-independent public cookbook, beginning with DeepSeek
V4.1 Flash on six RTX PRO 6000 Blackwell 96 GB GPUs with x16 PCIe links.
Provide English documentation, exact dependency pins, portable launch commands,
source-attributed patches, benchmark tools and honest measured limitations.

Each recipe owns its runtime Dockerfile, launch profile, patch explanations,
acceptance tests and measured results. Shared documentation explains PCIe P2P,
CUDA graphs and measurement methodology. Add further model/hardware/backend
combinations as independent recipes once they have evidence; do not advertise
unmeasured compatibility.

The initial runtime preserves the qualified checkpoint and kernel changes.
The public package removes machine-specific access and operational configuration.
A user supplies six GPU UUIDs, a downloaded checkpoint path and a writable cache
path. The launcher binds the HTTP endpoint to loopback by default and fails
before model loading when its six-device/x16/96-GB requirements are not met.
The profile keeps GPU Engram, zero CPU weight offload, TP2/PP3, partition
8,12,20, one simultaneous request, FP8 KV, fixed workspace budgets and full
1,048,576-token total context. Publication is a recipe release, with no change
to an existing serving installation.

Publish original material under AGPL-3.0-only and retain upstream license notices
for copied snippets and patch contexts. Include public contributor CI, meaningful
CPU tests, in-runtime regressions and opt-in GPU/API qualification commands.
Historical results must identify their setup and protocol and distinguish the
qualified runtime from the portable packaging checks. Benchmarks must include
short-prompt latency, memory pressure and rejected experiments, not only gains.

Implementation plan: (1) extract only public-safe assets into a fresh history;
(2) implement the portable launcher and benchmark interface; (3) document the
recipe and measured evidence; (4) build and test the pinned runtime and CPU
checks; (5) review all publishable files/history and dependencies; (6) publish
the approved repository and verify CI, visibility and cloneability.
