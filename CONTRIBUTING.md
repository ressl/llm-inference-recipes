# Contributing

Open an issue or pull request on GitHub. Include the concrete model, hardware,
backend versions and the problem or measurement your change addresses.

Add each tested combination under `recipes/<model>/<hardware>/`. Start from
the [recipe template](docs/recipe-template.md). Keep shared tools model-neutral;
model-specific flags, patches and acceptance prompts belong with the recipe.

A runnable recipe needs an immutable checkpoint revision and runtime pins,
hardware and memory requirements, an explicit launch command, correctness
checks, measured results and known limitations. Label proposals and untested
combinations clearly. Record unsuccessful experiments when they help explain
the selected configuration.

For performance claims, provide baseline and candidate runs using the same
protocol, input/output lengths and concurrency. Include TTFT, decode rate,
memory pressure and any errors. Preserve per-run numeric data; do not submit
private prompts, credentials, machine access details or unreviewed server logs.

Run the two CPU check commands in the root README. Changes to GPU kernels,
parallelism, allocation or runtime pins also need the recipe's runtime tests
and fresh GPU/API qualification. CPU CI cannot establish GPU correctness.
Do not replace a hash guard with a permissive string replacement to accommodate
an unrelated upstream version.

New original contributions use AGPL-3.0-only. Identify copied code and preserve
its upstream license and attribution in THIRD_PARTY_NOTICES.md.
