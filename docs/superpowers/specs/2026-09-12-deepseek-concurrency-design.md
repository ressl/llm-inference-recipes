# DeepSeek V4.1 Flash concurrent serving

The approved next step is a measured multi-request profile on the same six x16
GPUs. Keep the accepted single-request runtime/profile as the rollback point.
Start with the same checkpoint, TP2/PP3 partition, GPU Engram, zero CPU weight
offload, 1.8 GiB per-worker KV budget, 1M context limit and 1536 prefill budget.

Generalize the DeepSeek-specific indexer workspace bound to multiple scheduled
requests. Its existing request/query splitter must ensure each processed chunk
fits the fixed workspace and logits budget. Verify mixed request lengths,
compression ratios, request offsets and complete query coverage before GPU use.
Other model types retain the upstream workspace policy.

Qualify concurrency 2, 4, 8, 16 and 24 sequentially, with corresponding decode
graph captures. Exercise expanding and shrinking batches and unique request
markers to detect cross-request corruption. Keep fixed pins and change one
capacity setting at a time. If memory pressure appears, compare a 1024 prefill
budget; shorter-context profiles are optional alternatives with explicit limits.

Measure actual running requests separately from queued clients, aggregate and
per-stream decode, TTFT/latency, cache occupancy/hits/preemptions, VRAM, allocator
warnings and worker restarts. Use 32K/128K/256K inputs where the aggregate cache
budget permits; overload cases must be labeled as queueing tests. Include
uncached and shared-prefix traffic, basic/tool/stream correctness and exact 1M
regression. A chosen candidate needs at least one hour of realistic multi-turn
agent-style traffic before promotion.

Implementation sequence: record rollback and baseline; add a failing workspace
regression; implement and build the bounded multi-request runtime; pass CPU and
runtime regressions; run staged GPU/API qualification; run the one-hour soak;
select the useful stable profile (or retain baseline); publish evidence and
reproduction instructions through the existing checked source/export workflow.

No promise is made that 24 requests or 20 long contexts fit. More GPU KV cache
is a separate optimization after actual concurrent working-memory needs are
known. Findings and unsuccessful variants belong with the results.
