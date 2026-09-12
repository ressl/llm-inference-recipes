# CUDA graphs

CUDA graphs can reduce repeated CPU launch overhead during decode. They require
stable execution and storage across replay; a successful capture alone does
not demonstrate that later requests use current inputs. See the
[vLLM CUDA graph design](https://docs.vllm.ai/en/latest/design/cuda_graphs/).

The first recipe uses `FULL_DECODE_ONLY`, capture size 1 and compilation mode
0. Prefill remains eager. Its pipeline-parallel ranks need the current token
count both during ordinary execution and graph replay. The included patches
and runtime tests cover those paths and empty KV-cache groups.

For this setup, switching from eager decode to the corrected graph path moved
the measured decode rate from roughly 15 to 61 tokens/s. This was the largest
individual improvement. The later 103–105 tokens/s configuration also changes
kernel and collective implementations, so the total gain cannot be attributed
to graphs alone.

When changing a graph-compatible collective, test changing inputs and multiple
dependent collectives across repeated replays. Check basic answers and tools
again after switching kernels. Graphs, allocator settings and IPC buffer
registration must be qualified together; the initial recipe records a rejected
legacy custom-all-reduce experiment for exactly this reason.
