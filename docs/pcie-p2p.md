# PCIe topology and P2P

Record the actual GPU topology before selecting devices:

```sh
nvidia-smi --query-gpu=uuid,name,pci.bus_id,pcie.link.width.current,memory.total --format=csv
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
```

Link width, peer-access capability and the collective implementation answer
different questions. An x16 link alone does not establish that peer transfers
work between two GPUs. Check the selected pairs, then verify transfers and
collectives in the serving runtime.

`NCCL_P2P_DISABLE=0` permits NCCL's P2P transport. `NCCL_P2P_LEVEL=PHB` allows it
within a PCI host bridge; it does not create a missing hardware capability.
See [NVIDIA's NCCL environment-variable reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
The first recipe was measured with all six selected GPUs on one NUMA node.
Keep adjacent UUIDs in the launcher list as the intended two-GPU TP pairs and
confirm the resulting rank layout in startup logs.

FlashInfer's PCIe IPC all-reduce is a separate implementation selected by
`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` in the pinned runtime. Keep vLLM's
legacy custom all-reduce disabled in that recipe. Its separate graph replay
probe exercises changing inputs and chained collectives:

```sh
torchrun --standalone --nproc-per-node=2 /opt/recipe/benchmarks/qualify_pcie_graph.py
```

Run this inside the built runtime, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
and writable `XDG_CACHE_HOME` / `HF_HOME` paths, with **only the
two idle GPUs being tested exposed**, once per TP pair. It is an opt-in GPU
qualification test, not a background health check. Do not run it alongside a
measurement or a serving process that occupies the same cards.

On the measured setup, enabling NCCL P2P improved decode by about 1%; choosing
FlashInfer PCIe IPC later added about 3% to a different baseline. These are
separate experiments and are not universal expectations. The
[benchmark report](../recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell/benchmarks/README.md)
provides the comparisons. Recheck topology and behavior after driver or kernel
changes. Host-wide IOMMU/ACS changes are outside this recipe.
