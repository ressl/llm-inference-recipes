"""Check FlashInfer PCIe IPC with changing graph inputs on two x16 GPUs.

Run with torchrun --standalone --nproc-per-node=2, the two explicit GPU UUIDs
in CUDA_VISIBLE_DEVICES, and expandable_segments:True. Run independently
of performance measurements; this probe checks correctness, not speed.
"""
import json, os
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.flashinfer_pcie_ipc_all_reduce import FlashInferPcieIpcAllReduce

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=180))
cpu = dist.new_group(backend='gloo')
for dtype in [torch.bfloat16, torch.float16]:
    ar = FlashInferPcieIpcAllReduce(dist.group.WORLD, cpu, torch.device('cuda',rank))
    ar.setup(hidden_dim=5120,dtype=dtype,capture_sizes=[1],tune_cache=Path('/tmp/recipe-pcie-probe-'+str(dtype)+'.json'))
    assert ar.initialized
    x=torch.zeros((1,5120),device='cuda',dtype=dtype)
    assert ar.should_use(x)
    for step in range(3):
        x.fill_(rank+step)
        y=ar.all_reduce(x)
        torch.cuda.synchronize()
        assert torch.all(y==1+2*step).item()
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    graph=torch.cuda.CUDAGraph()
    with ar.capture():
        with torch.cuda.graph(graph,stream=stream):
            y=ar.all_reduce(x+0)
            z=ar.all_reduce(y+rank)
    for step in range(20):
        x.fill_(rank+step)
        graph.replay(); torch.cuda.synchronize()
        assert torch.all(y==1+2*step).item()
        assert torch.all(z==3+4*step).item()
    print(json.dumps({'rank':rank,'dtype':str(dtype),'changing_input_chained_graph_replays':20,'status':'pass','allocator':os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}),flush=True)
    del graph,y,z,x
    dist.barrier(); ar.destroy()
dist.destroy_process_group()
