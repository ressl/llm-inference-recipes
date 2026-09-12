"""Exhaustive packed-byte correctness and bounded temporary allocations."""
import ast
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode

torch.set_num_threads(1)
path=Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/b12x.py')
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_canonicalize_fp4_zero_signs_')
ns={'torch':torch}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
normalize=ns[node.name]

def reference(value):
    lo=value&15; hi=value>>4
    return (0 if lo==8 else lo) | ((0 if hi==8 else hi)<<4)

for x in [torch.arange(256,dtype=torch.uint8),torch.arange(256,dtype=torch.uint8).reshape(16,16).T,torch.tensor(0x88,dtype=torch.uint8),torch.empty((0,16),dtype=torch.uint8)]:
    expected=torch.tensor([reference(v) for v in x.flatten().tolist()],dtype=torch.uint8).reshape(x.shape)
    normalize(x);assert torch.equal(x,expected)
    normalize(x);assert torch.equal(x,expected)

class AllocationWatch(TorchDispatchMode):
    max_bytes=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}))
        if any(name in str(func) for name in ['bitwise_and.Tensor','bitwise_or.Tensor','bitwise_left_shift.Tensor','bitwise_right_shift.Tensor']):
            self.max_bytes=max(self.max_bytes,out.numel()*out.element_size())
        return out

x=torch.full((64,512*1024),0x88,dtype=torch.uint8)
with AllocationWatch() as watch:normalize(x)
assert torch.count_nonzero(x).item()==0
assert 0<watch.max_bytes<=16*1024*1024,watch.max_bytes
print('All 256 packed bytes, strided/scalar/empty tensors and idempotence pass; max temporary bytes:',watch.max_bytes)
