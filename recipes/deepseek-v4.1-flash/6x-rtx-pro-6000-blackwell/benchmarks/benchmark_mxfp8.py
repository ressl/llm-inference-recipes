"""Compare installed MXFP8 kernels on representative TP2 matrix shapes.

Run inside the pinned image with CUDA_VISIBLE_DEVICES set to one x16 UUID.
These resident-weight CUDA-graph timings identify candidates; they do not
predict whole-model throughput. Selected paths are checked against the same
FP32 product of the stored FP8 weights and block scales.
"""
import argparse,json,torch
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutlassMxfp8LinearKernel
from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
from vllm.model_executor.kernels.linear.mxfp8.b12x import B12xMxfp8LinearKernel

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--kernels',nargs='+',choices=['flashinfer','marlin','b12x'],default=['flashinfer','marlin'])
parser.add_argument('--tokens',nargs='+',type=int,default=[1,1024,2048])
args=parser.parse_args()
assert all(m>0 for m in args.tokens)
kernel_types={'flashinfer':FlashInferCutlassMxfp8LinearKernel,'marlin':MarlinMxfp8LinearKernel,'b12x':B12xMxfp8LinearKernel}

torch.set_default_dtype(torch.bfloat16);torch.manual_seed(20260912)
for n,k in [(512,5120),(1280,5120),(16384,1280),(5120,4096),(2304,5120),(5120,1152)]:
 w=(torch.randn(n,k,device='cuda')*0.1).to(torch.float8_e4m3fn)
 sc=torch.randint(125,129,(n,k//32),dtype=torch.uint8,device='cuda')
 refw=w.float()*torch.exp2(sc.float()-127).repeat_interleave(32,dim=1)
 layers=[]
 for cls in [kernel_types[name] for name in args.kernels]:
  kernel=cls(Mxfp8LinearLayerConfig());layer=torch.nn.Module()
  layer.weight=torch.nn.Parameter(w.clone(),requires_grad=False)
  layer.weight_scale=torch.nn.Parameter(sc.clone(),requires_grad=False)
  layer.input_size_per_partition=k;layer.output_size_per_partition=n
  kernel.process_weights_after_loading(layer);layers.append((cls.__name__,kernel,layer))
 for m in args.tokens:
  x=torch.randn(m,k,device='cuda');ref=x.float()@refw.T
  for name,kernel,layer in layers:
   for _ in range(3):y=kernel.apply_weights(layer,x)
   err=((y.float()-ref).square().mean()/ref.square().mean()).sqrt().item()
   assert torch.isfinite(y).all().item() and err<0.07,(name,n,k,m,err)
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g,stream=stream):
    for _ in range(20): y=kernel.apply_weights(layer,x)
   torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   a.record()
   for _ in range(20):g.replay()
   b.record();b.synchronize()
   print(json.dumps({'kernel':name,'m':m,'n':n,'k':k,'relative_rmse':err,'microseconds':a.elapsed_time(b)*1000/400}),flush=True)
   del g,y
 del w,sc,refw,layers,x,ref
 torch.cuda.empty_cache()
