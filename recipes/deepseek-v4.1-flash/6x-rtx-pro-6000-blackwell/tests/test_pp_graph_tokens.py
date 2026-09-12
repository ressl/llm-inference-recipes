"""Exercise real graph-input preparation without a GPU or graph allocation."""
import ast
import contextlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    '/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/cudagraph_utils.py'
)
tree = ast.parse(path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
           and n.name == 'ModelCudaGraphManager')
# Execute the actual capture method, replacing only GPU/attention plumbing.
cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'capture']

class Base:
    def capture(self, factory, description):
        for warmup in (True, False):
            factory(NS(num_tokens=2, num_reqs=1, num_active_loras=0,
                       cg_mode=2, max_query_len=1), warmup)(0)

class Model:
    def __init__(self, needs_ids, expected):
        self.needs_ids, self.expected, self.calls = needs_ids, expected, 0
    def __call__(self, **inputs):
        assert inputs['input_ids'] == self.expected, (
            'Graph warmup/capture lost required raw token IDs', inputs['input_ids'])
        self.calls += 1
        return [17, 23]

interfaces = ModuleType('vllm.model_executor.models.interfaces')
interfaces.requires_raw_input_tokens = lambda model: model.needs_ids
sys.modules[interfaces.__name__] = interfaces
namespace = {
    'CudaGraphManager': Base,
    'CUDAGraphMode': NS(NONE=0, PIECEWISE=1, FULL=2),
    'set_forward_context': lambda *a, **kw: contextlib.nullcontext(),
    'prepare_inputs_to_capture': lambda *a, **kw: ({}, {}),
    'torch': NS(empty_like=lambda x: [0] * len(x)),
}
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
    ast.alias(name='annotations')], level=0), cls], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
Manager = namespace['ModelCudaGraphManager']
for first in (True, False):
    for needs_ids in (True, False):
        manager = Manager()
        manager.use_breakable_cg = False
        manager.cudagraph_mode = NS(has_piecewise_cudagraphs=lambda: False)
        manager.max_num_reqs = 1
        manager.dp_size = 1
        manager.is_first_pp_rank = first
        manager.is_last_pp_rank = True
        manager.vllm_config = NS()
        manager.hidden_states = None
        model = Model(needs_ids, [11, 29] if first or needs_ids else None)
        buffers = NS(input_ids=[11, 29, 37], positions=[0, 1, 2],
                     is_padding=NS(fill_=lambda value: None, __getitem__=None))
        class Padding:
            def fill_(self, value): pass
            def __getitem__(self, key): return [True, True]
        buffers.is_padding = Padding()
        state = NS(prepare_dummy_inputs=lambda *a: {})
        manager.capture(model, state, buffers, [0, 0], None, [], None)
        assert model.calls == 2
print('Graph warmup and capture preserve required PP raw IDs; generic PP stays compatible')
