"""Verify that the bounded workspace covers the native 1M prefill schedule."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch


source = Path(
    '/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py'
)
tree = ast.parse(source.read_text())
names = {'get_max_prefill_buffer_size', 'split_indexer_prefill_chunks'}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
namespace = {'torch': torch, 'VllmConfig': object}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
workspace = namespace['get_max_prefill_buffer_size']
split = namespace['split_indexer_prefill_chunks']


def config(model_type='deepseek_v41', sequences=1):
    return NS(
        model_config=NS(max_model_len=1048576, hf_config=NS(model_type=model_type)),
        scheduler_config=NS(max_num_seqs=sequences),
    )


assert workspace(config()) == 1048576, 'Do not reserve 40 requests for one request'
assert workspace(config(sequences=2)) == 1048576
assert workspace(config(model_type='deepseek_v32')) == 40 * 1048576

for budget_mb in (128, 512):
    for ratio in (1, 2):
        capacity = workspace(config()) // ratio
        for sequence_length in (128, 4096, 65536, 262144, 524288, 1048576):
            keys = sequence_length // ratio
            query_count = min(sequence_length, 2048)
            chunks = split(torch.tensor([keys]), torch.tensor([query_count]), capacity,
                           budget_mb * 1024 * 1024)
            queries = []
            for req, query in chunks:
                assert (req.start, req.stop) == (0, 1)
                assert keys <= capacity, 'Gathered keys must fit the physical workspace'
                assert (query.stop - query.start) * keys * 4 <= budget_mb * 1024 * 1024
                queries.extend(range(query.start, query.stop))
            assert queries == list(range(query_count)), 'Every query must run exactly once'

print('Indexer workspace and 1M prefill chunk bounds passed')
