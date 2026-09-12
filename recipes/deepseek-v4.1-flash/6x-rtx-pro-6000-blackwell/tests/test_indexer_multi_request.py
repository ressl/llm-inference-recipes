"""Exercise real indexer chunking and its physical-capacity call site on CPU."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

SOURCE = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py')
tree = ast.parse(SOURCE.read_text())
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
         and node.name in {'get_max_prefill_buffer_size', 'split_indexer_prefill_chunks'}]
scope = {'torch': torch, 'VllmConfig': object}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
workspace = scope['get_max_prefill_buffer_size']
split = scope['split_indexer_prefill_chunks']
call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == 'split_indexer_prefill_chunks')
capacity_expression = compile(ast.Expression(call.args[2]), str(SOURCE), 'eval')


def config(sequences=1, model='deepseek_v41'):
    return NS(model_config=NS(max_model_len=1048576, hf_config=NS(model_type=model)),
              scheduler_config=NS(max_num_seqs=sequences))


def capacity(ratio, sequences=24, model='deepseek_v41'):
    cfg = config(sequences, model)
    owner = NS(vllm_config=cfg, compress_ratio=ratio,
               max_prefill_buffer_size=workspace(cfg))
    return eval(capacity_expression, {'self': owner})


class MultiRequestIndexerTests(unittest.TestCase):
    def test_fixed_workspace_at_all_concurrencies(self):
        for sequences in (1, 2, 4, 8, 16, 24, 32):
            with self.subTest(sequences=sequences):
                self.assertEqual(workspace(config(sequences)), 1048576)
        self.assertEqual(workspace(config(24, 'deepseek_v32')), 40 * 1048576)

    def test_metadata_bound_matches_physical_compressed_allocation(self):
        # The model allocates max_total_seq_len = workspace // compress_ratio.
        # Exercise the actual metadata-builder argument, not a copy of its policy.
        for ratio in (1, 2):
            for sequences in (1, 2, 24):
                with self.subTest(ratio=ratio, sequences=sequences):
                    self.assertEqual(capacity(ratio, sequences), 1048576 // ratio)
        self.assertEqual(capacity(2, model='deepseek_v32'), 40 * 1048576)

    def check_chunks(self, lengths, queries, ratio, offset, budget):
        keys = [length // ratio for length in lengths]
        physical_capacity = capacity(ratio)
        chunks = split(torch.tensor(keys), torch.tensor(queries), physical_capacity,
                       budget, request_offset=offset)
        seen = [[] for _ in queries]
        for req, query in chunks:
            start, stop = req.start - offset, req.stop - offset
            self.assertTrue(0 <= start < stop <= len(queries))
            gathered = sum(keys[start:stop])
            self.assertLessEqual(gathered, physical_capacity)
            self.assertLessEqual((query.stop - query.start) * gathered * 4, budget)
            self.assertTrue(0 <= query.start < query.stop <= sum(queries[start:stop]))
            chunk_offset = 0
            for request in range(start, stop):
                low = max(query.start, chunk_offset)
                high = min(query.stop, chunk_offset + queries[request])
                seen[request].extend(range(low - chunk_offset, high - chunk_offset))
                chunk_offset += queries[request]
        for actual, count in zip(seen, queries):
            self.assertEqual(actual, list(range(count)), 'Missing or repeated query rows')

    def test_mixed_requests_offsets_and_logits_splitting(self):
        cases = [
            ([128, 4096, 32768, 131072], [128, 512, 512, 384]),
            ([1048576, 1048576], [512, 512]),
            ([65536] * 24, [8] * 24),
            ([262144] * 8, [64] * 8),
            ([1048576, 128, 524288, 7], [1, 16, 8, 7]),
            ([1, 0, 3], [1, 0, 3]),
        ]
        for ratio in (1, 2):
            for offset in (0, 3):
                for budget in (128 * 1024 * 1024, 512 * 1024 * 1024):
                    for lengths, queries in cases:
                        with self.subTest(ratio=ratio, offset=offset, budget=budget,
                                          requests=len(lengths)):
                            self.check_chunks(lengths, queries, ratio, offset, budget)


if __name__ == '__main__':
    unittest.main()
