import importlib.util
from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('concurrency_benchmark', TOOLS / 'benchmark_concurrency.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def metrics(requests, tokens, hits=0, preemptions=0):
    values = {'request_success_total': requests, 'generation_tokens_total': tokens,
              'prefix_cache_hits_total': hits, 'request_decode_time_seconds_sum': requests * 2,
              'num_preemptions_total': preemptions}
    return '\n'.join(f'vllm:{key}{{model_name="test",engine="0"}} {value}'
                     for key, value in values.items())


class ConcurrentBenchmarkTests(unittest.TestCase):
    def test_seed_reproduces_inputs_but_requests_remain_distinct(self):
        first = benchmark.fixture_ids(42, 24, False)
        self.assertEqual(first, benchmark.fixture_ids(42, 24, False))
        self.assertNotEqual(first, benchmark.fixture_ids(43, 24, False))
        self.assertEqual(24, len({marker for marker, _ in first}))
        self.assertEqual(24, len({prefix for _, prefix in first}))
        shared = benchmark.fixture_ids(42, 24, True)
        self.assertEqual(1, len({prefix for _, prefix in shared}))
        self.assertEqual(24, len({marker for marker, _ in shared}))

    def test_exact_group_usage_and_uncached_contract(self):
        before = metrics(10, 1000)
        result = benchmark.validate_group(before, metrics(14, 3048), 'test', 4, 512, False)
        self.assertEqual(result['request_success_total'], 4)
        for bad in (metrics(15, 3048), metrics(14, 3047), metrics(14, 3048, hits=64)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                benchmark.validate_group(before, bad, 'test', 4, 512, False)

    def test_shared_cache_and_preemption_are_reported(self):
        result = benchmark.validate_group(metrics(10, 1000), metrics(14, 3048, 4096, 2),
                                          'test', 4, 512, True)
        self.assertEqual(result['prefix_cache_hits_total'], 4096)
        self.assertEqual(result['preemptions'], 2)

    def test_sample_ignores_other_models_and_similar_metric_names(self):
        raw = ('vllm:num_requests_waiting{model_name="test"} 3\n'
               'vllm:num_requests_waiting{model_name="other"} 90\n'
               'vllm:num_requests_waiting_by_reason{model_name="test",reason="capacity"} 3\n')
        self.assertEqual(benchmark.metric(raw, 'num_requests_waiting', 'test'), 3)
        with self.assertRaises(ValueError):
            benchmark.metric(raw, 'num_requests_running', 'test')


if __name__ == '__main__':
    unittest.main()
