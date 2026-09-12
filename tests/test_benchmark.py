import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).resolve().parents[1] / "tools/benchmark_vllm.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def event(value):
    return b"data: " + json.dumps(value).encode() + b"\n"


def metrics(requests, tokens, hits, seconds):
    values = [requests, tokens, hits, seconds]
    return "\n".join(f'vllm:{key}{{model_name="test",engine="0"}} {value}'
                     for key, value in zip(benchmark.COUNTERS, values))


class BenchmarkTests(unittest.TestCase):
    def test_content_timing_ignores_empty_chunks_and_checks_usage(self):
        lines = [event({"choices": [{"text": ""}]}),
                 event({"choices": [{"text": "a"}]}),
                 event({"choices": [{"text": "b", "finish_reason": "length"}]}),
                 event({"choices": [], "usage": {"prompt_tokens": 1024,
                                                    "completion_tokens": 512}}),
                 b"data: [DONE]\n"]
        times = iter([2, 7, 8])
        result = benchmark.consume_stream(lines, 1, 1024, 512, lambda: next(times))
        self.assertEqual(result["ttft_s"], 1)
        self.assertEqual(result["decode_tokens_per_second"], 511 / 5)
        for invalid in (lines[:-1], lines[:3] + lines[-1:], lines[:2] + lines[3:]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                benchmark.consume_stream(invalid, 1, 1024, 512, lambda: 2)

    def test_metrics_detect_cache_traffic_reset_and_wrong_model(self):
        before = metrics(10, 1000, 0, 10)
        after = metrics(11, 1512, 0, 15)
        unrelated = metrics(99, 99999, 999, 999).replace('"test"', '"other"')
        result = benchmark.validate_window(before, after + "\n" + unrelated, "test", 512)
        self.assertEqual(result["server_decode_tokens_per_second"], 511 / 5)
        for invalid in (metrics(12, 1512, 0, 15), metrics(11, 1513, 0, 15),
                        metrics(11, 1512, 64, 15), metrics(1, 512, 0, 1), unrelated):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                benchmark.validate_window(before, invalid, "test", 512)


if __name__ == "__main__":
    unittest.main()
