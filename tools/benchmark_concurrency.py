#!/usr/bin/env python3
"""Concurrent vLLM streams with request markers and sampled running/queued counts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import threading
import time
import urllib.request
import uuid

from benchmark_vllm import consume_stream, counters


def fixture_ids(seed, concurrency, shared):
    generator = random.Random(seed) if seed is not None else None
    def identifier():
        return (f'{generator.getrandbits(128):032x}' if generator is not None
                else uuid.uuid4().hex)
    group = identifier()
    return [('REQUEST_' + identifier()[:16], group if shared else identifier())
            for _ in range(concurrency)]


def metric(raw, name, model):
    values = []
    for line in raw.splitlines():
        if not line.startswith('vllm:' + name + '{'):
            continue
        labels = dict((key, json.loads(value)) for key, value in
                      re.findall(r'(\w+)=("(?:[^"\\]|\\.)*")', line))
        if labels.get('model_name') == model:
            values.append(float(line.rsplit(' ', 1)[1]))
    if not values:
        raise ValueError('Missing model metric: ' + name)
    return sum(values)


def validate_group(before, after, model, concurrency, output, shared):
    start, end = counters(before, model), counters(after, model)
    delta = {key: end[key] - start[key] for key in start}
    if (delta['request_success_total'] != concurrency
            or delta['generation_tokens_total'] != concurrency * output
            or (not shared and delta['prefix_cache_hits_total'] != 0)):
        raise ValueError('Unexpected traffic, usage or prefix reuse: ' + str(delta))
    delta['preemptions'] = (metric(after, 'num_preemptions_total', model)
                            - metric(before, 'num_preemptions_total', model))
    return delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:30000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--prompt-tokens', type=int, default=32768)
    parser.add_argument('--output-tokens', type=int, default=512)
    parser.add_argument('--shared-prefix', action='store_true')
    parser.add_argument('--fixture-seed', type=int,
                        help='Repeat identical inputs; use a fresh vLLM cache salt per group')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.url.startswith(('http://', 'https://')):
        parser.error('--url must use HTTP(S)')
    if not 1 <= args.concurrency <= 64 or args.prompt_tokens < 256 or args.output_tokens < 64:
        parser.error('Require concurrency 1..64, input >=256 and output >=64')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = args.url.rstrip('/')

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        # Operator-selected CLI URL constrained to HTTP(S) above.
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        return urllib.request.urlopen(req, timeout=900)

    def metrics():
        # Same validated endpoint, literal metrics path.
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        with urllib.request.urlopen(base + '/metrics', timeout=10) as response:
            return response.read().decode()

    # Salting avoids accidental prefix reuse across runs of the same fixture.
    # Keep this salt private; it is not part of the model-visible prompt.
    cache_options = ({'cache_salt': uuid.uuid4().hex + uuid.uuid4().hex}
                     if args.fixture_seed is not None else {})
    prompts = []
    for marker, prefix in fixture_ids(args.fixture_seed, args.concurrency, args.shared_prefix):
        text = ('Benchmark ' + prefix + '\nReference notes:\n'
                + 'A service accepts requests and records a status value for each operation.\n'
                * (args.prompt_tokens // 8 + 100)
                + '\nBegin your response with exactly this line: ' + marker
                + '\nThen write a complete Python HTTP service with routing, validation, '
                'logging and a long unittest suite. Continue until every test is complete.\n')
        with post('/tokenize', {'model': args.model,
                  'messages': [{'role': 'user', 'content': text}],
                  'add_generation_prompt': True,
                  'chat_template_kwargs': {'enable_thinking': False}}) as response:
            ids = json.load(response)['tokens']
        if len(ids) <= args.prompt_tokens:
            raise ValueError('Fixture is shorter than requested input')
        prompts.append((marker, ids[:args.prompt_tokens - 128] + ids[-128:]))
    if args.shared_prefix:
        with post('/v1/completions', {'model': args.model, 'prompt': prompts[0][1],
                  'max_tokens': 64, 'ignore_eos': True, 'temperature': 0,
                  **cache_options}) as response:
            warmup = json.load(response)
        (args.output_dir / 'prefix-warmup.json').write_text(json.dumps(warmup, indent=2))

    before = metrics()
    if metric(before, 'num_requests_running', args.model) or metric(before, 'num_requests_waiting', args.model):
        raise ValueError('Server must be idle before the measurement')
    (args.output_dir / 'metrics-before.txt').write_text(before)
    stop = threading.Event()
    barrier = threading.Barrier(args.concurrency)
    samples, errors = [], []

    def sample():
        while not stop.is_set():
            try:
                raw = metrics()
                samples.append({'time': time.time(), **{name: metric(raw, name, args.model)
                                for name in ('num_requests_running', 'num_requests_waiting',
                                             'kv_cache_usage_perc')}})
            except Exception as exc:
                errors.append(str(exc))
            stop.wait(0.25)

    def request(item):
        marker, ids = item
        output_text = []
        def collect(lines):
            for raw in lines:
                if raw.startswith(b'data:') and raw[5:].strip() != b'[DONE]':
                    data = json.loads(raw[5:])
                    output_text.extend(choice.get('text') or '' for choice in data.get('choices', []))
                yield raw
        barrier.wait(timeout=30)
        started = time.monotonic()
        with post('/v1/completions', {'model': args.model, 'prompt': ids,
                  'max_tokens': args.output_tokens, 'temperature': 0,
                  'ignore_eos': True, 'stream': True,
                  'stream_options': {'include_usage': True}, **cache_options}) as response:
            result = consume_stream(collect(response), started, args.prompt_tokens, args.output_tokens)
        text = ''.join(output_text)
        if not text.lstrip().startswith(marker):
            raise ValueError('Missing or incorrect per-request marker: ' + repr(text[:100]))
        return {'marker': marker, 'response_sha256': hashlib.sha256(text.encode()).hexdigest(),
                **result}

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(request, prompts))
        elapsed = time.monotonic() - started
    finally:
        stop.set()
        sampler.join(timeout=15)
        (args.output_dir / 'samples.json').write_text(json.dumps(samples, indent=2))
        (args.output_dir / 'sampler-errors.json').write_text(json.dumps(errors, indent=2))
    after = metrics()
    (args.output_dir / 'metrics-after.txt').write_text(after)
    delta = validate_group(before, after, args.model, args.concurrency,
                           args.output_tokens, args.shared_prefix)
    if errors or not samples:
        raise ValueError('Metrics sampling incomplete: ' + repr(errors))
    summary = {'concurrency': args.concurrency, 'prompt_tokens': args.prompt_tokens,
               'output_tokens': args.output_tokens, 'shared_prefix': args.shared_prefix,
               'fixture_seed': args.fixture_seed,
               'elapsed_s': elapsed,
               'aggregate_output_tokens_per_second': args.concurrency * args.output_tokens / elapsed,
               'median_stream_decode_tokens_per_second': statistics.median(r['decode_tokens_per_second'] for r in results),
               'median_ttft_s': statistics.median(r['ttft_s'] for r in results),
               'maximum_running': max(s['num_requests_running'] for s in samples),
               'maximum_waiting': max(s['num_requests_waiting'] for s in samples),
               'maximum_cache_usage': max(s['kv_cache_usage_perc'] for s in samples),
               'metrics_delta': delta, 'requests': results}
    (args.output_dir / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'requests'}), flush=True)


if __name__ == '__main__':
    main()
