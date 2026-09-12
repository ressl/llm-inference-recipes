#!/usr/bin/env python3
"""Exercise growing chat histories, tool results and changing client counts.

Tool results are synthetic fixtures. No generated code or shell commands execute.
This tests OpenAI-compatible agent traffic; it does not invoke an agent harness.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.request
import uuid

from benchmark_concurrency import metric
from benchmark_vllm import counters


def consume_chat(lines, clock=time.monotonic):
    content, reasoning, calls = [], [], {}
    first = finish = usage = None
    done = False
    for raw in lines:
        if not raw.startswith(b'data:'):
            continue
        payload = raw[5:].strip()
        if payload == b'[DONE]':
            done = True
            break
        data = json.loads(payload)
        if data.get('error'):
            raise ValueError('API stream error: ' + str(data['error']))
        usage = data.get('usage') or usage
        for choice in data.get('choices', []):
            finish = choice.get('finish_reason') or finish
            delta = choice.get('delta', {})
            text = delta.get('content') or ''
            thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
            tool_parts = delta.get('tool_calls', [])
            if first is None and (text or thought or tool_parts):
                first = clock()
            content.append(text)
            reasoning.append(thought)
            for part in tool_parts:
                call = calls.setdefault(part['index'], {'id': '', 'type': 'function',
                                       'function': {'name': '', 'arguments': ''}})
                if part.get('id'):
                    call['id'] = part['id']
                for key in ('name', 'arguments'):
                    call['function'][key] += part.get('function', {}).get(key) or ''
    if not done or finish not in ('stop', 'length', 'tool_calls') or not usage or first is None:
        raise ValueError('Incomplete chat stream or missing usage')
    if usage.get('completion_tokens', 0) <= 0 or usage.get('prompt_tokens', 0) <= 0:
        raise ValueError('Invalid token usage')
    message = {'role': 'assistant', 'content': ''.join(content) or None}
    if calls:
        message['tool_calls'] = [calls[index] for index in sorted(calls)]
    if any(reasoning):
        message['reasoning'] = ''.join(reasoning)
    return message, {'usage': usage, 'finish_reason': finish, 'first_token_time': first}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:30000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--clients', type=int, default=24)
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.url.startswith(('http://', 'https://')) or not 1 <= args.clients <= 32 or args.seconds < 30:
        parser.error('Require HTTP(S), 1..32 clients and at least 30 seconds')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = args.url.rstrip('/')
    stop = threading.Event()
    lock = threading.Lock()
    results, errors, samples = [], [], []
    started = time.monotonic()
    started_utc = time.time()
    deadline = started + args.seconds
    run_id = uuid.uuid4().hex

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        # Operator-selected HTTP(S) CLI URL, checked above.
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        return urllib.request.urlopen(req, timeout=600)

    def metrics():
        # Same validated endpoint and a literal metrics path.
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        with urllib.request.urlopen(base + '/metrics', timeout=10) as response:
            return response.read().decode()

    def record(filename, item):
        with lock:
            with (args.output_dir / filename).open('a') as handle:
                handle.write(json.dumps(item) + '\n')

    def fail(worker, exc):
        item = {'time': time.time(), 'worker': worker, 'error': str(exc)}
        with lock:
            errors.append(item)
        record('errors.jsonl', item)
        stop.set()

    before = metrics()
    counters(before, args.model)
    (args.output_dir / 'metrics-before.txt').write_text(before)

    def sample():
        while not stop.is_set():
            try:
                raw = metrics()
                row = {'time': time.time(), **{key: metric(raw, key, args.model) for key in
                       ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc',
                        'num_preemptions_total')}}
                samples.append(row)
                record('metrics.jsonl', row)
            except Exception as exc:
                fail('metrics', exc)
            stop.wait(1)

    def chat(worker, cycle, phase, messages, **extra):
        begin = time.monotonic()
        with post('/v1/chat/completions', {'model': args.model, 'messages': messages,
                  'temperature': 0, 'max_tokens': 512, 'stream': True,
                  'stream_options': {'include_usage': True},
                  'chat_template_kwargs': {'enable_thinking': False}, **extra}) as response:
            message, info = consume_chat(response)
        row = {'worker': worker, 'cycle': cycle, 'phase': phase, 'time': time.time(),
               'elapsed_s': time.monotonic() - begin,
               'ttft_s': info.pop('first_token_time') - begin, **info}
        with lock:
            results.append(row)
        record('requests.jsonl', row)
        return message, row

    def client(worker):
        cycle = 0
        # 24 clients: 12 small, 8 medium, 3 long and one very long history.
        target = [8192, 32768, 131072, 262144][
            0 if worker % 24 < 12 else 1 if worker % 24 < 20 else 2 if worker % 24 < 23 else 3]
        try:
            while time.monotonic() < deadline and not stop.is_set():
                # Repeat quiet and busy periods to exercise padded CUDA-graph transitions.
                phase = int((time.monotonic() - started) // 45) % 8
                active = min(args.clients, [2, 4, 8, 16, args.clients, args.clients,
                                           args.clients, args.clients][phase])
                if worker >= active:
                    stop.wait(1)
                    continue
                marker = f'CASE_{run_id}_{worker}_{cycle}'
                fixture = ('Repository fixture ' + marker + '\n'
                           + 'The request handler validates its input and records the operation status.\n'
                           * (target // 14))
                messages = [{'role': 'system', 'content': 'You review a synthetic software project. '
                             'Follow the current user instruction and keep every case separate.'},
                            {'role': 'user', 'content': fixture + '\nRead the test report with '
                             f'get_test_report(case_id="{marker}"). Call the tool before answering.'}]
                tool = {'type': 'function', 'function': {'name': 'get_test_report',
                        'description': 'Read a synthetic test report by case ID.',
                        'parameters': {'type': 'object', 'properties': {'case_id': {'type': 'string'}},
                                       'required': ['case_id'], 'additionalProperties': False}}}
                call, row = chat(worker, cycle, 'tool_call', messages, tools=[tool], tool_choice='auto')
                calls = call.get('tool_calls', [])
                if (row['finish_reason'] != 'tool_calls' or len(calls) != 1
                        or calls[0]['function']['name'] != 'get_test_report'
                        or json.loads(calls[0]['function']['arguments']) != {'case_id': marker}):
                    raise ValueError('Incorrect tool call: ' + repr(call))
                messages.extend([call, {'role': 'tool', 'tool_call_id': calls[0]['id'],
                    'content': json.dumps({'case_id': marker, 'passed': 17, 'failed': 0,
                                          'report_code': marker})},
                    {'role': 'user', 'content': f'Begin with exactly {marker} on its own line. '
                     'Then propose a detailed Python test suite for validation, missing inputs, '
                     'timeouts and retries. Show code and explain the assertions.'}])
                answer, _ = chat(worker, cycle, 'tool_result', messages)
                if not (answer.get('content') or '').lstrip().startswith(marker):
                    raise ValueError('Incorrect report marker: ' + repr(answer))
                messages.extend([answer, {'role': 'user', 'content':
                    f'What was the report code returned by the tool? Reply with exactly {marker}.'}])
                follow, _ = chat(worker, cycle, 'follow_up', messages, max_tokens=128)
                if (follow.get('content') or '').strip() != marker:
                    raise ValueError('Incorrect follow-up marker: ' + repr(follow))
                cycle += 1
                stop.wait(1 + worker % 5)
        except Exception as exc:
            fail(worker, exc)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        with ThreadPoolExecutor(max_workers=args.clients) as pool:
            jobs = [pool.submit(client, worker) for worker in range(args.clients)]
            while not all(job.done() for job in jobs):
                stop.wait(1)
                if int(time.monotonic() - started) % 30 == 0:
                    print(json.dumps({'elapsed_s': round(time.monotonic() - started),
                          'completed_requests': len(results), 'errors': len(errors),
                          'latest_metrics': samples[-1] if samples else None}), flush=True)
                if stop.is_set():
                    break
            for job in jobs:
                job.result()
    finally:
        stop.set()
        sampler.join(timeout=15)
    elapsed = time.monotonic() - started
    after = metrics()
    (args.output_dir / 'metrics-after.txt').write_text(after)
    first_counts, last_counts = counters(before, args.model), counters(after, args.model)
    summary = {'started_unix_s': started_utc, 'ended_unix_s': time.time(), 'elapsed_s': elapsed,
               'requested_duration_s': args.seconds, 'clients': args.clients,
               'completed_requests': len(results), 'errors': errors,
               'completed_conversations': sum(r['phase'] == 'follow_up' for r in results),
               'own_input_tokens': sum(r['usage']['prompt_tokens'] for r in results),
               'own_output_tokens': sum(r['usage']['completion_tokens'] for r in results),
               'maximum_running': max((r['num_requests_running'] for r in samples), default=0),
               'maximum_waiting': max((r['num_requests_waiting'] for r in samples), default=0),
               'maximum_cache_usage': max((r['kv_cache_usage_perc'] for r in samples), default=0),
               'server_counters_delta': {k: last_counts[k] - first_counts[k] for k in first_counts},
               'server_preemptions': metric(after, 'num_preemptions_total', args.model)
                                    - metric(before, 'num_preemptions_total', args.model),
               'median_ttft_s': statistics.median(r['ttft_s'] for r in results) if results else None}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if errors or elapsed < args.seconds or not summary['completed_conversations']:
        raise SystemExit('Soak qualification failed')


if __name__ == '__main__':
    main()
