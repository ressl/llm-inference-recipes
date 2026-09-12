"""Warm representative prefill/decode shapes before the container health check passes."""
import json
import time
import urllib.request

BASE = 'http://127.0.0.1:30000'
MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'
# The container health check retries while the API is loading.
with urllib.request.urlopen(BASE + '/health', timeout=5) as response:
    assert response.status == 200


def post(path, body):
    request = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


# Use the real tokenizer and original model. Exact prompt sizes cover full
# prefill chunks, long-context paths and the captured single-token decode.
text = ('Startup kernel warmup.\n' +
        'A service records a status value for each operation.\n' * 5000 +
        '\nWrite a Python HTTP server with routing and request validation.')
tokens = post('/tokenize', {
    'model': MODEL, 'messages': [{'role': 'user', 'content': text}],
    'add_generation_prompt': True,
    'chat_template_kwargs': {'enable_thinking': False},
})['tokens']
assert len(tokens) > 32768
for length in (1024, 32768):
    started = time.monotonic()
    result = post('/v1/completions', {
        'model': MODEL, 'prompt': tokens[:length - 128] + tokens[-128:],
        'max_tokens': 16, 'ignore_eos': True, 'temperature': 0,
    })
    assert result['usage']['prompt_tokens'] == length
    assert result['usage']['completion_tokens'] == 16
    print(f'Kernel warmup passed: {length} prompt tokens, '
          f'{time.monotonic() - started:.2f}s', flush=True)
