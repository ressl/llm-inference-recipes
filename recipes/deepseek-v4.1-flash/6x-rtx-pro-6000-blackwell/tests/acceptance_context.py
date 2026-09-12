"""Recall three synthetic records while exercising an exact context boundary."""
import argparse
import json
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:30000')
p.add_argument('--context', type=int, required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
model = 'deepseek-ai/DeepSeek-V4.1-Flash'
output_budget = 128
target = a.context - output_budget
filler = 'Archivzeile: Dieser Eintrag enthält ausschließlich Fülltext.\n'
records = ['ANFANG_CODE=KOBALT-7249', 'MITTE_CODE=FARN-3816', 'ENDE_CODE=AMBER-5927']


def post(path, body):
    req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=7200) as response:
        return json.load(response)


def tokens(text):
    return post('/tokenize', {
        'model': model,
        'messages': [{'role': 'user', 'content': text}],
        'add_generation_prompt': True,
        'chat_template_kwargs': {'enable_thinking': False},
    })['tokens']


def content(n):
    return ('Merke dir die drei einmaligen Codes in diesem Archiv.\n'
            + filler * (n // 10) + records[0] + '\n'
            + filler * (4 * n // 10) + records[1] + '\n'
            + filler * (4 * n // 10) + records[2] + '\n'
            + filler * (n - n // 10 - 2 * (4 * n // 10))
            + '\nGib nur die drei Codes aus: zuerst ANFANG_CODE, dann MITTE_CODE, '
              'dann ENDE_CODE. Ein Code pro Zeile. Ignoriere den Fülltext.')


baseline = len(tokens(content(0)))
per_line = len(tokens(filler * 4)) - len(tokens(filler * 3))
n = max(100, (target - baseline) // per_line)
prompt = tokens(content(n))
# Correct the final few BPE-boundary tokens inside a filler region, far from
# the three records, instructions and role delimiters. Preserve the complete
# chat template while sending its token IDs through the completions endpoint.
delta = target - len(prompt)
assert abs(delta) < 128, (delta, per_line, n)
offset = len(prompt) // 4
if delta > 0:
    prompt[offset:offset] = [1000] * delta
elif delta < 0:
    del prompt[offset:offset - delta]
assert len(prompt) == target
print(f'Starting {len(prompt)} input + {output_budget} forced output tokens', flush=True)
started = time.monotonic()
response = post('/v1/completions', {
    'model': model, 'prompt': prompt, 'temperature': 0,
    'max_tokens': output_budget, 'ignore_eos': True,
})
result = {'context': a.context, 'seconds': time.monotonic() - started,
          'expected_codes': [r.split('=')[1] for r in records],
          'filler_correction_tokens': delta, 'response': response}
Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False), flush=True)
assert response['model'] == model
assert response['usage']['prompt_tokens'] == target
assert response['usage']['completion_tokens'] == output_budget
assert response['usage']['total_tokens'] == a.context
text = response['choices'][0]['text']
positions = [text.find(c) for c in result['expected_codes']]
assert all(i >= 0 for i in positions) and positions == sorted(positions), text
print('Exact context boundary and three-record recall passed', flush=True)
