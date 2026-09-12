"""Qualify real image understanding, request limits and mixed traffic.

Requires Pillow. Fixtures contain answers absent from the text prompts.
"""
import argparse
import base64
import concurrent.futures
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:30000')
p.add_argument('--output', required=True)
p.add_argument('--expect-disabled', action='store_true')
p.add_argument('--concurrency', type=int, default=8)
a = p.parse_args()
model = 'deepseek-ai/DeepSeek-V4.1-Flash'
results = []
fonts = ['/System/Library/Fonts/Supplemental/Arial.ttf',
         '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
         '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']
font = next((ImageFont.truetype(f, 76) for f in fonts if Path(f).exists()),
            ImageFont.load_default(size=76))


def picture(index, size=(1400, 1000)):
    im = Image.new('RGB', (1400, 1000), 'white')
    draw = ImageDraw.Draw(im)
    draw.ellipse((140, 140, 540, 540), fill='red')
    draw.rectangle((850, 140, 1250, 540), fill='blue')
    code = f'RABE-{5831 + index}'
    draw.text((300, 700), code, font=font, fill='black')
    im = im.resize(size)
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    return {'type': 'image_url', 'image_url': {
        'url': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()}}, code


def chat(messages):
    body = {'model': model, 'messages': messages, 'temperature': 0,
            'max_tokens': 160, 'chat_template_kwargs': {'enable_thinking': False}}
    request = urllib.request.Request(a.url + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
        data = json.loads(error.read())
    return {'status': status, 'seconds': time.monotonic() - start, 'response': data}


def record(name, result):
    results.append({'name': name, **result})
    Path(a.output).write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(name, json.dumps(result, ensure_ascii=False), flush=True)


def user(text, images=()):
    return {'role': 'user', 'content': [{'type': 'text', 'text': text}, *images]}


image, code = picture(0)
messages = [user('Beschreibe die Formen links und rechts samt ihren Farben. '
                 'Lies außerdem den aufgedruckten Code. Antworte kurz auf Deutsch.', [image])]
result = chat(messages)
record('single_image', result)
if a.expect_disabled:
    assert result['status'] == 400, result
    print('Current text-only server rejects the real image fixture as expected')
    raise SystemExit(0)

assert result['status'] == 200, result
text = result['response']['choices'][0]['message']['content'].lower()
assert all(x in text for x in ['rot', 'kreis', 'blau', code.lower()]), text
assert any(x in text for x in ['quadrat', 'rechteck']), text
messages.append(result['response']['choices'][0]['message'])
messages.append(user('Welche Farbe hat die Form rechts? Antworte nur mit der Farbe.'))
followup = chat(messages)
record('image_followup', followup)
assert 'blau' in followup['response']['choices'][0]['message']['content'].lower()

images, codes = zip(*(picture(i, (1600, 1100) if i % 2 else (700, 1000))
                      for i in range(4)))
multiple = chat([user('Lies die vier aufgedruckten Codes in Bildreihenfolge. '
                      'Antworte nur mit den Codes.', images)])
record('four_images', multiple)
assert multiple['status'] == 200, multiple
text = multiple['response']['choices'][0]['message']['content']
positions = [text.find(c) for c in codes]
assert all(i >= 0 for i in positions) and positions == sorted(positions), text

rejected = chat([user('Beschreibe diese Bilder.', [*images, image])])
record('five_images_rejected', rejected)
assert rejected['status'] == 400, rejected


def concurrent_case(i):
    if i % 2:
        img, expected = picture(100 + i)
        request = user('Lies nur den aufgedruckten Code.', [img])
    else:
        expected = str(1700 + i)
        request = user(f'Antworte ausschließlich mit der Zahl: Was ist 1700 + {i}?')
    result = chat([request])
    assert result['status'] == 200, result
    text = result['response']['choices'][0]['message']['content']
    assert expected in text, (i, expected, text)
    return {'case': i, 'expected': expected, **result}


with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as executor:
    for result in executor.map(concurrent_case, range(a.concurrency)):
        record('mixed_concurrency', result)
print('Image shapes, OCR, follow-up, four-image order, limit rejection and mixed traffic passed')
