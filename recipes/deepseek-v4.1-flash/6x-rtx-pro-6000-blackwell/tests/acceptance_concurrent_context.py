"""Independent exact-size contexts, recall and demonstrably overlapping decode."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import urllib.request
import uuid

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:30000")
p.add_argument("--context", type=int, default=1048576)
p.add_argument("--sessions", type=int, default=2)
p.add_argument("--output-tokens", type=int, default=2048)
p.add_argument("--output", required=True)
a = p.parse_args()
model = "deepseek-ai/DeepSeek-V4.1-Flash"
barrier = threading.Barrier(a.sessions)


def request(path, body):
    return urllib.request.Request(a.url + path, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})


def tokenize(text):
    with urllib.request.urlopen(request("/tokenize", {
        "model": model, "messages": [{"role": "user", "content": text}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }), timeout=600) as r:
        return json.load(r)["tokens"]


def one(index):
    identity = uuid.uuid4().hex
    codes = [f"{name}-{identity[i:i+6]}" for name, i in
             (("KOBALT", 0), ("FARN", 6), ("AMBER", 12))]
    filler = f"Archivzeile {index}: Dieser Eintrag enthält ausschließlich Fülltext.\n"
    def content(n):
        # A unique first line prevents meaningful sharing between requests.
        return (f"Unabhängige Sitzung {identity}. Merke die drei Codes.\n"
                + filler * (n // 10) + "ANFANG_CODE=" + codes[0] + "\n"
                + filler * (4 * n // 10) + "MITTE_CODE=" + codes[1] + "\n"
                + filler * (4 * n // 10) + "ENDE_CODE=" + codes[2] + "\n"
                + filler * (n - n // 10 - 2 * (4 * n // 10))
                + "\nGib nur die drei Codes aus, in der Reihenfolge ANFANG, MITTE, ENDE. "
                  "Ein Code pro Zeile. Ignoriere den Fülltext.")
    target = a.context - a.output_tokens
    baseline = len(tokenize(content(0)))
    per_line = len(tokenize(filler * 4)) - len(tokenize(filler * 3))
    n = (target - baseline) // per_line
    tokens = tokenize(content(n))
    delta = target - len(tokens)
    assert abs(delta) < 128, (delta, n, per_line)
    offset = len(tokens) // 4
    if delta > 0:
        tokens[offset:offset] = [1000] * delta
    elif delta < 0:
        del tokens[offset:offset-delta]
    assert len(tokens) == target
    print(f"Session {index}: {target} input + {a.output_tokens} output ready", flush=True)
    body = {"model": model, "prompt": tokens, "temperature": 0,
            "max_tokens": a.output_tokens, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    barrier.wait(timeout=600)
    start = time.time()
    first = last = None
    chunks = []
    usage = None
    done = False
    with urllib.request.urlopen(request("/v1/completions", body), timeout=7200) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            item = json.loads(data)
            if item.get("usage"):
                usage = item["usage"]
            for c in item.get("choices", []):
                # Forced generation may emit hidden special tokens after the
                # answer. They still occupy the full context and run decode.
                last = time.time()
                if c.get("text"):
                    if first is None:
                        first = last
                        print(f"Session {index}: first token after {first-start:.1f}s", flush=True)
                    chunks.append(c["text"])
    text = "".join(chunks)
    result = {"session": index, "expected_codes": codes, "started": start,
              "first_token": first, "last_token": last,
              "seconds": time.time()-start, "usage": usage,
              "done": done, "text": text}
    Path(a.output + f".session-{index}.json").write_text(json.dumps(result, indent=2))
    assert done and usage and usage["prompt_tokens"] == target, result
    assert usage["completion_tokens"] == a.output_tokens, usage
    assert usage["total_tokens"] == a.context, usage
    positions = [text.find(code) for code in codes]
    assert all(x >= 0 for x in positions) and positions == sorted(positions), text[:1000]
    print(f"Session {index}: exact context and recall passed ({result['seconds']:.1f}s)", flush=True)
    return result


with ThreadPoolExecutor(max_workers=a.sessions) as executor:
    results = list(executor.map(one, range(a.sessions)))
overlap = min(r["last_token"] for r in results) - max(r["first_token"] for r in results)
report = {"context_per_session": a.context, "sessions": a.sessions,
          "overlapping_decode_seconds": overlap, "results": results}
Path(a.output).write_text(json.dumps(report, indent=2))
assert overlap > 0, "No simultaneous full-context decode was demonstrated"
print(f"All {a.sessions} sessions passed; overlapping decode: {overlap:.2f}s", flush=True)
