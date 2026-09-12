#!/usr/bin/env python3
"""Serial, uncached vLLM streaming benchmark with server-counter validation."""
import argparse
import json
import math
from pathlib import Path
import re
import time
import urllib.request
import uuid

CASES = [("warmup", 256, 64), ("1k-1", 1024, 512),
         ("1k-2", 1024, 512), ("1k-3", 1024, 512),
         ("32k-1", 32768, 512), ("32k-2", 32768, 512),
         ("128k-1", 131072, 512)]
COUNTERS = ("request_success_total", "generation_tokens_total",
            "prefix_cache_hits_total", "request_decode_time_seconds_sum")


def counters(raw, model):
    totals = {}
    for line in raw.splitlines():
        match = re.fullmatch(r'vllm:(\w+)\{([^}]*)\}\s+(\S+)(?:\s+\S+)?', line)
        if not match or match[1] not in COUNTERS:
            continue
        labels = dict((key, json.loads(value)) for key, value in
                      re.findall(r'(\w+)=("(?:[^"\\]|\\.)*")', match[2]))
        if labels.get("model_name") != model:
            continue
        value = float(match[3])
        if not math.isfinite(value):
            raise ValueError("Non-finite server counter")
        totals[match[1]] = totals.get(match[1], 0.0) + value
    if set(totals) != set(COUNTERS):
        raise ValueError("Required vLLM metrics are missing for the served model")
    return totals


def validate_window(before, after, model, output_tokens):
    start, end = counters(before, model), counters(after, model)
    delta = {key: end[key] - start[key] for key in COUNTERS}
    if (delta["request_success_total"] != 1
            or delta["generation_tokens_total"] != output_tokens
            or delta["prefix_cache_hits_total"] != 0
            or delta["request_decode_time_seconds_sum"] <= 0):
        raise ValueError(f"Contaminated, cached, reset or incomplete metric window: {delta}")
    delta["server_decode_tokens_per_second"] = (
        (output_tokens - 1) / delta["request_decode_time_seconds_sum"])
    return delta


def consume_stream(lines, started, prompt_tokens, output_tokens, clock=time.monotonic):
    first = last = usage = finish = None
    done = False
    for raw in lines:
        if not raw.startswith(b"data:"):
            continue
        data = raw[5:].decode().strip()
        if data == "[DONE]":
            done = True
            break
        chunk = json.loads(data)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            finish = choice.get("finish_reason") or finish
            if choice.get("text"):
                now = clock()
                first = now if first is None else first
                last = now
    elapsed = clock() - started
    if (not done or not usage or finish != "length"
            or usage.get("prompt_tokens") != prompt_tokens
            or usage.get("completion_tokens") != output_tokens
            or first is None or last <= first):
        raise ValueError("Incomplete stream, unexpected usage or insufficient content chunks")
    return {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
            "ttft_s": first - started, "elapsed_s": elapsed,
            "decode_tokens_per_second": (output_tokens - 1) / (last - first),
            "end_to_end_output_tokens_per_second": output_tokens / elapsed,
            "finish_reason": finish}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    if not base.startswith(("http://", "https://")):
        parser.error("--url must use HTTP or HTTPS")
    # Refuse to replace earlier evidence.
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def post(path, body):
        request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=600)

    def metrics():
        with urllib.request.urlopen(base + "/metrics", timeout=20) as response:
            return response.read().decode()

    results = []
    for label, length, output in CASES:
        body = ("Benchmark " + uuid.uuid4().hex + "\nReference notes:\n"
                + "A service accepts requests and records a status value for each operation.\n"
                * (length // 8 + 100)
                + "\nWrite a complete Python HTTP service with routing, validation, "
                "authentication hooks, logging, graceful shutdown, and a long unittest suite. "
                "Output only Python code. Continue until the implementation and every test "
                "are complete.\n")
        with post("/tokenize", {"model": args.model,
                  "messages": [{"role": "user", "content": body}],
                  "add_generation_prompt": True,
                  "chat_template_kwargs": {"enable_thinking": False}}) as response:
            ids = json.load(response)["tokens"]
        if len(ids) <= length:
            raise ValueError("Tokenizer produced fewer tokens than the requested fixture")
        ids = ids[:length - 128] + ids[-128:]
        before = metrics()
        counters(before, args.model)  # Fail before inference for unsupported metrics.
        (args.output_dir / f"{label}-metrics-before.txt").write_text(before)
        started = time.monotonic()
        with post("/v1/completions", {
            "model": args.model, "prompt": ids, "max_tokens": output,
            "temperature": 1.0, "top_p": 0.95, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True},
        }) as response:
            result = consume_stream(response, started, length, output)
        after = metrics()
        (args.output_dir / f"{label}-metrics-after.txt").write_text(after)
        result.update(label=label, warmup=label == "warmup",
                      metrics=validate_window(before, after, args.model, output))
        results.append(result)
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    print("Completed: exclude the warmup row when comparing results.")


if __name__ == "__main__":
    main()
