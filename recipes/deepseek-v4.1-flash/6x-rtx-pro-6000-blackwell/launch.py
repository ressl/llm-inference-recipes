#!/usr/bin/env python3
"""Launch the fixed six-GPU recipe; no server flags are silently overridden."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


def parse_gpus(value):
    uuids = value.split(",")
    if len(uuids) != 6 or len(set(uuids)) != 6 or not all(GPU_UUID.fullmatch(x) for x in uuids):
        raise ValueError("Provide exactly six distinct full GPU UUIDs, comma-separated")
    return uuids


def docker_command(args):
    uuids = parse_gpus(args.gpus)
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    model = Path(args.model_dir).expanduser().resolve()
    cache = Path(args.cache_dir).expanduser().resolve()
    if not (model / "config.json").is_file() or not (model / "model.safetensors.index.json").is_file():
        raise ValueError("Model directory must contain config.json and model.safetensors.index.json")
    if any("," in str(p) or "\n" in str(p) for p in (model, cache)):
        raise ValueError("Docker bind paths cannot contain commas or newlines")
    if model == cache or model in cache.parents or cache in model.parents:
        raise ValueError("Model and writable cache paths must be separate directories")
    profile = json.loads((HERE / "profile.json").read_text())
    selected = ",".join(uuids)
    cmd = ["docker", "run", "--detach", "--init", "--name", args.name,
           "--gpus", '"device=' + selected + '"',
           "--user", f"{os.getuid()}:{os.getgid()}",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
           "--memory", "96g", "--shm-size", "16g", "--stop-timeout", "180",
           "--publish", f"127.0.0.1:{args.port}:30000",
           "--mount", f"type=bind,src={model},dst=/models,readonly",
           "--mount", f"type=bind,src={cache},dst=/cache"]
    env = {**profile["env"], "RECIPE_EXPECTED_GPU_UUIDS": selected,
           "CUDA_VISIBLE_DEVICES": selected, "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"}
    for key, value in env.items():
        cmd += ["--env", f"{key}={value}"]
    return cmd + [args.image] + profile["args"], cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Six full GPU UUIDs in TP-pair order")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--image", default="llm-inference-recipes/deepseek-v41:2026-09-12")
    parser.add_argument("--name", default="deepseek-v41-recipe")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        cmd, cache = docker_command(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(shlex.join(cmd))
        return
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    print(f"Wait for container {args.name!r} to become healthy before sending requests.")


if __name__ == "__main__":
    main()
