"""Dependency-free syntax, public link and pinned recipe checks."""
import ast
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
for path in root.rglob("*.py"):
    if ".git" not in path.parts:
        ast.parse(path.read_text(), filename=str(path))
for path in root.rglob("*.json"):
    if ".git" not in path.parts:
        json.loads(path.read_text())
for path in root.rglob("*.md"):
    for target in re.findall(r'\]\(([^)\s]+)\)', path.read_text()):
        if target.startswith(("https://", "http://", "mailto:", "#")):
            continue
        destination = (path.parent / target.split("#", 1)[0]).resolve()
        if not destination.is_relative_to(root) or not destination.exists():
            raise ValueError(f"Broken or escaping local link in {path.relative_to(root)}: {target}")
for path in root.glob("recipes/*/*/profile*.json"):
    profile = json.loads(path.read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", profile["revision"]):
        raise ValueError(f"Unpinned checkpoint: {path}")
    dockerfile = path.with_name("Dockerfile").read_text()
    for base in re.findall(r"^FROM (\S+)", dockerfile, re.MULTILINE):
        if not re.search(r"@sha256:[0-9a-f]{64}$", base):
            raise ValueError(f"Unpinned runtime base: {base}")
print("Python syntax, JSON, local Markdown links and recipe pins passed.")
