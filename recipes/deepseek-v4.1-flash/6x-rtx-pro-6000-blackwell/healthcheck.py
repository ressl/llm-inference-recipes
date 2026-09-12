"""Run startup warmup once, then check API health without generating tokens."""
from pathlib import Path
import runpy
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=5) as response:
    assert response.status == 200
stamp = Path("/tmp/recipe-warmup.ok")
if not stamp.exists():
    runpy.run_path(str(Path(__file__).with_name("warmup.py")), run_name="__main__")
    stamp.touch()
