import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell"


def load(name):
    spec = importlib.util.spec_from_file_location(name, RECIPE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load("launch")
preflight = load("gpu_preflight")
GPUS = [f"GPU-00000000-0000-0000-0000-{i:012x}" for i in range(6)]


class LauncherTests(unittest.TestCase):
    def test_wrong_gpu_count_duplicates_and_indices_rejected(self):
        for value in [",".join(GPUS[:5]), ",".join(GPUS[:5] + [GPUS[0]]), "0,1,2,3,4,5"]:
            with self.assertRaises(ValueError):
                launch.parse_gpus(value)

    def inventory(self, width=16, memory=97344, name="NVIDIA RTX PRO 6000 Blackwell Server Edition"):
        return "\n".join(f"{g}, 0000:{i+1:02x}:00.0, {width}, {memory}, {name}" for i, g in enumerate(GPUS))

    def test_exact_allocation_accepted(self):
        self.assertEqual(6, len(preflight.validate(GPUS, GPUS, self.inventory())))

    def test_x8_wrong_memory_model_or_visibility_rejected(self):
        for inv in [self.inventory(width=8), self.inventory(memory=48000), self.inventory(name="NVIDIA RTX 6000 Ada")]:
            with self.assertRaises(ValueError):
                preflight.validate(GPUS, GPUS, inv)
        with self.assertRaises(ValueError):
            preflight.validate(GPUS, GPUS[::-1], self.inventory())
        with self.assertRaises(ValueError):
            preflight.validate(GPUS, GPUS, self.inventory() + "\n" + self.inventory().splitlines()[0])

    def test_command_preserves_paths_order_and_memory_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            model = root / "model with spaces"
            model.mkdir()
            for name in ["config.json", "model.safetensors.index.json"]:
                (model / name).write_text("{}")
            args = argparse.Namespace(gpus=",".join(GPUS), port=30000, model_dir=str(model),
                                      cache_dir=str(root / "cache"), name="recipe", image="recipe:test",
                                      profile="single")
            cmd, cache = launch.docker_command(args)
            self.assertIn(f"type=bind,src={model},dst=/models,readonly", cmd)
            self.assertEqual('"device=' + args.gpus + '"', cmd[cmd.index("--gpus") + 1])
            self.assertIn("127.0.0.1:30000:30000", cmd)
            self.assertIn("CUDA_VISIBLE_DEVICES=" + args.gpus, cmd)
            self.assertEqual("1048576", cmd[cmd.index("--max-model-len") + 1])
            self.assertEqual("0", cmd[cmd.index("--cpu-offload-gb") + 1])
            self.assertFalse(cache.exists(), "Dry command generation must not create directories")
            args.profile = "parallel"
            parallel, _ = launch.docker_command(args)
            self.assertEqual("24", parallel[parallel.index("--max-num-seqs") + 1])
            self.assertEqual("1048576", parallel[parallel.index("--max-model-len") + 1])
            graphs = json.loads(parallel[parallel.index("--compilation-config") + 1])
            self.assertEqual([1, 2, 4, 8, 16, 24], graphs["cudagraph_capture_sizes"])
            args.profile = "../profile"
            with self.assertRaises(ValueError):
                launch.docker_command(args)
            args.profile = "single"
            args.cache_dir = str(model / "cache")
            with self.assertRaises(ValueError):
                launch.docker_command(args)


if __name__ == "__main__":
    unittest.main()
