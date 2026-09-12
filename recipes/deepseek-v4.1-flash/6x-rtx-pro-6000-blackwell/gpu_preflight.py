"""Reject a mismatched GPU allocation before allocating model weights."""
import csv
import os
import subprocess


def validate(expected, visible, inventory):
    if len(expected) != 6 or len(set(expected)) != 6 or visible != expected:
        raise ValueError("Expected six distinct GPUs in the configured order")
    rows = [[value.strip() for value in row] for row in csv.reader(inventory.splitlines()) if row]
    if len(rows) != 6 or {row[0] for row in rows} != set(expected):
        raise ValueError("Container must expose exactly the six selected GPU UUIDs")
    for uuid, bus, width, memory, name in rows:
        if width != "16" or int(memory) < 97000 or "RTX PRO 6000 Blackwell" not in name:
            raise ValueError(f"Unsupported GPU {uuid} at {bus}: {name}, x{width}, {memory} MiB")
    return rows


def main():
    expected = os.environ["RECIPE_EXPECTED_GPU_UUIDS"].split(",")
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,pci.bus_id,pcie.link.width.current,memory.total,name",
        "--format=csv,noheader,nounits"], text=True)
    for uuid, bus, width, memory, name in validate(expected, visible, inventory):
        print(f"GPU preflight: {uuid} at {bus}, {name}, x{width}, {memory} MiB", flush=True)


if __name__ == "__main__":
    main()
