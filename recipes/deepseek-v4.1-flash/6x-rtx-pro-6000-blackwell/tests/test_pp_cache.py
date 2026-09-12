"""Regression: an empty PP cache group must not allocate another rank's layers."""
from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheLayout,
    UniformTypeKVCacheSpecs,
)


def test_empty_projected_group():
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16
    )
    remote = UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs={"remote_layer": spec}
    )
    groups = [KVCacheGroupSpec([], remote), KVCacheGroupSpec(["local_layer"], spec)]
    cache = SimpleNamespace(
        num_gpu_blocks_override=None,
        prefix_cache_retention_interval=None,
        get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
    )
    result = get_kv_cache_config_from_groups(
        SimpleNamespace(cache_config=cache), groups, 1024 * 1024
    )
    assert len(result.kv_cache_groups) == 2, "Global group IDs must stay stable"
    assert not result.kv_cache_groups[0].layer_names
    assert {name for tensor in result.kv_cache_tensors for name in tensor.layers} == {
        "local_layer"
    }
    assert result.num_blocks > 0


if __name__ == "__main__":
    test_empty_projected_group()
    print("PP cache projection regression passed")
