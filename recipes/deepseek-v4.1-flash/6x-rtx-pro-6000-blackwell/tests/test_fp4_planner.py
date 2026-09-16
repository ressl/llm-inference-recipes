"""Exercise the installed planner with the actual 40-layer TP2/PP3 geometry."""
import json
import os
from types import SimpleNamespace as NS
import torch
from vllm.v1.core import kv_cache_utils as planner
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec, SlidingWindowMLASpec, CircularBufferSpec,
    KVCacheLayout, get_kv_quant_mode,
)


def test_adapter_cache_pages():
    # Construct only the cache-spec owners, without model parameters or GPUs.
    # This catches inherited padding that a hand-written geometry would miss.
    from vllm.models.deepseek_v4_1.nvidia.b12x_fp4 import DeepseekV41B12xFP4Attention
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
    cfg = NS(cache_config=NS(block_size=64, cache_dtype="fp8_ds_mla"))
    owner = DeepseekV41B12xFP4Attention.__new__(DeepseekV41B12xFP4Attention)
    torch.nn.Module.__init__(owner)
    owner.is_kv_source = True
    owner.kv_cache_dtype = "fp8_ds_mla"
    owner.kv_cache_torch_dtype = torch.uint8
    owner.head_dim = 512
    for ratio in (1, 2):
        owner.compress_ratio = ratio
        spec = owner.get_kv_cache_spec(cfg)
        assert spec.state_content_size_bytes == 288
        assert spec.page_size_bytes == 64 // ratio * 288, spec
        assert spec.page_size_padded is None, spec
    swa = NS(cache_config=cfg.cache_config, recipe_b12x_fp4=True,
             block_size=64, head_dim=512, dtype=torch.uint8, window_size=128)
    spec = DeepseekV4SWACache.get_kv_cache_spec(swa, cfg)
    assert spec.state_content_size_bytes == 528
    assert spec.page_size_bytes == 64 * 528
    print("Actual adapter main/SWA cache pages and cleared padding passed")


def test_recipe_fp4_capacity():
    cfg = NS(
        model_config=NS(max_model_len=1048576, original_max_model_len=1048576,
                        hf_config=NS(model_type="deepseek_v41")),
        scheduler_config=NS(max_num_batched_tokens=1536, max_num_seqs=24,
                            disable_hybrid_kv_cache_manager=False),
        parallel_config=NS(decode_context_parallel_size=1, pipeline_parallel_size=3),
        cache_config=NS(num_gpu_blocks_override=None, prefix_cache_retention_interval=0,
                        get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC),
        speculative_config=None, max_in_flight_tokens=6144,
    )
    spec = {}
    common = dict(num_kv_heads=1, head_size=512, dtype=torch.uint8,
                  cache_dtype_str="fp8_ds_mla", alignment=16,
                  model_version="deepseek_v4", kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"))
    for i in range(40):
        name = f"model.layers.{i}.attn"
        spec[name + ".swa_cache"] = SlidingWindowMLASpec(
            **common, block_size=64, sliding_window=128, state_content_bytes=528)
        if i in (2, 8, 14, 20):
            ratio = 2 if i < 20 else 1
            spec[name] = MLAAttentionSpec(**common, block_size=64,
                                          tokens_per_state=ratio, state_content_bytes=288)
            spec[name + ".indexer.k_cache"] = MLAAttentionSpec(
                block_size=64 * ratio, num_kv_heads=1, head_size=132,
                dtype=torch.uint8, tokens_per_state=ratio, alignment=576)
            if ratio == 2:
                spec[name + ".compressor.state_cache"] = CircularBufferSpec(
                    block_size=8, num_kv_heads=1, head_size=1024,
                    head_size_v=0, dtype=torch.float32)
    workers = [{k: v for k, v in spec.items() if start <= int(k.split(".")[2]) < end}
               for start, end in ((0, 8), (8, 20), (20, 40))]
    os.environ["RECIPE_DSV41_FP4_CACHE"] = "1"
    configs = planner.get_kv_cache_configs(cfg, workers, [1932735283] * 3)
    scheduler = planner.generate_scheduler_kv_cache_config(configs)
    capacity, concurrency = planner.get_kv_cache_capacity(cfg, scheduler)
    assert scheduler.num_blocks == 57195
    assert capacity == 2101450, (capacity, concurrency)
    assert concurrency > 2
    for worker, config in zip(workers, configs):
        allocated = {name for tensor in config.kv_cache_tensors for name in tensor.layers}
        assert allocated == set(worker), (allocated, set(worker))
        assert planner._pool_bytes_per_block(config.kv_cache_groups) == 33792
    print(json.dumps({"pool_blocks": scheduler.num_blocks, "token_equivalent": capacity,
                      "full_contexts": concurrency, "stage_stride_bytes": 33792}))


if __name__ == "__main__":
    test_adapter_cache_pages()
    test_recipe_fp4_capacity()
