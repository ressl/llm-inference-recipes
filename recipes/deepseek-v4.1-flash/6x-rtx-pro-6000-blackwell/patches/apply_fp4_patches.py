"""Fail closed if the pinned runtime changed; enable only with an explicit flag."""
import hashlib
import os
from pathlib import Path
import shutil

ROOT = Path(os.environ.get("VLLM_PATCH_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm"))


def patch(relative, digest, replacements):
    path = ROOT / relative
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, f"Unexpected base: {relative}"
    text = raw.decode()
    for old, new in replacements:
        assert text.count(old) == 1, f"Non-unique patch anchor: {relative}: {old}"
        text = text.replace(old, new)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"FP4 patch verified: {relative}")


patch("models/deepseek_v4_1/nvidia/model.py",
      "530ed24c8fd2e9daeb5c3d340ef52246618786f19eace8ec217c8d3bdf271110", [(
    "    backend = vllm_config.attention_config.backend\n",
    '''    import os
    if os.environ.get("RECIPE_DSV41_FP4_CACHE") == "1":
        capability = current_platform.get_device_capability()
        if capability is None or capability.major != 12:
            raise ValueError("Recipe V4.1 FP4 cache requires SM120/SM121")
        from .b12x_fp4 import DeepseekV41B12xFP4Attention
        return DeepseekV41B12xFP4Attention
    backend = vllm_config.attention_config.backend
''')])

patch("models/deepseek_v4_1/compressor.py",
      "ed66374bb9e4c8c0201383d3fef811e061fd14f3cb0cb9db74bcb740f5a59009", [(
    "        kv_cache = k_cache_layer.kv_cache\n",
    '''        if getattr(k_cache_layer, "recipe_b12x_fp4", False):
            k_cache_layer.insert_compressed_cache(
                latent, positions, rotary_emb, k_cache_metadata
            )
            return
        kv_cache = k_cache_layer.kv_cache
''')])

patch("v1/attention/backends/mla/sparse_swa.py",
      "cc4c5f3d5dd914ce05f491b03b3bd7c45ba532bd7e566e2b12af348768acecbe", [(
    "            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,",
    '''            state_content_bytes=(528 if getattr(self, "recipe_b12x_fp4", False)
                                 else 584 if uses_fp8_ds_mla_layout else None),'''), (
    "            alignment=576 if uses_fp8_ds_mla_layout else 512,",
    '''            alignment=(16 if getattr(self, "recipe_b12x_fp4", False)
                       else 576 if uses_fp8_ds_mla_layout else 512),''')])

patch("v1/core/kv_cache_utils.py",
      "6f8a22163f6e09e3c8d1fd1875e193fee02a27e05b5dd2d8c3412d188514639f", [(
    "        if num_groups == 1:\n            groups.append(KVCacheGroupSpec(list(spec.kv_cache_specs), spec))",
    '''        # A globally wide attention group can project to a much narrower
        # PP-local group. Do not let two small circular states widen that whole
        # worker's shared pool. One group per state adds just two request blocks.
        if (
            __import__("os").environ.get("RECIPE_DSV41_FP4_CACHE") == "1"
            and vllm_config.parallel_config.pipeline_parallel_size > 1
            and isinstance(spec.first_spec, CircularBufferSpec)
            and vllm_config.model_config.hf_config.model_type == "deepseek_v41"
        ):
            num_groups = len(spec.kv_cache_specs)
        if num_groups == 1:
            groups.append(KVCacheGroupSpec(list(spec.kv_cache_specs), spec))''')])

shutil.copyfile(Path(__file__).with_name("b12x_fp4.py"),
                ROOT / "models/deepseek_v4_1/nvidia/b12x_fp4.py")
