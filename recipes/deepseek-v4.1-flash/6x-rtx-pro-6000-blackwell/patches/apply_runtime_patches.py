"""Apply the PP and SM120 fixes needed by the pinned V4.1 preview."""
import hashlib
from pathlib import Path

ROOT = Path('/usr/local/lib/python3.12/dist-packages/vllm')


def patch(relative_path, expected_hash, before, after):
    path = ROOT / relative_path
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == expected_hash, relative_path
    text = content.decode()
    assert text.count(before) == 1, relative_path
    path.write_text(text.replace(before, after))


patch(
    'v1/worker/gpu/cudagraph_utils.py',
    '211de2232fb71aeb761dedbd7fadb7e0832db9331eac6951a1eafb6d77f1be9c',
    '''                # Update for non-first PP ranks.
                model_inputs["input_ids"] = None''',
    '''                # Match normal PP execution: token-dependent models need
                # raw IDs during graph warmup and capture on every stage.
                from vllm.model_executor.models.interfaces import (
                    requires_raw_input_tokens,
                )

                if not requires_raw_input_tokens(model):
                    model_inputs["input_ids"] = None''',
)


patch(
    'v1/worker/gpu/model_runner.py',
    '33e49f3bd99e642971cea41bbcc73677cb74b863dbf16998d9ebfc94607a64e6',
    '''            # Update for non-first PP ranks.
            model_inputs["input_ids"] = None''',
    '''            # Later DeepSeek V4.1 stages need the raw IDs for Engram
            # hashing and token-dependent expert routing. PP broadcasts sampled
            # IDs to every rank, so their local input batches are kept current.
            if not requires_raw_input_tokens(self.model):
                model_inputs["input_ids"] = None''',
)

patch(
    'v1/core/kv_cache_utils.py',
    '0c4b312712598930d2dbdcf917c511de73fe531dd009f5b200e3c945b009bafe',
    '''    kv_cache_tensors = []
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec''',
    '''    kv_cache_tensors = []
    for group in kv_cache_groups:
        # Preserve global group IDs, but allocate only this PP worker's layers.
        # An empty projected group retains the global UniformType spec mapping.
        if not group.layer_names:
            continue
        group_spec = group.kv_cache_spec''',
)

patch(
    'models/deepseek_v4_1/attention.py',
    'ef13a8503b54172a63cca6932e2ee5a6d5d6ced81445949067f01b3f61ab6e5e',
    """            backend_cls=self.swa_backend_cls,
            block_size=32,""",
    """            backend_cls=self.swa_backend_cls,
            # FlashInfer SM120 primary-cache kernels require 64-token pages.
            block_size=64 if torch.cuda.get_device_capability()[0] == 12 else 32,""",
)

# With 64-token logical blocks, V4.1 ratio-2 sources use 32-token pages.
# The existing generic prefill template supports this layout but the pinned
# FlashInfer dispatcher only instantiates secondary sizes 2 and 64.
patch(
    '../flashinfer/data/csrc/sparse_mla_sm120_prefill.cu',
    '4596e198530866b2bd0fdd5abd862d408e811260cd908f3cfa83dec4edc77d94',
    """  } else if (extra_page_block_size == 2) {
    DISPATCH_BY_NH_PBSX(2);
  }
  return false;
#undef DISPATCH_BY_NH_PBSX""",
    """  } else if (extra_page_block_size == 2) {
    DISPATCH_BY_NH_PBSX(2);
  } else if (extra_page_block_size == 32) {
    DISPATCH_BY_NH_PBSX(32);
  }
  return false;
#undef DISPATCH_BY_NH_PBSX""",
)

patch(
    'models/deepseek_v4_1/nvidia/flashinfer_sparse.py',
    '19c8c2ebcffacd2e8fc370c005c7a2b619585597dfc1152e8010c7fecfd215c1',
    """    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]""",
    """    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        from vllm.platforms import current_platform

        capability = current_platform.get_device_capability()
        # DeepGEMM indexer metadata accepts physical pages of 32 or 64.
        # V4.1 has both ratio-1 and ratio-2 sources, so use logical size 64.
        return [64] if capability and capability.major == 12 else [128]""",
)

patch(
    'v1/attention/backends/mla/indexer.py',
    '392d93da110ec942db3ce17895dc358bf92501edb01a8b9c19b6bdfab917bb7f',
    'return [64 if current_platform.is_device_capability_family(90) else 128]',
    """if current_platform.is_device_capability_family(120):
            return [64, 128]
        return [64 if current_platform.is_device_capability_family(90) else 128]""",
)

patch(
    'models/deepseek_v4_1/attention.py',
    'e1c5eb9bcf72118c1c153b6c93d42cd498b183d6cfb9f2702c39222da0b5dbc0',
    '            block_size=self.cache_config.block_size,',
    """            # SM120 FP8 paged indexer kernels require 64 physical entries.
            # Ratio-2 index caches therefore own 128 logical tokens per block;
            # their block table is independent of the attention-cache table.
            block_size=(
                64 * self.compress_ratio
                if torch.cuda.get_device_capability()[0] == 12
                else self.cache_config.block_size
            ),""",
)

patch(
    'v1/attention/backends/mla/indexer.py',
    'd611939c7e4d35b28283831d50c33df9097489ee8b521c836dcd7650e8854f73',
    '    return max_model_len * 40',
    """    # Process arbitrarily many requests in bounded request/query chunks.
    # One complete request must fit, but their aggregate context need not fit
    # simultaneously in this gathered-key workspace. The metadata splitter
    # must use the same physical capacity after KV compression.
    if (
        getattr(vllm_config.model_config.hf_config, "model_type", None)
        == "deepseek_v41"
    ):
        return max_model_len
    return max_model_len * 40""",
)


patch(
    'v1/attention/backends/mla/indexer.py',
    'ca2a61583028eda860aa4e978a830bf4f6cbd95a6b25e81a3fc819d7bb2e986f',
    """                self.max_prefill_buffer_size,
                max_logits_bytes,""",
    """                # V4.1 allocates gathered keys after compression; enforce
                # that physical bound across multiple prefill requests.
                (
                    self.max_prefill_buffer_size // self.compress_ratio
                    if getattr(
                        self.vllm_config.model_config.hf_config, "model_type", None
                    ) == "deepseek_v41"
                    else self.max_prefill_buffer_size
                ),
                max_logits_bytes,""",
)

# Native V4.1 expert matrices exceed 2 GiB. Normalize zero signs in bounded
# groups of outer rows rather than materializing several full-size temporaries.
patch(
    'model_executor/layers/fused_moe/b12x.py',
    'ee76cc6aa801e24be8ec78a9e8502061ad48667d5ff8711e43f68fba912425bd',
    """    packed = packed.view(torch.uint8)
    magnitude = packed & 0x77
    nonzero = (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11
    packed.bitwise_and_(0x77 | (nonzero << 3))""",
    """    packed = packed.view(torch.uint8)
    row_elements = packed.numel() // max(1, packed.shape[0]) if packed.ndim else 1
    rows_per_chunk = max(1, (16 * 1024 * 1024) // max(1, row_elements))
    chunks = packed.split(rows_per_chunk, dim=0) if packed.ndim else (packed,)
    for chunk in chunks:
        magnitude = chunk & 0x77
        nonzero = (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11
        chunk.bitwise_and_(0x77 | (nonzero << 3))""",
)
