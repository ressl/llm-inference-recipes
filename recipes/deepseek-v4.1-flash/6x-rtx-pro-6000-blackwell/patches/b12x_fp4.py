# SPDX-License-Identifier: AGPL-3.0-only
"""Native V4.1 cache on SM120; retain upstream projections and index selection.

The main latent is NVFP4 (288 bytes); the SWA latent is block-scaled FP8
(528 bytes). The indexer remains FP8. Cache storage is opaque uint8 throughout.
"""
from dataclasses import replace

import torch
from b12x.attention import compressed_sparse_mla as mla
from b12x.attention.compressed_sparse_mla.preparation import rotate

from vllm.logger import init_logger
from vllm.models.deepseek_v4_1.common.ops import compute_global_topk_indices_and_lens
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


def byte_pages(cache):
    # flatten only the within-page dimensions, preserving the pool's stride.
    # reshape() must never silently copy the pooled cache.
    return cache.view(cache.shape[0], -1)


def rotated(x, positions, cos_sin_cache, ratio=1):
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rotate(x, positions, cos_sin_cache, out=out, rope_dim=64, ratio=ratio)
    return out


class DeepseekV41B12xFP4Attention(DeepseekV4FlashInferSM120Attention):
    recipe_b12x_fp4 = True

    def __init__(self, vllm_config, *args, **kwargs):
        if vllm_config.speculative_config is not None:
            raise ValueError("Recipe V4.1 FP4 cache is qualified without speculation")
        super().__init__(vllm_config, *args, **kwargs)
        self.swa_cache_layer.recipe_b12x_fp4 = True
        self._recipe_config = vllm_config
        self._recipe_plans = {}
        self._recipe_graph_resources = []
        logger.info_once(
            "Recipe B12X V4.1 cache: main NVFP4 288 B, SWA FP8 528 B, "
            "indexer FP8; native attention, no speculative decoding."
        )

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        # dataclasses.replace preserves the old fp8_ds_mla page padding unless
        # explicitly cleared. Keeping it defeats the compact record geometry.
        return replace(spec, state_content_bytes=288, alignment=16,
                       page_size_padded=None) if spec else None

    def _ensure_plans(self, device):
        if self._recipe_plans:
            return
        cfg = self._recipe_config
        decode_rows = max(
            cfg.scheduler_config.max_num_seqs,
            cfg.compilation_config.max_cudagraph_capture_size or 0,
        )
        main_page = cfg.cache_config.block_size // max(self.compress_ratio, 1)
        for mode, rows in (("decode", decode_rows),
                           ("extend", self.max_num_batched_tokens)):
            # Vision can widen SWA in prefill. Decode always has the base window.
            swa_width = self.window_size + (self.max_image_tokens if mode == "extend" else 0)
            indexed_width = 512 if self.compress_ratio else 0
            plan = mla.plan(mla.Caps(
                device=device, num_q_heads=self.padded_heads, max_q_rows=rows,
                max_width=swa_width + indexed_width,
                swa_width=swa_width, indexed_width=indexed_width,
                swa_page_size=self.swa_cache_layer.block_size,
                indexed_page_size=main_page, cache_format="deepseek_v41",
                mode=mode, use_cuda_graph=True,
                # Extend uses a single-pass kernel. Avoid reserving a decode
                # split-K tensor for every prefill row (several GiB).
                max_chunks_per_row=1 if mode == "extend" else 64,
            ))
            self._recipe_plans[mode] = plan
            current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())

    def _reserve_empty_forward_workspace(self):
        self._ensure_plans(torch.device("cuda", torch.cuda.current_device()))

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        if not isinstance(attn_metadata, dict):
            return super()._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        # V4.1 does not apply an extra per-head Q norm here.
        assert self.n_local_heads == self.padded_heads
        meta = attn_metadata[self.swa_cache_layer.prefix]
        cos_sin = self.rotary_emb.cos_sin_cache
        q_rotated = rotated(q, positions, cos_sin)
        kv_rotated = rotated(kv, positions, cos_sin)
        mla.write_cache(
            kv_rotated[:meta.slot_mapping.numel()],
            byte_pages(self.swa_cache_layer.kv_cache), meta.slot_mapping,
            page_size=meta.block_size, cache_kind="swa",
        )
        return q_rotated

    def insert_compressed_cache(self, latent, positions, rotary_emb, metadata):
        count = metadata.slot_mapping.numel()
        # Compressor output is defined only on complete compression groups.
        # Mask before publication even if a future metadata builder emits slots
        # for their incomplete rows.
        slots = torch.where(
            (positions[:count] + 1).remainder(self.compress_ratio) == 0,
            metadata.slot_mapping, -1,
        )
        kv = rotated(latent[:count], positions[:count],
                     rotary_emb.cos_sin_cache, self.compress_ratio)
        mla.write_cache(kv, byte_pages(self.kv_cache), slots,
                        page_size=metadata.block_size // self.compress_ratio,
                        cache_kind="indexed")

    def _run_b12x(self, mode, q, output, swa_indices, swa_lengths,
                  main_cache, main_indices, main_lengths, main_page):
        self._ensure_plans(q.device)
        plan = self._recipe_plans[mode]
        scratch = current_workspace_manager().get_simultaneous(*plan.shapes_and_dtypes())
        binding = plan.bind(
            scratch=scratch, q=q, swa_indices=swa_indices, swa_lengths=swa_lengths,
            indexed_indices=main_indices, indexed_lengths=main_lengths,
        )
        if torch.cuda.is_current_stream_capturing():
            self._recipe_graph_resources.append(binding)
        mla.run(
            binding=binding, swa_k_cache=byte_pages(self.swa_cache_layer.kv_cache),
            indexed_k_cache=byte_pages(main_cache) if main_cache is not None else None,
            swa_page_size=self.swa_cache_layer.block_size,
            indexed_page_size=main_page, sm_scale=self.scale,
            attn_sink=self.attn_sink, out=output, cache_format="deepseek_v41",
        )

    def _forward_decode(self, q, kv_cache, swa_metadata, attn_metadata,
                        swa_only, output):
        count = swa_metadata.num_decode_tokens
        indices = lengths = page = None
        if not swa_only:
            page = attn_metadata.block_size // self.compress_ratio
            indices, lengths = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:count],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:swa_metadata.num_decodes], page,
                swa_metadata.is_valid_token[:count],
            )
        self._run_b12x("decode", q, output, swa_metadata.decode_swa_indices,
                       swa_metadata.decode_swa_lens, kv_cache, indices, lengths, page)

    def _forward_prefill(self, q, compressed_k_cache, swa_k_cache, output,
                         attn_metadata, swa_metadata):
        start = swa_metadata.num_decode_tokens
        end = start + swa_metadata.num_prefill_tokens
        indices = lengths = page = None
        if self.compress_ratio:
            page = attn_metadata.block_size // self.compress_ratio
            indices, lengths = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[start:end],
                swa_metadata.token_to_req_indices[start:end],
                attn_metadata.block_table, page,
                swa_metadata.is_valid_token[start:end],
            )
        self._run_b12x("extend", q, output, swa_metadata.prefill_swa_indices,
                       swa_metadata.prefill_swa_lens, compressed_k_cache,
                       indices, lengths, page)
