"""GPU qualification for Recipe's TP2 cache geometry; no model weights required."""
import json
import math
import torch
from b12x.attention import compressed_sparse_mla as mla
from b12x.attention.compressed_sparse_mla.preparation import rotate
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference, pack_deepseek_v41_cache_reference,
)


@torch.inference_mode()
def qualify(main_page, mode, swa_only=False):
    torch.manual_seed(410128 + main_page)
    device = "cuda:0"
    rows = 24 if mode == "decode" else 1536
    def alloc(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=device)
    def rand(*shape):
        return torch.randn(shape, device=device, dtype=torch.bfloat16)

    # Simulate actual block-outermost shared pools, including inter-page gaps.
    def cache(kind, page, pages=12):
        width = 528 if kind == "swa" else 288
        storage = alloc(pages, 33792, dtype=torch.uint8).fill_(0xAB)
        view = storage[:, :page * width]
        values = rand(pages * page, 512) * 0.7
        slots = torch.arange(len(values), dtype=torch.int64, device=device)
        mla.write_cache(values, view, slots, page_size=page, cache_kind=kind)
        expected = pack_deepseek_v41_cache_reference(values, page_size=page, cache_kind=kind)
        # Exact writer verification is also covered by upstream tests; here
        # allow the hardware's negative-zero FP4 representation equivalence.
        if kind == "swa":
            assert torch.equal(view, expected)
        assert bool((storage[:, page * width:] == 0xAB).all())
        return view

    swa = cache("swa", 64)
    main = None if swa_only else cache("indexed", main_page)
    q = rand(rows, 32, 512) * 0.2
    indices = torch.arange(128, device=device, dtype=torch.int32).repeat(rows, 1) + 64
    lengths = alloc(rows, dtype=torch.int32).fill_(128)
    lengths[0] = 0
    indexed = None if swa_only else torch.arange(512, device=device, dtype=torch.int32).repeat(rows, 1).remainder(main_page * 12)
    main_lengths = None if swa_only else alloc(rows, dtype=torch.int32).fill_(512)
    if main_lengths is not None:
        main_lengths[0] = 0
    sink = torch.linspace(-2, 2, 32, device=device)
    plan = mla.plan(mla.Caps(
        device=device, num_q_heads=32, max_q_rows=rows,
        max_width=128 + (0 if swa_only else 512), swa_width=128,
        indexed_width=0 if swa_only else 512,
        swa_page_size=64, indexed_page_size=main_page,
        mode=mode, cache_format="deepseek_v41", use_cuda_graph=True,
        max_chunks_per_row=1 if mode == "extend" else 64,
    ))
    spec, = plan.scratch_specs()
    scratch = torch.empty(spec.shape, device=device, dtype=spec.dtype)
    binding = plan.bind(scratch=scratch, q=q, swa_indices=indices,
                        swa_lengths=lengths, indexed_indices=indexed,
                        indexed_lengths=main_lengths)
    out = torch.empty_like(q)
    def run():
        mla.run(binding=binding, swa_k_cache=swa, indexed_k_cache=main,
                swa_page_size=64, indexed_page_size=main_page,
                sm_scale=1 / math.sqrt(512), attn_sink=sink, out=out,
                cache_format="deepseek_v41")
    def check():
        pick = torch.tensor([0, 1, rows // 2, rows - 1], device=device)
        expected = compressed_sparse_mla_reference(
            q[pick], swa, indices[pick], lengths[pick],
            extra_k_cache=main,
            extra_indices=indexed[pick] if indexed is not None else None,
            extra_topk_lengths=main_lengths[pick] if main_lengths is not None else None,
            swa_page_size=64, extra_page_size=main_page,
            sm_scale=1 / math.sqrt(512), attn_sink=sink,
            cache_format="deepseek_v41",
        )
        torch.testing.assert_close(out[pick], expected, rtol=0.035, atol=0.035)
        assert bool(torch.isfinite(out).all())
    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.mul_(0.7)
    indices.add_(64)
    lengths[1:] = 73
    if main_lengths is not None:
        main_lengths[1:] = 257
    graph.replay()
    torch.cuda.synchronize()
    check()
    print(json.dumps({"mode": mode, "rows": rows, "main_page": main_page,
                      "swa_only": swa_only, "scratch_MiB": scratch.numel() / 2**20,
                      "changed_input_graph": "passed"}), flush=True)


@torch.inference_mode()
def qualify_rope():
    x = torch.randn((11, 32, 512), device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 2, 3, 31, 32, 1023, 1024, 4095, 4096, 8191], device="cuda")
    phases = torch.randn((8192, 32), device="cuda")
    cs = torch.cat((phases.cos(), phases.sin()), -1)
    for ratio in (1, 2):
        out = torch.empty_like(x)
        rotate(x, positions, cs, out=out, ratio=ratio)
        p = positions // ratio * ratio
        c, s = cs[p, :32, None].transpose(1, 2), cs[p, 32:, None].transpose(1, 2)
        tail = x[..., 448:].float().reshape(11, 32, 32, 2)
        expected = x.clone()
        expected[..., 448::2] = tail[..., 0] * c - tail[..., 1] * s
        expected[..., 449::2] = tail[..., 1] * c + tail[..., 0] * s
        torch.testing.assert_close(out, expected, atol=0.032, rtol=0.008)
    print('{"rope_ratio_1_and_2": "passed"}', flush=True)


if __name__ == "__main__":
    qualify_rope()
    for mode in ("decode", "extend"):
        for page in (32, 64):
            qualify(page, mode)
        qualify(64, mode, swa_only=True)
