# Third-party notices

Original documentation, launch tooling, tests and changes in this repository
are licensed under AGPL-3.0-only. Upstream code excerpts embedded in the runtime
patch script remain subject to their original terms; the repository license
does not replace those notices.

| Component | Source used | License and included notice |
| --- | --- | --- |
| vLLM | Public `vllm/vllm-openai` image pinned in the recipe; package `0.1.dev20904+g179dd0fa9` | Apache-2.0; [license](licenses/vLLM-Apache-2.0.txt) |
| FlashInfer | [c05407ceffb7d9ce111a73553a8e37ad752adb62](https://github.com/flashinfer-ai/flashinfer/tree/c05407ceffb7d9ce111a73553a8e37ad752adb62) | Apache-2.0 and the additional notices in its [LICENSE](licenses/FlashInfer-LICENSE.txt) |
| B12X | [1.3.0](https://pypi.org/project/b12x/1.3.0/) | Apache-2.0; [license from the wheel](licenses/B12X-LICENSE.txt) |

The vLLM and FlashInfer search/replacement fragments in
`recipes/deepseek-v4.1-flash/6x-rtx-pro-6000-blackwell/patches/apply_runtime_patches.py`
are modified upstream code. The script identifies the affected files and checks
their original SHA-256 hashes before modification. The B12X normalization fix
changes vLLM's B12X adapter; B12X itself is installed from its pinned wheel.
The Docker build retains dependencies' installed license files.

[DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277)
is downloaded separately under its upstream MIT license. No model weights are
distributed by this repository. NVIDIA CUDA, PyTorch and other base-image
components are separate dependencies with their own terms and notices.
