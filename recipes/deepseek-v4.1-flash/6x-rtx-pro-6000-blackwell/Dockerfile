FROM vllm/vllm-openai@sha256:4f3c8bcf6328305b8cb6f61146dbfa125b0b04ae98eef3d091019ec30f564a9b AS flashinfer-builder

# The preview's FlashInfer 0.6.18 cannot execute V4.1's SM120 attention shapes.
# Build the official source at an immutable revision; EP transport is unused.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && git init /tmp/flashinfer \
    && git -C /tmp/flashinfer remote add origin https://github.com/flashinfer-ai/flashinfer.git \
    && git -C /tmp/flashinfer fetch --depth 1 origin c05407ceffb7d9ce111a73553a8e37ad752adb62 \
    && git -C /tmp/flashinfer checkout --detach FETCH_HEAD \
    && git -C /tmp/flashinfer submodule update --init --recursive --depth 1 \
    && python3 -m pip install --no-deps setuptools==80.9.0 wheel==0.45.1 \
    && BUILD_NVEP=0 python3 -m pip wheel --no-deps --no-build-isolation \
         --wheel-dir /wheels /tmp/flashinfer

FROM vllm/vllm-openai@sha256:4f3c8bcf6328305b8cb6f61146dbfa125b0b04ae98eef3d091019ec30f564a9b

LABEL org.opencontainers.image.title="DeepSeek V4.1 Flash six-GPU recipe"
LABEL org.opencontainers.image.description="Pinned vLLM runtime for six RTX PRO 6000 Blackwell GPUs"
LABEL org.opencontainers.image.flashinfer-revision="c05407ceffb7d9ce111a73553a8e37ad752adb62"
COPY --from=flashinfer-builder /wheels/flashinfer_python-0.7.0-py3-none-any.whl /opt/recipe/
RUN python3 -m pip uninstall -y flashinfer-jit-cache flashinfer-cubin \
    && python3 -m pip install --no-deps --force-reinstall /opt/recipe/flashinfer_python-0.7.0-py3-none-any.whl \
    && rm /opt/recipe/flashinfer_python-0.7.0-py3-none-any.whl
COPY patches/apply_runtime_patches.py /opt/recipe/apply_runtime_patches.py
RUN python3 /opt/recipe/apply_runtime_patches.py
COPY --chmod=755 entrypoint.sh /opt/recipe/
COPY gpu_preflight.py healthcheck.py profile.json /opt/recipe/
COPY warmup.py /opt/recipe/warmup.py
RUN python3 -m pip install --no-deps \
    'https://files.pythonhosted.org/packages/d2/bc/f28f6b3ade2522785f57caaebb2b02e35e7f189e647f47b0da8e7f581e36/b12x-1.3.0-py3-none-any.whl#sha256=c97d88635521a7fdd4f67717c1835a017d0dcea49d49cc7aa5b4299f290d19e3'
COPY tests /opt/recipe/tests
COPY benchmarks/*.py /opt/recipe/benchmarks/
USER 1000:1000
HEALTHCHECK --interval=15s --timeout=180s --start-period=30m --retries=3 \
    CMD ["python3", "/opt/recipe/healthcheck.py"]
ENTRYPOINT ["/opt/recipe/entrypoint.sh"]
