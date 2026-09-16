# syntax=docker/dockerfile:1.7
#
# The kernel differential harness  (sm_86/sm_120 runbook, Phase 2 step 4).
#
# ONE image, both nodes -- and that is the whole design. The arch/ images are
# deliberately two (each compiled for one card); this one is deliberately one,
# because the thing under test is the *prebuilt vLLM wheel's* kernels. If the
# two nodes ran different test code or different dependency versions, a
# disagreement between them would no longer be attributable to the architecture.
#
# Nothing here rebuilds vLLM. Phase 2 step 1 established that the pinned
# vllm/vllm-openai:v0.29.0 wheel runs natively on sm_120, so its kernels are a
# legitimate subject: torch.ops._C.silu_and_mul on the 5090 matches a float
# reference exactly. A source build is still required to *modify* kernels; it is
# not required to *differentiate* them.
FROM vllm/vllm-openai:v0.29.0

# The test dependencies tests/kernels actually needs, and no more. Versions are
# the ones pinned in requirements/test/cuda.txt at the v0.29.0 tag.
#
# --no-deps is load-bearing, not tidiness. The full test requirements resolve to
# several hundred packages including lm-eval and ray, and would be free to move
# torch off 2.13.0+cu130 -- which would mean the two nodes no longer share the
# runtime that makes them comparable. Verify after any edit here:
#     python3 -c "import torch; print(torch.__version__)"   -> 2.13.0+cu130
RUN pip install --no-cache-dir --no-deps \
        tblib==3.1.0 \
        pytest-asyncio==1.4.0 \
        einops==0.8.1 \
        hypothesis==6.131.0 \
        pytest-timeout==2.3.1

# The build context is a bare checkout of the v0.29.0 tag prepared by build.sh.
#
# It carries tests/ and pyproject.toml and NOT the vllm/ source directory, which
# is the trap this layout exists to avoid: pytest puts the working directory on
# sys.path, so a context containing vllm/ makes `import vllm` resolve to
# uncompiled source instead of the installed wheel. Every kernel test would then
# fail on a missing extension module -- or, worse, appear to run against
# something that is not the wheel under test. Confirm with:
#     python3 -c "import vllm; print(vllm.__file__)"
# which must print a path under /usr/local/lib/python3.12/dist-packages.
WORKDIR /kernels
COPY tests ./tests
COPY pyproject.toml ./

ENV PYTHONPATH=/kernels \
    HF_HUB_OFFLINE=1 \
    VLLM_LOGGING_LEVEL=ERROR

ENTRYPOINT ["python3", "-m", "pytest"]
