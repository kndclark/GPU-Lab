# syntax=docker/dockerfile:1.7
#
# One Dockerfile, two arch targets  (sm_86/sm_120 runbook, Phase 0 step 6).
#
# The two images MUST differ only in the arch arguments below. Everything else --
# base image, package set, source -- is identical, so any measured difference
# between the nodes is attributable to the architecture and not the environment.
#
#   desktop  RTX 3090        sm_86     TORCH_CUDA_ARCH_LIST=8.6
#   laptop   RTX 5090 Laptop sm_120    TORCH_CUDA_ARCH_LIST=12.0+PTX
#
ARG CUDA_BASE=nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04
FROM ${CUDA_BASE}

# ---- the only per-node variables ----
ARG TORCH_CUDA_ARCH_LIST
ARG NVCC_GENCODE

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    NVCC_GENCODE=${NVCC_GENCODE} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
        git build-essential ninja-build cmake ccache ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY arch_probe.cu .

# Compile natively for this node's arch. No PTX-only fallback on the desktop;
# the laptop additionally embeds PTX for forward-compat within the 12.x family.
# NOTE: `nvcc -arch=sm_XX` silently embeds PTX as well as SASS, so a mistargeted
# image JIT-compiles instead of failing. Explicit -gencode is what gives the desktop
# a cubin-only binary that errors out loudly on the wrong hardware.
RUN nvcc ${NVCC_GENCODE} -o /usr/local/bin/arch-probe arch_probe.cu \
    && echo "--- fatbin contents ---" \
    && cuobjdump /usr/local/bin/arch-probe | grep -E "Fatbin|arch =" 

CMD ["arch-probe"]
