#!/usr/bin/env bash
# Build the arch-specific image for whichever node this runs on.
# Usage: ./build.sh [sm86|sm120]   (auto-detects if omitted)
set -euo pipefail

target="${1:-auto}"
if [[ "$target" == "auto" ]]; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
    case "$cap" in
        86)  target=sm86  ;;
        120) target=sm120 ;;
        *)   echo "unrecognised compute cap '$cap' - pass sm86 or sm120 explicitly" >&2; exit 1 ;;
    esac
    echo "auto-detected compute cap $cap -> $target"
fi

case "$target" in
    # Cubin only, no PTX. Running this image on non-sm_86 hardware fails loudly
    # instead of silently JIT-compiling, which would poison a cross-arch measurement.
    sm86)
        ARCH_LIST="8.6"
        GENCODE="-gencode arch=compute_86,code=sm_86"
        ;;
    # Family target (sm_120f, not sm_120a -- runbook 02) plus PTX, exactly as
    # TORCH_CUDA_ARCH_LIST=12.0+PTX specifies. PTX here is deliberate, not accidental.
    sm120)
        ARCH_LIST="12.0+PTX"
        GENCODE="-gencode arch=compute_120f,code=sm_120f -gencode arch=compute_120f,code=compute_120f"
        ;;
    *)
        echo "usage: $0 [sm86|sm120]" >&2; exit 1
        ;;
esac

echo "building gpu-lab:$target"
echo "  TORCH_CUDA_ARCH_LIST = $ARCH_LIST"
echo "  gencode              = $GENCODE"
exec sudo docker build \
    --build-arg "TORCH_CUDA_ARCH_LIST=$ARCH_LIST" \
    --build-arg "NVCC_GENCODE=$GENCODE" \
    -t "gpu-lab:$target" .
