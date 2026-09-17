#!/usr/bin/env bash
# Run the Phase 2b QLoRA proof on this node.
#   ./run.sh              train, then verify (detaches; returns immediately)
#   ./run.sh --fg         stay in the foreground
#   ./run.sh --verify     verify an existing adapter, no training
#
# Structured after kernels/overnight.sh, and for the same reasons.
#
# 1. THE DESKTOP ONLY. Phase 2b puts training on the 3090 deliberately: the
#    Laptop 5090 has no VRAM advantage (24463 vs 24576 MiB), sm_86 is the
#    better-supported training target, and a multi-hour job at 175 W throttles
#    while the desktop sustains 400 W. Running this on the laptop would produce
#    a thermally-limited number that looks like a result.
#
# 2. SERVING GOES DOWN, AND COMES BACK. QLoRA on an 8B needs most of the card,
#    and llama-swap loads on demand -- one inbound request during the run would
#    contend for the memory the trainer is using. Serving stops for the
#    duration and is restored from an EXIT trap whether this passes, fails or
#    is killed.
#
#    Note what this means for the front door: with the desktop down, LiteLLM is
#    down too, and qwen3-embed on the laptop becomes unreachable through it.
#    Whether that is acceptable is a decision recorded in docs/, not here.
#
# 3. THE ADAPTER IS VERIFIED, NOT ASSUMED. verify.py runs an A/B against
#    held-out probes and reads both answers. An adapter that loads without
#    error has proven nothing; an adapter of all zeros loads fine too.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/.." && pwd)

IMAGE=gpu-lab:training
ADAPTERS=/srv/model-cache/adapters
MODEL=${MODEL:-Qwen/Qwen3-8B}
NAME=${NAME:-qwen3-8b-gpulab}

cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
if [ "$cap" != "86" ]; then
    echo "this is not the 3090 (compute cap $cap)." >&2
    echo "Phase 2b trains on sm_86 deliberately -- see the header of this file." >&2
    echo "set FORCE=1 to override." >&2
    [ "${FORCE:-}" = 1 ] || exit 1
fi

sudo docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "missing image $IMAGE -- build it first:" >&2
    echo "    cd $here && sudo docker build -t $IMAGE ." >&2
    exit 1
}

run_container() {
    # --ipc=host for the dataloader; /srv/model-cache is the NFS-backed weight
    # cache both nodes share, and adapters land in it so either node can serve
    # what this produced.
    sudo docker run --rm --name gpu-lab-training --init --gpus all --ipc=host \
        -v /srv/model-cache:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
        -v "$ADAPTERS":/adapters \
        -v "$here":/training \
        --entrypoint python3 "$IMAGE" "$@"
}

do_verify() {
    echo
    echo "=== verifying the adapter (A/B against held-out probes) ==="
    run_container /training/verify.py --model "$MODEL" --adapter "/adapters/$NAME"
}

run_all() {
    echo "=== Phase 2b QLoRA: $MODEL -> $NAME  ($(date -u +%FT%TZ)) ==="
    nvidia-smi --query-gpu=name,compute_cap,memory.total,temperature.gpu --format=csv,noheader

    sudo mkdir -p "$ADAPTERS"
    sudo chown "$(id -u):$(id -g)" "$ADAPTERS"

    # BEFORE anything is stopped. The first run of this script took the whole
    # serving plane down, loaded an 8B model in 4-bit, and then died on a
    # TrainingArguments keyword that transformers 5.x had removed. Preflight
    # costs seconds against a healthy lab; that cost three minutes against a
    # stopped one.
    echo "--- preflight (serving still up) ---"
    run_container /training/preflight.py "$MODEL"

    echo "--- stopping the serving plane for the duration ---"
    "$repo/bin/lab" down || true
    trap 'echo "--- restoring the serving plane ---"; "$repo/bin/lab" up || true' EXIT

    run_container /training/qlora.py --model "$MODEL" --out "/adapters/$NAME" "$@"
    do_verify

    echo
    echo "=== finished $(date -u +%FT%TZ) ==="
    echo "adapter:  $ADAPTERS/$NAME"
    echo "reports:  train-report.json, verify-report.json"
}

case "${1:-}" in
    --verify) do_verify; exit $? ;;
    --fg)     shift; run_all "$@"; exit $? ;;
esac

mkdir -p "$here/runs"
log="$here/runs/qlora-$(date -u +%Y%m%dT%H%M%SZ).log"
setsid nohup "$0" --fg "$@" > "$log" 2>&1 < /dev/null &
echo "detached: Phase 2b QLoRA run, pid $!"
echo "log:      $log"
echo "watch:    tail -f $log"
