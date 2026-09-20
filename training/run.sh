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
# 2. ONLY THE GPU GOES DOWN. QLoRA on an 8B needs most of the card, and
#    llama-swap loads on demand -- one inbound request during the run would
#    contend for the memory the trainer is using. But LiteLLM, Prometheus and
#    Grafana are CPU containers that request no GPU, so stopping them frees
#    nothing and costs the front door.
#
#    That distinction is what makes the runbook's Phase 2b promise -- "the
#    desktop trains, the laptop serves, and your tools never notice" -- actually
#    achievable. A full "lab down" stops LiteLLM, which IS the front door, so no
#    laptop-side failover can work however the config is written. "lab down
#    --gpu-only" frees the card and leaves routing up: qwen3-embed keeps
#    answering from the laptop on the same base URL, and Prometheus keeps
#    scraping it, so the run stays observable while it happens.
#
#    Restored from an EXIT trap whether this passes, fails or is killed.
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
NAME=${NAME:-qwen3-8b-research}

cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
if [ "$cap" != "86" ]; then
    if [ "${FORCE:-0}" != "1" ]; then
        echo "=== Running on non-3090 node (compute cap $cap) -> dispatching to RTX 3090 (llm) ==="
        rsync -az --exclude='__pycache__' --exclude='runs' "$here/" llm:~/gpu-lab/training/
        exec ssh llm "cd ~/gpu-lab/training && ./run.sh $*"
    fi
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

    echo "--- freeing the GPU (front door stays up) ---"
    "$repo/bin/lab" down --gpu-only || true
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
