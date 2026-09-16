#!/usr/bin/env bash
# Run the whole kernel suite on this node, detached, one directory at a time.
# Usage: ./overnight.sh          (detaches and returns immediately)
#        ./overnight.sh --fg     (stay in the foreground, for debugging)
#
# Three things this exists to get right, none of which a single long pytest
# invocation does:
#
# 1. ONE PYTEST PROCESS PER FILE. Two separate reasons, both learned the hard
#    way on the night of 2026-09-15.
#
#    pytest writes its junit XML at the end of a run, so a twelve-hour
#    invocation interrupted at hour eleven produces nothing at all.
#
#    Worse: a test that kills the CUDA context takes every later test in the
#    same process down with it, and they are recorded as ordinary failures. On
#    sm_120 one device-side assert in test_silu_mul_fp8_quant_deep_gemm turned
#    into 1882 bogus MoE "failures", and a second in test_block_fp8 voided 6758
#    quantization tests -- including the INT8 results that the earlier targeted
#    run had measured cleanly. Nothing in the output distinguishes a real
#    failure from collateral damage.
#
#    So each FILE gets its own pytest process. A context kill then voids the
#    rest of that one file instead of the rest of the night. Directories are
#    still ordered highest-value first, because quantization and moe are where
#    the two cards differ.
#
# 2. THE LAPTOP MUST NOT SUSPEND. A lid-close ends the run and wedges the I226-V
#    NIC on the way down (host/README.md), so the morning's symptom would be an
#    unreachable node rather than an obviously aborted run. systemd-inhibit holds
#    sleep, idle and the lid switch for the duration.
#
# 3. NEITHER NODE MAY SERVE WHILE IT MEASURES. llama-swap loads on demand, so a
#    single inbound request takes the card out from under the suite (23.2 GiB for
#    the coder on the desktop, 8.3 GiB for the embedding model on the laptop) and
#    the allocation failures read as architecture findings. Serving goes down for
#    the run and comes back up straight afterwards, whether the run passed, failed
#    or was killed. Note this puts the whole lab offline for the duration: with
#    the desktop down there is nothing for LiteLLM to fall back to anyway.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/.." && pwd)

# helion, turboquant and scripts collect zero tests in this image and are left
# out deliberately -- an empty run would otherwise look like a clean pass.
DIRS=(quantization moe attention core ir mamba)

cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
case "$cap" in
    86)  arch=sm_86;  node=desktop ;;
    120) arch=sm_120; node=laptop  ;;
    *)   echo "unrecognised compute cap '$cap'" >&2; exit 1 ;;
esac

run_all() {
    local started; started=$(date -u +%FT%TZ)
    echo "=== overnight kernel differential: $arch ($node), started $started ==="

    echo "--- stopping the serving plane so nothing loads a model mid-run ---"
    "$repo/bin/lab" down || true
    # Whatever happens next -- pass, fail, kill -- serving comes back.
    trap 'echo "--- restoring the serving plane ---"; "$repo/bin/lab" up || true' EXIT

    for d in "${DIRS[@]}"; do
        echo
        echo "=== $d  ($(date -u +%FT%TZ)) ==="
        # The tests live inside the image, not on the host, so the image is what
        # knows which files exist.
        files=$(sudo docker run --rm --entrypoint bash gpu-lab:kernels \
                    -c "find tests/kernels/$d -name 'test_*.py' | sort" 2>/dev/null)
        if [ -z "$files" ]; then
            echo "!!! no test files found under tests/kernels/$d -- skipping"
            continue
        fi
        for f in $files; do
            name="$d-$(basename "$f" .py)"
            echo "--- $name  ($(date -u +%FT%TZ)) ---"
            # run.sh is the single entry point, so the overnight run and a
            # hand-run produce identically-named, identical artifacts.
            SUITE="$name" "$here/run.sh" "$f" || echo "!!! $name did not complete"
        done
    done

    echo
    echo "=== finished $(date -u +%FT%TZ) ==="
    ls -la "$here/results"
}

if [ "${1:-}" = "--fg" ]; then
    run_all
    exit $?
fi

log="$here/results/overnight-${arch}.log"
mkdir -p "$here/results"

# setsid detaches from this shell so the run survives the terminal, the ssh
# session and the agent that started it.
inhibit=()
[ "$node" = laptop ] && inhibit=(systemd-inhibit --what=sleep:idle:handle-lid-switch
                                 --why="kernel differential run" --mode=block)

setsid nohup "${inhibit[@]}" "$0" --fg > "$log" 2>&1 < /dev/null &
echo "detached: $arch overnight run, pid $!"
echo "log:      $log"
echo "watch:    tail -f $log"
