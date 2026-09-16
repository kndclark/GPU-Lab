#!/usr/bin/env bash
# Run the whole kernel suite on this node, detached, one directory at a time.
# Usage: ./overnight.sh          (detaches and returns immediately)
#        ./overnight.sh --fg     (stay in the foreground, for debugging)
#
# Three things this exists to get right, none of which a single long pytest
# invocation does:
#
# 1. ONE XML PER DIRECTORY. pytest writes its junit XML at the end of the run,
#    so a twelve-hour invocation that gets interrupted at hour eleven produces
#    nothing at all. Directories are run separately and in a fixed order, so an
#    overnight that does not finish still leaves complete, comparable results
#    for the directories that did. Highest-value first, for the same reason:
#    quantization and moe are where the two cards differ.
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
        # run.sh is the single entry point, so the overnight run and a hand-run
        # produce identically-named, identically-produced artifacts.
        SUITE="$d" "$here/run.sh" "tests/kernels/$d" || echo "!!! $d did not complete -- continuing to the next directory"
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
