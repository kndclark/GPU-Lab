#!/usr/bin/env bash
# Find this node's real training ceiling by escalating load one step at a time.
#
#   ./ramp.sh              run the default ladder
#   ./ramp.sh --steps "2:16 4:16 8:32"   batch:rank pairs, in order
#
# Why a ladder rather than one big run. The question "how far can this card be
# pushed" has two failure modes with different signatures: OOM is immediate and
# harmless, thermal/power limiting is gradual and only visible in the response
# curve. Jumping to maximum load answers neither -- it either OOMs instantly or
# produces a single endpoint number with nothing to compare it against.
#
# Every step is guarded by watchdog.py, which can stop the run; qlora.py cannot
# stop itself. A step that throttles ends the ladder, because every step above
# it is throttled by definition.
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STEPS=${STEPS:-"2:16 4:16 8:16 8:32"}
EPOCHS=${EPOCHS:-0.25}
NAME=${NAME:-qwen3-8b-ramp}
OUT=${OUT:-$here/runs/ramp-$(date -u +%Y%m%dT%H%M%SZ)}

while [ $# -gt 0 ]; do
    case "$1" in
        --steps) STEPS=$2; shift 2 ;;
        --epochs) EPOCHS=$2; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT"
echo "ramp: steps [$STEPS], $EPOCHS epochs each"
echo "out:  $OUT"

for step in $STEPS; do
    batch=${step%%:*}
    rank=${step##*:}
    label="b${batch}r${rank}"
    log="$OUT/$label.log"
    wdlog="$OUT/$label.watchdog.log"

    echo
    echo "=== step $label: batch $batch, rank $rank ==="

    FORCE=1 NAME="$NAME" NO_VERIFY="${NO_VERIFY:-1}" "$here/run.sh" --fg \
        --epochs "$EPOCHS" --batch-size "$batch" --rank "$rank" \
        ${MAXLEN:+--max-len "$MAXLEN"} ${EXTRA:-} \
        > "$log" 2>&1 &
    run_pid=$!

    python3 "$here/watchdog.py" --container gpu-lab-training \
        --wait 600 --interval 5 --report "$OUT/$label.watchdog.json" \
        > "$wdlog" 2>&1 &
    wd_pid=$!

    wait $run_pid
    run_rc=$?
    kill $wd_pid 2>/dev/null
    wait $wd_pid 2>/dev/null

    aborted=$(python3 -c "
import json,sys
try:
    print(json.load(open('$OUT/$label.watchdog.json')).get('aborted', False))
except Exception:
    print('unknown')
" 2>/dev/null)

    if grep -qiE "out of memory|OutOfMemoryError|CUDA out of memory" "$log"; then
        echo "  OOM at batch $batch rank $rank -- ceiling is below this step"
        echo "$label OOM" >> "$OUT/summary.txt"
        break
    fi

    if [ "$aborted" = "True" ]; then
        echo "  WATCHDOG ABORTED this step -- ladder stops here"
        echo "$label ABORTED" >> "$OUT/summary.txt"
        break
    fi

    if [ $run_rc -ne 0 ]; then
        echo "  step failed (rc=$run_rc), see $log"
        echo "$label FAILED rc=$run_rc" >> "$OUT/summary.txt"
        break
    fi

    python3 - "$OUT/$label.watchdog.json" "$label" "$OUT/summary.txt" <<'PY'
import json, sys
path, label, summary = sys.argv[1:4]
samples = json.load(open(path)).get("samples", [])
if not samples:
    sys.exit(0)
temps = [s["temp_c"] for s in samples]
powers = [s["power_w"] for s in samples]
limits = [s["power_limit_w"] for s in samples if s["power_limit_w"] == s["power_limit_w"]]
throttled = sum(1 for s in samples if s["hw_thermal"] or s["sw_thermal"])
line = (f"{label} ok  peak {max(temps):.0f}C  "
        f"power {max(powers):.0f}W of {max(limits):.0f}W enforced  "
        f"throttled {throttled}/{len(samples)} samples")
print("  " + line)
open(summary, "a").write(line + "\n")
PY
done

echo
echo "=== ramp summary ==="
cat "$OUT/summary.txt" 2>/dev/null || echo "(no steps completed)"
