#!/bin/bash
# Prefill-width sweep on the pool, as bench/baseline-pool-llama31-70b-prefill-batch.json
# was run by hand: one full `lab pool up` per --max-num-batched-tokens value (B), one
# throwaway request (the first after pool up is slow), then bench/prefill_curve.py.
# The KV lines come from the head's serve log. Each B ends with `lab pool down`, which
# restores what pool up stopped. Run from the laptop; bin/lab is the deployed copy.
# usage: bench/prefill_sweep.sh OUTDIR B [B...]
set -u
out=$1; shift
mkdir -p "$out"
LAB=$HOME/gpu-lab/bin/lab
here=$(cd "$(dirname "$0")/.." && pwd)
for B in "$@"; do
  echo "== B=$B $(date +%T)"
  if ! POOL_MAX_BATCHED_TOKENS=$B "$LAB" pool up > "$out/pool-up-$B.log" 2>&1; then
    echo "  pool up FAILED"; tail -5 "$out/pool-up-$B.log"
    "$LAB" pool down > "$out/pool-down-$B.log" 2>&1; exit 1
  fi
  ssh llm "sudo docker exec ray-head grep -E 'KV cache|Maximum concurrency' /tmp/vllm-pool.log" \
    > "$out/kv-$B.log" 2>&1
  python3 "$here/bench/prefill_curve.py" --samples 1 --counts 975 > /dev/null 2>&1
  python3 "$here/bench/prefill_curve.py" --label "pool Qwen3-14B, --max-num-batched-tokens $B" \
    --out "$out/prefill-$B.json" > "$out/prefill-$B.log" 2>&1
  echo "  curve exit $?"
  "$LAB" pool down > "$out/pool-down-$B.log" 2>&1
  echo "  pool down exit $?"
done
