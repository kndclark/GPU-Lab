#!/usr/bin/env bash
# Run the kernel differential on whichever node this is.
# Usage: ./run.sh [pytest-target ...]      (default: tests/kernels/quantization)
#
# Writes kernels/results/<arch>-<suite>.xml plus a .env.json recording the
# conditions it ran under, and a .log of the console output. compare.py diffs
# two of those XMLs.
#
# Why quantization is the default target: it is where the two cards are least
# alike. The 3090 is CC 8.6 and has no FP8 compute at all (that needs >= 8.9),
# while sm_120's native CUTLASS NVFP4 path is the one the runbook flags as
# producing garbage rather than falling back. A disagreement here is a finding;
# a disagreement in tests/kernels/core would more likely be a harness bug.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
results="$here/results"
mkdir -p "$results"

cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
case "$cap" in
    86)  arch=sm_86  ;;
    120) arch=sm_120 ;;
    *)   echo "unrecognised compute cap '$cap' -- this harness knows only the lab's two cards" >&2; exit 1 ;;
esac

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(tests/kernels/quantization)

# Name the output after the targets, so two runs on one node do not overwrite
# each other and the filename says what was actually run. Override with SUITE=
# when the target list is long -- the name has to survive being a filename, and
# both nodes must use the same one or compare.py has nothing to pair up.
suite="${SUITE:-$(printf '%s\n' "${targets[@]}" | sed 's#^tests/kernels/##; s#/#-#g; s#\.py$##' | paste -sd+ -)}"
stem="${arch}-${suite}"

# A loaded model owns most of the card, and these tests allocate. The failure
# mode is not a clean OOM either: individual tests fail on allocation and read
# as architecture findings. Worse on the desktop, where llama-swap loads on
# demand -- one inbound request can take 23.2 GiB out from under the suite.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 2000 ]; then
    echo "refusing to run: GPU already holds ${used} MiB." >&2
    echo "  a model is loaded; kernel tests would fail on allocation and look like findings." >&2
    echo "  stop the serving plane first:  bin/lab down" >&2
    exit 1
fi
echo "GPU idle at ${used} MiB"
echo "note: on the desktop an inbound request can still make llama-swap load a"
echo "      model mid-run -- 'lab down' first if this run is going in the results."

image_id=$(sudo docker images --no-trunc --format '{{.ID}}' gpu-lab:kernels | head -1)
[ -n "$image_id" ] || { echo "gpu-lab:kernels not present -- run kernels/build.sh" >&2; exit 1; }

# Provenance beside the results. A cross-arch diff is only readable if you can
# show the two runs differed in architecture and in nothing else.
cat > "$results/${stem}.env.json" <<JSON
{
  "arch": "${arch}",
  "compute_cap": "${cap}",
  "node": "$(hostname)",
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
  "driver": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)",
  "image_id": "${image_id}",
  "targets": "${targets[*]}",
  "started_utc": "$(date -u +%FT%TZ)"
}
JSON

echo "running ${targets[*]} on ${arch}"
# --name so 'lab down' can sweep it: a killed run otherwise leaves a container
# holding the card, which silently blocks the next model load.
# The results directory is mounted so the XML survives --rm.
# -p no:cacheprovider keeps pytest from writing .pytest_cache into the mount.
sudo docker run --rm --gpus all --shm-size=4g \
    --name gpu-lab-kernels \
    -v "$results:/out" \
    gpu-lab:kernels \
    "${targets[@]}" \
    -q --no-header -p no:cacheprovider \
    --timeout=600 \
    --junitxml="/out/${stem}.xml" \
    2>&1 | tee "$results/${stem}.log" || true

# The container runs as root; hand the artifacts back so they are editable and
# committable without sudo.
sudo chown "$(id -u):$(id -g)" "$results/${stem}.xml" "$results/${stem}.log" 2>/dev/null || true

if [ ! -s "$results/${stem}.xml" ]; then
    echo "FAILED: no results XML was produced -- treat this run as void, not as a pass" >&2
    exit 1
fi

# Store compressed. A full attention run is 22 MB of XML and 0.3 MB gzipped,
# and pushing here deploys, so uncompressed artefacts would bloat every clone
# of this repo forever. compare.py reads .gz directly.
gzip -f "$results/${stem}.xml"
echo
echo "wrote kernels/results/${stem}.xml.gz"
