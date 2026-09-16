#!/usr/bin/env bash
# Build the kernel differential harness image.
# Usage: ./build.sh [vllm-tag]        (default: the tag the lab serves, v0.29.0)
#
# The tests come from the vLLM source tree, which is not vendored here: 24 MB of
# upstream test code has no business in this repo, and an immutable tag is a
# better pin than a copy anyone could edit. So the build context is assembled in
# a temp dir at build time and thrown away afterwards.
#
# The image is NOT built per node. Build it once and move it across the direct
# link with `docker save | docker load`, exactly as the serving image was moved
# in Phase 2 step 1, so both nodes run a bit-identical harness:
#
#     sudo docker save gpu-lab:kernels | ssh llm 'sudo docker load'
#
# Rebuilding independently on each node would almost certainly produce the same
# thing, and "almost certainly" is not the standard a differential is held to.
set -euo pipefail

TAG="${1:-v0.29.0}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# The image must match the wheel it tests. The tests in vLLM's tree track the
# kernels in the same commit, so a mismatched tag produces failures that look
# like architecture findings and are not.
installed=$(sudo docker run --rm --entrypoint python3 "vllm/vllm-openai:${TAG}" \
                -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)
expected="${TAG#v}"
if [ "$installed" != "$expected" ]; then
    echo "refusing to build: vllm/vllm-openai:${TAG} reports version '${installed}', expected '${expected}'" >&2
    exit 1
fi
echo "base image vllm/vllm-openai:${TAG} carries vllm ${installed}"

ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT

echo "fetching vLLM tests at ${TAG}"
git clone --quiet --depth 1 --branch "$TAG" \
    https://github.com/vllm-project/vllm.git "$ctx/src"
# Only these two. Copying the repo wholesale would bring vllm/ with it and
# shadow the installed wheel -- see the Dockerfile's COPY comment.
cp -r "$ctx/src/tests" "$ctx/tests"
cp "$ctx/src/pyproject.toml" "$ctx/pyproject.toml"
rm -rf "$ctx/src"

cp "$here/Dockerfile" "$ctx/Dockerfile"

echo "building gpu-lab:kernels"
sudo docker build -t gpu-lab:kernels "$ctx"

echo
echo "verifying the wheel is not shadowed by source"
sudo docker run --rm --entrypoint python3 gpu-lab:kernels \
    -c 'import vllm, torch; print("vllm   :", vllm.__file__); print("version:", vllm.__version__); print("torch  :", torch.__version__)'
