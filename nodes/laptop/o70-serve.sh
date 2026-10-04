#!/bin/bash
# O70: Llama-3.1-70B (GPTQ-INT4) on this card alone, part of its layers in
# pinned RAM. llama-swap starts it (llama-swap.yaml, llama-70b-onecard) and
# stops it with docker stop; the settings are O70's measured ones (moe-qlora
# docs/next-model-plan.md): eager is mandatory, CUDA graphs do not fit.
#
# A wrapper rather than a bare docker run because llama-swap runs cmd as a
# plain command, not through a shell, and this model needs two checks first:
#
#   RAM   with the model up, O70 measured 32.3 GiB used and 30.4 GiB available
#         of 61 (26.1 GiB shared, the pinned layers). From ~4 GiB used idle
#         the load takes ~28 GiB, so 35 GiB available leaves ~7. Below that
#         the load drives the laptop into swap instead of failing cleanly.
#   VRAM  vLLM needs 0.92 of the board free at start. llama-swap unloads
#         qwen3-embed first (one model at a time), but nothing unloads the
#         pool's stage or a stray container; name the clients instead of
#         leaving vLLM's memory error as the only clue.
set -u
name=${1:?usage: o70-serve.sh <container name>}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-35}

avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
if [ "$avail" -lt "$MIN_AVAIL_GIB" ]; then
    echo "o70: ${avail} GiB RAM available, needs ${MIN_AVAIL_GIB}" >&2
    exit 1
fi

read -r free total < <(nvidia-smi --query-gpu=memory.free,memory.total \
    --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
need=$(( total * 92 / 100 ))
if [ "$free" -lt "$need" ]; then
    echo "o70: ${free} MiB free on the card, vLLM needs ${need}. GPU clients:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader >&2
    exit 1
fi

exec docker run --name "$name" --init --rm --gpus all --ipc=host \
    -v /srv/model-cache:/hf:ro -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
    -p 8103:8000 \
    vllm/vllm-openai:v0.29.0 \
    --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
    --revision 1b0ae7f9d6da8b79f36fdc24912f950ecb2b6e91 \
    --served-model-name llama-70b-onecard \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --max-num-batched-tokens 512 \
    --no-enable-flashinfer-autotune \
    --enforce-eager \
    --cpu-offload-gb 21.5
