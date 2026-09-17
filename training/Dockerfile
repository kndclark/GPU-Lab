# syntax=docker/dockerfile:1.7
#
# The QLoRA training container  (sm_86/sm_120 runbook, Phase 2b).
#
# Built FROM the pinned serving image rather than from a PyTorch base, for the
# same reason kernels/ is: the adapter this produces is loaded back into
# vllm/vllm-openai:v0.29.0 for inference, and an adapter trained against one
# torch/transformers build and served by another is a version-skew bug waiting
# to be misread as a training result. One torch (2.13.0+cu130), one
# transformers (5.16.1), on both sides of the adapter.
#
# The image already carries torch, transformers and accelerate. The four
# packages added here are the training stack and nothing else.
FROM vllm/vllm-openai:v0.29.0

# The constraint file is what keeps the resolver honest. datasets and trl pull
# real dependency trees (pyarrow, pandas, ...), so --no-deps would leave them
# unimportable -- but an unconstrained install is free to move torch. Pinning
# every package the serving image already has, to the version it has, lets pip
# add what is missing and forbids it from touching what is shared.
#
# bitsandbytes 0.50.2 was verified on the 3090 inside this image before this
# file existed: NF4 Linear4bit matmul ran on cc 8.6 against torch cu130 with
# relative error 0.09 -- so the CUDA 13 binary in the manylinux wheel is real.
RUN pip freeze | grep -E '^(torch|torchvision|torchaudio|transformers|accelerate|numpy|triton|safetensors|tokenizers|huggingface-hub)==' \
        > /tmp/constraints.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt \
        peft==0.21.0 \
        bitsandbytes==0.50.2 \
        trl==1.13.0 \
        datasets==5.0.1 \
    && python3 -c "import torch, transformers; assert torch.__version__=='2.13.0+cu130', torch.__version__; assert transformers.__version__=='5.16.1', transformers.__version__; import peft, bitsandbytes, trl, datasets; print('training stack ok:', peft.__version__, bitsandbytes.__version__, trl.__version__, datasets.__version__)"

WORKDIR /training
ENV HF_HOME=/hf \
    TOKENIZERS_PARALLELISM=false

# No ENTRYPOINT: the run script chooses between train.py and a serving check.
