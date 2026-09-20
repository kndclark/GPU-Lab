#!/usr/bin/env python3
"""Find out what CPU offload actually does before committing to a 70B download.

transformers refuses a CPU/disk device_map for a 4-bit model unless
llm_int8_enable_fp32_cpu_offload is set, and the error text says the offloaded
modules are kept "in 32-bit". If that is literally true, a 70B is impossible on
this machine: the CPU share would be roughly 30B params at 4 bytes = 120 GB
against 64 GB of RAM. If offloaded weights instead stay quantised or bf16, the
runbook's "QLoRA on 70B, single node" is reachable.

The question is answered by loading a model we already have with the GPU budget
deliberately starved, then reading back where each parameter landed and in what
dtype. No 140 GB download required to find out.
"""
import argparse
import collections

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--gpu-gib", type=float, default=8.0,
                    help="starve the GPU on purpose so layers must spill to CPU")
    ap.add_argument("--cpu-gib", type=float, default=40.0)
    ap.add_argument("--explicit-map", action="store_true",
                    help="build a layer-by-layer device_map instead of 'auto'")
    ap.add_argument("--gpu-layers", type=int, default=20)
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    if args.explicit_map:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        n = cfg.num_hidden_layers
        dm = {"model.embed_tokens": 0, "model.norm": 0, "model.rotary_emb": 0, "lm_head": 0}
        for i in range(n):
            dm[f"model.layers.{i}"] = 0 if i < args.gpu_layers else "cpu"
        print(f"explicit map: {args.gpu_layers} of {n} layers on GPU, rest on CPU", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, dtype=torch.bfloat16, device_map=dm,
        )
    else:
        max_memory = {0: f"{args.gpu_gib}GiB", "cpu": f"{args.cpu_gib}GiB"}
        print(f"loading {args.model} with max_memory={max_memory}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory,
        )

    # bytes and dtypes by device: the whole point of the probe
    stats = collections.defaultdict(lambda: collections.Counter())
    bytes_by_dev = collections.Counter()
    for _, p in model.named_parameters():
        dev = str(p.device)
        stats[dev][str(p.dtype)] += p.numel()
        bytes_by_dev[dev] += p.numel() * p.element_size()

    print("\n=== where parameters landed ===", flush=True)
    for dev in sorted(stats):
        print(f"  {dev}: {bytes_by_dev[dev]/1024**3:.2f} GiB", flush=True)
        for dt, n in stats[dev].most_common():
            print(f"      {dt:<16} {n/1e9:.3f}B params", flush=True)

    quantised = sum(n for dev in stats for dt, n in stats[dev].items()
                    if "uint8" in dt or "int8" in dt)
    print(f"\n  params in a quantised dtype: {quantised/1e9:.3f}B", flush=True)

    cpu_bytes = bytes_by_dev.get("cpu", 0)
    if cpu_bytes:
        per_param = cpu_bytes / max(sum(stats["cpu"].values()), 1)
        print(f"  CPU-resident bytes per parameter: {per_param:.2f}", flush=True)
        print(f"  -> a 70B with this split would need "
              f"{70e9 * per_param / 1024**3:.0f} GiB of RAM if fully offloaded", flush=True)
    else:
        print("  nothing landed on the CPU -- raise --gpu-gib starvation", flush=True)

    print(f"\n  GPU allocated now: {torch.cuda.memory_allocated()/1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
