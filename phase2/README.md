# Phase 2 — the dev plane (laptop, sm_120)

## Step 1 is answered: the prebuilt wheel works on sm_120

The runbook's instruction was to try the prebuilt wheel before any source build,
because a `no kernel image` failure is a twenty-minute answer rather than an
overnight one. It did not fail.

`vllm/vllm-openai:v0.29.0` — the *same pinned image the desktop runs*, moved
across the direct link with `docker save | docker load` so the digest is
identical (`sha256:c2914767…`) rather than re-resolved from a registry — serves
on the RTX 5090 Laptop GPU with no source build:

```
FlashAttention version 2                       (selected, not fallen back to)
torch.compile took 14.67 s
Available KV cache memory: 5.05 GiB
GPU KV cache size: 47,264 tokens, max concurrency 5.77x at 8k
init engine (profile, create kv cache, warmup) took 55.81 s
Application startup complete.
```

**This was verified by reading the output, not by checking for HTTP 200** — the
runbook is explicit that on SM120 a broken path returns garbage rather than an
error. Three sentences embedded and compared by cosine:

| pair | cosine | expected |
|---|---|---|
| "The cat sat on the mat." / "A feline rested upon the rug." | **0.7262** | high |
| "The cat sat on the mat." / "Quantum chromodynamics describes the strong nuclear force." | **0.1957** | low |

1024-dim, L2-normalised to 1.0000, 1023–1024 non-zero components. The vectors
are real and semantically ordered, so sm_120 kernels genuinely ran.

The toolchain was independently confirmed native beforehand, via `arch_probe.cu`
in the `gpu-lab:sm120` image — `__CUDA_ARCH__ = 1200`, gencode
`compute_120f`, `RESULT: OK - native cubin matches hardware`. No PTX JIT
fallback, which is the confound the whole two-arch premise depends on avoiding.

## What runs here now

`llama-swap` v255 (byte-identical to the desktop's, sha256 verified) on
`:8080`, serving `qwen3-embed` on pinned `:8102`. Managed by
`phase1/lab`, which is now node-aware.

```bash
phase1/lab up --warm     # starts llama-swap, preloads the embedding model
phase1/lab status        # reports which node it is on
```

**Why embeddings moved here.** The coder alone takes 23.2 of the 3090's
24.5 GiB, so on the desktop the two models could not be co-resident and
llama-swap evicted one to serve the other at ~90 s. That is now gone: they are
on different cards. `--gpu-memory-utilization` is 0.30 rather than the desktop's
0.20 because this board drives a display and loses ~1.25 GiB before vLLM starts;
measured steady state is 8.4 GiB of 24463 MiB, leaving ~16 GiB free for
experiments.

## Not done

- **Source build against `compute_120f`, FlashInfer, CUTLASS.** Step 1 succeeding
  does not make this unnecessary — it makes it *elective for serving* and still
  required for kernel work. Nothing here has exercised the NVFP4 MoE path, so the
  `--moe-backend marlin` correction remains untested rather than disproven.
- **llama.cpp with expert offload** for a 120B-class MoE (`--n-cpu-moe` ≈ 21).
- **`pytest tests/kernels` on both nodes.** This is where the differential is
  born and it is the only route to the "one genuine sm_86/sm_120 disagreement,
  identified and understood" exit criterion. **Phase 2 is not complete without
  it.**

## Still unsettled

The runbook's §05 fork — engine development or application development — is not
recorded as answered. The work above was chosen because it pays off either way.
The source build and the kernel differential do not: they serve engine work only.
