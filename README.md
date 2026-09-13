# GPU-Lab

A two-node heterogeneous CUDA lab: **sm_86** (RTX 3090) and **sm_120** (RTX 5090 Laptop).

The point is not raw throughput. It is that two GPU architectures, one generation
and one power class apart, can be made identical in *every respect except the
architecture* — so that any measured difference between them is attributable.

Design doc and build plan (published Artifact, versioned separately):
<https://claude.ai/code/artifact/ef6ac6fe-d2c7-4b18-9a31-7342e0473826>

## Nodes

| | Desktop | Laptop |
|---|---|---|
| Host | `davids-llm-server` | `david-Legion-Pro-7-16IAX10H` |
| GPU | RTX 3090, sm_86, 24576 MiB | RTX 5090 Laptop, sm_120, 24463 MiB |
| Power cap | 400 W | 175 W |
| Host RAM | 31.9 GB | 63.4 GB |
| Role | always-on serving + training | source builds + the sm_120 experiment |

Both: Ubuntu 26.04.1, kernel 7.0.0-31-generic, driver 595.91.07, CUDA 13.2,
docker-ce 29.8.0, nvidia-container-toolkit 1.20.0-1.

Connected by a direct 2.5GbE cable on a private /30 (`10.10.0.1` / `10.10.0.2`),
0.55 ms RTT. See [phase0/](phase0/).

## Contents

    Dockerfile        one image recipe, two architecture targets
    arch_probe.cu     runtime proof that a container's kernels match its host GPU
    build.sh          builds gpu-lab:sm86 or gpu-lab:sm120 (auto-detects by compute cap)
    phase0/           the host configuration that is not reproducible from code alone

## Build

    ./build.sh          # auto-detect from nvidia-smi compute_cap
    ./build.sh sm86     # explicit
    ./build.sh sm120

## The one thing to understand before changing the build

`nvcc -arch=sm_XX` emits **both** SASS and PTX. An image built for the wrong
architecture therefore does not fail — the driver JIT-compiles the PTX and runs it.
You get correct output at a speed that reflects JIT overhead and a generically
derived kernel, and no error anywhere.

For a project whose entire purpose is comparing two architectures, that is the
worst possible failure: stable, repeatable, plausible, wrong. Repeat sampling and
variance discipline cannot detect it, because the numbers are not noisy.

So the gencode is always stated explicitly:

    sm86    -gencode arch=compute_86,code=sm_86
            cubin only. Running this on non-sm_86 hardware MUST fail with
            cudaErrorNoKernelImageForDevice.

    sm120   -gencode arch=compute_120f,code=sm_120f
            -gencode arch=compute_120f,code=compute_120f
            cubin + PTX, because TORCH_CUDA_ARCH_LIST=12.0+PTX asks for it.
            The PTX here is deliberate. It still will not run on sm_86 —
            PTX only JITs forward.

`arch_probe` enforces this at runtime and exits non-zero on a mismatch:

    0   native cubin matches the hardware
    2   no kernel image for this device (wrong-arch image — correct behaviour)
    3   kernel arch != hardware arch

Verify both directions after any build change. Both images passing on their own
hardware is *also* consistent with both being mistargeted-but-JIT-capable.

## Repo layout across the two nodes

The laptop is where you work. `git push` reaches **both** GitHub and the desktop:

    origin  git@github.com:kndclark/GPU-Lab.git   (fetch)
    origin  git@github.com:kndclark/GPU-Lab.git   (push)
    origin  llm:/home/david/gpu-lab.git           (push)

The desktop's repo is `/home/david/gpu-lab.git` with its work tree at
`/home/david/gpu-lab`. A `post-receive` hook checks out `main` on every push, so
the desktop cannot sit on a stale commit. That is deliberate: a stale working copy
builds the wrong image, and a wrong image yields plausible numbers rather than an
error.

The desktop also has its own key on GitHub and tracks `origin/main`, so it can
recover on its own if the laptop is unavailable:

    ssh llm 'cd /home/david/gpu-lab && git pull'

That is the fallback, not the normal path. Pushing from the laptop stays primary
because it is the only route that forces a checkout — a `pull` the desktop never
runs leaves it silently stale. It also works with the internet down, over the
direct cable. `ssh llm` is the direct-link alias (see [phase0/](phase0/)).

To confirm both nodes agree before a measurement run:

    git rev-parse --short HEAD
    ssh llm 'cd /home/david/gpu-lab && git rev-parse --short HEAD'

## Status

Phase 0 complete. Next: Phase 1 — vLLM behind llama-swap on the 3090,
Prometheus/Grafana on `/metrics`, LiteLLM as a single front door across both nodes.
