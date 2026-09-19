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
0.55 ms RTT. See [host/](host/).

## Contents

Organised by what runs where, not by the phase that introduced it. Phases are a
plan axis; they stopped describing the tree the moment a second node appeared and
Phase 2's endpoint had to be configured in Phase 1's files.

    bin/lab           control the serving plane on either node; it is node-aware
    bin/check.py      pre-push validation -- a config must be able to serve before it deploys
    arch/             one image recipe, two architecture targets, and the runtime arch proof
    nodes/            per-node llama-swap config + systemd unit (desktop sm_86, laptop sm_120)
    serving/          LiteLLM, the single front door across both nodes
    monitoring/       Prometheus, Grafana, and the compose stack behind them
    bench/            the benchmark and its recorded baseline
    kernels/          the cross-arch kernel differential harness and its C4 repros
    kernels/sm-differ the Phase 3 regression differ (Rust); `cargo test` runs in the push gate
    training/         the QLoRA pipeline (runs on the desktop, sm_86)
    host/             host configuration that is not reproducible from code alone
    docs/             what each phase delivered, and the upstream candidate ledger

## Build

    arch/build.sh          # auto-detect from nvidia-smi compute_cap
    arch/build.sh sm86     # explicit
    arch/build.sh sm120

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
direct cable. `ssh llm` is the direct-link alias (see [host/](host/)).

To confirm both nodes agree before a measurement run:

    git rev-parse --short HEAD
    ssh llm 'cd /home/david/gpu-lab && git rev-parse --short HEAD'

## Status

**Phases 0, 1, 2 and 2b are complete** — which is the runbook's own finish line.
It is explicit that Phase 2, not Phase 3, is where the setup is finished; Phase 3
is a project you run *on* the setup and adds no capability the lab does not
already have.

Two things the build taught that the plan did not know. The source build against
`compute_120f` was never needed: the prebuilt wheel serves on sm_120, and the
kernel differential runs on it too. A source build is required to *modify*
kernels, not to *differentiate* them. And QLoRA does not inject facts — the
pipeline works, both adapters confabulated, and the right tool for "the model
should know my lab" is retrieval.

The runbook's §05 fork was settled on 2026-09-17: **engine development**. It
costs no application capability, because Phase 1 already delivered that and it
keeps running.

Remaining work is elective. The live work list is
[docs/contributions.md](docs/contributions.md) — eight upstream candidates, the
evidence each has, and the step still missing before it could be filed. Nothing
has been filed. Phase 2's one unbuilt item is the llama.cpp 120B endpoint, which
is also the only way to test the `--moe-backend marlin` and `--n-cpu-moe` claims.

Both nodes hold their driver metapackages (`apt-mark showhold`) so an `apt
upgrade` cannot move 595.91.07 out from under a measurement.

Pushing deploys. Enable the pre-push check once per clone:

    git config core.hooksPath .githooks
