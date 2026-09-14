# Phase 1 — serving plane

The desktop (sm_86, RTX 3090) is the always-on, instrumented node. One base URL
answers for every model on either machine.

## Endpoints

| Service | Address | Notes |
|---|---|---|
| **LiteLLM** (use this) | `http://10.10.0.1:4000/v1` | the single front door; needs the master key |
| llama-swap | `http://10.10.0.1:8080/v1` | desktop-local model router, no auth |
| vLLM coder | `127.0.0.1:8101` | pinned so Prometheus has a stable target |
| vLLM embed | `127.0.0.1:8102` | pinned, same reason |
| Prometheus | `http://10.10.0.1:9090` | 90d retention |
| Grafana | `http://10.10.0.1:3000` | admin/admin, dashboard "GPU Lab" |

Master key: `/etc/gpu-lab/litellm.env` on the desktop, mode 0600, not in git.

    sudo grep -oP 'LITELLM_MASTER_KEY=\K.*' /etc/gpu-lab/litellm.env

## Models

| Name | What | Footprint |
|---|---|---|
| `qwen3-coder` | Qwen3-Coder-30B-A3B-Instruct AWQ 4-bit, 32k ctx | 23.2 / 24.5 GiB, KV pool 36,032 tokens |
| `qwen3-embed` | Qwen3-Embedding-0.6B, 1024-dim, 8k ctx | ~5 GiB |

They **cannot be co-resident** — the coder alone takes 23.2 GiB — so llama-swap
evicts one to start the other, costing a ~90 s reload. Shrinking the coder to fit
both is the wrong fix: it would cut KV from 3.3 GiB to ~1.1 GiB and cap context
near 12k. If simultaneous serving matters, put the embedding model on the laptop
and add it to `litellm-config.yaml`.

## Baseline

Recorded 2026-09-13, 50 samples, `bench/baseline-qwen3-coder-32k.json`:

    TTFT         p50  0.016 s   p95  0.031 s
    decode rate  p50  175.9 tok/s   cv 0.5%
    thermals     40 -> 70 C, 321 W, no throttle

Reproduce with `python3 bench/bench.py --repeats 10`. **Do not edit the prompt
set in `bench.py`** without resetting the baseline — comparing across different
prompts compares nothing.

Two facts worth keeping: decode drifted 177.6 -> 175.9 tok/s as the card went
58 -> 70 C, which is why single samples are banned here. And "maximum concurrency
1.10x" in the vLLM log is `36,032 / max_model_len` — an arithmetic restatement
assuming every request fills the whole window, not a measurement. Real concurrency
at typical request sizes is far higher: measured 2 concurrent streams with zero
preemptions and aggregate throughput of 305 tok/s.

## Corrections to the runbook found here

- `vllm:time_per_output_token_seconds` **no longer exists** in v0.29. Use
  `vllm:inter_token_latency_seconds` (per token) or
  `vllm:request_time_per_output_token_seconds` (per request).
- `gpu_cache_usage_perc` -> `kv_cache_usage_perc` was right, but the advice to
  "scrape both" is now wrong: the old name is fully removed, not deprecated.
- llama-swap's `/metrics` is **system and GPU stats only**; it does not proxy the
  upstream. vLLM engine metrics must be scraped from vLLM directly, which is why
  the ports here are pinned rather than using `${PORT}`.
- vLLM sizes a KV cache from `max_position_embeddings` even under
  `--runner pooling`, so an embedding model needs an explicit `--max-model-len`
  or it tries to reserve 3.5 GiB and dies.

## Operating it

Use `bin/lab` on the desktop. From the laptop, prefix with `ssh llm`.

    lab up                 start everything
    lab up --warm          ...and preload the coder (~90s), so the first real
                           request is not the one that pays for the cold load
    lab down               stop everything; services still return on reboot
    lab down --boot-off    ...and disable autostart
    lab status             what is running, and whether the GPU is actually free
    lab logs model         follow the loaded model's log
    lab logs llama-swap    follow the router (also: litellm|prometheus|grafana)

Order is not arbitrary. Coming up, llama-swap starts before LiteLLM so the front
door never advertises an upstream that cannot answer. Going down, LiteLLM stops
first so nothing new arrives while models are being evicted.

`down` verifies the GPU actually drops below 1 GiB rather than assuming it. It
also force-stops orphaned `qwen3-*` containers: llama-swap normally runs each
model's `cmdStop` on shutdown, but if it ever dies uncleanly the upstream outlives
it and keeps the GPU pinned, which silently blocks the next start with an
out-of-memory failure that looks like a config problem.

**After a reboot everything comes back on its own** -- llama-swap is a systemd
unit and the compose services use `restart: unless-stopped`. A manual `lab down`
is respected across a Docker daemon restart, but not across a machine reboot;
use `--boot-off` for that.

Editing config:

    # after editing llama-swap.yaml
    sudo systemctl restart llama-swap
    # after editing litellm-config.yaml
    bin/lab reload litellm
    # after editing prometheus.yml
    bin/lab reload prometheus

Remember both nodes track git: edit on the laptop, `git push`, and the desktop
checks out automatically. Editing directly on the desktop will be overwritten by
the next push.

### Rule: every new component goes into `lab`

**When you add a service to the lab, wire it into `bin/lab` in the same change**
-- `up`, `down`, and `status`. Never leave starting or stopping it as a step
someone has to remember.

A control script that covers only some of what is running is worse than no script,
because it creates false confidence that `lab down` released the machine. That
matters more than usual here: an orphaned vLLM upstream keeps ~23 GiB pinned, and
the *next* start then fails with an out-of-memory error that reads like a config
bug rather than leftover state.

Checklist for a new component:

- `cmd_up` starts it in dependency order -- routers before front doors, so the
  front door never advertises an upstream that cannot answer.
- `cmd_down` stops it in reverse -- front doors first, so nothing new arrives
  while models are being evicted.
- If it can hold GPU memory, add it to the orphan sweep in `cmd_down`
  (currently `grep -E '^qwen3-'`), and keep the check that the GPU actually drops
  below 1 GiB rather than assuming it did.
- `cmd_status` reports its health; `cmd_logs` can follow it.
- Decide explicitly whether it should survive a reboot (systemd unit, or
  `restart: unless-stopped`) and make sure `--boot-off` covers it if not.

### The laptop shares this script

The repo is checked out on both nodes, so the laptop gets a **node-aware `lab`**,
not a second script -- two files would drift. Add node detection (hostname, or
`nvidia-smi --query-gpu=compute_cap`) and a `--node desktop|laptop|all` selector.

Two things make the laptop genuinely different, and the script has to respect both:

- **It is not always-on.** It is the interactive machine, so `up` there should not
  assume boot-time autostart the way the desktop does.
- **The nodes must stay independently controllable.** Phase 2b runs training on the
  desktop while the laptop keeps serving through the LiteLLM front door. A script
  that could only drive both at once would break that failover.
