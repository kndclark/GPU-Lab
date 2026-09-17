# Handoff — resuming after Phase 2b

Originally written 2026-09-13 at the end of the session that completed Phase 1.
**Updated 2026-09-17**, when Phase 2 and Phase 2b both completed. This document
exists so a new session can pick up without re-deriving anything.

**Read these three first, in this order:**

1. `docs/contributions.md` — every upstream candidate, the evidence each has,
   and the step still missing before it could be filed. Nothing has been filed.
2. `docs/phase2-dev-plane.md` — the kernel differential and what it found.
3. The lab report artifact, which is the narrative version of both:
   https://claude.ai/artifact/PbAw4JBv8mPiGpRcBUvnj9

**The one decision still open** is the runbook's §05 fork — engine development
or application development. It is recorded nowhere as settled, and it decides
whether Phase 3 is worth doing at all. Ask; do not assume.

**Source of truth for the plan is the runbook, not this file:**
https://claude.ai/code/artifact/ef6ac6fe-d2c7-4b18-9a31-7342e0473826
(§06 holds the phases; §05 the end state; §08 the elective training pool.)
This file carries the *state*, the *traps* and the *settled decisions* — the
things the runbook does not know because they happened after it was written.

---

## Start here (first five minutes)

Run this before anything else. It answers "is the lab alive" in one shot:

```bash
ssh llm '/home/david/gpu-lab/bin/lab status'
```

Expect: four endpoints OK, GPU near 33 MiB idle, and a `direct link` section
showing `enp5s0 10.10.0.1 2500Mb/s` with the peer reachable at ~0.4 ms.

**If `ssh llm` times out**, do not start diagnosing the desktop. The overwhelmingly
likely cause is the laptop's NIC (see Traps, first entry). Check in this order:

```bash
ping -c3 10.10.0.1            # direct link
ping -c3 192.168.0.190        # desktop over Wi-Fi — if this answers, the desktop is fine
ssh llm-wifi 'hostname'       # the fallback route into the desktop
sudo /usr/local/sbin/gpu-lab-igc-resume-repair    # fixes the laptop NIC
```

If the lab is down rather than unreachable: `ssh llm '/home/david/gpu-lab/bin/lab up'`
(add `--warm` to preload the coder, ~90 s).

---

## Where things stand

| Phase | State | Evidence |
|---|---|---|
| Phase 0 — Foundation | **Complete** 2026-09-13 | all four exit criteria verified |
| Phase 1 — Serving plane | **Complete** 2026-09-13 | all four exit criteria verified; baseline recorded |
| Phase 2 — Dev plane | **Complete** 2026-09-16 | 195 files on both nodes; five findings understood |
| Phase 2b — Adapter training | **Complete** 2026-09-17 | all three exit criteria met |
| Phase 3 — Differential harness | Elective. The runbook is explicit that the finish line is Phase 2. | |

**This is the runbook's stated finish line.** Everything remaining is elective
and should be chosen deliberately rather than continued by momentum.

### What changed on 2026-09-17

- **`lab down --gpu-only`** stops llama-swap and anything holding the card and
  leaves LiteLLM, Prometheus and Grafana running. They request no GPU, so
  stopping them frees nothing and costs the front door — which is what made the
  runbook's "the desktop trains, the laptop serves" promise impossible before.
  `training/run.sh` uses it. Verified mid-run: desktop GPU at 33 MiB with the
  trainer resident, embeddings answered through `10.10.0.1:4000` by the laptop.
- **`training/`** holds the QLoRA pipeline: `preflight.py` (runs *before*
  anything is stopped), `qlora.py`, `verify.py`, `dataset.py`, `run.sh`.
  Two adapters exist at `/srv/model-cache/adapters/`.
- **`docs/contributions.md`** tracks eight upstream candidates, C1–C8.
- Grounding that doc in the result XML produced three corrections to
  `phase2-dev-plane.md`: the INT8 failure is two distinct causes, not one; the
  MLA file is 43 kernel errors plus 6 OOMs, not 49; and `test_nvfp4_qutlass`
  fails 132/132 on sm_120 and appears in no earlier write-up.

### The Phase 2b result worth knowing before repeating it

QLoRA **does not inject facts**, and no amount of tuning will change that.
Run 1 trained on lab facts alone: loss 3.996 → 0.400 in 36 steps, textbook
catastrophic forgetting, one probe answered with 160 consecutive `1`s. Run 2
mixed general instruction data 2:1 and halved the LR: forgetting gone, fluency
preserved, loss 3.490 → 1.653 — **and it still invented an "RTX 3050 Laptop
GPU" and put the front door at "12345 Main Street."** LoRA teaches style,
format and behaviour. For "the model should know my lab," the right tool is
retrieval, and `qwen3-embed` is already running for it.

**Live right now:** the front door is `http://10.10.0.1:4000/v1` (LiteLLM).
Behind it, on the desktop: llama-swap on `:8080`, vLLM on pinned `:8101`
(coder) and `:8102` (embed), Prometheus `:9090` (90 d retention), Grafana
`:3000` (folder "GPU Lab"; the password is NOT admin/admin — the compose env
only seeds a fresh DB). Master key lives in
`/etc/gpu-lab/litellm.env` on the desktop, mode 0600, never in git.

Pinned versions: vLLM v0.29.0, llama-swap v255, Prometheus v3.14.0,
Grafana 13.2.1, LiteLLM v1.100.1.

**Models:** `qwen3-coder` (Qwen3-Coder-30B-A3B-Instruct AWQ 4-bit, 32k ctx) and
`qwen3-embed` (Qwen3-Embedding-0.6B, 1024-dim, 8k ctx).

**Recorded baseline** (50 samples, the reference point for every later claim):
TTFT p50 0.016 s / p95 0.031 s, decode 175.9 tok/s p50, cv 0.5%, 40→70 °C.
Stored at `bench/baseline-qwen3-coder-32k.json`; reproduce with
`python3 bench/bench.py --repeats 10`.

---

## The two nodes

| | Desktop — `davids-llm-server` | Laptop — `david-Legion-Pro-7-16IAX10H` |
|---|---|---|
| GPU | RTX 3090, **sm_86**, 24576 MiB, 400 W | RTX 5090 Laptop (GB203M), **sm_120**, 24463 MiB, 175 W |
| CPU / RAM | i9-10850K, 31.9 GB | Ultra 9 275HX, **63.4 GB** |
| Role | always-on server, headless | primary interactive; **Phase 2 happens here** |
| Direct link | `enp5s0` = 10.10.0.1/30 | `enp129s0` = 10.10.0.2/30 |
| Wi-Fi | 192.168.0.190 | 192.168.0.93 |
| Driver | 595.91.07, headless precompiled (`no-dkms`), no MOK | 595.91.07, **DKMS** `-open`, MOK enrolled, Secure Boot ON |
| SSH from laptop | `ssh llm` (direct) / `ssh llm-wifi` (fallback) | — |

The direct 2.5GbE cable measures **0.4–0.55 ms**; Wi-Fi between the same two
machines measures 34–187 ms with wild jitter. Always use the 10.10.0.x path for
inter-node work. `/srv/model-cache` is NFSv4.2 from the desktop, exported only
to `10.10.0.2/32`, measured 294 MB/s read.

**The laptop has the RAM, which is why Phase 2 is laptop work** — a source build
of vLLM plus FlashInfer and CUTLASS does not fit comfortably in the desktop's
31.9 GB.

---

## What Phase 2 is

*Goal (runbook §06): "The laptop becomes the experiment, and the first real
sm_86/sm_120 differential appears." Estimated weeks 2–3.*

1. **Try the prebuilt wheel first.** Wheels cover CC 7.5+ on CUDA 12.9 and
   SM120 support has landed upstream. If it serves, most of Phase 2 is saved;
   if it throws `no kernel image`, that is a twenty-minute answer instead of an
   overnight build. **Do this before any source build.**
2. **Then build from source** against `compute_120f`, plus FlashInfer and
   CUTLASS.
3. **llama.cpp with expert offload** for a 120B-class MoE as the slow-thinking
   endpoint — also on the laptop, for the same RAM reason.
4. **`pytest tests/kernels` on both machines.** This is where the differential
   is born; every disagreement between sm_86 and sm_120 is a lead.

**The runbook's corrections to its own earlier advice, which still apply:**

- **`compute_120f`, not `sm_120a`.** The community fix for the SM120 FP4 path
  landed on `120f`; NVIDIA prefers the family target for forward-compat across
  sm_120/sm_121. Use `a` only for a strictly sm_120-only feature.
- **Serve with `--moe-backend marlin`.** The native CUTLASS NVFP4 MoE path on
  SM120 does not merely fall back — it produces garbage or crashes
  (cutlass#3096, flashinfer#2723). Marlin is the correct path, not a
  consolation prize. **Read the generated text; do not just check for HTTP 200.**
- **Cap `--n-cpu-moe` around 21.** Putting all experts on CPU loads ~60 GB and
  hard-crashes a 64 GB machine; the laptop has 63.4 GB. Expect roughly 8 tok/s.

**Exit criteria — all three:**
- Both nodes complete `tests/kernels` and the two result sets are saved side by side
- At least one genuine sm_86/sm_120 disagreement is identified **and understood**
- A completion from the sm_120 node returns correct text, **read by eye**

**Integration point already prepared:** `serving/litellm-config.yaml` has a
commented laptop block ready for the new endpoint —
`api_base: http://10.10.0.2:8080/v1`. Adding the laptop is a config block, not
a client change. That is the whole reason LiteLLM is there.

---

## Phase 2b — runs in parallel, on the desktop

QLoRA on the **3090, not the 5090**, for three compounding reasons the runbook
verified: there is no VRAM advantage (the Laptop 5090 is 24463 MiB, the same
budget as the 3090 — online "RTX 5090 QLoRA" guidance describes the 32 GB
desktop card and does not transfer); sm_86 is the better-supported training
target (flash-attention for sm_120 needs CUDA ≥ 12.8 and a community wheel,
since `setup.py` only emits sm_120 gencode from 12.8 — any cu126 wheel silently
carries no Blackwell kernels, while on Ampere it is a `pip install`); and a
multi-hour job at 175 W throttles, while the desktop sustains 400 W.

What fits in 24 GB: 7–8B comfortable; 13–14B comfortable with gradient
checkpointing, batch 1–2; 30–32B dense is the tight edge (seq ≤ 512–1024,
batch 1, gradient checkpointing, 8-bit optimizer); 70B does not fit.

**Training takes the serving node down** — which is what the LiteLLM front door
earns back. Point a fallback at the laptop's endpoint for the duration of a run.

Exit criteria: a 7–8B QLoRA run completes on the 3090 and the adapter loads back
for inference; GPU temperature stays under threshold for the whole run (check,
do not assume); inference keeps answering on the same base URL throughout,
served from the laptop.

---

## Decide this before starting Phase 2

**The fork the runbook asks you to settle (§05): engine development or
application development?**

- *Engine & kernel development* — contributing to vLLM, CUTLASS, FlashInfer,
  fixing the SM120 FP4 path. Phases 2–3 serve this superbly; the two-arch
  differential is a genuinely rare asset.
- *Application development* — agents, RAG, tool use, evals. **Phase 1 already
  delivered essentially all of this.** Phases 2–3 contribute almost nothing to
  it.

This is not recorded as settled anywhere. David is proceeding to Phase 2, which
points at engine work, and has separately asked about the §08 training pool —
but ask rather than assume, because the honest answer changes how much of
Phase 2 is worth doing.

---

## Settled decisions — do not re-litigate

- **Finish line is Phase 2, not Phase 3.** Phase 3 is elective.
- **Rust, not Go**, if tooling gets written. A CLI tool, not a control-plane
  service — the job has no scheduling in it.
- **LiteLLM in front of both nodes**, llama-swap behind it per host.
- **The two current models cannot be co-resident.** The coder alone takes 23.2
  of 24.5 GiB, so llama-swap evicts one to load the other at ~90 s. **Do not
  "fix" this by shrinking the coder** — that cuts KV from 3.3 GiB to ~1.1 GiB
  and caps context near 12k. The right fix is moving the embedding model to the
  laptop, which is Phase 2 work.
- **The 48 GB cross-node training pool (§08) waits until after Phase 2b**, and
  needs all three triggers: 2b complete, a named ceiling hit, and a real 70B
  target with a dataset and an eval.
- **Both nodes are on driver 595.91.07 by convergence, not by pinning.**
  `apt-mark hold` the driver metapackages on both before any benchmark sweep.

---

## Traps that have already cost real time

1. **The laptop's I226-V NIC does not survive suspend.** After a lid-close the
   netdev still appears in `ip link show` but every operation returns ENODEV,
   the PHY is dead, and *the desktop therefore shows `enp5s0 DOWN` with no
   address* — which reads as a desktop failure and is not one. systemd-networkd
   does not address a link with no carrier, so the missing 10.10.0.1 is a
   symptom, not lost netplan config. Automated by `gpu-lab-igc-resume.service`
   on the laptop; manual command is
   `sudo /usr/local/sbin/gpu-lab-igc-resume-repair`. Details in
   `host/README.md`. **Caveat: the hook has not yet been through a real
   suspend cycle** — it was verified by driving the unit and by wedging the NIC
   for real, but the lid-close trigger itself is verified by construction.
   Check `journalctl -u gpu-lab-igc-resume` after the first real resume.
2. **`nvcc -arch=sm_XX` silently embeds PTX alongside SASS.** An image built
   for the wrong architecture still runs — the driver JIT-compiles the PTX — so
   the harness measures JIT overhead and a generic kernel while believing it
   measured an architecture. Correct output, wrong numbers, no error. Use
   explicit `-gencode arch=compute_86,code=sm_86` (cubin only), which raises
   `cudaErrorNoKernelImageForDevice` on foreign hardware. **This attacks the
   premise of the whole project; nothing else in the plan catches it.**
3. **A failed kernel launch surfaces through `cudaGetLastError()` immediately
   after the launch, not through `cudaDeviceSynchronize()`.** Checking only the
   sync lets a kernel that never ran report success with an unwritten buffer.
   Any arch-verification code must check `cudaGetLastError()` first.
4. **Never run `ubuntu-drivers install --gpgpu` on the laptop.** It selects the
   headless line, strips `libnvidia-gl` and `xserver-xorg-video-nvidia`, and the
   next boot has no driver for the GPU the panel is attached to. The running
   session survives because deleted libraries stay mapped — it looks fine right
   up until reboot. Desktop only.
5. **Single-sample benchmarks are banned.** Decode drifted 177.6 → 175.9 tok/s
   as the card went 58 → 70 °C. **Never edit the prompt set in `bench/bench.py`**
   without resetting the baseline.
6. **vLLM's "maximum concurrency 1.10x" is not a measurement.** It is
   `KV_pool / max_model_len` assuming every request fills the whole window. Real
   concurrency is far higher — 2 concurrent streams ran with zero preemptions at
   305 tok/s aggregate. Lowering `max_model_len` frees **no** memory; it only
   changes that ratio.

---

## How to work in this repo

- **Repo:** `git@github.com:kndclark/GPU-Lab.git` (private), working copy at
  `/home/david/gpu-lab` on the laptop. Currently clean at `cd41bea`.
- **One `git push` reaches both GitHub and the desktop.** `origin` has two push
  URLs; the desktop's bare repo has a post-receive hook that runs
  `checkout -f main`. So **pushing is deploying** — never push casually, and
  never branch for work that must reach the desktop (the hook only checks out
  `main`).
- Confirm both nodes agree before any measurement run:
  `git rev-parse --short HEAD` and
  `ssh llm 'cd /home/david/gpu-lab && git rev-parse --short HEAD'`.
- **Every new component gets wired into `bin/lab`** — `up`, `down`, *and*
  `status` — in the same change that introduces it. Never leave starting or
  stopping something as a remembered manual step. `lab down` must still verify
  the GPU actually drops below 1 GiB, and anything that can hold GPU memory
  joins the orphan sweep.
- **The laptop shares that script**, node-aware, rather than getting a fork of
  it. Note the node-awareness is *specified but not yet built* — `cmd_up`,
  `cmd_down` and most of `cmd_status` still assume the desktop. Phase 2 adds
  the laptop's endpoint, so this is the moment that gap has to close. Two
  constraints it must respect: the laptop is not always-on, and the nodes must
  stay independently controllable (Phase 2b trains on the desktop while the
  laptop serves).
- **Commit messages follow 50/72** — subject ≤ 50 chars, imperative, blank
  line, body wrapped at 72. No `Co-Authored-By` or tool-attribution trailers,
  ever.
- **`runbook.html` is gitignored on purpose** — the artifact has its own version
  history and a local copy would drift.
- `host/` holds host configuration no code reproduces (direct-link addressing,
  NFS export and mount options, driver/Secure Boot notes, the NIC resume fix).
  Netplan is never committed verbatim — the desktop's carries a cleartext Wi-Fi
  PSK.

---

## Loose ends carried into Phase 2

- The desktop's git says `ahead of origin/main by 11 commits` — a stale fetch on
  its side only; its working tree is correct. `ssh llm 'cd /home/david/gpu-lab && git fetch'`
  quiets it.
- NFS write is capped at 189 MB/s by the `sync` export option. `async` would
  raise it, defensible here only because the weights are re-downloadable. Not done.
- Phase 1's `qwen3-embed` is tagged `arch: sm_120_pending` in the LiteLLM config
  — it is still served from the desktop and is meant to move to the laptop.
- The runbook contains one **unreconciled contradiction**: Phase 2b rules the
  laptop out for training ("a multi-hour job at 175 W throttles"), while §08's
  pipeline pool puts half of every training step on that same 175 W laptop. Not
  fatal — a pipeline stage is bursty rather than a sustained duty cycle — but
  the laptop half will throttle first and pace both nodes. Do not plan §08
  around the average of the two cards.
