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
`bin/lab`, which is now node-aware.

```bash
bin/lab up --warm     # starts llama-swap, preloads the embedding model
bin/lab status        # reports which node it is on
```

**Why embeddings moved here.** The coder alone takes 23.2 of the 3090's
24.5 GiB, so on the desktop the two models could not be co-resident and
llama-swap evicted one to serve the other at ~90 s. That is now gone: they are
on different cards. `--gpu-memory-utilization` is 0.30 rather than the desktop's
0.20 because this board drives a display and loses ~1.25 GiB before vLLM starts;
measured steady state is 8.4 GiB of 24463 MiB, leaving ~16 GiB free for
experiments.

## Step 4: the kernel differential runs on the prebuilt wheel

The runbook puts `pytest tests/kernels` after a source build against
`compute_120f`. It does not need to be: **the differential runs against the
pinned wheel, with no source build at all.** A source build is required to
*modify* kernels. It is not required to *differentiate* them, and that is
several days of the Phase 2 estimate.

`kernels/` holds the harness. One image, `gpu-lab:kernels` -- the pinned
`vllm/vllm-openai:v0.29.0` plus five pinned test dependencies and the `tests/`
tree from the matching upstream tag. It is built once and moved across the
direct link with `docker save | docker load` (65 s), so both nodes run a
bit-identical harness; verified by image ID, `sha256:97e2441087b4...` on each.

```bash
kernels/build.sh                                    # build (laptop)
sudo docker save gpu-lab:kernels | ssh llm 'sudo docker load'
SUITE=quant-core kernels/run.sh $(grep -v '^#' kernels/SUITE.txt)   # each node
kernels/compare.py kernels/results/sm_86-quant-core.xml \
                   kernels/results/sm_120-quant-core.xml
kernels/overnight.sh                                # whole suite, detached
```

### The first genuine sm_86/sm_120 disagreement

Same image, same six test files, same command:

| | sm_86 (3090) | sm_120 (5090 Laptop) |
|---|---|---|
| collected | 274 | **344** |
| passed | 270 | 342 |
| skipped | 4 | 2 |

**The two cards do not even run the same tests.** 72 NVFP4 tests exist on
sm_120 and do not exist on sm_86 -- `test_nvfp4_quant.py` and
`test_nvfp4_scaled_mm.py` are skipped at *collection* on the 3090, so their
tests never materialise. Of the 272 tests both nodes do run, **zero disagree**,
which is what makes the structural gap the finding rather than noise.

The cause is one line at the top of each of those files:

```python
if not current_platform.has_device_capability(100):
    pytest.skip(reason="Nvfp4 Requires compute capability of 10 or above.",
                allow_module_level=True)
```

`has_device_capability` compares `major * 10 + minor`, confirmed on both cards:
sm_86 is 86, below 100, so the module is skipped; sm_120 is **120**, above 100,
so it is admitted.

**That is a family assumption enforced as a numeric threshold, and the
arithmetic admits a family the reason string does not name.** "Compute
capability 10 or above" means SM100 -- datacenter Blackwell, which has NVFP4
tensor cores. Consumer Blackwell is the 12.x family, and per the runbook (§02)
the 10.x and 12.x families are not cross-compatible in either direction. sm_120
runs these tests because 120 > 100, not because anyone decided it should.

And they pass -- all 72. That is worth stating carefully, because it **narrows a
runbook claim rather than confirming it.** The runbook warns that the SM120
NVFP4 path "produces garbage output or crashes" (cutlass#3096,
flashinfer#2723). Those reports are specifically about CUTLASS *grouped
block-scaled* GEMM, the MoE path. The ops covered here -- NVFP4 quantisation,
swizzled and padded scale-factor layouts, and plain (non-grouped) NVFP4 GEMM --
are correct on consumer Blackwell. The `--moe-backend marlin` correction remains
untested, not disproven; `tests/kernels/moe` is where it would show, and it is
second in the overnight order for that reason.

A quieter result in the same run: **all 241 FP8 quantisation tests pass on the
3090**, which has no FP8 compute at all (that needs CC >= 8.9). This is the
runbook's own correction observed directly -- FP8 *checkpoints* are fine on
Ampere via dequantisation; only FP8 *math* is not. The tests exercise the
quantisation ops, not tensor-core FP8 GEMM.

### Traps this harness had to design around

- **The source tree shadows the wheel.** pytest puts the working directory on
  `sys.path`, so running from a vLLM checkout makes `import vllm` resolve to
  uncompiled source instead of the installed package. The image therefore
  carries `tests/` and `pyproject.toml` and *not* `vllm/`. Check with
  `python3 -c "import vllm; print(vllm.__file__)"` -- it must print a path under
  `dist-packages`.
- **The full test requirements would move torch.** They resolve to several
  hundred packages including ray and lm-eval. `pip install --no-deps` of five
  pinned packages keeps torch at `2.13.0+cu130` on both nodes, which is the
  thing that makes them comparable. Re-check after any edit to the Dockerfile.
- **Collect errors without a GPU are fiction.** Collecting `tests/kernels` in a
  container started without `--gpus all` reports import errors in five
  quantization files; with the GPU attached the same command collects 7090 tests
  and zero errors. The difference is `libcuda.so.1`. Never size or triage this
  suite from a CPU-only container.
- **An interrupted pytest writes no XML at all.** The junit file is written at
  the end of the run, so a twelve-hour invocation killed at hour eleven yields
  nothing. `overnight.sh` runs one directory per invocation, highest-value
  first, so a partial night still leaves complete results.
- **A run holds the whole card.** `run.sh` refuses to start above 2000 MiB, and
  `overnight.sh` takes the serving plane down for the duration and restores it
  from an EXIT trap. On the laptop it also holds `systemd-inhibit
  --what=sleep:idle:handle-lid-switch`, because a lid-close would end the run
  *and* wedge the I226-V NIC, making an aborted run look like an unreachable
  node.

## Not done

- **Source build against `compute_120f`, FlashInfer, CUTLASS.** Still required
  to *change* kernels; no longer required to measure them, which is the part
  that has now been demonstrated rather than assumed.
- **llama.cpp with expert offload** for a 120B-class MoE (`--n-cpu-moe` ~ 21).
- **The whole suite on both nodes.** Only the six-file `quant-core` subset has
  run on both. `overnight.sh` covers quantization, moe, attention, core, ir and
  mamba; `moe` is the one that matters, because it is where the NVFP4 MoE claim
  lives.
- **Nothing is pushed.** The harness exists on both nodes -- on the desktop as
  untracked files copied over ssh, not via the deploy path -- and the desktop's
  `bin/lab` is still the version without the kernel-container sweep.

## Still unsettled

The runbook's §05 fork -- engine development or application development -- is
not recorded as answered. Phase 2's remaining items serve engine work only.
