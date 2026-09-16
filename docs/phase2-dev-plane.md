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
kernels/compare.py kernels/results/sm_86-quant-core.xml.gz \
                   kernels/results/sm_120-quant-core.xml.gz
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

### The second disagreement: the newer card lost a capability

Found while the overnight run was starting, and stronger than the first,
because it is an *outcome* disagreement rather than a structural one -- the two
nodes run the same test and get different answers.

**All 400 INT8 CUTLASS scaled-GEMM tests fail on sm_120. The identical test ids
pass on sm_86.** Verified by running one nodeid on each node by hand, not by
inference:

```
test_cutlass_int8_gemm[True-b_scale_group_shape0-a_scale_group_shape0-1-256-128]
  sm_86  : 1 passed
  sm_120 : 1 failed
```

It is not a numerical mismatch or a tolerance question. vLLM's own C++ dispatch
refuses outright:

```
RuntimeError: dispatch_scaled_mm,
csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:34,
Int8 not supported on SM120. Use FP8 quantization instead, or run on older
arch (SM < 100).
```

**The 3090 can do something the 5090 cannot.** INT8 W8A8 CUTLASS GEMM is
supported on Ampere and dropped on consumer Blackwell, and upstream's remedy in
the error string -- "use FP8 quantization instead" -- is only available on the
newer card. FP8 blockwise scaled GEMM does pass here, 19/19, so the suggested
path is real. But it means an INT8-quantised checkpoint is a desktop-only model
in this lab, and the LiteLLM config should never route one to the laptop.

There is a contribution-shaped observation sitting on top of this. The NVFP4
tests are *gated* -- sm_86 skips them cleanly at collection. The INT8 tests are
*not* gated for SM120, so they fail instead of skipping. Upstream knows the
capability is absent, because the dispatcher raises a specific error naming
SM120; the test suite just has not been taught the same fact. That asymmetry is
exactly the kind of thing the two-arch lab exists to notice.

Note what this does to the first finding's framing: the numeric
`has_device_capability(100)` gate lets sm_120 *into* NVFP4 tests it passes,
while the absence of any gate lets it into INT8 tests it cannot pass. Capability
gating in this codebase is inconsistent in both directions.

## The overnight run, 2026-09-15 into 09-16

Both nodes completed all six directories and restored their serving planes from
the EXIT trap; the front door answered 200 afterwards. Raw counts:

| directory | sm_86 (3090) | sm_120 (5090) |
|---|---|---|
| quantization | 5899 tests, 2307P 197F 17E 3378S | 10200 tests, 582P 3104F 3675E 2839S |
| moe | 8004, 6044P 47F 1913S | 8792, 5298P 1939F 11E 1544S |
| attention | 8272, 3426P **2757F** 2089S | 8280, 5968P 375F 1937S |
| core | 3308, 3181P 71F 56S | 3308, 3208P 53F 47S |
| ir | 1627, 1443P 0F 184S | 1627, 1443P 0F 184S |
| mamba | 929, 895P 8F 26S | 929, 895P 8F 26S |

**Read the last two rows first.** `ir` and `mamba` are identical on both cards,
to the test. That is the control: the harness is not manufacturing differences,
so the differences elsewhere are the cards.

### Most of those failure counts are not findings

The quantization and moe columns for sm_120 are largely **collateral from a
killed CUDA context.** A device-side assert destroys the context, and every
later test in the same pytest process then fails with
`CUDA error: unspecified launch failure` -- recorded as an ordinary failure,
indistinguishable from a real one.

- **moe, sm_120:** context dies at test 6856 of 8792. 1894 of the 1936 results
  after it are collateral. The honest count is ~45 real failures, not 1939.
- **quantization, sm_120:** context dies at test 634 of 10200, and 6777 results
  after it are collateral. **This voided the INT8 result** -- those tests run
  after `test_block_fp8` and were swallowed, reported as launch failures rather
  than as the clean `Int8 not supported on SM120` rejection the targeted run
  had already measured. The targeted `quant-core` run stands; the overnight
  quantization column does not, past index 634.

`compare.py` now detects this and refuses to let the number pass unqualified.
Its heuristic deliberately does not anchor on the *first* launch failure: moe
has an isolated one at index 3508 that the run recovered from (a test that forks
a subprocess), and treating that as the cascade would have written off 3300
healthy tests. It takes the earliest point after which >= 70% of everything
fails.

**`overnight.sh` now runs one pytest process per FILE rather than per
directory.** A context kill then voids the rest of that one file instead of the
rest of the night.

### The third disagreement: a hard crash on sm_120 that Ampere does not have

`tests/kernels/moe/test_silu_mul_fp8_quant_deep_gemm.py`:

| | sm_86 | sm_120 |
|---|---|---|
| | **23 passed** | 1 passed, 22 failed, 3 error |

Reproduced in isolation, so it is not itself collateral. The root cause is a
device-side assertion inside vLLM's bundled DeepGEMM:

```
Assertion failed: vllm/third_party/deep_gemm/include/deep_gemm/impls/
smxx_layout.cuh:131, condition: (values[j] & 0x807fffffu) == 0
```

That mask is the sign bit plus the mantissa of an IEEE-754 float. The kernel is
asserting that every scale factor is **exponent-only** -- a UE8M0 scale, the
format DeepGEMM's block quantisation expects. On sm_120 something upstream hands
it a scale that is not, and the kernel aborts the context rather than returning
an error.

This is the most serious of the three findings, and the most upstream-shaped:
it is not a missing capability that a gate should have caught, it is a kernel
crashing on hardware it was compiled for. It is also the cause of the moe
cascade above -- one assert, ~1900 corrupted results.

### The fourth: the same gap, pointing the other way

sm_86's 2757 attention failures are **not** a cascade; they are real and they
share one cause:

```
ValueError: type fp8e4nv not supported in this architecture.
The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')
```

2632 of them are Triton refusing to compile E4M3 FP8 kernels for Ampere
(1296 in `test_merge_attn_states`, 1113 in `test_triton_unified_attention`).
This is the CC >= 8.9 FP8 requirement showing up in the Triton compiler rather
than in a capability gate, and it is the mirror image of the sm_120 results:
**the 5090 fails quantization and MoE where the 3090 succeeds, and the 3090
fails attention where the 5090 succeeds.** Neither card is the better one.

A further 87 sm_86 attention failures are CUDA OOM, on a card that had the whole
24 GiB free. Not yet investigated -- it may be a genuine capacity limit on the
larger attention shapes, and it is the one number here I would not quote yet.

### Known harness artefact

17 quantization tests error in setup with
`huggingface_hub.errors.LocalEntryNotFoundError`, because the image sets
`HF_HUB_OFFLINE=1`. Both nodes hit it identically so it cancels out of the
differential, but it is noise, not a finding.

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
