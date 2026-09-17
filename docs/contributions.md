# Upstream contribution candidates

Every observation this lab has produced that might belong upstream rather than
in a local note. One row per candidate, each with the evidence that exists
today and the specific thing still missing before it could be filed.

**Nothing here has been filed.** Nothing here has been reproduced against
upstream `main` either -- see [Before filing anything](#before-filing-anything),
which is the gate every candidate has to pass and none has passed yet.

The bar for this file is deliberately higher than "sm_86 and sm_120 disagree".
A disagreement is a lead. It earns a row here only when it survives three
questions: is it a cascade artifact, is it a harness artifact, and does
upstream already know? The [Not candidates](#not-candidates) section at the
bottom exists because several things that looked like findings failed one of
those and should not be re-raised.

## Environment of record

Any issue filed from this lab has to carry this, because most of it is unusual
enough that upstream cannot assume it.

| | |
|---|---|
| vLLM | `v0.29.0`, the **prebuilt wheel** -- no source build |
| Harness image | `gpu-lab:kernels`, `sha256:97e2441087b4...`, byte-identical on both nodes (verified by image ID) |
| Tests | `tests/` from the matching `v0.29.0` tag; `torch 2.13.0+cu130` |
| sm_86 node | RTX 3090, 24576 MiB, 400 W, i9-10850K / 31.9 GB |
| sm_120 node | RTX 5090 **Laptop** (GB203M), 24463 MiB, 175 W, Ultra 9 275HX / 63.4 GB |
| Both nodes | Ubuntu 26.04.1, kernel 7.0.0-31, driver 595.91.07, CUDA 13.2 |
| Run | per-file isolation (one pytest process per file), 195 files, 2026-09-16 |

The RTX 5090 *Laptop* GPU matters: it is sm_120 like the desktop 5090, but at
24 GB and 175 W. Upstream readers will assume 32 GB unless told otherwise, and
a few of these results are close enough to a memory ceiling for that to matter.

## The ledger

| ID | Candidate | Kind | Readiness |
|---|---|---|---|
| [C1](#c1) | `test_cutlass_mla_decode.py` gates on the wrong function | test gate | **Strongest.** Fix is known and sits in the same file |
| [C2](#c2) | INT8 CUTLASS tests are not gated for SM120 | test gate | Ready once reproduced on `main` |
| [C3](#c3) | `cutlass_gemm_caller` `Error Internal` on 216 azp tests | kernel or gate -- unknown | Needs triage. Distinct from C2 |
| [C4](#c4) | DeepGEMM device-side assert kills the CUDA context | kernel bug | Needs a minimal repro. Highest severity |
| [C5](#c5) | qutlass NVFP4 fused-quantize fails 132/132 on sm_120 | kernel or gate -- unknown | Needs triage |
| [C6](#c6) | FlashInfer `trtllm` backend raises instead of skipping | test gate | Ready once reproduced. Low severity |
| [C7](#c7) | NVFP4 *emulation* path will not compile on sm_86 | kernel or test | Needs triage |
| [C8](#c8) | 2757 Ampere attention tests fail on `fp8e4nv` rather than skipping | test gate | Needs a "does upstream care" check first |

C1, C2, C6 and C8 are all the same species: **the codebase knows a capability
is absent and the test suite has not been told.** If any one of them is worth
filing, they are probably worth filing together as one issue about capability
gating, with four instances. That is a judgement call to make when the first
one is actually reproduced on `main`, not now.

---

<a id="c1"></a>
## C1 -- `test_cutlass_mla_decode.py` gates on the wrong function

**The file contains its own fix, three lines above the bug.**

```python
CUTLASS_MLA_UNSUPPORTED_REASON = (
    "Cutlass MLA Requires compute capability of 100 or above."
    if not current_platform.is_device_capability_family(100)   # <- correct
    else "Cutlass MLA is supported"
)

@pytest.mark.skipif(
    not current_platform.has_device_capability(100),           # <- wrong
    reason=CUTLASS_MLA_UNSUPPORTED_REASON,
)
```

`has_device_capability` compares `major * 10 + minor`, so sm_120 scores 120,
clears the 100 threshold, and is admitted to tests written for SM100 hardware.
`is_device_capability_family` does `(capability // 10) == (target // 10)` --
the correct family comparison -- and is used only to *word the skip reason*,
never to decide whether to skip. Both were called directly on the laptop:
`has_device_capability(100)` is `True` and `is_device_capability_family(100)`
is `False` for the same device.

**Observed:**

| | sm_86 (3090) | sm_120 (5090 Laptop) |
|---|---|---|
| `test_cutlass_mla_decode.py` | 49 skipped, reason *"Cutlass MLA Requires compute capability of 100 or above"* | 49 run: **43 fail, 6 OOM** |

The 43 are a real kernel error, not a Python exception:

```
RuntimeError: runMla,
csrc/libtorch_stable/attention/mla/sm100_cutlass_mla_kernel.cu:209, Error Internal
```

**The other 6 are not evidence and should not be cited.** They are
`torch.OutOfMemoryError` -- one asks for 4.85 GiB on top of 18.63 GiB already
in use on a 24 GB card. On a 32 GB desktop 5090 they would likely pass. Only
the 43 speak to the gate.

The reason string the test would print if it skipped correctly describes
exactly the mistake being made: CUTLASS MLA needs SM100 (tensor memory, UMMA),
and consumer Blackwell is the 12.x family, which per the runbook (§02) is not
cross-compatible with 10.x in either direction.

**Blast radius, measured.** 21 files in the suite gate on
`has_device_capability(100)`. Only one other, `test_flashinfer_mla_decode.py`,
repeats the identical module-level shape *and* the wrong-reason-string wording
-- and it is harmless there: 9/9 pass on sm_120, because FlashInfer's MLA
kernel happens to run on consumer Blackwell where CUTLASS's does not.
`test_flashinfer_nvfp4_scaled_mm.py` uses **both** functions correctly, gating
its SM100-only cute-dsl backend on the family check and its SM120-specific
b12x backend on the numeric one -- proof the codebase knows the distinction
where someone applied it carefully.

**Still needed:** reproduce on upstream `main` (the gate may already have been
fixed since v0.29.0), then a one-line PR swapping the skipif condition.

<a id="c2"></a>
## C2 -- INT8 CUTLASS tests are not gated for SM120

vLLM's own C++ dispatcher refuses INT8 on SM120 by name:

```
RuntimeError: dispatch_scaled_mm,
csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:34,
Int8 not supported on SM120. Use FP8 quantization instead, or run on older
arch (SM < 100).
```

**189 tests in `test_cutlass_scaled_mm.py` fail with exactly that message on
sm_120. The identical test ids pass on sm_86.** Confirmed by running one
nodeid by hand on each node rather than by inference:

```
test_cutlass_int8_gemm[True-b_scale_group_shape0-a_scale_group_shape0-1-256-128]
  sm_86  : 1 passed
  sm_120 : 1 failed
```

Upstream knows the capability is absent -- the dispatcher raises a message
naming SM120 specifically. The test suite has not been taught the same fact,
so these fail instead of skipping. Compare the NVFP4 files, which skip cleanly
at collection on sm_86.

**The 3090 does something the 5090 cannot.** INT8 W8A8 CUTLASS GEMM is
supported on Ampere and dropped on consumer Blackwell, and the remedy in the
error string -- "use FP8 quantization instead" -- is only available on the
newer card. FP8 blockwise scaled GEMM does pass on sm_120. Lab consequence,
independent of any upstream filing: **an INT8-quantised checkpoint is a
desktop-only model here, and the LiteLLM config must never route one to the
laptop.**

**Still needed:** reproduce on `main`; decide whether to file with C1.

<a id="c3"></a>
## C3 -- `cutlass_gemm_caller` `Error Internal`, 216 azp tests

**This was previously reported as part of C2 and it is not.** The doc said
"all 400 INT8 tests fail with the dispatch rejection". The real split of the
405 sm_120 failures in `test_cutlass_scaled_mm.py` is:

| cause | count |
|---|---|
| `dispatch_scaled_mm ... Int8 not supported on SM120` (C2) | 189 |
| `cutlass_gemm_caller ... c3x/cutlass_gemm_caller.cuh:62, Error Internal` | **216** |

The 216 are `test_cutlass_int8_azp*` -- the activation-zero-point variants --
and they fail *past* the dispatcher, inside the GEMM caller, with a generic
internal error rather than the specific capability refusal. sm_86: 0 failures
in the whole file.

Two readings, and the evidence does not currently choose between them:

1. The same missing INT8 capability surfacing through a second code path that
   lacks C2's explicit check -- in which case it is one bug with two faces.
2. A genuinely separate failure in the azp path.

**Still needed:** run one azp nodeid in isolation on sm_120 with
`CUDA_LAUNCH_BLOCKING=1` and read the real CUTLASS status code behind
`Error Internal`. Until then this is a lead, not a finding, and it should not
be described as INT8-related in anything that leaves this repo.

<a id="c4"></a>
## C4 -- DeepGEMM device-side assert kills the CUDA context

**Highest severity of anything here**, because it is not a missing capability
a gate should have caught -- it is a kernel crashing on hardware it was
compiled for, in a way the process cannot recover from.

```
Assertion failed:
vllm/third_party/deep_gemm/include/deep_gemm/impls/smxx_layout.cuh:131,
condition: (values[j] & 0x807fffffu) == 0
```

That mask is the sign bit plus the mantissa of an IEEE-754 float. The kernel
asserts every scale factor is **exponent-only** -- a UE8M0 scale. On sm_120
something hands it a scale that is not, and it aborts the device context
rather than returning an error.

**Observed at two independent call sites**, which is what makes it a bug rather
than a flaky test:

- `test_silu_mul_fp8_quant_deep_gemm` -- sm_86: **23/23 pass**; sm_120: 1 pass,
  22 fail, 22 teardown errors
- `test_w8a8_block_fp8_deep_gemm_matmul` -- same root cause, different test

Both reproduce in isolation, so neither is cascade collateral. The visible
symptom downstream is `torch.AcceleratorError: CUDA error: unspecified launch
failure` on every subsequent test in the same process -- this is the crash that
caused the moe cascade in the first overnight run, and the reason
`overnight.sh` now runs one process per file.

**Still needed:** a minimal repro outside pytest -- construct the scale tensor
the kernel receives, show a non-exponent-only value reaching it, and identify
whether the bad scale originates in the sm_120 quantisation path or in the
layout conversion. This is the one candidate where the repro is real work
rather than a formality, and also the one most likely to be worth it.

<a id="c5"></a>
## C5 -- qutlass NVFP4 fused-quantize fails 132/132 on sm_120

`test_nvfp4_qutlass.py`: **every test fails on sm_120**; the file is skipped at
collection on sm_86.

```
RuntimeError: run, .deps/qutlass-src/qutlass/csrc/fused_quantize_nv.cu:101, Error Internal
RuntimeError: runGemmNv, .deps/qutlass-src/qutlass/csrc/fused_quantize_nv_sm100.cu:187, Error Internal
```

Note the second filename: **`fused_quantize_nv_sm100.cu`**, executing on an
sm_120 device. That is suggestive of the same family-vs-threshold confusion as
C1 -- an SM100 kernel being dispatched on consumer Blackwell -- but the file
name alone is not proof, and `Error Internal` says nothing about why.

This file appears in none of the earlier write-ups; it surfaced while grounding
this document in the result XML rather than in the prose.

**Still needed:** read the qutlass dispatch logic to find out what selects the
`sm100` path on an sm_120 device, and whether qutlass is vendored (`.deps/`
suggests a submodule -- so the fix may belong in qutlass upstream, not vLLM).
Triage before treating this as related to C1.

<a id="c6"></a>
## C6 -- FlashInfer `trtllm` backend raises instead of skipping

18 of 180 tests in `test_flashinfer_nvfp4_scaled_mm.py` fail on sm_120 with:

```
flashinfer.utils.BackendSupportedError: mm_fp4 does not support backend
'trtllm' with capability (12, 0)
```

The library raises a clean, specific, *correct* error -- the backend genuinely
does not support this capability. The test then reports that as a failure
rather than catching it as a skip. Same species as C2, lower stakes, and the
fix is a `pytest.skipif` or a `pytest.raises` on the parametrised backend.

Worth noting this is the same file that gets C1's distinction *right* elsewhere
in its gating. Correct in one place, missing in another.

**Still needed:** reproduce on `main`; bundle with C1/C2/C8.

<a id="c7"></a>
## C7 -- NVFP4 emulation will not compile on sm_86

`test_nvfp4_emulation.py`: **84 failures on sm_86, 0 on sm_120** -- the
opposite direction from everything above.

```
triton.compiler.errors.CompilationError
```

The interesting part is what the file is *for*. An emulation path exists so
that hardware without native NVFP4 can still execute the semantics. The
hardware without native NVFP4 here is the 3090 -- and that is precisely where
it fails to compile.

**Still needed:** the full Triton error body (the XML holds only the first
line of the compilation traceback; the `.log` alongside it has the rest), to
establish whether this is an unsupported dtype on Ampere -- likely the same
`fp8e4nv` wall as C8 -- or something the emulation path could reasonably
support. If it is `fp8e4nv`, this folds into C8 and stops being its own row.

<a id="c8"></a>
## C8 -- 2757 Ampere attention tests fail on `fp8e4nv` rather than skipping

**Every one of the 2757 sm_86 attention failures traces to one cause**: Triton
cannot compile E4M3 FP8 kernels for Ampere (`type fp8e4nv not supported in
this architecture`). Verified by matching the string across every failure in
the directory -- not a sample, and cleaner than the 2632 estimated during the
run.

| file | failures |
|---|---|
| `test_merge_attn_states` | 1296 |
| `test_triton_unified_attention` | 1200 |
| `test_trtllm_kvfp8_dequant` | 199 |
| `test_cache` | 38 |
| `test_triton_decode_attention` | 16 |
| `test_pack_unpack_triton` | 8 |

FP8 compute needs CC >= 8.9 and the 3090 is 8.6, so the *absence* is correct
and expected -- this is the runbook's own FP8 correction observed directly. The
candidate is not "Ampere lacks FP8"; it is that 2757 tests express a known,
static hardware fact as a failure rather than a skip, across six files.

**And this is the one most likely to be a non-issue.** Ampere is old and
well-trodden; if this were unwelcome noise, someone would very likely have
fixed it years ago. Which raises the real question below.

**Still needed:** before anything else, find out whether upstream CI runs
`tests/kernels/attention` on Ampere at all. If it does not, this is invisible
to them and the report has value; if it does and these failures are tolerated,
there is a reason and it should be learned rather than argued with.

---

## Not candidates

Kept deliberately. Each of these looked like a finding at some point, and
re-deriving them wastes a session.

- **The 87 sm_86 attention OOM failures** from the first overnight run. Gone
  entirely under per-file isolation -- CUDA allocator fragmentation from
  thousands of GEMM-heavy tests in one long-lived pytest process, not a
  capacity limit on the card. A harness artifact. *General rule earned here:
  never trust a memory-pressure result from a multi-file run.*
- **The 6 MLA decode OOMs** on sm_120 (C1). A 24 GB Laptop 5090 running a test
  sized for a larger card. Not architectural.
- **17 `LocalEntryNotFoundError` errors** in `test_nvfp4_emulation.py` on
  **both** nodes. The harness sets `HF_HUB_OFFLINE=1` and those tests want a
  checkpoint that is not in the cache. Harness configuration, identical on both
  nodes, therefore not a differential at all.
- **The first overnight run's failure counts.** A device-side assert destroys
  the CUDA context and every later test in the process reports an
  indistinguishable `unspecified launch failure`. moe's 1939 sm_120 failures
  were ~45 real; quantization's column was void past index 634. Superseded by
  the per-file run. Keep the numbers only as the cascade story.
- **NVFP4 non-grouped ops passing on sm_120** (all 72). This *narrows* a
  runbook claim rather than refuting anything upstream: the known SM120 NVFP4
  breakage (cutlass#3096, flashinfer#2723) is specifically CUTLASS **grouped**
  block-scaled GEMM, the MoE path. Quantisation, swizzled/padded scale layouts
  and plain NVFP4 GEMM are correct on consumer Blackwell. Documentation-shaped,
  not contribution-shaped -- it belongs in the runbook, and it is already in
  [phase2-dev-plane.md](phase2-dev-plane.md).
- **241 FP8 quantisation tests passing on the 3090**, which has no FP8 compute.
  Correct: they exercise quantisation ops, not tensor-core FP8 GEMM. Confirms
  the runbook's FP8 correction; nothing to file.
- **`ir` and `mamba` identical test-for-test on both cards** (1627 and 929).
  This is the control, and it is the reason the disagreements elsewhere are
  believable. Not a finding -- but the first thing to re-check if the harness
  is ever suspected.

## Before filing anything

None of these has cleared this list. It applies to every row above.

1. **Reproduce on upstream `main`, not on `v0.29.0`.** Everything here was
   observed on a wheel pinned in September 2026. A fixed bug filed as new is
   worse than silence.
2. **Search the issue tracker first**, including closed issues. C2 and C8
   describe hardware facts upstream demonstrably knows -- the odds someone has
   already raised the gating question are not small.
3. **Reduce to a minimal repro** that does not require this lab: one nodeid, or
   a standalone script. "It fails in my two-node differential harness" is not a
   reproduction anyone can act on.
4. **State the environment of record**, above -- particularly *Laptop* 5090,
   24 GB, 175 W.
5. **Separate the test-suite fixes from the kernel bugs.** C1/C2/C6/C8 are PRs
   against `tests/`. C4 is an issue against a kernel. Filing them as one thread
   makes the kernel bug harder to see, and it is the one that matters most.
6. **Do not file C3, C5 or C7 as anything yet.** Each is one triage step away
   from either becoming real or folding into another row.
