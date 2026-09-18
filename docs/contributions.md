# Upstream contribution candidates

Every observation this lab has produced that might belong upstream rather than
in a local note. One row per candidate, each with the evidence that exists
today and the specific thing still missing before it could be filed.

**Nothing here has been filed.** Both gates have now been run. The first --
*does upstream still have this?* -- was run against `main` by source inspection
on 2026-09-17 and killed C1. The second -- *does it still happen on a `main`
build?* -- was run on 2026-09-18 against an actual upstream nightly, and every
surviving candidate reproduced. See [Runtime reproduction on
main](#runtime-repro).

**What source inspection can and cannot settle.** It can *kill* a candidate
outright: if the code upstream now reads the way the fix would, there is
nothing to file, and C1 died exactly that way in about thirty seconds. It
cannot *confirm* one. An unchanged test gate plus a silently fixed kernel still
produces a passing test. That is why the runtime pass exists, and why it was
worth doing: it changed nothing about C2 and C6, and it produced the decisive
evidence for C5.

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
| vLLM (primary) | `v0.29.0`, the **prebuilt wheel** -- no source build |
| vLLM (`main` check) | `vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910`, reporting `0.29.1rc1.dev347+gdee37d891`, pulled 2026-09-18 |
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
| [C1](#c1) | `test_cutlass_mla_decode.py` gates on the wrong function | test gate | ~~Strongest~~ **DEAD -- fixed upstream 2026-09-17** |
| [C2](#c2) | INT8 CUTLASS tests are not gated for SM120 | test gate | **File-ready.** Repro on `main`; upstream CI avoids it with `-k 'fp8'` |
| [C3](#c3) | `cutlass_scaled_mm_azp` routes Blackwell to the SM90 kernel | **product bug** | **File-ready, and promoted.** Not a test issue at all |
| [C4](#c4) | The test feeds uninitialised padding rows to a validating kernel | **test bug** | **File-ready. One-line fix, and it is not architecture-specific** |
| [C5](#c5) | qutlass ships SM100-only kernels while its test gate admits SM120 | **build + test gate** | **Strongest. Proven by binary inspection of a `main` build** |
| [C6](#c6) | FlashInfer `trtllm` backend raises instead of skipping | test gate | **File-ready.** Repro on `main`. Low severity |
| [C7](#c7) | NVFP4 *emulation* path will not compile on sm_86 | test gate | **FOLDED into C8 -- same cause, same fix** |
| [C8](#c8) | Ampere FP8 tests fail rather than skip, in two directories | test gate | **File-ready.** Upstream CI cannot see this -- no Ampere in the fleet |

Four candidates are the same species: **the codebase knows a capability is
absent and the test suite has not been told.** That is C2, C6, C8 and (folded
into C8) C7. C1 was the fifth, and its merged fix is the precedent to cite if
the rest are filed together.

C3, C4 and C5 are not that species and should not be filed with them. **C3 is
a bug in shipped library code**, C4 is a test that corrupts its own input, and
C5 is a build-system selector that silently drops an architecture. Filing them
in one thread with four test-gate fixes would bury all three.

<a id="runtime-repro"></a>
## Runtime reproduction on `main` -- 2026-09-18

Gate item 1 said "reproduce on upstream `main`, not on `v0.29.0`". Done, using
the official nightly image rather than a source build -- which is the same
lesson Phase 2 step 4 already produced: a source build is needed to *modify*
kernels, not to *observe* them.

| ID | Reproduced on `main`? | How |
|---|---|---|
| C2 | **Yes** | `kernels/repros/c3_azp_missing_guard.py` in the nightly image: same dispatcher refusal, verbatim |
| C3 | **Yes** | same script, same run: `cutlass_gemm_caller ... Error Internal`, verbatim |
| C4 | **Yes (runtime)** | `kernels/repros/c4_padding_rows.py` against `main`'s own test file in the nightly image: **518,864** non-conforming scales, **0** in real rows -- identical to v0.29.0 |
| C5 | **Yes (binary)** | `cuobjdump` on the nightly's own `_qutlass_C.abi3.so`: **sm_100 only**, while `_C` and `_moe_C` in the same wheel carry sm_120 |
| C6 | **Yes** | `main`'s test file run in the nightly image: **18 failed, 18 skipped**, matching v0.29.0 exactly |
| C8 | **N/A** | Not a "has it been fixed" question -- see C8, where the question was whether upstream can see it at all |

**The C5 result is the one that justifies the exercise.** A pytest run would
only have reproduced the symptom. Inspecting the shipped binary produced the
*cause*, and it is not in the test suite at all.

<a id="ci-fleet"></a>
## What upstream CI actually runs, and on what

Read from `.buildkite/test_areas/kernels.yaml` on `main`, 2026-09-18. The old
`test-pipeline.yaml` was deprecated on 2026-02-18 and now only points here.

| Job | Hardware | Compute capability |
|---|---|---|
| Attention Kernels Shard | **L4** | **8.9** |
| Quantization Kernels Shard | **L4** | **8.9** |
| MoE Kernels Shard | **L4** | **8.9** |
| Core Operation Kernels, Mamba, vLLM IR | H200 MIG | 9.0 |
| MLA, DeepGEMM, FusedMoE, FP8 MoE, Helion, DiffKV | H100 | 9.0 |
| Kernels Shard, FP4 MoE, Miscellaneous | B200 | 10.0 |
| Spark b12x Linear Kernels (nightly) | DGX Spark | 12.x -- inferred from the `b12x` job name, not confirmed |

Three facts follow, and each one lands on a candidate below.

1. **The lowest-capability NVIDIA GPU in the whole kernels fleet is the L4, at
   8.9 -- which is exactly the threshold.** Triton's own CUDA backend reads:

   ```python
   if capability >= 89:
       supported_fp8_dtypes.add("fp8e4nv")
   ```
   (`triton/backends/nvidia/compiler.py:193`, in the installed wheel.)

   The L4 is Ada Lovelace, compute capability 8.9, so it clears that bar by a
   single step and `fp8e4nv` compiles. The 3090 is 8.6 and does not. There is no
   Ampere anywhere in the fleet -- no 8.0, no 8.6 -- so every FP8 Triton failure
   this lab sees on the 3090 is *invisible* upstream rather than tolerated. That
   is C8, and it is the answer C8 was waiting for.
2. **Only the B200 job runs `test_nvfp4_qutlass.py`**, and B200 is sm_100. The
   one 12.x machine in the fleet, the DGX Spark, runs a single unrelated file
   (`test_b12x_linear.py`). So *nothing in upstream CI ever runs qutlass on the
   12.x family* -- which is exactly the support its test gate claims. That is
   C5.
3. **The B200 job runs `test_cutlass_scaled_mm.py -k 'fp8'`.** The INT8 tests
   are filtered out at the CI invocation rather than gated in the file. Upstream
   is already working around the missing skip; the workaround just lives in the
   wrong place. That is C2, and it is the strongest argument for the fix.

---

<a id="gate-survey"></a>
## How widespread the wrong gate is -- surveyed 2026-09-18

C1's fix was one line, but the construct it fixed is everywhere. Counted across
the whole `tests/` tree at `v0.29.0`:

| | files |
|---|---|
| gate on `has_device_capability(100)` (numeric, `>=`) | **28** |
| use `is_device_capability_family(...)` anywhere | 48 |
| use **both**, deliberately, in the same file | **3** |

The three careful ones are `test_cutlass_mla_decode.py` (after its fix),
`test_flashinfer_nvfp4_scaled_mm.py` and `tests/models/quantization/test_nvfp4.py`.

**Most of the 28 are not bugs, and this number must not be reported as a defect
count.** `has_device_capability(N)` is correct whenever the intent really is
"this capability or newer" -- C6's `b12x` gate is a good example. It is wrong
only where the feature belongs to one family, and deciding that requires
reading each site. What the survey establishes is the *shape* of the risk:
the numeric form is the default reach, the family form is the exception, and
only three files in the tree show anyone weighing the two.

The four sites this lab has actually shown to be wrong are C1's (now fixed),
C4's line 258, C5's module gate, and C3's -- which is not in `tests/` at all.

<a id="c1"></a>
## C1 -- `test_cutlass_mla_decode.py` gates on the wrong function

**DEAD. Fixed upstream, confirmed 2026-09-17.** Both `skipif` sites (lines 45
and 220) now call `is_device_capability_family(100)`, and
`test_flashinfer_mla_decode.py` was fixed the same way.

Kept because it is the **precedent** every other gate candidate cites, and
because the shape recurs: a numeric `has_device_capability(N)` is a `>=`
comparison on `major*10+minor`, so sm_120 scores 120 and clears any threshold
meant for SM100. `is_device_capability_family(N)` does `(cap // 10) == (N // 10)`
and is the correct comparison when the intent is "this family".

Observed before the fix: `test_cutlass_mla_decode.py` skipped 49 on sm_86 and
ran 49 on sm_120, of which **43 failed with a real kernel error** and 6 were
OOM. The 6 are not evidence -- a 24 GB Laptop 5090 running a test sized for a
larger card -- and were never cited.

**The same wrong construct is still live in three places found so far:** C4's
line 258, C5's module gate, and C3's C++ dispatcher. Two are in tests; one is
not.

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
nodeid by hand on each node rather than by inference, and again on 2026-09-18
by the standalone repro, on both `v0.29.0` and `main`.

**How the refusal is actually implemented** -- worth knowing, because it is not
a capability check. In `scaled_mm_helper.hpp`:

```cpp
if constexpr (!std::is_same_v<Int8Func, std::nullptr_t>) {
  int8_func(c, a, b, a_scales, b_scales, bias);
} else {
  int32_t version_num = get_sm_version_num();
  STD_TORCH_CHECK(false, "Int8 not supported on SM", version_num, ...);
}
```

It is a **compile-time** test of whether an INT8 functor was supplied at all.
The SM120 build supplies `nullptr`, and the message is generated from that. So
the absence is a build-time decision, and the runtime message is downstream of
it. This matters for C3, where the same absence surfaces with no message.

**Upstream already works around this, in the wrong place.** The B200 CI job
runs `pytest test_cutlass_scaled_mm.py -k 'fp8'` -- the INT8 tests are excluded
by the invocation. A `pytest.skipif` in the file would make the exclusion
visible, portable and correct for anyone running the file outside CI, which is
what this lab did.

**The 3090 does something the 5090 cannot.** INT8 W8A8 CUTLASS GEMM is
supported on Ampere and dropped on consumer Blackwell, and the remedy in the
error string -- "use FP8 quantization instead" -- is only available on the
newer card. FP8 blockwise scaled GEMM does pass on sm_120. Lab consequence,
independent of any upstream filing: **an INT8-quantised checkpoint is a
desktop-only model here, and the LiteLLM config must never route one to the
laptop.**

**Still needed:** search the issue tracker, then file with C6 and C8.

<a id="c3"></a>
## C3 -- `cutlass_scaled_mm_azp` routes Blackwell to the SM90 kernel

**This was previously filed under C2 as "a second face of the same bug", and
before that reported as part of C2's failure count. Both were wrong about what
it is.** It is not a test-suite problem. It is a routing bug in shipped library
code, and it is the only candidate here that affects users who never run a
test.

**The dispatcher, `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu`.**
The non-azp entry point knows about Blackwell:

```cpp
#if defined ENABLE_SCALED_MM_SM120 && ENABLE_SCALED_MM_SM120
  if (version_num >= 120) { cutlass_scaled_mm_sm120(...); return; }
#endif
#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
  if (version_num >= 100 && version_num < 120) { cutlass_scaled_mm_sm100(...); return; }
#endif
```

The azp entry point does not. Its first branch is:

```cpp
#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  if (version_num >= 90) {
    cutlass_scaled_mm_azp_sm90(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }
#endif
```

There is no SM100 or SM120 branch anywhere in it. **Every Blackwell capability
-- 100, 103, 120, 121 -- scores `>= 90` and is handed the Hopper kernel.**
This is C1's construct again, in C++, in the product rather than in a test.

**Observed, both architectures, both vLLM versions.** The repro
(`kernels/repros/c3_azp_missing_guard.py`) builds one set of INT8 tensors and
sends it through both entry points on the same device:

```
sm_120 (RTX 5090 Laptop), v0.29.0 and main -- identical output:
  cutlass_scaled_mm      -> RuntimeError: dispatch_scaled_mm ...
                            Int8 not supported on SM120. Use FP8 ...
  cutlass_scaled_mm_azp  -> RuntimeError: cutlass_gemm_caller,
                            .../c3x/cutlass_gemm_caller.cuh:62, Error Internal

sm_86 (RTX 3090), same script, control:
  cutlass_scaled_mm      -> returned (64, 32) torch.bfloat16
  cutlass_scaled_mm_azp  -> returned (64, 32) torch.bfloat16
```

The `c3x` in the failing path is CUTLASS 3.x -- the SM90+ code -- which is
independent confirmation that the `version_num >= 90` branch is the one that
fired.

**Scale:** 216 `test_cutlass_int8_azp*` tests fail this way on sm_120; sm_86
has 0 failures in the entire file. Those 216 plus C2's 189 are the 405 sm_120
failures in `test_cutlass_scaled_mm.py`.

**Status on `main` (checked 2026-09-18): LIVE.** The azp dispatcher is
unchanged -- still `version_num >= 90`, still no Blackwell branch.

**Why this is the serious one.** vLLM deliberately refuses INT8 on SM120 with
an actionable message. Any user running an asymmetrically-quantised INT8 W8A8
checkpoint (activation zero-points -- the azp path) on an RTX 5090 gets
`Error Internal` instead, with nothing to act on. Upstream's own stated intent
is not enforced on this path.

**One honest caveat.** A repro that trips a conformality check tells you
nothing: `b` must be **column-major** and `N`, `K` multiples of 16, or both
calls die in `scaled_mm_entry.cu` long before dispatch. The first version of
this repro did exactly that and had to be corrected.

**Still needed:** search the tracker for the azp path specifically, then file
as its own issue -- not with the test-gate group.

<a id="c4"></a>
## C4 -- The test feeds uninitialised padding rows to a validating kernel

**Re-diagnosed for the third time on 2026-09-18, and this one is grounded in a
measurement rather than a hypothesis.** The two earlier write-ups in this file
were both wrong: it is not a DeepGEMM bug, and it is not the bf16 ceiling. It
is also not, in the end, primarily an sm_120 story.

**The source, `tests/kernels/moe/test_silu_mul_fp8_quant_deep_gemm.py`:**

```python
ref_q   = torch.empty((E, T, H), dtype=fp8_dtype, device=DEVICE)
ref_s_f32 = torch.empty((E, T, cdiv(H, group_size)), dtype=torch.float32, device=DEVICE)

for e in range(E):
    nt = tokens_per_expert[e].item()
    if nt == 0:
        continue
    ref_q[e, :nt], ref_s_f32[e, :nt] = silu_mul_quant(...)   # only [:nt] written
```

`tokens_per_expert` is drawn as `randint(low=0, high=T)`, and `high` is
exclusive -- so `nt < T` **always**, and every expert carries at least one row
that is never written. The whole tensor, written rows and untouched rows
alike, is then passed to `transform_sf_into_required_layout`, whose pack kernel
asserts that every FP32 value is exponent-only:

```
Assertion failed: /usr/local/lib/python3.12/dist-packages/vllm/third_party/
deep_gemm/include/deep_gemm/impls/smxx_layout.cuh:131,
condition: (values[j] & 0x807fffffu) == 0
```

That is the observed site, 1094 times in one run's log, in the DeepGEMM that
`v0.29.0` vendors under `vllm/third_party`.

Uninitialised device memory does not satisfy that. **The assert is correct and
DeepGEMM is not at fault.**

**Measured, not argued.** `kernels/repros/c4_padding_rows.py` reproduces the
test's exact setup for all 21 distinct shapes, classifies every non-conforming
scale by whether its row index is inside that expert's token count, and never
calls the pack kernel -- so the context survives and all cases run in one
process. On sm_120:

| | count |
|---|---|
| non-conforming scales, total | **518,864** |
| ...in real rows (index < nt) | **0** |
| ...in padding rows (index >= nt) | **518,864** |

Five of the 21 shapes reported zero. That is not a refutation -- it is the
allocator handing back zeroed pages for those allocations, and it is exactly
the nondeterminism this diagnosis predicts. `0.0` has a clear sign and
mantissa, so it passes the assert.

**Two independent defects, and they need different fixes.**

1. **The test corrupts its own input.** `torch.empty` -> `torch.zeros` for
   `ref_s_f32`, or mask the padding before the transform. This is the real bug,
   it is one line, and **it is not architecture-specific** -- the same
   uninitialised rows are handed to the same assert on SM100, which is the
   architecture the block is written for. Whether B200 CI has been lucky with
   allocator state or this is a known flake is the one thing still unchecked.
2. **The gate is C1's gate.** Line 258 is
   `current_platform.has_device_capability(100)`, still unchanged on `main`.
   Fixing only this would *hide* the bug on consumer Blackwell while leaving it
   live on SM100. Worth reporting, worth not conflating.

**Note what the first fix costs the second.** Had the gate been "fixed" first,
this lab would have recorded a resolved candidate and never found the defect
that actually matters.

**What the sm_86 result means.** The earlier claim that "23/23 pass on the
3090" implied Ampere computes this correctly. It does not compute it at all:
`support_deep_gemm()` is `False` there, so the scale format resolves to
`FLOAT32` and the DeepGEMM path never executes. A pass on Ampere here is
evidence of non-execution, not of correctness.

**Reproductions kept:** `c4_padding_rows.py` (the diagnosis),
`c4_ue8m0_pack_assert.py` (the one-value minimal case),
`c4_check_bf16_ceiling.py` (the dead hypothesis, kept so it is not re-derived).

**Still needed:** check whether B200 CI has ever flaked on this file, then file
the `torch.empty` fix as its own issue.

<a id="c5"></a>
## C5 -- qutlass ships SM100-only kernels while its test gate admits SM120

**The strongest candidate, and the only one whose cause was found in a binary
rather than in source.** `test_nvfp4_qutlass.py` fails **132/132 on sm_120**
and is skipped at collection on sm_86.

```
RuntimeError: run, .deps/qutlass-src/qutlass/csrc/fused_quantize_nv.cu:101, Error Internal
RuntimeError: runGemmNv, .deps/qutlass-src/qutlass/csrc/fused_quantize_nv_sm100.cu:187, Error Internal
```

**The cause: the shipped wheel contains no sm_120 qutlass code at all.**
`cuobjdump --list-elf` on the extension modules, in both the pinned v0.29.0
image and the `main` nightly:

| module | architectures present |
|---|---|
| `_C_stable_libtorch.abi3.so` | sm_75 80 86 89 90 90a 100 **120** |
| `_moe_C_stable_libtorch.abi3.so` | sm_75 80 86 89 90 100 **120** |
| `_flashkda_C.abi3.so` | sm_90a 100 **120** |
| **`_qutlass_C.abi3.so`** | **sm_100 -- and nothing else** |

No sm_120 cubin, and no PTX to JIT from. The rest of the wheel carries
consumer Blackwell; qutlass does not.

**Why, in upstream's own words.** `cmake/external_projects/qutlass.cmake`:

```cmake
# QUTLASS uses TARGET_CUDA_ARCH as a single preprocessor selector for all its
# sources. Do not compile a mixed SM100/SM120 arch list with one selector; prefer
# SM100 when both families are requested because that is the primary deployed
# target for this extension today.
if(QUTLASS_SM100_ARCHS)
  set(QUTLASS_ARCHS "${QUTLASS_SM100_ARCHS}")
  set(QUTLASS_TARGET_CC 100)
  if(QUTLASS_SM120_ARCHS)
    message(WARNING "[QUTLASS] Both SM100 and SM120 archs were requested; selecting SM100 ...")
```

A release wheel requests both families, so SM120 loses, every time. The build
emits a CMake **warning** that nobody reads in a wheel build log, and the
resulting binary is SM100-only -- which is why the runtime error names
`fused_quantize_nv_sm100.cu` while executing on an sm_120 device. This is the
runbook's §02 claim observed in the wild: 10.x and 12.x cubins are not
cross-compatible in either direction, and here a 10.x-only extension is shipped
to 12.x users.

**The test gate then invites them in.** Identical in `v0.29.0` and `main`:

```python
if not (
    current_platform.has_device_capability(100)
    or current_platform.has_device_capability(120)
):
    pytest.skip(reason="Tests require compute capability 10.0 (100) or 12.0 (120).")
```

`has_device_capability` is `>=`, so **the second clause is dead code** -- any
device that satisfies `>= 120` already satisfied `>= 100`. The condition reads
as a two-family allowlist and compiles to `>= 100`, admitting sm_103, sm_110,
sm_121 and anything future along with sm_120. The reason string states the
intent the code does not implement.

**Upstream already has the correct pattern, one directory over.**
`_flashmla_C.abi3.so` is *also* built without sm_120 -- and
`test_flashmla.py` skips all **144** tests cleanly on sm_120, because it gates
on `is_flashmla_dense_supported()`, a capability query from the library itself
rather than a numeric threshold. Same situation, correct gate, clean skip. That
is the fix pattern to cite, and it makes this a report with a precedent rather
than a proposal.

**Nothing in upstream CI would catch it.** Only the B200 job (sm_100) runs
`test_nvfp4_qutlass.py`; the one 12.x machine in the fleet runs an unrelated
file. See [the CI fleet](#ci-fleet).

**Two defects again, and again they need different fixes.** The gate should use
family checks. The build should either compile qutlass twice with different
selectors or stop claiming 12.x support. Only the second one actually gets
consumer Blackwell working.

**Lab consequence:** the prebuilt wheel has no working qutlass NVFP4
fused-quantize on the laptop, and no configuration change will produce one.

**Still needed:** decide whether this is filed against vLLM (which owns the
cmake and the test) or `IST-DASLab/qutlass` (which owns
`TARGET_CUDA_ARCH` being a single selector). The cmake is vLLM's; the
single-selector constraint it is working around is not.

<a id="c6"></a>
## C6 -- FlashInfer `trtllm` backend raises instead of skipping

18 of 180 tests in `test_flashinfer_nvfp4_scaled_mm.py` fail on sm_120:

```
flashinfer.utils.BackendSupportedError:
mm_fp4 does not support backend 'trtllm' with capability 120
```

The library raises a clean, specific, *correct* error -- the backend genuinely
does not support this capability. The test then reports that as a failure
rather than catching it as a skip.

**Reproduced on `main` 2026-09-18**, using `main`'s own test file in the
nightly image: **18 failed, 18 skipped, 153 deselected** -- the same 18, and the
18 skips are the float16 parametrisations that the file's one existing guard
does catch. (The message wording changed from `capability (12, 0)` to
`capability 120`; the defect did not.)

**This file is the instructive one**, because it gets the distinction right
twice and misses it once:

```python
if "trtllm" in backend and dtype == torch.float16:      # dtype guard only
if backend == "cute-dsl" and not current_platform.is_device_capability_family(100):
if backend == "b12x" and not current_platform.has_device_capability(120):
```

`cute-dsl` uses the **family** check correctly. `b12x` uses the numeric check,
correctly, because it genuinely means "120 or above". `trtllm` has no
capability guard at all. Three backends, three different amounts of care, in
one parametrize list.

**Still needed:** search the tracker; file with C2 and C8 as one test-gate PR.

<a id="c7"></a>
## C7 -- NVFP4 emulation will not compile on sm_86 -- FOLDED INTO C8

**Resolved 2026-09-18 and folded.** C7's open question was whether the 84
sm_86 failures in `test_nvfp4_emulation.py` were an unsupported dtype or
something the emulation path could reasonably support. The full Triton error
body, read from the desktop's log, settles it:

```
scale = scale.to(tl.float8e4nv).to(tl.float32)
        ^
type fp8e4nv not supported in this architecture.
The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')
```

Identical to C8's cause. All **84** failures are one test function,
`test_triton_nvfp4_quant_dequant`, across 84 parametrisations; 2 tests in the
file pass and the 17 errors are the known `HF_HUB_OFFLINE` artifact.

**It keeps one observation that C8 does not have**, and that observation should
travel with the C8 report: NVFP4's scale factor *is* an E4M3 FP8 value, so the
"emulation" path -- which exists so hardware without native NVFP4 can execute
the semantics -- still requires FP8 casting, and therefore still requires
CC >= 8.9. On the exact hardware it is meant to serve, it does not compile.
Fixing that is more than adding a skip; the E4M3 rounding would have to be
emulated in fp32 or integer arithmetic. The skip is still the right immediate
fix.

**Note on where the evidence lived.** `*.log` is gitignored, so the sm_86 logs
exist only on the desktop and the sm_120 logs only on the laptop. This file
previously said "the `.log` alongside it" without saying which node. It is
`10.10.0.1:~/gpu-lab/kernels/results/`.

<a id="c8"></a>
## C8 -- Ampere FP8 tests fail rather than skip, in two directories

**Every one of the 2757 sm_86 attention failures traces to one cause**: Triton
cannot compile E4M3 FP8 kernels for Ampere (`type fp8e4nv not supported in
this architecture`). Verified by matching the string across every failure in
the directory -- not a sample.

| file | failures | directory |
|---|---|---|
| `test_merge_attn_states` | 1296 | attention |
| `test_triton_unified_attention` | 1200 | attention |
| `test_trtllm_kvfp8_dequant` | 199 | attention |
| `test_cache` | 38 | attention |
| `test_triton_decode_attention` | 16 | attention |
| `test_pack_unpack_triton` | 8 | attention |
| `test_nvfp4_emulation` (was C7) | **84** | **quantization** |
| **total** | **2841** | |

FP8 compute needs CC >= 8.9 and the 3090 is 8.6, so the *absence* is correct
and expected -- this is the runbook's own FP8 correction observed directly. The
candidate is not "Ampere lacks FP8"; it is that 2841 tests express a known,
static hardware fact as a failure rather than a skip, across seven files in two
directories.

**The "surely someone would have fixed this" objection is answered, and the
answer is the opposite of what it assumed.** Upstream *does* run
`tests/kernels/attention` and `tests/kernels/quantization` in CI -- on an
**L4**, compute capability **8.9**. On 8.9, `fp8e4nv` compiles. There is no
Ampere GPU anywhere in the kernels CI fleet: the NVIDIA hardware is L4 (8.9),
H100/H200 (9.0), B200 (10.0) and one DGX Spark (12.1). See
[the CI fleet](#ci-fleet).

So these failures are not tolerated by upstream -- they are **structurally
invisible** to upstream. That makes the report worth more than it looked, and
it changes its framing from "please clean this up" to "here is a class of
failure your fleet cannot observe."

**Still needed:** search the tracker, then file with C2 and C6 as one test-gate
PR, leading with the CI-visibility argument.

---

<a id="not-candidates"></a>
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
- **`_flashmla_C` shipping without sm_120.** Looks like C5's twin and is not:
  `test_flashmla.py` gates on a library capability query and skips all 144
  tests cleanly on sm_120. Same situation, correct handling. It is the
  *counter-example* that makes C5 a defect rather than a policy, and it belongs
  in the C5 report as the fix pattern.
- **`ir` and `mamba` identical test-for-test on both cards** (1627 and 929).
  This is the control, and it is the reason the disagreements elsewhere are
  believable. Not a finding -- but the first thing to re-check if the harness
  is ever suspected.

## Before filing anything

None of these has been filed. Items 1 and 4 are now satisfied for every
surviving row; items 2, 3 and 5 are not.

1. ~~**Reproduce on upstream `main`, not on `v0.29.0`.**~~ **Done 2026-09-18**
   for C2, C3, C4, C5 and C6 -- see [Runtime reproduction on
   main](#runtime-repro). C8 is not a "has it been fixed" question.
2. **Search the issue tracker first**, including closed issues. **Still
   outstanding for every row.** C2 and C8 describe hardware facts upstream
   demonstrably knows -- the odds someone has already raised the gating question
   are not small. C3 and C5 are the two least likely to be known, because
   neither is visible from upstream's CI.
3. **Reduce to a minimal repro that does not require this lab.** Done for C3
   (`c3_azp_missing_guard.py`) and C4 (`c4_padding_rows.py`,
   `c4_ue8m0_pack_assert.py`). C5 needs no repro -- `cuobjdump` on the published
   wheel is the evidence. **Still outstanding for C2, C6 and C8**, though all
   three are one nodeid each.
4. ~~**State the environment of record.**~~ Above, and it now includes the
   nightly image used for the `main` checks.
5. **Separate the test-suite fixes from the product bugs.** Three threads, not
   one:
   - **Test gates** (C2, C6, C8+C7): one PR against `tests/`, citing C1's
     merged fix as precedent and leading with the CI-visibility argument.
   - **C3**: an issue against the library. Asymmetric INT8 on any Blackwell
     silently routes to a Hopper kernel.
   - **C4** and **C5**: one issue each. C4 is a one-line test fix with a
     measurement behind it; C5 is a build-system selector that drops an
     architecture from a shipped wheel.
6. **C7 is folded and C1 is dead.** Do not re-raise either as its own row.
