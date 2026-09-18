#!/usr/bin/env python3
"""C4 -- the non-conforming scale is uninitialised padding, not a computed value.

`ref_with_scale_fmt` allocates the reference scale tensor with torch.empty over
the full (E, T, H/128) shape and then writes only rows [e, :nt] for each
expert's token count nt. Because tokens_per_expert is drawn as
randint(low=0, high=T), nt is always strictly less than T, so *every* expert
carries at least one row that is never written.

The test then hands that whole tensor -- written rows and untouched rows alike
-- to transform_sf_into_required_layout, whose pack kernel asserts that every
FP32 value is exponent-only (sign and mantissa clear). Uninitialised device
memory does not satisfy that, and the assert is correct to fire.

This script classifies every non-conforming value as either a real row
(index < nt, computed by the reference) or a padding row (index >= nt, never
written). It never calls the pack kernel, so the CUDA context stays alive and
all cases can be checked in one process.

    python3 c4_padding_rows.py
"""

import sys

import torch

# Default to the pinned harness tree; point C4_TEST_MODULE at a mounted copy of
# main's test file to run the same analysis against upstream.
import importlib  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, "/kernels")
sys.path.insert(0, os.environ.get("C4_TEST_DIR", "/kernels"))
_mod = importlib.import_module(
    os.environ.get(
        "C4_TEST_MODULE", "tests.kernels.moe.test_silu_mul_fp8_quant_deep_gemm"
    )
)
CASES = _mod.CASES
DeepGemmQuantScaleFMT = _mod.DeepGemmQuantScaleFMT
ref_with_scale_fmt = _mod.ref_with_scale_fmt
token_random = _mod.token_random
from vllm.platforms import current_platform  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

GROUP = 128
CONFORM_MASK = 0x807FFFFF  # sign bit + mantissa: must be zero for a power of two


def nonconforming(sf: torch.Tensor) -> torch.Tensor:
    """Boolean mask of values that are NOT exponent-only floats."""
    bits = sf.view(torch.int32)
    return (bits & CONFORM_MASK) != 0


def check_case(E, T, H):
    set_random_seed(42)
    tokens_per_expert = torch.randint(
        low=0, high=T, size=(E,), dtype=torch.int32, device="cuda"
    )
    y = token_random(E, T, 2 * H, tokens_per_expert)
    gate, up = y[..., :H].to(torch.bfloat16), y[..., H:].to(torch.bfloat16)

    _q, _s = ref_with_scale_fmt(
        E, T, H, GROUP, tokens_per_expert, gate, up,
        scale_fmt=DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
    )

    bad = nonconforming(_s)
    if not bad.any():
        return E, T, H, 0, 0, 0
    # Split the offenders by whether their row index is within that expert's
    # token count. Anything at or beyond nt was never written by the reference.
    real = padding = 0
    for e in range(E):
        nt = int(tokens_per_expert[e].item())
        real += int(bad[e, :nt].sum().item())
        padding += int(bad[e, nt:].sum().item())
    return E, T, H, int(bad.sum().item()), real, padding


def main():
    cap = current_platform.get_device_capability()
    print(f"device: {torch.cuda.get_device_name()}  sm_{cap.major}{cap.minor}")
    print(f"support_deep_gemm: {current_platform.support_deep_gemm()}")
    print(f"gate the test uses: has_device_capability(100) = "
          f"{current_platform.has_device_capability(100)}   "
          f"(family100 = {current_platform.is_device_capability_family(100)})")
    print()
    print(f"{'E':>4}{'T':>6}{'H':>7}{'bad':>8}{'in real rows':>14}{'in padding':>12}")
    print("-" * 51)

    tot_real = tot_pad = 0
    seen = set()
    for E, T, H, *_rest in CASES:
        if (E, T, H) in seen:
            continue
        seen.add((E, T, H))
        try:
            e, t, h, bad, real, pad = check_case(E, T, H)
        except torch.OutOfMemoryError:
            print(f"{E:>4}{T:>6}{H:>7}{'OOM':>8}{'-':>14}{'-':>12}")
            continue
        tot_real += real
        tot_pad += pad
        print(f"{e:>4}{t:>6}{h:>7}{bad:>8}{real:>14}{pad:>12}")

    print("-" * 51)
    print(f"{'TOTAL':>17}{tot_real + tot_pad:>8}{tot_real:>14}{tot_pad:>12}")
    print()
    if tot_pad and not tot_real:
        print("VERDICT: every non-conforming scale sits in a padding row that")
        print("         ref_with_scale_fmt allocated with torch.empty and never wrote.")
        print("         The computed scales are all conforming. The kernel assert is correct;")
        print("         the test is feeding it uninitialised memory.")
        return 0
    if tot_real:
        print("VERDICT: some non-conforming scales are in REAL rows -- the reference")
        print("         computation itself produces them. Padding is not the whole story.")
        return 1
    print("VERDICT: nothing non-conforming found -- allocator handed back zeroed pages.")
    print("         This is the nondeterminism the diagnosis predicts, not a refutation.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
