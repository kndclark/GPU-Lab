#!/usr/bin/env python3
"""C3 -- the azp INT8 entry point has no SM120 capability guard.

Two vLLM entry points, the same INT8 tensors, one device. On sm_120:

    cutlass_scaled_mm      -> RuntimeError naming SM120 and suggesting FP8
    cutlass_scaled_mm_azp  -> RuntimeError "Error Internal", no diagnosis

On sm_86 both return a tensor. That asymmetry is the finding: the capability
is absent on consumer Blackwell either way, and only one of the two paths
knows how to say so.

    python3 c3_azp_missing_guard.py

No pytest, no vLLM test tree, no weights. Run it inside gpu-lab:kernels.
"""

import sys

import torch
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

M, N, K = 64, 32, 32  # N,K multiples of 16: cutlass_scaled_mm_azp asserts it


def make_int8_inputs():
    """One set of tensors, used unchanged by both entry points."""
    torch.manual_seed(0)
    dev = "cuda"
    # a is row-major (M, K); b must be COLUMN-major (K, N), which is why it is
    # built transposed. Getting this wrong trips a conformality check in
    # scaled_mm_entry.cu long before dispatch, and the run tells you nothing.
    aq_i8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=dev)
    bq_i8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=dev).t()
    scale_a = torch.rand((M, 1), dtype=torch.float32, device=dev)
    scale_b = torch.rand((N, 1), dtype=torch.float32, device=dev).t().contiguous()
    # azp_adj is the column sum of b, exactly as the upstream test builds it.
    azp_adj = bq_i8.to(dtype=torch.int32).sum(dim=0, keepdim=True, dtype=torch.int32)
    azp = torch.randint(-128, 127, (M, 1), dtype=torch.int32, device=dev)
    assert bq_i8.stride(0) == 1, "b must be column-major"
    return aq_i8, bq_i8, scale_a, scale_b, azp_adj, azp


def attempt(label, fn):
    try:
        out = fn()
        print(f"  {label:<26} -> returned {tuple(out.shape)} {out.dtype}")
        return None
    except Exception as exc:  # noqa: BLE001 -- the exception *is* the result
        line = " ".join(str(exc).strip().split())
        print(f"  {label:<26} -> {type(exc).__name__}: {line[:400]}")
        return line


def main():
    cap = current_platform.get_device_capability()
    print(f"device   : {torch.cuda.get_device_name(0)}")
    print(f"capability: {cap.to_int()}  (has>=100={current_platform.has_device_capability(100)}, "
          f"family100={current_platform.is_device_capability_family(100)})")
    print()

    a, b, sa, sb, azp_adj, azp = make_int8_inputs()

    print("same int8 tensors, two entry points:")
    plain = attempt(
        "cutlass_scaled_mm",
        lambda: ops.cutlass_scaled_mm(a, b, sa, sb, torch.bfloat16),
    )
    azp_err = attempt(
        "cutlass_scaled_mm_azp",
        lambda: ops.cutlass_scaled_mm_azp(a, b, sa, sb, torch.bfloat16, azp_adj, azp),
    )

    print()
    if plain is None and azp_err is None:
        print("VERDICT: both paths work here -- this is not an sm_120 node.")
        return 0
    if plain is not None and azp_err is not None:
        named = "SM120" in plain or "Int8 not supported" in plain
        generic = "Error Internal" in azp_err
        if named and generic:
            print("VERDICT: one bug, two faces. The capability is absent on both paths;")
            print("         only cutlass_scaled_mm has the guard that says so.")
            print("         cutlass_scaled_mm_azp reaches the GEMM caller and dies generically.")
            return 0
    print("VERDICT: the two paths did NOT split as expected -- re-read the output above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
