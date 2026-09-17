"""Minimal standalone repro: DeepGEMM UE8M0 pack assert, vLLM v0.29.0, sm_120.

    python3 minimal_c4.py good   -> scales that ARE powers of two: succeeds
    python3 minimal_c4.py bad    -> one scale that is not:         aborts the CUDA context

No pytest, no vLLM test tree, no model weights. Derived from
tests/kernels/moe/test_silu_mul_fp8_quant_deep_gemm.py:275, which fails on
consumer Blackwell with:

    Assertion failed: deep_gemm/include/deep_gemm/impls/smxx_layout.cuh:131,
    condition: (values[j] & 0x807fffffu) == 0

DeepGEMM packs four FP32 scales into one int32 by shifting each exponent into a
byte, so every scale must be an exact power of two. The assert enforces exactly
that. Run the two modes in separate processes: the failing one aborts the
context, so nothing after it in the same process is trustworthy.
"""
import struct, sys
import torch
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import transform_sf_into_required_layout, is_deep_gemm_e8m0_used

MN, K, RECIPE, NUM_GROUPS = 4, 128, (1, 128, 128), 1


def describe(v):
    b = struct.unpack(">I", struct.pack(">f", float(v)))[0]
    return f"{float(v)!r:>22}  bits=0x{b:08x}  sign+mantissa=0x{b & 0x807FFFFF:06x}  pow2={(b & 0x807FFFFF) == 0}"


mode = sys.argv[1] if len(sys.argv) > 1 else "bad"
cap = current_platform.get_device_capability()
print(f"device: {torch.cuda.get_device_name()}  sm_{cap.major}{cap.minor}")
print(f"support_deep_gemm: {current_platform.support_deep_gemm()}   "
      f"e8m0 cast enabled: {is_deep_gemm_e8m0_used()}")
if not current_platform.support_deep_gemm():
    print("DeepGEMM unsupported here -- this path never runs on this card.")
    sys.exit(0)

if mode == "good":
    # Exact powers of two, which is what the pack kernel's contract requires.
    sf = torch.full((NUM_GROUPS, MN, K // 128), 0.015625, device="cuda", dtype=torch.float32)
else:
    # The value the upstream test's own bf16 reference ceiling can produce:
    # exp2(ceil(log2(s))) evaluated entirely in bfloat16 does not reliably land
    # on an exact power of two once converted back to float32.
    sf = torch.full((NUM_GROUPS, MN, K // 128), 0.015625, device="cuda", dtype=torch.float32)
    sf[0, 0, 0] = 0.00872802734375          # bits 0x3c0f0000 -- mantissa 0x0f0000

print(f"\nsf {tuple(sf.shape)} float32, mn={MN}, k={K}, recipe={RECIPE}, num_groups={NUM_GROUPS}, is_sfa=True")
for i, v in enumerate(sf.flatten().tolist()):
    print(f"  [{i}] {describe(v)}")

print(f"\ncalling transform_sf_into_required_layout(...) -- mode={mode}")
try:
    out = transform_sf_into_required_layout(sf=sf, mn=MN, k=K, recipe=RECIPE,
                                            num_groups=NUM_GROUPS, is_sfa=True)
    torch.cuda.synchronize()
    print(f"  returned {tuple(out.shape)} {out.dtype}: {out.flatten()[:4].tolist()}")
    print("\nRESULT: completed without tripping the assert.")
except Exception as exc:
    print(f"\nRESULT: {type(exc).__name__}: {str(exc)[:200]}")
    print("The device-side assert fired and the CUDA context is now unusable.")
    sys.exit(1)
