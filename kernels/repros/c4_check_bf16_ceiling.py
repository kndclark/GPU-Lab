"""Does the bf16 UE8M0 ceiling itself produce non-powers-of-two?

Reproduces the FLOAT32_CEIL_UE8M0 reference exactly as the upstream test's
do_quant() computes it -- every intermediate in bfloat16 -- for the shape that
actually failed, (E=1, T=4, H=128), and checks the invariant the pack kernel
asserts. This is the one link in the C4 chain that was inferred rather than seen.
"""
import struct, torch
G = 128
torch.manual_seed(0)
tot = bad = 0
for E, T, H in [(1, 1, 128), (1, 4, 128), (2, 4, 256), (8, 16, 512), (17, 31, 768)]:
    y = torch.randn((E, T, 2 * H), device="cuda", dtype=torch.bfloat16)
    g, u = y[..., :H], y[..., H:]
    act = (g.to(torch.float32) / (1.0 + torch.exp(-g.to(torch.float32)))).to(torch.bfloat16)
    x = (act * u).reshape(-1, G).to(torch.bfloat16)
    eps = torch.tensor([1e-10], device="cuda", dtype=torch.bfloat16)
    one = torch.tensor([1.0], device="cuda", dtype=torch.bfloat16)
    fp8max = torch.tensor([448.0], device="cuda", dtype=torch.bfloat16)
    amax = x.abs().amax(dim=1).clamp(min=eps)
    s = amax * (one / fp8max)
    # the FLOAT32_CEIL_UE8M0 ceiling, entirely in bfloat16, as upstream writes it
    sc = torch.exp2(torch.ceil(torch.log2(s).to(torch.bfloat16)).to(torch.bfloat16)).to(torch.bfloat16)
    b = sc.to(torch.float32).contiguous().view(torch.int32)
    v = (b & 0x807FFFFF) != 0
    n = int(v.sum()); tot += sc.numel(); bad += n
    flag = "  <-- VIOLATES" if n else ""
    print(f"  E={E:<4} T={T:<5} H={H:<5} {n:>5}/{sc.numel():<7} non-power-of-two{flag}")
    if n:
        i = int(v.nonzero()[0][0]); val = float(sc.flatten()[i])
        bb = struct.unpack(">I", struct.pack(">f", val))[0]
        print(f"        e.g. {val!r}  bits=0x{bb:08x}  sign+mantissa=0x{bb & 0x807FFFFF:06x}")
print(f"\ntotal: {bad}/{tot} scales from the bf16 ceiling are not exact powers of two")
print("CONFIRMED: the bf16 reference ceiling is the source." if bad else
      "NOT the source: the bf16 ceiling is exact at these shapes, so the bad scale\n"
      "reaching the pack kernel must originate elsewhere in the test path.")
