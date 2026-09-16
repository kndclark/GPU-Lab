#!/usr/bin/env python3
"""Diff two kernel-test runs and surface the architecture disagreements.

Usage: kernels/compare.py results/sm_86-quantization.xml results/sm_120-quantization.xml

The interesting output is not the pass count. It is the three ways two runs of
byte-identical test code can disagree:

  MISSING    a test exists on one node and not the other. Parametrisation is
             capability-gated, so the suites are not the same size on both
             cards -- this is where "sm_86 has no FP8 compute" shows up as a
             structural fact rather than as a failure.
  OUTCOME    the same test passes on one card and fails or skips on the other.
             A pass/fail split here is the strongest kind of finding.
  SKIPREASON both skipped, but for different stated reasons.

A skip is evidence, not an absence of evidence: the reason string usually names
the capability that decided it.
"""
import gzip
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def load(path):
    """nodeid -> (outcome, reason). Outcome is pass/fail/error/skip."""
    out = {}
    for case in _parse(path).getroot().iter("testcase"):
        nodeid = f"{case.get('classname', '')}::{case.get('name', '')}"
        outcome, reason = "pass", ""
        for kind, label in (("failure", "fail"), ("error", "error"), ("skipped", "skip")):
            child = case.find(kind)
            if child is not None:
                outcome = label
                reason = (child.get("message") or "").strip().replace("\n", " ")
                break
        out[nodeid] = (outcome, reason)
    return out


def _parse(path):
    """Results are stored gzipped -- 22 MB of XML is 0.3 MB, and this repo
    deploys on push, so uncompressed artefacts would bloat every clone."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            return ET.parse(fh)
    return ET.parse(path)


CONTEXT_KILL = ("unspecified launch failure", "CUDA error: an illegal memory access")


def cascade(path):
    """Find where a killed CUDA context starts poisoning everything after it.

    A device-side assert or illegal access destroys the CUDA context, and every
    later test in the same pytest process then fails with the same message --
    recorded as ordinary failures, indistinguishable from real ones. On the
    night of 2026-09-15 this turned one bad kernel into 1882 bogus MoE failures
    and voided 6758 quantization tests. Any count taken past this index is not
    evidence.

    Returns (index, total, poisoned) or None.
    """
    outcomes = []
    for case in _parse(path).getroot().iter("testcase"):
        bad, msg = False, ""
        for kind in ("failure", "error"):
            child = case.find(kind)
            if child is not None:
                bad, msg = True, (child.get("message") or "")
                break
        outcomes.append((bad, msg))
    hits = [i for i, (bad, m) in enumerate(outcomes)
            if bad and any(k in m for k in CONTEXT_KILL)]
    if not hits:
        return None
    # Anchoring on the FIRST context kill is wrong: a test that forks a
    # subprocess can raise one and recover, and the run carries on healthy for
    # thousands of tests. (tests/kernels/moe has exactly this at index 3508,
    # long before the kill at 6856 that actually ended the run.) So take the
    # earliest point after which nearly everything fails, not the earliest
    # point where a launch failure appears at all.
    for i in hits:
        after = outcomes[i:]
        poisoned = sum(1 for bad, _ in after if bad)
        if poisoned >= 50 and poisoned >= 0.7 * len(after):
            return i, len(outcomes), poisoned
    return None


def short(reason, n=110):
    return reason[:n] + ("..." if len(reason) > n else "")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = load(a_path), load(b_path)
    a_name, b_name = a_path.split("/")[-1], b_path.split("/")[-1]

    def totals(d):
        c = Counter(o for o, _ in d.values())
        return f"{len(d)} tests: {c['pass']} pass, {c['fail']} fail, {c['error']} error, {c['skip']} skip"

    print(f"{a_name}\n  {totals(a)}")
    print(f"{b_name}\n  {totals(b)}")

    for label, path in ((a_name, a_path), (b_name, b_path)):
        c = cascade(path)
        if c:
            i, total, poisoned = c
            print(f"\n  !! {label}: CUDA context killed at test {i} of {total}.")
            print(f"     {poisoned} results after that point are collateral, not findings.")
            print("     Re-run this file alone before trusting anything past that index.")

    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    print(f"\n== MISSING ==  only in {a_name}: {len(only_a)}   only in {b_name}: {len(only_b)}")
    for label, ids, src in ((a_name, only_a, a), (b_name, only_b, b)):
        if not ids:
            continue
        # Group by test function rather than by parameter: one gated parameter
        # axis produces hundreds of ids and one fact.
        fams = Counter(i.split("[")[0] for i in ids)
        print(f"  only in {label}:")
        for fam, n in fams.most_common(12):
            print(f"    {n:5}  {fam}")
        if len(fams) > 12:
            print(f"    ... and {len(fams) - 12} more test functions")

    shared = set(a) & set(b)
    diffs = [(i, a[i], b[i]) for i in sorted(shared) if a[i][0] != b[i][0]]
    print(f"\n== OUTCOME ==  {len(diffs)} of {len(shared)} shared tests disagree")
    fams = Counter(i.split("[")[0] + f"  [{x[0]} -> {y[0]}]" for i, x, y in diffs)
    for fam, n in fams.most_common(20):
        print(f"  {n:5}  {fam}")
    if diffs:
        print("\n  first few, with reasons:")
        for i, (ao, ar), (bo, br) in diffs[:5]:
            print(f"    {i}")
            print(f"      {a_name}: {ao}  {short(ar)}")
            print(f"      {b_name}: {bo}  {short(br)}")

    both_skip = [(i, a[i][1], b[i][1]) for i in sorted(shared)
                 if a[i][0] == b[i][0] == "skip" and a[i][1] != b[i][1]]
    print(f"\n== SKIPREASON ==  {len(both_skip)} tests skipped on both, for different reasons")
    seen = Counter((short(x, 70), short(y, 70)) for _, x, y in both_skip)
    for (x, y), n in seen.most_common(10):
        print(f"  {n:5}  {a_name}: {x}\n         {b_name}: {y}")

    print("\n== SKIP REASONS BY NODE ==")
    for label, d in ((a_name, a), (b_name, b)):
        c = Counter(short(r, 70) for o, r in d.values() if o == "skip")
        print(f"  {label}:")
        for r, n in c.most_common(10):
            print(f"    {n:5}  {r or '(no reason given)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
