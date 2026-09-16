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
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def load(path):
    """nodeid -> (outcome, reason). Outcome is pass/fail/error/skip."""
    out = {}
    for case in ET.parse(path).getroot().iter("testcase"):
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
