#!/usr/bin/env python3
"""Compare saved research_eval runs, with the uncertainty the headline
percentages leave out.

  compare.py --table research-eval-*.json    every run, headline metrics,
                                             Wilson 95% intervals
  compare.py A.json B.json                   A vs B, paired item by item:
                                             exact McNemar p, bootstrap 95%
                                             interval on the difference
  compare.py A.json B.json --flips hit_and_grounded --split held_out
                                             the items that changed, with
                                             both final answers

Runs are paired by item id, so both must come from the same item build
(same --seed, --window and item sets); compare refuses otherwise. Scores are
read as saved: after a scorer change, `research_eval.py --rescore` each file
first.

Why paired: 90 held-out questions give a Wilson interval about +-9 points
wide at 70%, so two runs' intervals overlap for most real differences. The
same 90 questions asked of both models is far more informative than two
independent samples -- only the items where the runs disagree carry signal,
and McNemar's test uses exactly those.
"""
import argparse, json, math, os, random, sys

V2_FLAG = ("held_out2", "two_flag", "fix_cmd", "task", "trap2", "trap2_control")
ROCKY_FLAG = ("rocky_held_out", "rocky_task", "rocky_trap", "rocky_trap_control")
# (label, splits pooled, score key, True if higher is better). A row prints
# only for runs that contain its splits.
HEADLINE = [
    ("held-out right flag, nothing invented", ("held_out",), "hit_and_grounded", True),
    ("seen-tool right flag, nothing invented", ("seen_tool",), "hit_and_grounded", True),
    ("held-out looked up", ("held_out",), "looked_up", True),
    ("trap denied (heuristic)", ("trap",), "denied_heuristic", True),
    ("trap control wrongly denied", ("trap_control",), "denied_heuristic", False),
    ("flag Qs citing an invented flag", ("held_out", "seen_tool", "trap", "trap_control"), "!grounded", False),
    ("flag Qs claiming an unrun lookup", ("held_out", "seen_tool", "trap", "trap_control"),
     "claims_unrun_lookup", False),
    ("no-tool over-trigger", ("no_tool",), "over_trigger", False),
    ("no-tool correct (scorable)", ("no_tool",), "correct", True),
    # item set v2 (research_eval.py --set v2)
    ("new-tool right flag, nothing invented", ("held_out2",), "hit_and_grounded", True),
    ("new-tool looked up", ("held_out2",), "looked_up", True),
    ("two flags both right, nothing invented", ("two_flag",), "hit_and_grounded", True),
    ("fix-the-command right, nothing invented", ("fix_cmd",), "hit_and_grounded", True),
    ("operator task right, nothing invented", ("task",), "hit_and_grounded", True),
    ("asserted fake denied (heuristic)", ("trap2",), "denied_heuristic", True),
    ("asserted real wrongly denied", ("trap2_control",), "denied_heuristic", False),
    ("v2 flag Qs citing an invented flag", V2_FLAG, "!grounded", False),
    ("v2 flag Qs claiming an unrun lookup", V2_FLAG, "claims_unrun_lookup", False),
    # item set rocky (research_eval.py --set rocky)
    ("rocky-tool right flag, nothing invented", ("rocky_held_out",), "hit_and_grounded", True),
    ("rocky-tool looked up", ("rocky_held_out",), "looked_up", True),
    ("farm task right, nothing invented", ("rocky_task",), "hit_and_grounded", True),
    ("rocky asserted fake denied (heuristic)", ("rocky_trap",), "denied_heuristic", True),
    ("rocky asserted real wrongly denied", ("rocky_trap_control",), "denied_heuristic", False),
    ("rocky flag Qs citing an invented flag", ROCKY_FLAG, "!grounded", False),
    ("rocky flag Qs claiming an unrun lookup", ROCKY_FLAG, "claims_unrun_lookup", False),
    # item set promql (research_eval.py --set promql [--promql-catalog])
    ("live question answered correctly", ("promql",), "correct", True),
    ("used the promql tool", ("promql",), "used_promql", True),
    ("stated a live value it did not query", ("promql",), "invented_value", False),
    # item set general (research_eval.py --set general)
    ("general question answered correctly", ("general",), "correct", True),
    ("general question: called a tool", ("general",), "over_trigger", False),
    # item set alert (research_eval.py --set alert)
    ("alert rule passes promtool check", ("alert",), "valid", True),
    ("alert rule fires and stays quiet correctly", ("alert",), "correct", True),
    ("alert answer claims an unrun lookup", ("alert",), "claims_unrun_lookup", False),
    # item set trap3 (research_eval.py --set trap3)
    ("missing/misspelled tool noticed", ("trap3",), "noticed", True),
    ("missing tool: flag denied, tool not noticed", ("trap3",), "flag_denied_only", False),
    ("missing tool's usage fabricated", ("trap3",), "fabricated", False),
    ("trap3 answer claims an unrun lookup", ("trap3",), "claims_unrun_lookup", False),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact p on the discordant pairs: b items only A passes, c
    only B passes. Under no difference each is Binomial(b + c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def values(run, splits, key):
    """{item id: bool} for the items in splits that carry key."""
    neg = key.startswith("!")
    key = key.lstrip("!")
    out = {}
    for r in run["results"]:
        if r["split"] in splits and key in r["score"]:
            v = r["score"][key]
            out[r["id"]] = (not v) if neg else bool(v)
    return out


def load(path):
    d = json.load(open(path))
    d["_path"] = path
    d["_name"] = d.get("label") or os.path.basename(path)
    return d


def check_paired(a, b):
    qa = {r["id"]: r["question"] for r in a["results"]}
    qb = {r["id"]: r["question"] for r in b["results"]}
    shared = set(qa) & set(qb)
    if not shared or any(qa[i] != qb[i] for i in shared):
        sys.exit(f"{a['_name']} and {b['_name']} were not built from the same items; "
                 "a paired comparison would be meaningless")
    return shared


def pct(x):
    return f"{100 * x:5.1f}%"


def table(runs):
    for label, splits, key, higher in HEADLINE:
        if not any(values(d, splits, key) for d in runs):
            continue
        print(f"\n{label}   [{'/'.join(splits)} . {key}]  ({'higher' if higher else 'lower'} is better)")
        for d in runs:
            v = values(d, splits, key)
            if not v:
                continue
            k, n = sum(v.values()), len(v)
            lo, hi = wilson(k, n)
            print(f"  {d['_name']:24s} {pct(k / n)}  [{pct(lo)} - {pct(hi)}]  {k}/{n}")


def pair(a, b, boot=10000, seed=0):
    shared = check_paired(a, b)
    rng = random.Random(seed)
    print(f"A = {a['_name']}   B = {b['_name']}   ({len(shared)} shared items)")
    print(f"{'metric':42s} {'A':>7s} {'B':>7s} {'B-A':>7s}  {'95% CI of B-A':>17s}  "
          f"{'A only':>6s} {'B only':>6s}  {'McNemar p':>9s}")
    for label, splits, key, higher in HEADLINE:
        va, vb = values(a, splits, key), values(b, splits, key)
        ids = sorted(set(va) & set(vb))
        if not ids:
            continue
        xa = [va[i] for i in ids]
        xb = [vb[i] for i in ids]
        n = len(ids)
        only_a = sum(x and not y for x, y in zip(xa, xb))
        only_b = sum(y and not x for x, y in zip(xa, xb))
        diff = (sum(xb) - sum(xa)) / n
        # Paired bootstrap: resample items, keeping each item's pair intact.
        ds = sorted((sum(xb[j] - xa[j] for j in s) / n)
                    for s in ([rng.randrange(n) for _ in range(n)] for _ in range(boot)))
        lo, hi = ds[int(0.025 * boot)], ds[int(0.975 * boot) - 1]
        p = mcnemar_exact(only_a, only_b)
        mark = "*" if p < 0.05 else " "
        print(f"{label[:42]:42s} {pct(sum(xa) / n):>7s} {pct(sum(xb) / n):>7s} {100 * diff:+6.1f}  "
              f"[{100 * lo:+6.1f}, {100 * hi:+6.1f}]  {only_a:6d} {only_b:6d}  {p:9.4f}{mark}")
    print("\n* p < 0.05, uncorrected: across nine metrics expect about one false alarm in two "
          "comparisons by chance alone.")


def flips(a, b, key, split, width):
    check_paired(a, b)
    splits = (split,) if split else tuple({r["split"] for r in a["results"]})
    va, vb = values(a, splits, key), values(b, splits, key)
    ra = {r["id"]: r for r in a["results"]}
    rb = {r["id"]: r for r in b["results"]}
    for direction, want in (("A only", (True, False)), ("B only", (False, True))):
        ids = [i for i in sorted(set(va) & set(vb)) if (va[i], vb[i]) == want]
        print(f"\n==== {key}: {direction} ({len(ids)})")
        for i in ids:
            r = ra[i]
            print(f"\n-- {i}  [{r.get('flag') or r.get('fake_flag') or ''}]  {r['question']}")
            for name, rr in ((a["_name"], ra[i]), (b["_name"], rb[i])):
                calls = ", ".join(f"{c['name']}:{c['outcome']}" for c in rr["run"]["calls"]) or "none"
                fin = (rr["run"]["final"] or f"<{rr['run']['status']}>").replace("\n", " / ")
                print(f"   {name} [calls {calls}]: {fin[:width]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--table", action="store_true", help="per-run rates with intervals")
    ap.add_argument("--flips", metavar="KEY", help="list items whose KEY differs between A and B")
    ap.add_argument("--split", default="", help="restrict --flips to one split")
    ap.add_argument("--width", type=int, default=300, help="chars of each answer --flips shows")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    runs = [load(p) for p in a.runs]
    if a.table:
        table(runs)
    elif len(runs) != 2:
        ap.error("pairwise modes take exactly two runs")
    elif a.flips:
        flips(runs[0], runs[1], a.flips, a.split, a.width)
    else:
        pair(runs[0], runs[1], a.boot)


if __name__ == "__main__":
    main()
