#!/usr/bin/env python3
"""Summarise ramp runs from watchdog samples, correctly.

ramp.sh's inline summary takes max(draw) and max(limit) independently across
every sample, which produces two lies at once: it compares numbers from
different moments (yielding "119 W of 110 W enforced"), and it includes idle
samples taken while the model loads, when Dynamic Boost advertises a high
ceiling precisely because nothing is drawing against it.

Power-limited is a per-sample question -- was draw at the ceiling *at that
moment* -- and only samples under real load can answer it.

  ./ramp_report.py runs/ramp-A [runs/ramp-B ...]
"""
import argparse
import glob
import json
import os

LOAD_THRESHOLD = 50.0
PINNED_FRACTION = 0.95


def finite(x):
    return x == x and x is not None


def analyse(path):
    samples = json.load(open(path)).get("samples", [])
    loaded = [s for s in samples if s.get("util_pct", 0) >= LOAD_THRESHOLD]
    if not loaded:
        return None

    with_limit = [s for s in loaded if finite(s.get("power_limit_w"))]
    pinned = [s for s in with_limit
              if s["power_w"] >= PINNED_FRACTION * s["power_limit_w"]]
    throttled = [s for s in loaded if s.get("hw_thermal") or s.get("sw_thermal")]
    limits = sorted({round(s["power_limit_w"]) for s in with_limit})

    return {
        "loaded": len(loaded),
        "pinned_pct": 100.0 * len(pinned) / len(with_limit) if with_limit else float("nan"),
        "limit_lo": limits[0] if limits else float("nan"),
        "limit_hi": limits[-1] if limits else float("nan"),
        "draw_mean": sum(s["power_w"] for s in loaded) / len(loaded),
        "draw_max": max(s["power_w"] for s in loaded),
        "temp_max": max(s["temp_c"] for s in loaded),
        "vram_max": max(s["mem_mib"] for s in loaded),
        "throttled": len(throttled),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()

    header = (f"{'run / rung':<34}{'load':>5}{'pinned':>8}{'limit W':>10}"
              f"{'draw avg/max':>14}{'temp':>6}{'VRAM MiB':>10}{'throttled':>10}")
    print(header)
    print("-" * len(header))

    for d in args.dirs:
        label = os.path.basename(d.rstrip("/"))
        for f in sorted(glob.glob(os.path.join(d, "*.watchdog.json"))):
            rung = os.path.basename(f).replace(".watchdog.json", "")
            a = analyse(f)
            name = f"{label}/{rung}"
            if a is None:
                print(f"{name:<34}{'-':>5}  (no samples under load)")
                continue
            print(f"{name:<34}{a['loaded']:>5}{a['pinned_pct']:>7.0f}%"
                  f"{a['limit_lo']:>5.0f}-{a['limit_hi']:<4.0f}"
                  f"{a['draw_mean']:>7.0f}/{a['draw_max']:<6.0f}"
                  f"{a['temp_max']:>5.0f}C{a['vram_max']:>10.0f}"
                  f"{a['throttled']:>10}")

        summary = os.path.join(d, "summary.txt")
        if os.path.exists(summary):
            for line in open(summary):
                if "OOM" in line or "ABORTED" in line or "FAILED" in line:
                    print(f"{label + '/' + line.split()[0]:<34}{line.split(None, 1)[1].strip()}")


if __name__ == "__main__":
    main()
