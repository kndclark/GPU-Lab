#!/usr/bin/env python3
"""Thermal/power guard for a training run. Samples, streams, and aborts.

qlora.py's GpuMonitor only accumulates samples and reports them after
trainer.train() returns, so a run that climbs has nothing watching it and no
way to be stopped. This is the missing half: it can kill the container.

Two independent triggers, because temperature alone is not the question:

  1. The card reports it is being actively slowed (hw_thermal_slowdown). This
     is authoritative -- it is the hardware saying so, not an inference from a
     number we picked.
  2. A conservative absolute ceiling, defaulting below the card's own target
     temperature rather than the 95 C constant qlora.py assumes, which is a
     desktop figure and wrong for the laptop (target 87 C).

Sustained read failure also aborts: being blind during a thermal stress run is
itself the unsafe condition.
"""
import argparse
import json
import signal
import subprocess
import sys
import time

QUERY = (
    "temperature.gpu,power.draw,enforced.power.limit,memory.used,utilization.gpu,"
    "clocks_throttle_reasons.hw_thermal_slowdown,"
    "clocks_throttle_reasons.sw_thermal_slowdown,"
    "clocks_throttle_reasons.hw_power_brake_slowdown"
)


def target_temp():
    """The card's own thermal target, when the driver will say."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-q", "-d", "TEMPERATURE"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        for line in out.stdout.splitlines():
            if "GPU Target Temperature" in line:
                value = line.split(":")[1].strip().split()[0]
                return int(value)
    except Exception:
        pass
    return None


def sample():
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    parts = [p.strip() for p in out.stdout.strip().split("\n")[0].split(",")]
    def number(raw):
        try:
            return float(raw)
        except ValueError:
            return float("nan")

    return {
        "temp_c": number(parts[0]),
        "power_w": number(parts[1]),
        "power_limit_w": number(parts[2]),
        "mem_mib": number(parts[3]),
        "util_pct": number(parts[4]),
        "hw_thermal": parts[5].lower() == "active",
        "sw_thermal": parts[6].lower() == "active",
        "hw_power_brake": parts[7].lower() == "active",
    }


def abort(container, reason, history, report_path):
    print(f"\n!!! ABORT: {reason}", flush=True)
    killed = subprocess.run(
        ["sudo", "docker", "kill", container], capture_output=True, text=True
    )
    print(f"!!! docker kill {container}: rc={killed.returncode} {killed.stderr.strip()}",
          flush=True)
    if report_path:
        with open(report_path, "w") as fh:
            json.dump({"aborted": True, "reason": reason, "samples": history[-60:]},
                      fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="gpu-lab-training")
    ap.add_argument("--max-temp", type=float, default=None,
                    help="absolute ceiling; defaults to the card's target minus 3 C")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--violations", type=int, default=3,
                    help="consecutive breaches before aborting, to ride out spikes")
    ap.add_argument("--report", default=None)
    ap.add_argument("--wait", type=float, default=0,
                    help="seconds to wait for the container to appear before giving up")
    # For a saturation run, throttling is the measurement, not the emergency:
    # aborting on it guarantees we never observe whether the card settles into a
    # stable throttled equilibrium. The card's own hardware slowdown and
    # shutdown remain in force underneath either way.
    ap.add_argument("--allow-throttle", action="store_true",
                    help="record throttle events instead of aborting on them")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop the run cleanly after this many hours")
    args = ap.parse_args()

    target = target_temp()
    if args.max_temp is None:
        args.max_temp = (target - 3) if target else 84.0

    print(f"watchdog: ceiling {args.max_temp} C "
          f"(card target {target if target else 'unknown'} C), "
          f"abort after {args.violations} consecutive breaches, "
          f"container {args.container}", flush=True)

    history = []
    breaches = 0
    read_failures = 0
    started = time.time()
    seen = False

    # ramp.sh SIGTERMs this process when a rung ends, which otherwise kills it
    # between samples and loses the report entirely.
    def on_term(signum, frame):
        if args.report:
            with open(args.report, "w") as fh:
                json.dump({"aborted": False, "terminated": True,
                           "samples": history}, fh, indent=2)
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    while True:
        try:
            s = sample()
            read_failures = 0
        except Exception as exc:
            read_failures += 1
            print(f"watchdog: read failed ({read_failures}): {exc}", flush=True)
            if read_failures >= 10:
                abort(args.container, f"blind for {read_failures} consecutive polls",
                      history, args.report)
                return 2
            time.sleep(args.interval)
            continue

        s["t"] = round(time.time() - started, 1)
        history.append(s)

        throttling = s["hw_thermal"] or s["sw_thermal"]
        over = s["temp_c"] >= args.max_temp
        fatal = over or (throttling and not args.allow_throttle)
        breaches = breaches + 1 if fatal else 0

        flags = "".join([
            "H" if s["hw_thermal"] else ".",
            "S" if s["sw_thermal"] else ".",
            "P" if s["hw_power_brake"] else ".",
        ])
        print(f"[{s['t']:7.1f}s] {s['temp_c']:5.1f}C "
              f"{s['power_w']:6.1f}/{s['power_limit_w']:.0f}W "
              f"{s['util_pct']:3.0f}% {s['mem_mib']:7.0f}MiB throttle={flags}"
              + (f"  BREACH {breaches}/{args.violations}" if breaches else ""),
              flush=True)

        if breaches >= args.violations:
            why = (f"temp {s['temp_c']} C >= ceiling {args.max_temp} C" if over
                   else "hardware thermal slowdown active")
            abort(args.container, why, history, args.report)
            return 1

        if args.max_hours and (time.time() - started) >= args.max_hours * 3600:
            abort(args.container, f"reached the {args.max_hours} h cap", history, args.report)
            return 3

        running = subprocess.run(
            ["sudo", "docker", "inspect", "-f", "{{.State.Running}}", args.container],
            capture_output=True, text=True,
        ).stdout.strip() == "true"
        if running:
            seen = True
        elif seen or time.time() - started > args.wait:
            print("watchdog: container is not running, exiting", flush=True)
            if args.report:
                with open(args.report, "w") as fh:
                    json.dump({"aborted": False, "samples": history}, fh, indent=2)
            return 0

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
