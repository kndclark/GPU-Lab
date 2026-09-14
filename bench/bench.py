#!/usr/bin/env python3
"""
Baseline benchmark for the serving plane.

Measures TTFT and decode rate over a fixed prompt set, with repeats, and reports
spread rather than a single number -- a lone sample cannot distinguish a
regression from thermal noise, and on a throttling node noise is the expected
state, not the edge case.

Records driver version and GPU temperature alongside every run, because a result
that cannot be attributed to a known hardware state is not a baseline.

Stdlib only: the desktop runs Python 3.14 with no pip.
"""
import argparse, json, statistics, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

# Fixed prompt set. Never edit these without resetting the baseline -- comparing
# across different prompts is comparing nothing.
PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number iteratively. Code only.",
    "Explain the difference between a process and a thread in three sentences.",
    "Write a SQL query that finds the second-highest salary in an employees table.",
    "Refactor this to be idiomatic Python: result = []\nfor i in range(len(xs)):\n    if xs[i] % 2 == 0:\n        result.append(xs[i] * 2)",
    "What does the CAP theorem say? Answer in two sentences.",
]


def gpu_state():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=driver_version,temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().split(", ")
        return {"driver": out[0], "temp_c": float(out[1]),
                "power_w": float(out[2]), "sm_mhz": float(out[3])}
    except Exception as e:
        return {"error": str(e)}


def one_request(base, model, prompt, max_tokens):
    """Stream a completion; return (ttft_s, decode_tok_s, completion_tokens)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    usage_tokens = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage_tokens = chunk["usage"].get("completion_tokens")
            for ch in chunk.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    ntok += 1
    total = time.perf_counter() - t0
    if ttft is None:
        raise RuntimeError("no content tokens received")
    ntok = usage_tokens or ntok
    decode_s = total - ttft
    return ttft, (ntok - 1) / decode_s if decode_s > 0 and ntok > 1 else 0.0, ntok


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarise(name, xs, unit):
    if not xs:
        return f"  {name:<22} (no samples)"
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    mean = statistics.mean(xs)
    cv = (sd / mean * 100) if mean else 0.0
    return (f"  {name:<22} p50 {pct(xs,50):7.3f}  p95 {pct(xs,95):7.3f}  "
            f"mean {mean:7.3f}  sd {sd:6.3f}  cv {cv:5.1f}%  {unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8101")
    ap.add_argument("--model", default="qwen3-coder")
    ap.add_argument("--repeats", type=int, default=5, help="passes over the prompt set")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    print(f"=== baseline: {a.label or a.model} ===")
    print(f"  endpoint {a.base}  repeats {a.repeats}  "
          f"max_tokens {a.max_tokens}  concurrency {a.concurrency}")
    before = gpu_state()
    print(f"  gpu before: {before}")

    for _ in range(a.warmup):
        try:
            one_request(a.base, a.model, PROMPTS[0], 32)
        except Exception as e:
            print(f"  warmup failed: {e}", file=sys.stderr)
            return 1

    ttfts, rates, temps = [], [], []
    t_start = time.perf_counter()
    for r in range(a.repeats):
        if a.concurrency == 1:
            for p in PROMPTS:
                ttft, rate, _ = one_request(a.base, a.model, p, a.max_tokens)
                ttfts.append(ttft); rates.append(rate)
        else:
            with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
                futs = [ex.submit(one_request, a.base, a.model, p, a.max_tokens)
                        for p in PROMPTS for _ in range(a.concurrency)]
                for f in futs:
                    ttft, rate, _ = f.result()
                    ttfts.append(ttft); rates.append(rate)
        temps.append(gpu_state().get("temp_c", 0))
        print(f"  pass {r+1}/{a.repeats} done ({len(ttfts)} samples, "
              f"gpu {temps[-1]:.0f}C)")
    wall = time.perf_counter() - t_start
    after = gpu_state()

    print(f"\n  samples {len(ttfts)}   wall {wall:.1f}s")
    print(summarise("TTFT", ttfts, "s"))
    print(summarise("decode rate", rates, "tok/s"))
    print(f"  gpu after : {after}")
    print(f"  temp rise : {after.get('temp_c',0) - before.get('temp_c',0):+.0f} C "
          f"(max during run {max(temps) if temps else 0:.0f} C)")
    if max(temps or [0]) >= 83:
        print("  WARNING: node crossed 83 C -- treat these numbers as thermally poisoned")

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"label": a.label, "model": a.model,
                       "max_tokens": a.max_tokens, "concurrency": a.concurrency,
                       "repeats": a.repeats, "samples": len(ttfts),
                       "ttft_p50": pct(ttfts,50), "ttft_p95": pct(ttfts,95),
                       "ttft_mean": statistics.mean(ttfts),
                       "rate_p50": pct(rates,50), "rate_p95": pct(rates,95),
                       "rate_mean": statistics.mean(rates),
                       "gpu_before": before, "gpu_after": after,
                       "temp_max": max(temps) if temps else None}, f, indent=2)
        print(f"  wrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
