#!/usr/bin/env python3
"""Regime C: single-stream prefill scaling against the pooled endpoint.

Measures TTFT as a function of prompt length. Prompts are built from unique
random 4-digit numbers, regenerated for every sample, so prefix caching cannot
serve any of them -- the point is to time real prefill work.

max_tokens is tiny so the measurement is dominated by prefill, and TTFT is read
off the stream's first content chunk rather than the total latency.
"""
import argparse, json, random, statistics, sys, time, urllib.request

MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"


def ttft(base, model, numbers, max_tokens=8, timeout=300):
    """One streamed request. Returns (ttft_s, prompt_tokens, total_s)."""
    prompt = ("Here is a list of reference numbers: "
              + ", ".join(str(n) for n in numbers)
              + ". Reply with the single word OK.")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    ptoks = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if first is None and chunk.get("choices"):
                d = chunk["choices"][0].get("delta", {})
                if d.get("content") or d.get("role") == "assistant" and d.get("content") is not None:
                    first = time.perf_counter() - t0
            if chunk.get("usage"):
                ptoks = chunk["usage"]["prompt_tokens"]
    return first, ptoks, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://10.10.0.1:8200")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    # counts chosen to land near the prompt_tokens of the recorded regime C
    ap.add_argument("--counts", default="20,70,160,325,480,650,850,975")
    a = ap.parse_args()

    rng = random.Random(20260922)
    rows = []
    for count in [int(c) for c in a.counts.split(",")]:
        samples = []
        try:
            for _ in range(a.samples):
                nums = rng.sample(range(1000, 10000), count)
                t, ptoks, tot = ttft(a.base, a.model, nums)
                samples.append((t, ptoks, tot))
                time.sleep(0.4)
        except Exception as e:
            print(f"  {count} numbers: skipped ({e})", flush=True)
            continue
        ts = sorted(s[0] for s in samples)
        row = {
            "numbers": count,
            "prompt_tokens": samples[0][1],
            "ttft_s": round(statistics.median(ts), 4),
            "ttft_min_s": round(ts[0], 4),
            "ttft_max_s": round(ts[-1], 4),
            "samples": a.samples,
            "prefill_tok_s_cumulative": round(samples[0][1] / statistics.median(ts), 1),
        }
        rows.append(row)
        print(f"  {row['prompt_tokens']:5d} tok  ttft={row['ttft_s']:.3f}s "
              f"[{row['ttft_min_s']:.3f}-{row['ttft_max_s']:.3f}]  "
              f"cum={row['prefill_tok_s_cumulative']} tok/s", flush=True)

    print("\n  marginal rate between adjacent points:")
    for i in range(1, len(rows)):
        dt = rows[i]["ttft_s"] - rows[i - 1]["ttft_s"]
        dn = rows[i]["prompt_tokens"] - rows[i - 1]["prompt_tokens"]
        rows[i]["marginal_tok_s"] = round(dn / dt, 1) if dt > 0 else None
        print(f"  {rows[i-1]['prompt_tokens']:5d} -> {rows[i]['prompt_tokens']:5d}: "
              f"{rows[i]['marginal_tok_s']} tok/s")

    if a.out:
        json.dump({"label": a.label, "series": rows}, open(a.out, "w"), indent=2)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
