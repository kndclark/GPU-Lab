#!/usr/bin/env python3
"""N concurrent long-context requests. Unique prompts, so no prefix-cache help."""
import argparse, json, random, statistics, threading, time, urllib.request

MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"


def one(base, model, nums, max_tokens, out, i):
    prompt = ("Here is a list of reference numbers: " + ", ".join(str(n) for n in nums)
              + ". Summarise in one short sentence what kind of data this is.")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "min_tokens": max_tokens, "ignore_eos": True,
                       "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; n = 0; ptoks = ctoks = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            p = line[6:]
            if p == "[DONE]":
                break
            ch = json.loads(p)
            if ch.get("choices") and ch["choices"][0].get("delta", {}).get("content"):
                n += 1
                if first is None:
                    first = time.perf_counter() - t0
            if ch.get("usage"):
                ptoks = ch["usage"]["prompt_tokens"]; ctoks = ch["usage"]["completion_tokens"]
    end = time.perf_counter() - t0
    out[i] = {"ttft_s": first, "decode_tok_s": (ctoks - 1) / (end - first) if first and end > first else None,
              "prompt_tokens": ptoks, "completion_tokens": ctoks, "total_s": end}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://10.10.0.1:8200")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--numbers", type=int, default=950)
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()
    # A FIXED seed makes every run send identical prompts, which the prefix
    # cache then serves: TTFT collapses and the run measures nothing. Seed
    # from the clock so each run is cold.
    rng = random.Random(time.time_ns())
    out = [None] * a.concurrency
    ts = [threading.Thread(target=one, args=(a.base, a.model,
          rng.sample(range(1000, 10000), a.numbers), a.max_tokens, out, i))
          for i in range(a.concurrency)]
    t0 = time.perf_counter()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.perf_counter() - t0
    ttft = sorted(r["ttft_s"] for r in out)
    dec = sorted(r["decode_tok_s"] for r in out)
    comp = sum(r["completion_tokens"] for r in out)
    print(f"  c={a.concurrency}  prompt_tokens={out[0]['prompt_tokens']}  wall={wall:.1f}s")
    print(f"  ttft p50={statistics.median(ttft):.3f}s  min={ttft[0]:.3f}  max={ttft[-1]:.3f}")
    print(f"  per-request decode p50={statistics.median(dec):.3f} tok/s")
    print(f"  aggregate end-to-end={comp/wall:.1f} tok/s  ({comp} completion tokens)")


if __name__ == "__main__":
    main()
