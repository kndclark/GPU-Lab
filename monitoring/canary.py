#!/usr/bin/env python3
"""Probe the serving plane for faults no scraped metric can show.

Runs on each node (gpu-lab-canary.timer) against the vLLM engines that node
can reach, and writes Prometheus text to the exporter's textfile directory,
where gpu_exporter.py serves it. It never starts a model: an engine that does
not answer /metrics is skipped, and llama-swap is never contacted.

Two things it measures, both found the hard way:

  KV capacity   vllm:cache_config_info carries num_gpu_blocks as a LABEL, so
                PromQL cannot compare it with anything. The script parses it
                and publishes it next to the floor from canary.yml.
  Adapter       a LoRA adapter whose weights were skipped (wrong key names,
                unreadable file) is still listed in /v1/models and still
                answers -- as the base model. The only proof is behavioural:
                the same greedy prompts to adapter and base must differ.

A completion that times out is reported too: a pipeline-parallel deadlock
leaves /metrics answering while generation never returns, and the canary's own
request is what makes an otherwise idle pool show running > 0.
"""
import argparse
import json
import os
import re
import socket
import sys
import time
import urllib.request

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = "/var/lib/gpu-lab/textfile/canary.prom"


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def post(url, body, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def endpoints(prom_cfg, node):
    """vLLM engines this node should probe, read from the scrape config so the
    addresses are written in one place. The pool's single endpoint is the
    desktop's; the laptop does not probe it (it holds no front door)."""
    found = []
    for job in prom_cfg.get("scrape_configs", []):
        if job["job_name"] not in ("vllm", "vllm-pool"):
            continue
        for sc in job.get("static_configs", []):
            labels = sc.get("labels", {})
            mine = labels.get("node") == node or (
                job["job_name"] == "vllm-pool" and node == "desktop")
            if mine:
                found.extend((t, labels) for t in sc["targets"])
    return found


def kv_blocks(metrics):
    m = re.search(r'^vllm:cache_config_info\{[^}]*num_gpu_blocks="(\d+)"', metrics, re.M)
    return int(m.group(1)) if m else None


def esc(v):
    return str(v).replace("\\", "\\\\").replace('"', '\\"')


def lbl(**kw):
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in kw.items()) + "}"


def probe(target, labels, cfg, node, log):
    out = []
    base_url = f"http://{target}"
    try:
        metrics = get(f"{base_url}/metrics")
        models = json.loads(get(f"{base_url}/v1/models"))["data"]
    except Exception as e:
        log(f"{target}: not answering ({type(e).__name__}) -- skipped, nothing started")
        return out
    model = labels.get("model", "?")
    ident = {"node": labels.get("node", node), "model": model}

    # ---- KV capacity ----
    blocks = kv_blocks(metrics)
    root = next((m.get("root") for m in models if not m.get("parent")), None)
    key = f"pool@{root}" if model == "pool" else f"{model}@{ident['node']}"
    floor = cfg.get("kv_floor_blocks", {}).get(key)
    if blocks is not None and floor is not None:
        out.append(f"gpulab_canary_kv_blocks{lbl(**ident)} {blocks}")
        out.append(f"gpulab_canary_kv_blocks_floor{lbl(**ident)} {floor}")
        log(f"{target}: KV {blocks} blocks, floor {floor} ({key}) "
            f"{'ok' if blocks >= floor else 'BELOW FLOOR'}")
    else:
        log(f"{target}: KV blocks={blocks}, no floor for {key} -- not judged")

    # ---- adapters ----
    prompts, n = cfg["adapter_prompts"], cfg.get("max_tokens", 64)
    tmo = cfg.get("request_timeout_s", 90)

    def ask(name, prompt):
        r = post(f"{base_url}/v1/chat/completions", {
            "model": name, "temperature": 0, "max_tokens": n,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}, tmo)
        return r["choices"][0]["message"].get("content") or ""

    adapters = [(m["id"], m["parent"]) for m in models if m.get("parent")]
    if model in cfg.get("no_generation_models", []):
        return out                    # embedding model: no chat route to probe
    if not adapters:
        # No adapter to compare; still prove generation returns, which is the
        # deadlock probe.
        try:
            ask(models[0]["id"], prompts[0])
            ok = 1
        except Exception as e:
            log(f"{target}: base completion FAILED ({type(e).__name__})")
            ok = 0
        out.append(f"gpulab_canary_probe_ok{lbl(**ident)} {ok}")
    for adapter, parent in adapters:
        try:
            differs = 0
            for p in prompts:
                if ask(adapter, p) != ask(parent, p):
                    differs = 1
                    break
            out.append(f"gpulab_canary_probe_ok{lbl(**ident)} 1")
            out.append(f"gpulab_canary_adapter_differs{lbl(**ident, adapter=adapter)} {differs}")
            log(f"{target}: adapter {adapter} vs {parent}: "
                f"{'differs' if differs else 'IDENTICAL -- served as the base'}")
        except Exception as e:
            log(f"{target}: adapter probe FAILED ({type(e).__name__})")
            out.append(f"gpulab_canary_probe_ok{lbl(**ident)} 0")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True, choices=["desktop", "laptop"])
    ap.add_argument("--prom-config", default=os.path.join(HERE, "prometheus.yml"))
    ap.add_argument("--config", default=os.path.join(HERE, "canary.yml"))
    ap.add_argument("--out", help="write the textfile here (default: print only)")
    args = ap.parse_args()
    socket.setdefaulttimeout(120)

    cfg = yaml.safe_load(open(args.config))
    prom = yaml.safe_load(open(args.prom_config))
    log = lambda m: print(m, file=sys.stderr)

    series = []
    for target, labels in endpoints(prom, args.node):
        series.extend(probe(target, labels, cfg, args.node, log))

    lines = [
        "# HELP gpulab_canary_last_run_timestamp_seconds When the canary last finished",
        "# TYPE gpulab_canary_last_run_timestamp_seconds gauge",
        f"gpulab_canary_last_run_timestamp_seconds {time.time():.0f}",
    ] + series
    text = "\n".join(lines) + "\n"
    if args.out:
        tmp = args.out + ".tmp"       # not *.prom: the exporter must never read half a file
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, args.out)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
