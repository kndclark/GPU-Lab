#!/usr/bin/env python3
"""Validate the lab's configuration before a push deploys it.

In this repo a push IS a deploy: origin has a second push URL whose post-receive
hook runs `checkout -f main` on the desktop. So CI that runs after the push is
too late -- by the time it goes red the serving node has already taken the
change. This runs as a pre-push hook instead.

Exit 0 = safe to push. Exit 1 = something would break on the other side.
"""
import os
import re
import subprocess
import sys

try:
    import yaml
except ImportError:
    sys.exit("check: PyYAML is not installed; cannot validate configs")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails, warns, oks = [], [], []


def rel(p):
    return os.path.relpath(p, REPO)


def load(path):
    full = os.path.join(REPO, path)
    if not os.path.exists(full):
        return None
    try:
        data = yaml.safe_load(open(full))
        oks.append(f"{path} parses")
        return data
    except Exception as e:
        fails.append(f"{path}: {e}")
        return None


# ---- 1. the control script must at least be syntactically runnable ----
lab = os.path.join(REPO, "bin", "lab")
if os.path.exists(lab):
    r = subprocess.run(["bash", "-n", lab], capture_output=True, text=True)
    if r.returncode:
        fails.append(f"bin/lab: bash syntax error: {r.stderr.strip()}")
    else:
        oks.append("bin/lab: syntax clean")

# ---- 2. every config must parse ----
litellm = load("serving/litellm-config.yaml")
prom = load("monitoring/prometheus.yml")
swap_desktop = load("nodes/desktop/llama-swap.yaml")
swap_laptop = load("nodes/laptop/llama-swap.yaml")
load("monitoring/docker-compose.yml")

# Which llama-swap config answers on which address. Both nodes listen on :8080.
BY_HOST = {
    "127.0.0.1": ("desktop", swap_desktop),
    "localhost": ("desktop", swap_desktop),
    "10.10.0.1": ("desktop", swap_desktop),
    "10.10.0.2": ("laptop", swap_laptop),
}


def served(cfg):
    return set((cfg or {}).get("models", {}).keys())


# ---- 3. every model LiteLLM publishes must exist on the node it points at ----
# This is the check that earns its keep: a typo here is a 404 at request time on
# a model the front door claims to serve, and nothing else catches it.
declared = set()
if litellm:
    for entry in litellm.get("model_list", []):
        name = entry.get("model_name")
        declared.add(name)
        params = entry.get("litellm_params", {}) or {}
        api_base = params.get("api_base", "")
        upstream = str(params.get("model", "")).split("/", 1)[-1]
        m = re.match(r"https?://([^:/]+):(\d+)", api_base)
        if not m:
            fails.append(f"litellm '{name}': unparseable api_base {api_base!r}")
            continue
        host, port = m.group(1), m.group(2)
        if host not in BY_HOST:
            warns.append(f"litellm '{name}': api_base host {host} is not a known node")
            continue
        node, cfg = BY_HOST[host]
        if cfg is None:
            fails.append(f"litellm '{name}': no llama-swap config found for the {node}")
        elif upstream not in served(cfg):
            fails.append(
                f"litellm '{name}': upstream '{upstream}' is not served by the "
                f"{node} (it serves: {', '.join(sorted(served(cfg))) or 'nothing'})"
            )
        else:
            oks.append(f"litellm '{name}' -> {node}:{port} serves '{upstream}'")

    # ---- 4. fallbacks must name models that actually exist ----
    for fb in (litellm.get("litellm_settings", {}) or {}).get("fallbacks", []) or []:
        for src, targets in fb.items():
            if src not in declared:
                fails.append(f"fallback source '{src}' is not a declared model_name")
            for t in targets:
                if t not in declared:
                    fails.append(f"fallback target '{t}' is not a declared model_name")
                elif t == src:
                    fails.append(f"fallback '{src}' points at itself")
            if src in declared and all(t in declared for t in targets):
                oks.append(f"fallback {src} -> {', '.join(targets)}")

# ---- 5. Prometheus must scrape the ports llama-swap actually proxies ----
if prom:
    proxied = {}
    for node, cfg in (("desktop", swap_desktop), ("laptop", swap_laptop)):
        for mname, mcfg in ((cfg or {}).get("models", {}) or {}).items():
            m = re.match(r"https?://[^:]+:(\d+)", str(mcfg.get("proxy", "")))
            if m:
                proxied.setdefault(node, {})[m.group(1)] = mname
    for job in prom.get("scrape_configs", []):
        if job.get("job_name") != "vllm":
            continue
        for sc in job.get("static_configs", []):
            for target in sc.get("targets", []):
                thost, _, tport = target.partition(":")
                node = BY_HOST.get(thost, (None, None))[0]
                if node is None:
                    warns.append(f"prometheus target {target}: unknown host")
                elif tport not in proxied.get(node, {}):
                    fails.append(
                        f"prometheus scrapes {target} but the {node}'s llama-swap "
                        f"proxies no model on :{tport}"
                    )
                else:
                    oks.append(f"prometheus {target} -> {node} {proxied[node][tport]}")

# ---- 6. installed unit drift (only meaningful on a real node) ----
unit = "/etc/systemd/system/llama-swap.service"
if os.path.exists(unit):
    for line in open(unit):
        if line.startswith("ExecStart="):
            m = re.search(r"--config\s+(\S+)", line)
            if m and not os.path.exists(m.group(1)):
                fails.append(
                    f"installed unit points at {m.group(1)}, which does not exist "
                    f"-- run 'lab install' (this is what a restructure breaks)"
                )
            elif m:
                oks.append(f"installed unit config exists ({rel(m.group(1))})")

for m in oks:
    print(f"  ok    {m}")
for m in warns:
    print(f"  warn  {m}")
for m in fails:
    print(f"  FAIL  {m}")
print(f"\n  {len(oks)} ok, {len(warns)} warnings, {len(fails)} failures")
sys.exit(1 if fails else 0)
