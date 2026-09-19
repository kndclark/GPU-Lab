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

# ---- 7. the Rust tools must still build and pass their own tests ----
#
# This section exists because the gate could not see Rust at all, and the hole
# was found the hard way: a crate that did not compile sat in the tree while
# every other check reported green. Nothing here globs -- same reason as the
# rest of this file -- so a new crate is a deliberate line, not a surprise.
#
# cargo is deliberately looked up by hand as well as on PATH. A git hook does
# not necessarily inherit a login shell's environment, and rustup installs to
# ~/.cargo/bin, so relying on PATH alone would turn "cargo is right there" into
# a silent skip.
RUST_CRATES = ["kernels/sm-differ"]


def find_cargo():
    from shutil import which

    return which("cargo") or next(
        (
            c
            for c in [os.path.expanduser("~/.cargo/bin/cargo"), "/usr/local/bin/cargo"]
            if os.path.exists(c)
        ),
        None,
    )


# `cargo test` rather than `cargo check`: it costs almost nothing more once the
# compilation is paid for (0.09 s against 0.26 s warm here, because check does
# not link but also does not run anything), and a crate that compiles while its
# own tests fail is not something to deploy either.
#
# The budget matters. Warm, this is instant; on a node with no target/ directory
# it is a full cold build of every dependency, and a pre-push hook that silently
# compiles for five minutes is its own kind of failure. A timeout is reported,
# not swallowed.
CARGO_BUDGET_S = 120

for crate in RUST_CRATES:
    manifest = os.path.join(REPO, crate, "Cargo.toml")
    if not os.path.exists(manifest):
        continue  # the crate is gone; so is the reason to check it
    cargo = find_cargo()
    if cargo is None:
        warns.append(
            f"{crate}: cargo not found, so it was NOT built or tested. "
            f"This node cannot verify Rust -- push from one that can, or "
            f"install rustup here."
        )
        continue
    try:
        r = subprocess.run(
            [cargo, "test", "--quiet", "--manifest-path", manifest],
            capture_output=True,
            text=True,
            timeout=CARGO_BUDGET_S,
        )
    except subprocess.TimeoutExpired:
        warns.append(
            f"{crate}: cargo test exceeded {CARGO_BUDGET_S}s and was stopped -- "
            f"probably a cold build. Run it yourself before trusting this push."
        )
        continue
    if r.returncode == 0:
        oks.append(f"{crate} builds and its tests pass")
    else:
        # Name the compiler error, or the test, by name. "test failed, to
        # rerun pass --test it" sends the reader back to cargo to find out
        # what broke; the point of a gate message is that it does not.
        # Under --quiet the per-test lines are suppressed, so a failing test
        # is identified by its captured-output header instead.
        blob = r.stderr + r.stdout
        detail = re.findall(r"^error\[[^\]]+\]: .*$", blob, re.M)
        detail += [f"test {m} failed" for m in re.findall(r"^---- (\S+) stdout ----$", blob, re.M)]
        if not detail:
            detail = [
                ln.strip()
                for ln in blob.splitlines()
                if ln.strip().startswith("error:") and "could not compile" not in ln
            ]
        detail = detail[:3]
        fails.append(f"{crate}: cargo test failed -- " + " | ".join(detail or ["see cargo output"]))

for m in oks:
    print(f"  ok    {m}")
for m in warns:
    print(f"  warn  {m}")
for m in fails:
    print(f"  FAIL  {m}")
print(f"\n  {len(oks)} ok, {len(warns)} warnings, {len(fails)} failures")
sys.exit(1 if fails else 0)
