#!/usr/bin/env python3
"""Codex against a minimal loop of our own, on a sealed task set.

Each run gets a fresh copy of a task's fixture/ as its working directory and
the task's prompt.md, verbatim, as its only instruction; then the task's
check.sh grades the final message and the workspace: PASS or FAIL.

Two harnesses, one model server, one shell:
  codex  `codex exec` over the Responses API: Codex's own prompt and its
         exec_command tool, configured as ~/.codex/config.toml is
  loop   research_eval.run_item's loop over /v1/chat/completions: the
         training bash tool, prompt.md as the only message, and vLLM's tool
         parser. run_item parses <tool_call> JSON itself, which misses the
         XML calls Lightning makes.

Every command either harness runs goes through Codex's Linux sandbox
(workspace-write, network on): read-only root, writes only in the workspace
and a /tmp of its own. agent_ab_bwrap.sh stands in for bwrap so the shell
sees the real machine (GPU nodes, host pids and /proc) but not the set, the
results or ssh keys. Every run uses the same cwd, so nothing in the context
names the task or the arm. The shell runs as its own uid (--agent-user), which
closes what opening 1 found open: the docker socket (group docker) and signals
to the lab user's services. `preflight` asserts that from inside the sandbox,
and the model server's health is still checked before and after each run.
Neither harness sets sampling: both get the server's defaults (Lightning's
generation_config: temperature 1.0, top_p 0.95).

The runner prints task id, arm and verdict only, never a prompt, answer or
FAIL reason: those land in --out, and reading them is part of opening the
set. Record every opening in the set's OPENINGS.md.

  agent_ab.py preflight            # the sandbox's walls, without opening the set
  agent_ab.py selftest --set DIR --out DIR --tag before
  agent_ab.py run      --set DIR --out DIR --modes off,on
  agent_ab.py summary  --out DIR
"""
import argparse, json, math, os, pwd, random, shutil, signal, subprocess, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
from tools import TOOLS  # noqa: E402

BASH = [t for t in TOOLS if t["function"]["name"] == "bash"]
MODELS = {"g6u": "lightning-g6u", "base": "lightning"}
HARNESSES = ("codex", "loop")
WORK = "/tmp/agentab/work"      # the one cwd every run sees
TMP = "/tmp/agentab/tmp"        # the sandbox's /tmp, emptied between runs
# One directory for everything the sandbox reads but may not write, and the
# agent user must read some of it, so it cannot be the lab user's ~/.cache
# (0700). bin/lab install makes it, owned by the lab user, 0755.
STATE = "/var/lib/agentab"
# Codex passes over a bwrap on PATH that the sandbox could write (the
# workspace, /tmp) and uses the system one, which hides nothing, so the shim
# must not live in either.
SHIM = STATE + "/shim"
# Codex's home: one fixed path, recreated per run, so nothing on PATH (its
# apply_patch helper lives here) names a task or an arm. It must stay outside
# the hidden paths, or that helper disappears.
CODEX_HOME = STATE + "/codex-home"
SANDBOX_HOME = STATE + "/sandbox-home"  # the loop's "codex sandbox"
OUT_CAP = 20000                 # characters of command output the loop returns, head and tail
CMD_TIMEOUT = 120               # seconds per command in the loop
MAX_CALLS = 60                  # loop tool calls per run; Codex has no such limit, only the task timeout
CODEX = "codex"                 # set from --codex
AGENT_USER = "labagent"         # set from --agent-user: the user every sandboxed shell runs as
SECRETS = ("KEY", "SECRET", "TOKEN")  # env names the loop's shell drops; codex_config() makes Codex drop them too


def lab_key():
    return subprocess.run(["sudo", "sed", "-n", "s/^LITELLM_MASTER_KEY=//p", "/etc/gpu-lab/litellm.env"],
                          capture_output=True, text=True, check=True).stdout.strip()


def state_dir():
    """Where the shim and Codex's homes live, made by bin/lab install because
    it takes root: /var/lib, so both users can read it and only one can write."""
    if not os.path.isdir(STATE):
        sys.exit(f"no {STATE}: run 'bin/lab install' on this node first")
    return STATE


def agent_uid():
    """The uid of the user the sandboxed shell runs as, and proof it exists."""
    p = subprocess.run(["id", "-u", "--", AGENT_USER], capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"no user {AGENT_USER!r}: run 'bin/lab install' on this node first")
    return int(p.stdout.strip())


def sandbox_env(a):
    """Environment for Codex and for the loop's sandboxed shell."""
    state_dir()
    if agent_uid() == os.getuid():
        sys.exit(f"{AGENT_USER!r} is the user running this: the shell needs a uid of its own")
    os.makedirs(SHIM, exist_ok=True)
    shim = os.path.join(SHIM, "bwrap")  # a copy: Codex runs a link's target, far from the lists
    if os.path.lexists(shim):
        os.remove(shim)
    shutil.copy(os.path.join(HERE, "agent_ab_bwrap.sh"), shim)
    paths = [os.path.realpath(os.path.expanduser(p)) for p in a.hide.split(":") + [a.set, a.out] if p]
    hide = sorted({p for p in paths if not any(p.startswith(q + "/") for q in paths)})
    with open(os.path.join(SHIM, "hide"), "w") as f:  # files, not env: see agent_ab_bwrap.sh
        f.write("".join(p + "\n" for p in hide))
    with open(os.path.join(SHIM, "tmp"), "w") as f:
        f.write(TMP)
    with open(os.path.join(SHIM, "user"), "w") as f:
        f.write(AGENT_USER + "\n")
    os.makedirs(SANDBOX_HOME, exist_ok=True)
    return dict(os.environ, PATH=SHIM + ":" + os.environ["PATH"], CODEX_HOME=SANDBOX_HOME)


def preflight(env):
    """Refuse to start unless one sandboxed command shows every wall standing:
    the hidden paths empty, /tmp private, a uid of its own, no key in its
    environment, the container socket shut, the lab user's processes and their
    environments beyond reach -- and the machine the sealed set promises still
    visible, host pids and GPUs. When Codex passes the wrapper over, every uid
    check fails loudly here instead of mid-run."""
    hide = open(os.path.join(SHIM, "hide")).read().split() + [SHIM]
    probe = "/tmp/agentab-preflight"
    want = agent_uid()
    fresh_work("")
    out = run_cmd(f'for p in {" ".join(hide)}; do n=$(ls -A "$p" 2>/dev/null | wc -l); '  # counts, never names
                  f'[ "$n" = 0 ] || echo "seen $p: $n entries"; done; touch {probe} 2>/dev/null; '
                  f'echo "uid=$(id -u)"; '
                  f'if [ ! -S /run/docker.sock ]; then echo socket=absent; '
                  f'elif ! command -v docker >/dev/null; then echo socket=noclient; '
                  f'elif docker ps >/dev/null 2>&1; then echo socket=open; else echo socket=shut; fi; '
                  f'kill -0 {os.getpid()} 2>/dev/null && echo runner=reachable || echo runner=separate; '
                  f'echo "secrets=$(env | grep -Eic "KEY|SECRET|TOKEN" || true)"; '
                  f'echo "runnerenv=$(cat /proc/{os.getpid()}/environ 2>/dev/null | wc -c)"; '
                  f'echo "pids=$(ps -e 2>/dev/null | wc -l)"; '
                  f'echo "gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"', env, 120)
    clean()
    leaked = os.path.exists(probe)
    if leaked:
        subprocess.run(["sudo", "-n", "rm", "-f", probe], check=False)  # the agent user's file, in a sticky /tmp
    got = {}
    for line in out.splitlines():
        k, _, v = line.partition("=")
        if k in ("uid", "socket", "runner", "secrets", "runnerenv", "pids", "gpus"):
            got[k] = v.strip()
    bad = []
    if leaked:
        bad.append("a write to its /tmp reached the host's")
    if "\nseen " in "\n" + out:
        bad.append("a hidden path is not empty")
    if not out.endswith("[exit code 0]"):
        bad.append("the probe did not finish")
    if got.get("uid") != str(want):
        bad.append(f"the shell runs as uid {got.get('uid')}, not {want} ({AGENT_USER})")
    if got.get("socket") == "open":
        bad.append("the container socket answers")
    if got.get("runner") != "separate":
        bad.append("the lab user's processes can be signalled")
    # The wrapper carries the env across the uid change, so what it drops is
    # worth asserting, and the key is also in the runner's own environment.
    if int(got.get("secrets") or 0):
        bad.append("a key, secret or token name reached the shell's environment")
    if int(got.get("runnerenv") or 0):
        bad.append("the runner's environment is readable")
    if int(got.get("pids") or 0) < 20:
        bad.append("ps cannot see the machine")      # the promised shell, not a wall
    if int(got.get("gpus") or 0) < 1:
        bad.append("nvidia-smi finds no GPU")
    if bad:
        sys.exit("preflight: " + "; ".join(bad) + f"\n{out}")
    print(f"preflight: uid {want} ({AGENT_USER}), container socket {got['socket']}, lab user's processes "
          f"separate, {len(hide)} paths hidden, /tmp private, no key in its environment; "
          f"{got['pids']} host pids and {got['gpus']} GPU(s) visible")


def healthy(a):
    try:
        with urllib.request.urlopen(a.base + "/health", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def wait_healthy(a, secs=600):
    t0 = time.time()
    while not healthy(a):
        if time.time() - t0 > secs:
            sys.exit(f"model server at {a.base} unhealthy for {secs}s: stopping; rerun resumes")
        time.sleep(15)


def codex_config(model, thinking, base):
    return "\n".join([
        f'model = "{model}"', 'model_provider = "lab"',
        f'model_reasoning_effort = "{"medium" if thinking else "none"}"', 'model_reasoning_summary = "none"',
        "model_context_window = 131072", "model_auto_compact_token_limit = 117964",
        'web_search = "disabled"', "check_for_update_on_startup = false",
        "[analytics]", "enabled = false", "[feedback]", "enabled = false",
        # plugins: each fresh CODEX_HOME clones the plugin marketplace from the network in the
        # background, still writing after Codex exits (it broke run_codex's rmtree)
        "[features]", "multi_agent = false", "goals = false", "plugins = false",
        "[sandbox_workspace_write]", "network_access = true",
        # without this the lab key (LAB_KEY) reaches Codex's shell, and the model can print it
        "[shell_environment_policy]", 'inherit = "all"', "ignore_default_excludes = false",
        'exclude = ["*KEY*", "*SECRET*", "*TOKEN*"]',
        "[model_providers.lab]", 'name = "GPU lab"', f'base_url = "{base}/v1"',
        'wire_api = "responses"', 'env_key = "LAB_KEY"', ""])


def wipe(d):
    """Remove a directory, with root for whatever the agent user left in it."""
    shutil.rmtree(d, ignore_errors=True)
    if os.path.exists(d):
        subprocess.run(["sudo", "-n", "rm", "-rf", d], check=True)


def share_work():
    """The shell runs as a uid of its own, so the workspace and the /tmp that
    stands in for the host's are shared with that uid, not handed to it:
    between the sandbox exiting and grading, Codex removes the mount targets
    it made in the workspace as the runner, which a directory the runner no
    longer owns refuses. An ACL instead, so both uids keep rwx on whatever the
    other creates -- the default entries carry that down into new directories
    -- while 0750 keeps the fixture off every other account on the machine."""
    if not shutil.which("setfacl"):
        sys.exit("no setfacl: 'apt install acl' on this node ('bin/lab install' reports it too)")
    me = pwd.getpwuid(os.getuid()).pw_name
    spec = ",".join(f"{d}u:{u}:rwx" for u in (me, AGENT_USER) for d in ("", "d:"))
    for d in (WORK, TMP):
        os.chmod(d, 0o750)
    subprocess.run(["setfacl", "-R", "-m", spec, WORK, TMP], check=True)


def fresh_work(fixture):
    for d in (WORK, TMP):
        wipe(d)
    os.makedirs(TMP)
    shutil.copytree(fixture, WORK, symlinks=True) if os.path.isdir(fixture) else os.makedirs(WORK)
    share_work()


def sweep():
    """Nothing of the agent user outlives its run. By user, not by name: it
    cannot match the runner, which is the point of a uid of its own."""
    subprocess.run(["sudo", "-n", "pkill", "-9", "-u", AGENT_USER], check=False)


def clean():
    sweep()
    for d in (WORK, TMP):
        wipe(d)
    if os.path.isdir(os.path.dirname(WORK)) and not os.listdir(os.path.dirname(WORK)):
        os.rmdir(os.path.dirname(WORK))


def grade(a, tid, answer, ws):
    p = subprocess.run(["bash", os.path.join("tasks", tid, "check.sh"), answer, ws], cwd=a.set,
                       capture_output=True, text=True, timeout=300, env=dict(os.environ, PROM_URL=a.prom))
    line = (p.stdout.strip().splitlines() or [""])[-1]
    return {"pass": p.returncode == 0 and line == "PASS", "line": line, "stderr": p.stderr[-2000:]}


# ---------------------------------------------------------------- the loop

def run_cmd(cmd, env, timeout):
    p = subprocess.Popen([CODEX, "sandbox", "-c", 'sandbox_mode="workspace-write"',
                          "-c", "sandbox_workspace_write.network_access=true", "--", "bash", "-lc", cmd],
                         cwd=WORK, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, start_new_session=True)
    try:
        out, rc, note = p.communicate(timeout=timeout)[0], p.returncode, ""
    except subprocess.TimeoutExpired:
        # The group kill reaches Codex and no further: the shell below it runs
        # as another uid, in its own session, holding the pipe this waits on.
        # Signalling it is the agent user's own business, so sweep by uid.
        os.killpg(p.pid, signal.SIGKILL)
        sweep()
        out, rc, note = p.communicate()[0], 124, f"\n[timed out after {timeout:.0f}s]"
    out = out.decode(errors="replace")
    if len(out) > OUT_CAP:
        out = out[:OUT_CAP // 2] + f"\n[... {len(out) - OUT_CAP} characters cut ...]\n" + out[-OUT_CAP // 2:]
    return f"{out}{note}\n[exit code {rc}]"


def chat(a, key, body, timeout):
    req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_loop(a, key, model, thinking, prompt, env, deadline):
    shell_env = {k: v for k, v in env.items() if not any(s in k.upper() for s in SECRETS)}
    messages = [{"role": "user", "content": prompt}]
    turns, calls, status, final = [], [], "error", ""
    try:
        while True:
            left = deadline - time.time()
            if left <= 0:
                status = "timeout"
                break
            body = {"model": model, "messages": messages, "tools": BASH,
                    "chat_template_kwargs": {"enable_thinking": thinking}}
            try:
                res = chat(a, key, body, left)
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode(errors="replace")
                status = "context_exhausted" if e.code == 400 and "context" in detail.lower() else "infra"
                turns.append({"error": f"HTTP {e.code} {detail}"})
                break
            except (TimeoutError, urllib.error.URLError) as e:
                status = "timeout" if time.time() >= deadline else "infra"
                turns.append({"error": repr(e)})
                break
            ch = res["choices"][0]
            m = ch["message"]
            tcs = m.get("tool_calls") or []
            turns.append({"usage": res.get("usage"), "finish_reason": ch.get("finish_reason"),
                          "reasoning_chars": len(m.get("reasoning") or m.get("reasoning_content") or ""),
                          "calls": len(tcs)})
            if not tcs:
                status = "truncated" if ch.get("finish_reason") == "length" else "answered"
                final = m.get("content") or ""
                break
            if len(calls) >= MAX_CALLS:
                status = "call_limit"
                break
            messages.append({"role": "assistant", "content": m.get("content"), "tool_calls": tcs})
            for tc in tcs:
                try:
                    cmd = json.loads(tc["function"]["arguments"]).get("command")
                except (ValueError, AttributeError):
                    cmd = None
                if tc["function"]["name"] != "bash" or not isinstance(cmd, str):
                    result = "error: the only tool is bash, called with a string `command`"
                else:
                    result = run_cmd(cmd, shell_env, max(1, min(CMD_TIMEOUT, deadline - time.time())))
                calls.append({"name": tc["function"]["name"], "command": cmd})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    except Exception as e:  # recorded, never silently dropped
        turns.append({"error": repr(e)})
    return {"status": status, "final": final, "calls": calls, "turns": turns, "messages": messages}


# ---------------------------------------------------------------- codex

def run_codex(a, key, model, thinking, prompt, env, timeout, rundir):
    home = CODEX_HOME
    shutil.rmtree(home, ignore_errors=True)
    os.makedirs(home)
    with open(os.path.join(home, "config.toml"), "w") as f:
        f.write(codex_config(model, thinking, a.base))
    last = os.path.join(rundir, "codex-last.txt")
    log = os.path.join(rundir, "codex.jsonl")
    with open(log, "w") as out:
        p = subprocess.run(["timeout", "-k", "10", str(timeout), CODEX, "exec", "--json", "--ephemeral",
                            "--skip-git-repo-check", "--strict-config", "-s", "workspace-write", "-C", WORK,
                            "-o", last, prompt],
                           env=dict(env, CODEX_HOME=home, LAB_KEY=key), stdin=subprocess.DEVNULL,
                           stdout=out, stderr=subprocess.STDOUT)
    calls, errors = [], []
    for line in open(log, errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        it = e.get("item") or {}
        if e.get("type") == "item.completed" and it.get("type") == "command_execution":
            calls.append({"name": "exec_command", "command": it.get("command")})
        if e.get("type") in ("turn.failed", "error"):
            errors.append(json.dumps(e)[:500])
    final = open(last).read() if os.path.exists(last) else ""
    if p.returncode in (124, 137):
        status = "timeout"
    elif errors or p.returncode:
        # a server that refused or dropped the request is infra; Codex's own failures count
        status = "infra" if any(s in " ".join(errors) for s in ("onnect", " 50", "disconnected")) else "failed"
    else:
        status = "answered"
    shutil.rmtree(home)  # config is codex_config(); the rest is Codex's own state
    return {"status": status, "final": final, "calls": calls, "rc": p.returncode, "errors": errors}


# ---------------------------------------------------------------- runner

def task_ids(a):
    root = os.path.join(a.set, "tasks")
    return sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "meta.json")))


def arms(modes):
    return [f"{h}-{m}-{mode}" for mode in modes for h in HARNESSES for m in MODELS]


def done_keys(path):
    if not os.path.exists(path):
        return set()
    return {(r["task"], r["arm"]) for r in map(json.loads, open(path)) if r["status"] != "infra"}


def cmd_run(a):
    key, env = lab_key(), sandbox_env(a)
    preflight(env)
    results = os.path.join(a.out, "results.jsonl")
    done = done_keys(results)
    order = task_ids(a)
    random.Random(a.seed).shuffle(order)
    for mode in a.modes.split(","):  # all of one mode before the next
        mode_arms = arms([mode])
        for i, tid in enumerate(order):
            meta = json.load(open(os.path.join(a.set, "tasks", tid, "meta.json")))
            prompt = open(os.path.join(a.set, "tasks", tid, "prompt.md")).read()
            for arm in mode_arms[i % len(mode_arms):] + mode_arms[:i % len(mode_arms)]:
                if (tid, arm) in done:
                    continue
                harness, mshort, _ = arm.split("-")
                rundir = os.path.join(a.out, "runs", tid, arm)
                if os.path.exists(rundir):  # an infra failure being rerun
                    shutil.rmtree(rundir)
                os.makedirs(rundir)
                wait_healthy(a)
                fresh_work(os.path.join(a.set, "tasks", tid, "fixture"))
                t0 = time.time()
                if harness == "codex":
                    r = run_codex(a, key, MODELS[mshort], mode == "on", prompt, env, meta["timeout_s"], rundir)
                else:
                    r = run_loop(a, key, MODELS[mshort], mode == "on", prompt, env, t0 + meta["timeout_s"])
                secs = round(time.time() - t0, 1)
                sweep()                   # Codex's --die-with-parent covers the usual exit, this the rest
                if not healthy(a):  # the server died under this run: not the model's fail
                    r["status"] = "infra"
                answer = os.path.join(rundir, "answer.txt")
                with open(answer, "w") as f:
                    f.write(r["final"] or "")
                g = grade(a, tid, answer, WORK)
                shutil.copytree(WORK, os.path.join(rundir, "ws"), symlinks=True)
                with open(os.path.join(rundir, "run.json"), "w") as f:
                    json.dump(dict(r, grade=g), f, indent=1)
                row = {"task": tid, "arm": arm, "pass": g["pass"], "status": r["status"],
                       "calls": len(r["calls"]), "secs": secs, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                with open(results, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{tid} {arm:15s} {'PASS' if g['pass'] else 'FAIL'} {r['status']:9s} "
                      f"calls={len(r['calls'])} {secs}s", flush=True)
    clean()


def cmd_selftest(a):
    path = os.path.join(a.out, f"selftest-{a.tag}.jsonl")
    ok = 0
    with open(path, "w") as f:
        for tid in task_ids(a):
            st = os.path.join(a.set, "tasks", tid, "selftest")
            ws = os.path.join(st, "ws")
            fresh_work(ws if os.path.isdir(ws) else os.path.join(a.set, "tasks", tid, "fixture"))
            g = grade(a, tid, os.path.join(st, "answer.txt"), WORK)
            ok += g["pass"]
            f.write(json.dumps({"task": tid, **g, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    clean()
    print(f"selftest {a.tag}: {ok}/{len(task_ids(a))} reference answers pass; failing ids:",
          " ".join(r["task"] for r in map(json.loads, open(path)) if not r["pass"]) or "none")


def mcnemar(b, c):
    """Exact two-sided McNemar p-value on b and c discordant pairs."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def cmd_summary(a):
    rows = [json.loads(l) for l in open(os.path.join(a.out, "results.jsonl"))]
    void = set()
    for tag in ("before", "after"):
        p = os.path.join(a.out, f"selftest-{tag}.jsonl")
        if os.path.exists(p):
            void |= {r["task"] for r in map(json.loads, open(p)) if not r["pass"]}
    if os.path.exists(os.path.join(a.out, "void.txt")):  # manual voids, one "tNN reason" per line
        void |= {l.split()[0] for l in open(os.path.join(a.out, "void.txt")) if l.strip()}
    by = {}
    for r in rows:
        if r["task"] not in void and r["status"] != "infra":
            by.setdefault(r["arm"], {})[r["task"]] = r["pass"]
    print(f"void: {' '.join(sorted(void)) or 'none'}")
    for arm in sorted(by):
        v = by[arm]
        print(f"{arm:15s} {sum(v.values()):2d}/{len(v)} pass")
    pairs = []
    for mode in ("off", "on"):
        for m in MODELS:
            pairs.append((f"codex-{m}-{mode}", f"loop-{m}-{mode}"))
        for h in HARNESSES:
            pairs.append((f"{h}-g6u-{mode}", f"{h}-base-{mode}"))
    for x, y in pairs:
        if x in by and y in by:
            common = sorted(set(by[x]) & set(by[y]))
            b = sum(by[x][t] and not by[y][t] for t in common)
            c = sum(by[y][t] and not by[x][t] for t in common)
            print(f"{x} vs {y}: {b} vs {c} discordant of {len(common)}, exact McNemar p={mcnemar(b, c):.3f}")


def cmd_preflight(a):
    """The walls on their own: no task, no prompt, nothing opened."""
    preflight(sandbox_env(a))


def main():
    global CODEX, AGENT_USER
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("preflight", "run", "selftest", "summary"))
    ap.add_argument("--set", default=os.path.expanduser("~/sealed/agent-v1"))
    ap.add_argument("--out", default="", help="required, except for preflight")
    ap.add_argument("--base", default="http://localhost:8300")
    ap.add_argument("--prom", default="http://lab-desktop:9090")
    ap.add_argument("--modes", default="off,on")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--tag", default="before")
    ap.add_argument("--codex", default=shutil.which("codex") or os.path.expanduser("~/.local/bin/codex"))
    ap.add_argument("--hide", default="~/sealed:~/.ssh", help="colon-separated; the set and --out are added")
    ap.add_argument("--agent-user", default=AGENT_USER, help="the user every sandboxed shell runs as")
    a = ap.parse_args()
    if not a.out:
        if a.cmd != "preflight":
            ap.error("--out is required")
        a.out = os.path.join(state_dir(), "preflight")  # nothing lands here; the hide list wants a path
    a.base, a.set, a.out = a.base.rstrip("/"), os.path.abspath(a.set), os.path.abspath(a.out)
    CODEX, AGENT_USER = a.codex, a.agent_user
    os.makedirs(a.out, exist_ok=True)
    {"preflight": cmd_preflight, "run": cmd_run,
     "selftest": cmd_selftest, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
