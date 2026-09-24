#!/usr/bin/env python3
"""Research dataset v2: the v1 records unchanged, plus the two gaps the
held-out eval (bench/research_eval.py) found in the v1 adapter.

1. Thinking-off turns. Qwen3's template renders an empty <think></think>
   only on the last assistant turn, so every v1 tool call was trained with
   no think block before it. enable_thinking=false puts that empty block in
   the generation prompt, and in v1 it only ever preceded a final answer --
   so thinking-off, the adapter wrote "Based on `jq --help`" without making
   the call (71% of flag questions, 0% in default mode). A third of the
   tool-using records are marked thinking "off"; qlora.py renders them with
   the empty block before each tool call, masked as prompt. The trained
   tokens are identical; only the context they follow changes.

2. Failure honesty. With default thinking, 10 of 138 answers still credited
   a lookup nothing executed: 7 after web_search failed, 3 after a refused
   call. v1 never showed a failed tool, so these add:
     web_fallback          web_search fails, the model runs --help instead
     man_fallback          --help fails, the model reads the man page
     lookup_failed         every lookup fails; the answer says so, gives the
                           flag as unverified memory, and claims no lookup
     trap_lookup_failed    the same, for an invented flag

3. Composition (--compose N, off by default; v3). On operator tasks the v1
   adapter scored 50% against base 85%: it answers every question with its
   one trained form, "the option is `-x`", so `ss -l` drops -t and -p. These
   records ask for two options of one v1 tool, both visible in one help
   window, and answer with both and a combined command. The phrasings are
   not the eval's two_flag templates.

4. Ordinary answers (--general N, --arith N; v3). v2's failure-honesty data
   overshot: asked 2^10 it read `bc` docs and answered with a flag; asked who
   wrote Pride and Prejudice it said it could not check a library database.
   v1's only tool-free records are four greetings repeated. These add
   human-written Dolly-15k answers (no context passage, nothing about
   commands or code, nothing the eval's no_tool split asks) and generated
   arithmetic, answered directly with no tool call.

5. Asserted traps (--asserted N; v3). Told "a colleague said `rsync
   --parallel` does X", v2 denied 50-58% of invented flags against v1's 83%:
   it read the help and then answered with the fake flag, or with the
   nearest real flag said to do the fake thing. v1's traps only ever asked
   neutrally. These put v1's own fake flags in a teammate's or runbook's
   mouth -- not the eval's phrasings -- and deny them after the lookup.
   The failure strings are deliberately not the eval's ("web_search is
   unavailable in this evaluation.", "refused by eval harness", "command
   failed or timed out"), so the eval measures the behaviour, not the words.
   Only v1's own tools are used; the eval's held-out tools stay unseen.

Deterministic for a given v1 file, seed and host man pages.
"""
import argparse, copy, json, os, random, re, subprocess

import dataset_generator

HERE = os.path.dirname(os.path.abspath(__file__))
MAN_ENV = dict(os.environ, MANPAGER="cat", PAGER="cat", MANWIDTH="100", TERM="dumb")

WEB_FAILURES = [
    "Error: search backend returned HTTP 503 (service unavailable)",
    "web_search failed: network is unreachable",
    "Search request timed out after 20 s; no results.",
]
BASH_FAILURES = [
    "error: command timed out after 30 seconds",
    "sandbox: this command is not permitted here",
    "Permission denied",
]


def call(name, **args):
    return {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def result(name, content):
    return {"role": "tool", "name": name, "content": content}


def man_page(page):
    try:
        r = subprocess.run(["man", page], capture_output=True, text=True, timeout=15,
                           env=MAN_ENV, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return re.sub(r".\x08", "", r.stdout)


def man_window(text, flag, max_lines=60):
    """Lines around the first line that defines flag, or None."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"\s*(-\S+,\s+)*" + re.escape(flag) + r"(?![\w-])", ln):
            start = max(0, i - 10)
            head = lines[:6] + ["... [sections omitted] ..."] if start > 6 else []
            return "\n".join(head + lines[start:start + max_lines])
    return None


def spec_of(answer):
    m = re.search(r"the option is `([^`]+)`", answer)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=os.path.join(HERE, "research_dataset.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "research_dataset_v2.json"))
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--off-fraction", type=float, default=1 / 3)
    ap.add_argument("--web-fallback", type=int, default=48)
    ap.add_argument("--man-fallback", type=int, default=36)
    ap.add_argument("--lookup-failed", type=int, default=36)
    ap.add_argument("--trap-failed", type=int, default=24)
    ap.add_argument("--compose", type=int, default=0)
    ap.add_argument("--general", type=int, default=0)
    ap.add_argument("--asserted", type=int, default=0)
    ap.add_argument("--arith", type=int, default=0)
    ap.add_argument("--dolly", default="/srv/model-cache/hub/datasets--databricks--databricks-dolly-15k/snapshots/"
                    "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    v1 = json.load(open(a.src))
    cli = [r for r in v1 if r["type"] == "cli_grounded"]
    # v1 repeats each trap 4x; draw from the distinct ones.
    traps = list({(r["tool"], r["fake_flag"]): r for r in v1 if r["type"] == "trap_refusal"}.values())
    new = []

    for r in rng.sample(cli, a.web_fallback):
        m = r["messages"]
        tool = r["tool"]
        desc = m[3]["content"].rsplit(f"`{tool}` to ", 1)[-1].rstrip(".")
        new.append({"type": "web_fallback", "tool": tool, "flag": r["flag"], "command": r["command"],
                    "messages": [m[0], call("web_search", query=f"{tool} option to {desc}"),
                                 result("web_search", rng.choice(WEB_FAILURES)), m[1], m[2], m[3]]})

    mans, skipped = {}, 0
    for r in rng.sample(cli, len(cli)):
        if sum(x["type"] == "man_fallback" for x in new) >= a.man_fallback:
            break
        page = r["tool"].replace(" ", "-")
        if page not in mans:
            mans[page] = man_page(page)
        spec = spec_of(r["messages"][3]["content"])
        win = man_window(mans[page], r["flag"]) if mans[page] else None
        if not win or not spec:
            skipped += 1
            continue
        m, tool = r["messages"], r["tool"]
        desc = m[3]["content"].rsplit(f"`{tool}` to ", 1)[-1].rstrip(".")
        answer = (f"`{r['command']}` failed here, so I read the man page instead. Based on "
                  f"`man {page}`, the option is `{spec}`.\n\nYou can use it with `{tool}` to {desc}.")
        new.append({"type": "man_fallback", "tool": tool, "flag": r["flag"], "command": r["command"],
                    "messages": [m[0], m[1], result("bash", rng.choice(BASH_FAILURES)),
                                 call("bash", command=f"man {page}"), result("bash", win),
                                 {"role": "assistant", "content": answer}]})

    for n, r in enumerate(rng.sample(cli, a.lookup_failed)):
        m, tool, cmd = r["messages"], r["tool"], r["command"]
        page = tool.replace(" ", "-")
        spec = spec_of(m[3]["content"]) or r["flag"]
        desc = m[3]["content"].rsplit(f"`{tool}` to ", 1)[-1].rstrip(".")
        msgs = [m[0], m[1], result("bash", rng.choice(BASH_FAILURES))]
        if n % 2:
            msgs += [call("bash", command=f"man {page}"), result("bash", f"No manual entry for {page}")]
            why = f"`{cmd}` failed and there is no man page for `{page}`"
        else:
            why = f"`{cmd}` failed"
        answer = (f"I couldn't check the documentation: {why}. So this is from memory and "
                  f"unverified: I believe the option to {desc} is `{spec}`. Please confirm it with "
                  f"`{cmd}` before relying on it.")
        new.append({"type": "lookup_failed", "tool": tool, "flag": r["flag"], "command": cmd,
                    "messages": msgs + [{"role": "assistant", "content": answer}]})

    for r in [rng.choice(traps) for _ in range(a.trap_failed)]:
        m, tool, cmd, fake = r["messages"], r["tool"], r["command"], r["fake_flag"]
        answer = (f"I couldn't check the documentation (`{cmd}` failed), so I can't verify this. "
                  f"I don't know of a `{fake}` option in `{tool}`, and I won't guess at its syntax. "
                  f"Run `{cmd}` to see whether it exists.")
        new.append({"type": "trap_lookup_failed", "tool": tool, "fake_flag": fake, "command": cmd,
                    "messages": [m[0], m[1], result("bash", rng.choice(BASH_FAILURES)),
                                 {"role": "assistant", "content": answer}]})

    compose_q = ["With {tool}, how do I {d1}, and also {d2}?",
                 "What {tool} options let me {d1}, and {d2}, in the same command?"]
    pairs = []
    junk = lambda x: re.search(r"--help|--version|(^|[\s,])-h\b", spec_of(x["messages"][3]["content"]) or "") or \
        re.search(r"--?[A-Za-z]|\bhelp\b", x["messages"][3]["content"].rsplit("` to ", 1)[-1])
    for r in cli if a.compose else []:  # drawing nothing keeps v2 byte-identical
        obs = r["messages"][2]["content"]
        for o in cli:
            if (o["tool"] == r["tool"] and o["flag"] != r["flag"] and o["command"] == r["command"]
                    and not junk(r) and not junk(o)
                    and spec_of(o["messages"][3]["content"]) and spec_of(r["messages"][3]["content"])
                    and spec_of(o["messages"][3]["content"]) in obs):
                pairs.append((r, o))
    seen = set()
    for r, o in (rng.sample(pairs, len(pairs)) if pairs else []):
        if len([x for x in new if x["type"] == "compose"]) >= a.compose:
            break
        key = tuple(sorted((r["flag"], o["flag"]))) + (r["tool"],)
        if key in seen:
            continue
        seen.add(key)
        tool, cmd = r["tool"], r["command"]
        d1, d2 = (x["messages"][3]["content"].rsplit(f"`{tool}` to ", 1)[-1].rstrip(".") for x in (r, o))
        s1, s2 = spec_of(r["messages"][3]["content"]), spec_of(o["messages"][3]["content"])
        answer = (f"Based on `{cmd}`, that takes two options:\n\n> `{s1}`: {d1}\n> `{s2}`: {d2}\n\n"
                  f"Use them together: `{tool} {r['flag']} {o['flag']}`.")
        new.append({"type": "compose", "tool": tool, "flag": r["flag"], "flag2": o["flag"], "command": cmd,
                    "messages": [{"role": "user", "content": rng.choice(compose_q).format(tool=tool, d1=d1, d2=d2)},
                                 r["messages"][1], r["messages"][2], {"role": "assistant", "content": answer}]})

    # Own generators, so --general and --arith leave every other draw alone.
    grng = random.Random(a.seed + 4)
    if a.general:
        techy = re.compile(r"\b(command|flag|option|linux|unix|shell|bash|terminal|git|docker|install|python|"
                           r"code|program|script|sql|api|cli|kubernetes|server|software|computer|"
                           r"pride and prejudice|austen|canberra|capital of australia|chemical symbol|gold|"
                           r"continents?|red planet|mars|romeo|haiku|golden retriever|race condition|thread|"
                           r"tcp|udp|binary search|idempotent|birthday)\b", re.I)
        dolly = [json.loads(l) for l in open(a.dolly)]
        pool = [d for d in dolly if d["category"] in ("open_qa", "general_qa", "brainstorming", "creative_writing",
                                                      "classification")
                and not d["context"].strip() and 20 <= len(d["response"]) <= 600
                and not techy.search(d["instruction"] + " " + d["response"])]
        for d in grng.sample(pool, a.general):
            new.append({"type": "general_answer", "source": f"dolly-15k/{d['category']}", "messages": [
                {"role": "user", "content": d["instruction"].strip()},
                {"role": "assistant", "content": d["response"].strip()}]})
    for _ in range(a.arith):
        x, y = grng.randint(13, 97), grng.randint(3, 49)
        q, ans = grng.choice([
            (f"What is {x} times {y}?", f"{x} × {y} = {x * y}."),
            (f"What is {x} plus {y}?", f"{x} + {y} = {x + y}."),
            (f"What is {x * y} divided by {y}?", f"{x * y} ÷ {y} = {x}."),
            (f"What is {y}% of {x * 20}?", f"{y}% of {x * 20} is {y * x * 20 / 100:g}."),
            (f"A cyclist rides at {y} km/h for {x // 10} hours. How far do they go?",
             f"Distance is speed times time: {y} × {x // 10} = {y * (x // 10)} km."),
        ])
        new.append({"type": "arithmetic", "messages": [{"role": "user", "content": q},
                                                       {"role": "assistant", "content": ans}]})

    arng = random.Random(a.seed + 5)
    asserted_q = ["My teammate swears `{tool} {fake}` is how you get {desc}. How do I run it?",
                  "Our runbook says to use `{tool} {fake}` for {desc}. What's the right way to call it?"]
    for n in range(a.asserted):
        r = traps[n % len(traps)]
        m, tool, cmd, fake = r["messages"], r["tool"], r["command"], r["fake_flag"]
        desc = next((d for t, f, _, d in dataset_generator.TRAP_SPECS if t == tool and f == fake), None)
        if desc is None:
            continue
        answer = (f"I checked `{cmd}`: `{tool}` has no `{fake}` option, so that command would fail. "
                  f"Whoever suggested it may have another tool or an old wrapper in mind; "
                  f"`{cmd}` lists what `{tool}` actually supports.")
        new.append({"type": "asserted_trap", "tool": tool, "fake_flag": fake, "command": cmd, "messages": [
            {"role": "user", "content": asserted_q[n // len(traps) % 2].format(tool=tool, fake=fake, desc=desc)},
            m[1], m[2], {"role": "assistant", "content": answer}]})

    out = [dict(copy.deepcopy(r), thinking="default") for r in v1] + new
    for r in new:
        r["thinking"] = "default"
    tool_using = [r for r in out if any("tool_calls" in m for m in r["messages"])]
    for r in rng.sample(tool_using, round(len(tool_using) * a.off_fraction)):
        r["thinking"] = "off"
    rng.shuffle(out)

    json.dump(out, open(a.out, "w"), indent=2)
    counts = {}
    for r in out:
        k = (r["type"], r["thinking"])
        counts[k] = counts.get(k, 0) + 1
    for k in sorted(counts):
        print(f"  {k[0]:22s} {k[1]:8s} {counts[k]}")
    print(f"{len(out)} records ({len(new)} new, {skipped} man candidates skipped) -> {a.out}")


if __name__ == "__main__":
    main()
