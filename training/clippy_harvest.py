#!/usr/bin/env python3
"""Harvest clippy diagnostics into a grounded corpus for training trajectories.

The CLI domains in dataset_generator.py use `--help` as their oracle: a flag
either appears in the tool's own output or it does not. This is the same
discipline against a different oracle. `cargo clippy --message-format=json`
emits, per diagnostic, the lint name, a message, the offending source line, a
span, and often a concrete suggested replacement -- all of it produced by the
compiler rather than written by anyone. A claim about code quality can then be
checked the same way a claim about a flag is.

Harvesting is separated from trajectory generation because it is the slow half:
it compiles every crate it inspects. Run it once, commit the corpus, and
generation stays deterministic and offline.

  ./clippy_harvest.py --projects ~/snake-rs ~/breakout-rs -o clippy_corpus.json
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys


def run_clippy(project: pathlib.Path, timeout: int, pedantic: bool):
    """Return raw compiler-message records for one cargo project."""
    cmd = ["cargo", "clippy", "--message-format=json", "--all-targets"]
    if pedantic:
        cmd += ["--", "-W", "clippy::pedantic"]
    env = dict(os.environ)
    env["PATH"] = f"{pathlib.Path.home()}/.cargo/bin:" + env.get("PATH", "")
    env["CARGO_TERM_COLOR"] = "never"
    try:
        res = subprocess.run(cmd, cwd=project, capture_output=True, text=True,
                             timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        print(f"  {project.name}: TIMEOUT after {timeout}s", file=sys.stderr)
        return []
    out = []
    for line in res.stdout.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("reason") == "compiler-message":
            out.append(d.get("message") or {})
    return out


def extract(messages, project_name):
    """Keep only diagnostics that can be verified against their own output.

    A record is usable when the lint name appears inside the rendered text, the
    same containment check Gate 4 applies to flags. Diagnostics from the crates
    registry are dropped: the corpus should describe code the user can open.
    """
    records = []
    for m in messages:
        code = (m.get("code") or {}).get("code") or ""
        if not code.startswith("clippy::"):
            continue
        rendered = m.get("rendered") or ""
        spans = [s for s in (m.get("spans") or []) if s.get("is_primary")]
        if not spans or not rendered:
            continue
        span = spans[0]
        file_name = span.get("file_name") or ""
        if "/.cargo/registry/" in file_name or file_name.startswith("/"):
            continue
        short = code.split("::", 1)[1]
        # the grounding invariant, checked at harvest rather than trusted later
        if short not in rendered and code not in rendered:
            continue
        suggestion = next(
            (s.get("suggested_replacement") for c in (m.get("children") or [])
             for s in (c.get("spans") or []) if s.get("suggested_replacement")),
            None,
        )
        doc_url = next(
            (c["message"].split("visit ", 1)[1].strip()
             for c in (m.get("children") or [])
             if c.get("message", "").startswith("for further information visit")),
            None,
        )
        records.append({
            "lint": code,
            "level": m.get("level"),
            "message": m.get("message"),
            "project": project_name,
            "file": file_name,
            "line": span.get("line_start"),
            "source_line": (span.get("text") or [{}])[0].get("text", "").strip(),
            "label": span.get("label"),
            "suggestion": suggestion,
            "applicability": span.get("suggestion_applicability"),
            "doc_url": doc_url,
            "rendered": rendered.rstrip(),
        })
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projects", nargs="+", required=True)
    ap.add_argument("-o", "--out", default="clippy_corpus.json")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--no-pedantic", action="store_true",
                    help="default lints only; pedantic yields far more variety")
    args = ap.parse_args()

    all_records = []
    for raw in args.projects:
        project = pathlib.Path(raw).expanduser()
        if not (project / "Cargo.toml").exists():
            print(f"  {project}: no Cargo.toml, skipping", file=sys.stderr)
            continue
        print(f"  harvesting {project.name} ...", flush=True)
        msgs = run_clippy(project, args.timeout, not args.no_pedantic)
        recs = extract(msgs, project.name)
        print(f"    {len(recs)} usable diagnostics from {len(msgs)} messages", flush=True)
        all_records.extend(recs)

    by_lint = {}
    for r in all_records:
        by_lint.setdefault(r["lint"], []).append(r)

    print(f"\n  total: {len(all_records)} diagnostics, {len(by_lint)} distinct lints")
    for lint, rs in sorted(by_lint.items(), key=lambda kv: -len(kv[1]))[:15]:
        withfix = sum(1 for r in rs if r["suggestion"])
        print(f"    {lint:<46} {len(rs):>4}  ({withfix} with a suggested fix)")

    with open(args.out, "w") as fh:
        json.dump(all_records, fh, indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
