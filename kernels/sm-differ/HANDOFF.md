# sm-differ handoff

Phase 3 of the sm_86/sm_120 GPU lab. Written 2026-09-18 for whoever finishes it.

## State

Committed and verified: `d99a813`, `05f2767`.
**Uncommitted and UNVERIFIED** — compiles, nothing else: `src/node.rs`,
`src/diff.rs` (both new), `src/lib.rs`, `src/main.rs` (modified).

Discard with:
`git checkout -- src/lib.rs src/main.rs && rm src/diff.rs src/node.rs`

## First thing to do

```
source ~/.cargo/env
cd /home/david/gpu-lab/kernels/sm-differ
cargo fmt && cargo clippy --all-targets && cargo test   # 45 tests were green before
cargo run -- diff --query 'llamaswap_gpu_util_percent' --bench gpu --metric util_pct
```
The `diff` command has **never been executed**. Expect it to be wrong.

## Rules that are not negotiable

1. **A missing measurement must never become a number.** This is why the
   project is Rust, not Go. `Measurement` is `Measured | Refused | NotRun`.
   No `unwrap_or(0.0)`, no default, no NULL-to-zero. Three real defects were
   found violating this; all are regression-tested.
2. **Never `rm`/overwrite files you did not create.** `.claude` and
   `.claude.json` in the repo root are David's. Leave them. Commit with
   explicit paths, never `git add -A`.
3. **Commit messages: 50-char subject, body wrapped at 72, why-only, no
   trailers** (no Co-Authored-By). See `~/.claude/CLAUDE.md`.
4. **Push = deploy.** `git push` auto-deploys to the desktop. Only push when
   David says "push" specifically.
5. **Verify by running, not reading.** Claiming something works without
   output is the failure this whole lab exists to catch.

## Architecture, and why

- **One crate, no async runtime, no workspace.** Tokio collapsed its own
  sub-crates (tokio#1318); the reason to split is releasing at different
  paces, which an internal tool cannot use. Tool is I/O-bound on multi-minute
  docker builds; ~2 concurrent tasks.
- **Deps chosen deliberately**: rusqlite (not sqlx — wants a runtime for a
  local file), ureq (not reqwest — its blocking client panics inside a
  runtime), hand-rolled serde for Prometheus (`prometheus-parse` dead since
  2023), quick-xml for JUnit when that lands. Shell out to `docker`, **not**
  bollard: the CLI uses BuildKit, the Engine API `/build` is the deprecated
  classic builder, so a library build can silently differ from a human's.
- **Verdict = two gates** (Criterion's structure): Welch t-test p<0.05 AND
  >1% relative change. Both required. Zero observed variance returns
  `Broken::NoiseNotEstimable`, never certainty — integer MiB and ms timers
  make flat samples ordinary, and a coarser instrument must not buy
  confidence.
- Metrics come from the **existing Prometheus** (`http://10.10.0.1:9090`),
  which already labels every series with `arch` and `node`. Values arrive as
  JSON **strings** (that is how NaN is transmitted) — parse loudly.
  Errors are HTTP **400 with a JSON body**, so ureq's status-as-error is off.
- Driver version is **not** in Prometheus (checked). `node.rs` gets it from
  `nvidia-smi` locally and over ssh. Compute capability is node identity,
  not hostname — follows `bin/lab`.

## Reference

**The runbook is the source of truth for the whole build.** It is a published
Artifact, not a repo file — `runbook.html` is gitignored precisely because a
copy here would drift. Read §05 (end state), §06 Phase 3 (the crate layout and
exit criteria), §07 (kill list) and §02 (claims ledger).

    https://claude.ai/artifact/WZj7niGuUjAen23XDrtFYZ
    (also linked from README.md:10 as .../code/artifact/ef6ac6fe-d2c7-4b18-9a31-7342e0473826)

A stale local copy may exist at `/home/david/gpu-lab/runbook.html` — trust the
URL over it. Current version as of this handoff: **v14**.

Companion artifact, a narrative summary of phases 0–2b:

    https://claude.ai/artifact/PbAw4JBv8mPiGpRcBUvnj9   ("Two Cards, One Question")

**Repo docs**

    docs/contributions.md      upstream candidate ledger C1–C8, with evidence
    docs/phase1-serving-plane.md   serving plane; the "wire it into lab" rule
    docs/phase2-dev-plane.md   kernel differential; the corrected failure counts
    README.md                  layout, and why it is organised by runtime
    bin/check.py               the push gate; §7 is the Rust check
    kernels/repros/            standalone C3/C4 reproductions

**Persistent memory** — read this first, it is the fastest context:

    ~/.claude/projects/-home-david/memory/MEMORY.md     index, 21 entries
    ~/.claude/projects/-home-david/memory/*.md          one fact per file

Most relevant here: `phase3-language-and-metrics.md` (why Rust, why
Prometheus), `gpu-lab-phase2-status.md`, `engine-fork-chosen.md`,
`gpu-lab-layout-and-deploy-gate.md`, `no-claude-git-attribution.md`,
`never-delete-files-i-did-not-create.md`.

**Session transcripts** (JSONL; this work spans the first one)

    ~/.claude/projects/-home-david/5142a11d-ac4a-418d-a2bd-f4f59a07a050.jsonl
    ~/.claude/projects/-home-david-gpu-lab/           earlier sessions, cwd=gpu-lab
    ~/.claude/projects/-home-david/                   other sessions, cwd=home

Sorted newest-first with `ls -t ~/.claude/projects/*/*.jsonl`.

**User instructions** — binding, read before working:

    ~/.claude/CLAUDE.md        fable method, 50/72 commit rule, no AI attribution

## Remaining work

1. Verify/fix `diff` (above). It probes nodes, checks the thermal gate
   (`llamaswap_gpu_temperature_celsius`, both nodes report it) *before*
   fetching the metric, records both sides, then compares.
2. No tests exist for `node.rs` or `diff.rs`. Add them.
3. JUnit XML parsing (quick-xml) — pytest emits a `<testsuites>` root and
   non-standard `file`/`line` attrs on `<testcase>`.
4. Docker build driver (shell out) + the `run` subcommand.
5. Phase 3 exit criteria are in the runbook §06. One is already met: a real
   sm_120 disagreement reduced to a minimal repro (`kernels/repros/`).

## Gotchas that cost time

- `cargo fmt` reformats code, so exact-string patching of source files
  silently no-ops. Assert your edits landed.
- The push gate (`bin/check.py` §7) now runs `cargo test`. It fails the push
  on a compile error or a failing test. `LAB_SKIP_CHECK=1` bypasses.
- Exit codes: 0 conclusive, 1 refused gates, 2 usage, 3 Broken verdict.
- `docs/contributions.md` is the upstream-candidate ledger; C1–C8 are closed
  but **nothing is filed** — the issue-tracker search is still open.
