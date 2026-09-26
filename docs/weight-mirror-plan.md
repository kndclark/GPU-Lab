# Weight mirror on the laptop — plan

**Status: plan only.** David approved writing it on 2026-09-26. Building any of
it needs his separate go; nothing below exists yet.

Tags: MEASURED = run for this plan on 2026-09-26; SOURCED = file:line;
ARITHMETIC = computed from those; UNKNOWN = not established. GB = 10^9 bytes;
GiB = 2^30, which is what `du -h` and `df -h` print as "G".

## The goal, and the constraint

The laptop loads every model from `/srv/model-cache`: NFS (the desktop's disk,
read over the network) across the direct 2.5GbE cable. Without the cable nothing
loads and `lab up` refuses [SOURCED bin/lab:203-209]. David wants a local copy
(a "mirror") so the laptop loads without the cable. The constraint is the lab's
premise: a cross-architecture comparison "is only meaningful if both nodes loaded
byte-identical weights" [SOURCED host/README.md:79-80]. So the copy must be
provably identical, and only the desktop may ever be written, or copies drift.

## What exists today (the survey, re-verified)

| Fact | Tag |
|---|---|
| Laptop mount `10.10.0.1:/srv/model-cache nfs4 rw,hard,...,x-systemd.automount,x-systemd.idle-timeout=600`; no `fsc`; `cachefilesd` not installed | SOURCED /etc/fstab:14; MEASURED dpkg |
| Export `10.10.0.2/32(rw,sync,...,all_squash,anonuid=1000,anongid=1000)`, `sec=sys`, in `/etc/exports:11`; no `/etc/exports.d` exists | MEASURED `ssh llm`, `exportfs -v` |
| Desktop cache: `hub/` 263.8 GB (246 GiB), `adapters/` 3.44 GB, `xet/` 6.6 MB | MEASURED `du -sb` |
| Of `hub/`: complete blobs 258.1 GB; 8 `.incomplete` files 5.68 GB, all Qwen3-14B, dated 2026-09-20 | MEASURED |
| Per repo, GiB: Lightning BF16 62, Qwen3-32B 62, Llama-3.1-70B GPTQ 38, Qwen3-14B 33 (incl. 5.3 incomplete), Lightning NVFP4 21, Coder-30B AWQ 17, Qwen3-8B 16, Embedding-0.6B 1.2, dolly 0.01 | MEASURED `du -sh` |
| 202 blobs (HF stores each file once, as `blobs/<hash>`): 122 named by 64 hex (sha256), 80 by 40 hex (git blob sha1). `sha256sum` and `git hash-object` each reproduced a sampled blob's name | MEASURED |
| `snapshots/<rev>/<file>` are relative symlinks to `../../blobs/<hash>`, 0 dangling; `refs/main` holds a 40-hex commit id; the two Nemotron repos have no `refs/` | MEASURED |
| Two files are root-owned 0600, unreadable by david: `trees/*.json` in the 70B and Qwen3-8B repos | MEASURED `find ! -readable` |
| Something has written into the cache root: `.cache/pip/`, `.agent_harnesses.json` | MEASURED `ls` |
| Laptop free: 679.2 GB (633 GiB) on `/`, the only Linux disk (the other NVMe is BitLocker) | MEASURED df, lsblk |
| Listing all of remote `hub/` over ssh (712 entries): 0.34 s | MEASURED |
| NFS read over the cable 294 MB/s; rsync over ssh: not measured | SOURCED host/README.md:91; UNKNOWN |
| The laptop runs no sshd, so control flows laptop → desktop only | SOURCED bin/lab:34-36 |

Laptop consumers of `/srv/model-cache`:

| Where | Use |
|---|---|
| nodes/laptop/llama-swap.yaml:30 | qwen3-embed, rw `-v`, no `HF_HUB_OFFLINE` (neither node's config sets it) |
| nodes/laptop/llama-swap.service:8 | `RequiresMountsFor=/srv/model-cache` |
| bin/lab:203-209 / :555-556 | `lab up` refuses without the link / pool `ray-worker` mount |
| training/run.sh:40,67,84-85 | **writes** adapters there, but only with `FORCE=1` on the laptop; by default it dispatches to the desktop (run.sh:44-51) |
| bench/research_eval.py:1382 | hashes a served adapter's file on the host (on the laptop: the mirror's copy, right only while current) |
| training/augment_research.py:121 | default dolly dataset path |
| ~/moe-qlora/probes: gpurun.sh:7; ssu_sm120/gpu/run.sh:6, patch/run_cta.sh:6, serve/serve_ab.sh:33; docstrings g1:34, g2:40, g4:30, g5:33 | `:ro` mount at `/hf` |
| g1_nf4_real_experts.py:53-54, g2_fidelity.py:145,177; g0_verify.py:9 | g1/g2 build paths under `$HF_HOME` (`/hf` in the container), so they follow the mount; g0's is a host path in a docstring. No edit |

## Decisions

1. **Mirror all of `hub/` and `adapters/`, minus an exclude list.** 261.5 GB is
   38% of the laptop's free space and leaves 417.7 GB [ARITHMETIC].
   *Lost: an allow-list* — each new download needs a second edit, and a
   missed one shows up offline, when it cannot be fixed. The desktop can
   outgrow the laptop (246 + 536 GiB free there, 633 GiB here [MEASURED df]),
   so the sync refuses if it would leave under 100 GB free. Excluded:
   `*.incomplete` (HF's in-progress downloads), `.locks/`, `xet/`, root
   dotfiles. First copy: ~15 min at the NFS rate, 14 min at line rate
   [ARITHMETIC: 261.5 GB / 294 MB/s = 889 s; / 312.5 MB/s = 837 s].
2. **The mirror lives at `/srv/model-mirror`, owned by david,** beside the path
   it serves. *Lost: `~/.cache/huggingface`* — any `hf download` on the laptop
   would write into it, making a second writer.
3. **Consumers switch by a read-only bind mount** (one directory shown at a
   second path): the laptop's `/srv/model-cache` becomes the mirror, read-only,
   and NFS moves to `/srv/model-cache-nfs`. Every consumer keeps its path and
   cable-up and cable-down read the same bytes, so no copy "wins" per run:
   **the laptop always reads its mirror, only the desktop is ever written, and
   the mirror changes only through `lab mirror sync`.** Read-only at the mount
   also stops root-run containers, which ignore file modes [`bind,ro`: SOURCED
   mount(8), util-linux 2.41.3]. Costs: one path means different things on the
   two nodes; a new desktop download is invisible here until the next sync; any
   consumer that writes under `/hf` now fails loudly (which do: UNKNOWN until
   steps 6-7). *Lost: a new path in every consumer* (17 lines in two trees, and
   the old path still works with the cable, so habit recreates two sources); *an
   env var* (`RequiresMountsFor` is static; unset silently means the old path);
   *switching only when the cable is down* (a container pins its mount at
   start, so one command would read different copies on different days).
4. **Copy with rsync over ssh, pulled from the laptop** (`ssh llm`; `llm-wifi`
   only with `--wifi`). rsync 3.4.1 is on both ends [MEASURED]; one code path
   serves both networks. *Lost: exporting NFS to Wi-Fi* — with `sec=sys` access
   is granted by IP address alone, no key or password, and the export is rw
   [MEASURED], so any machine on the Wi-Fi LAN holding the allowed address could
   rewrite the cache. *Lost: FS-Cache (`fsc`)* — a new package, and whether it
   serves files with the server gone is UNKNOWN. Gate, set now: if step 1
   measures ssh on the cable below 200 MB/s (68% of NFS's 294), the cable path
   reads from `/srv/model-cache-nfs` instead.
5. **No partial file is ever visible under a name a loader uses.** Per repo:
   (a) copy `blobs/`: rsync writes to a temp name and renames when complete
   [SOURCED rsync(1), `--inplace`, which describes that default], and
   `--partial-dir=.rsync-partial` keeps an interrupted shard to resume;
   (b) hash each newly arrived blob against its own name (`sha256sum` for 64
   hex, `git hash-object` for 40); a mismatch moves to `.lab-mirror/quarantine/`
   and the repo stops there; (c) only then copy the pointers: `snapshots/`,
   `refs/`, `trees/`. A loader reaches blobs only through pointers, so it sees
   the old revision or the complete, verified new one. *Lost: `--delay-updates`*
   — near-atomic renames, but it checks nothing. Trap: `--size-only` is right for
   blobs (the name is the hash) and wrong for `refs/main`, always 40 bytes, so a
   moved ref would never copy; pointers use rsync's default size+mtime check.
   Adapters are not content-named: default check plus rsync's transfer checksum.
6. **Stale files are reported, never deleted by default.** Stale = in the
   mirror, gone from the desktop. `lab mirror prune` is a dry-run report (repo,
   files, GB); `prune --apply` deletes only paths the sync itself recorded in
   `.lab-mirror/manifest` and the desktop no longer has, so it cannot touch
   anything a person put there (David's rule: never delete files you did not
   create). *Lost: `rsync --delete`* — one wrong source path empties the mirror.
7. **Unreadable source files are named, not fatal:** listed first by `find !
   -readable` over ssh, excluded, printed. Otherwise every run exits 23,
   "partial transfer due to error" [SOURCED rsync(1), EXIT VALUES].
8. **Sync is manual and refuses during measurements.** It shares the cable with
   pool NCCL (~280 MB/s [SOURCED README.md:45]) and the disk with model loads, so
   it refuses while `ray-worker`, `gpu-lab-kernels` or `gpu-lab-training` runs
   (`--force` overrides). One at a time (`flock` on `.lab-mirror/lock`). State
   and log live in the mirror, so removing it leaves no stale "last synced".

## Wiring into `lab`

Laptop only; on the desktop it says the desktop is the source, the way
`pool_require_driver` does (bin/lab:425-442).

    lab mirror status              per repo: current / behind (files, GB) / extra;
                                   last sync; free space. Read-only.
    lab mirror sync [--wifi] [--force] [repo...]
    lab mirror verify [repo...]    re-hash every blob against its name
    lab mirror prune [--apply]     report extras (decision 6)

Edits: usage block (bin/lab:2-15), both case lists (:129-132, :905-915),
`cmd_logs` (:881-903). Against the checklist in docs/phase1-serving-plane.md:120-131:

- **up** — never syncs (a 15-minute copy would surprise, and `up` must work
  offline). The :203-209 preflight warns instead of refusing when the mirror is
  mounted ("serving locally only; LiteLLM cannot reach this node"), and prints
  one freshness line.
- **down** — nothing to stop (no daemon, no GPU); a sync holding the lock is
  named with its pid rather than implying the machine is idle.
- **status** — a `mirror` section: mounted `ro` or not; last sync (time, route,
  GB, seconds, result); link up: current/behind/extra from a dry run; link
  down: "freshness unknown (link down)".
- **logs** — `lab logs mirror` follows `.lab-mirror/sync.log`. **boot** — the
  bind is an fstab line with `nofail`; no unit, so `--boot-off` has nothing to do.
- **pool** — `pool_up` checks `$POOL_MODEL` is current in the mirror: the worker
  loads from it (:556) offline, and a missing file would fail minutes in.

**bin/check.py**: the code stays in `bin/lab`, so the existing `bash -n` covers
it (check.py:42-49); check.py does not glob (:157-158), so a separate script
would need its own line. Add one warn-level check, laptop only: `/etc/fstab`'s
model-cache lines match `host/laptop-fstab-model-cache.line`, since `host/` is a
record nothing deploys.

## Steps

Precondition for 5-7: no container mounts `/srv/model-cache` (one did while this
was written: `ssu-ab-fi-simple`, ro). Commit only when David says.

| # | Step | Proven by |
|---|---|---|
| 1 | Measure: rsync-over-ssh rate, cable and Wi-Fi, on the 1.2 GiB Embedding repo into a scratch dir; sha256 rate on one shard | MB/s recorded here; decision 4's gate applied |
| 2 | `sudo install -d -o david -g david /srv/model-mirror`; write `lab mirror status` | desktop refuses with the message; laptop lists 9 repos + adapters behind, ~261.5 GB, and the 2 unreadable files by name |
| 3 | Write `sync` and `verify`; first sync over the cable | `status` 0 behind; `verify` 202 blobs, 0 mismatches; `g0_verify.py` passes on the mirror's Lightning snapshot against ~/moe-qlora/results/g0-manifest.json |
| 4 | Interrupt: `kill -9` a sync mid-shard, re-run | every final-named blob passes its hash; snapshots unchanged meanwhile; the re-run moves fewer bytes than the shard (resumed) |
| 5 | Switch the mount (sudo): NFS line → `/srv/model-cache-nfs`; add `/srv/model-mirror /srv/model-cache none bind,ro,nofail 0 0`; daemon-reload; update host/laptop-fstab-model-cache.line and host/README.md:76-99 | `findmnt` shows `/srv/model-cache` local and `ro`, `/srv/model-cache-nfs` nfs4 from 10.10.0.1; `docker run --rm -v /srv/model-cache:/hf --entrypoint touch vllm/vllm-openai:v0.29.0 /hf/x` fails "Read-only file system" |
| 6 | Consumers: `-e HF_HUB_OFFLINE=1` in nodes/laptop/llama-swap.yaml; the `up` preflight; the `pool_up` check; training/run.sh:40 node-aware (Q4) | **cable unplugged**: `lab up --warm` loads qwen3-embed, `/v1/embeddings` returns a 1024-dim vector, `lab status` says "freshness unknown (link down)" |
| 7 | Regression, cable in: `lab pool up` (Qwen3-14B); one moe-qlora probe via gpurun.sh | endpoint answers; load seconds recorded against the ~230 s NFS figure (bin/lab:96); probe log ends `exit 0` |
| 8 | `status`/`down`/`logs`/`prune` wiring; check.py warn; `lab check` | each output read; check green; a planted file outside the manifest is reported by `prune --apply` and survives it |

## Rollback

- **Mount:** drop the bind line, restore `/etc/fstab` line 14 as it is today
  (byte-identical to line 3 of host/laptop-fstab-model-cache.line [MEASURED
  diff]), unmount the bind, `daemon-reload`, start `srv-model\x2dcache.automount`.
  Proven when `findmnt /srv/model-cache` shows `10.10.0.1:/srv/model-cache nfs4`
  again, as it does now [MEASURED].
- **Code:** `git revert`; the push redeploys the desktop, which it never touched.
  **Data:** `/srv/model-mirror` stays, inert; deleting it is David's call.

## Open questions for David

1. **Repoint the laptop's `/srv/model-cache` at the mirror (decision 3)?** It
   changes what the path means on one node. *Recommend yes*: the only option
   where cable-up and cable-down read the same bytes with no consumer edits.
2. **Manual sync, or a timer?** *Recommend manual for now*; staleness shows in
   `lab up`/`status`, and a timer would share the cable with pool runs unless it
   learned every refusal rule.
3. **Desktop hygiene:** two root-owned 0600 `trees/*.json`, and 5.68 GB of
   Qwen3-14B `.incomplete` files from 2026-09-20. *Recommend* `chmod 644` on the
   two (the fix used on the adapters on 2026-09-25); the `.incomplete` files are
   yours to keep or delete. The mirror skips both either way.
4. **Laptop-side training writes:** `FORCE=1 training/run.sh` on the laptop
   writes `/srv/model-cache/adapters`, which becomes read-only (an adapter named
   `qwen3-8b-laptop-test` exists; which script wrote it is UNKNOWN). *Recommend*
   `/srv/model-cache-nfs/adapters` there; the desktop needs the cable to see it
   anyway.

**Answers (David, 2026-09-26):** 1 yes; 2 manual for now; 3 chmod, and delete the
`.incomplete` files if not needed; 4 as recommended. Done for 3 the same day
[MEASURED]: `chmod 644` on the two root-owned files (Llama-70B GPTQ, Qwen3-8B); the
eight Qwen3-14B `.incomplete` files each had a finished blob of the same hash that the
snapshot links to (15/15 links resolve), so all eight were deleted; 0 `.incomplete`
left, snapshot still 0 broken links, desktop free 536G -> 541G. Six more `trees/*.json`
are 0600 but owned by david, so readable by the sync; left as they are.
Implementation (steps above) not started.

## Survey corrections

The refusal is bin/lab:203-209, not 199-208. Missed consumers: training/run.sh
(a writer), bench/research_eval.py, training/augment_research.py, three
ssu_sm120 scripts. "246G"/"634G" were GiB (GB: 263.8 and 679.2). The g1/g2
snapshot paths follow `$HF_HOME`. New: the unreadable `trees/*.json`, no
`HF_HUB_OFFLINE` anywhere, writes into the cache root. Not ours to fix:
host/desktop-exports.d-model-cache.exports says `/etc/exports.d/`; the line is
in `/etc/exports:11` [MEASURED].
