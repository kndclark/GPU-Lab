#!/bin/bash
# Stands in for bwrap (agent_ab.py puts a copy first on PATH as "bwrap") under
# Codex's Linux sandbox. Codex gives each command a minimal /dev and its own
# pid namespace, so nvidia-smi and ps see nothing of the host; the sealed
# set promises a read-only shell on the real machine. So: bind the GPU
# device nodes, keep the host pid namespace, show its /proc read-only, and
# cover each path listed in ./hide (one per line, written by agent_ab.py
# beside the copy) with an empty tmpfs: the set itself, its results, ssh
# keys, and this directory. ./tmp names the directory that stands in for
# the host's /tmp, so one run cannot leave notes for the next.
#
# The shell runs as the user named in ./user, not as the lab user. Opening 1
# showed why: a shell with the lab user's uid reaches whatever that uid may
# reach, so one run removed the serving container over the docker socket
# (group docker) and another signalled a service by pid. Both are uid checks
# in the kernel, so a uid of its own closes them while the machine stays
# visible -- ps and nvidia-smi still work, the sealed set's promise. That uid
# owns nothing here, joins no group that opens a socket, and cannot read the
# lab user's 0700 directories or its processes' environments. agent_ab.py
# asserts each of those from inside the sandbox before a run.
#
# The lists are files, not env: Codex hands bwrap the env its
# shell_environment_policy leaves, so excluded variables never got here, and
# kept ones reach the shell. Carrying that env across the uid change is
# explicit, because sudo drops it and Codex passes bwrap no env arguments of
# its own. With no list, no user, or no --dev to hang the mounts on, it
# refuses to run.
#
# Every other argument (read-only root, writable workspace, network policy,
# seccomp) is Codex's. Codex runs each bwrap on PATH with --help first and
# passes over one whose help lacks --as-pid-1, so that goes straight through.
case "$1" in --help|--version) exec /usr/bin/bwrap "$1" ;; esac
here=$(dirname "$0")
for f in hide tmp user; do
  [ -r "$here/$f" ] || { echo "agent_ab_bwrap: no $here/$f" >&2; exit 1; }
done
user=$(cat "$here/user")
id -u -- "$user" >/dev/null 2>&1 || { echo "agent_ab_bwrap: no user $user" >&2; exit 1; }
gpu=(); for d in /dev/nvidia*; do [ -c "$d" ] && gpu+=(--dev-bind "$d" "$d"); done
hide=(--tmpfs "$here")
while IFS= read -r p; do [ -d "$p" ] && hide+=(--tmpfs "$p"); done < "$here/hide"
tmp=$(cat "$here/tmp")
envs=(--clearenv)
while IFS= read -r n; do
  case "$n" in *KEY*|*SECRET*|*TOKEN*) continue ;; esac
  envs+=(--setenv "$n" "${!n}")
done < <(compgen -e)
# HOME is the lab user's, which this uid cannot read. Point it at the private
# /tmp, emptied between runs, so a shell that writes a dotfile still works.
envs+=(--setenv HOME /tmp --setenv USER "$user" --setenv LOGNAME "$user")
out=(); hidden=
while [ $# -gt 0 ]; do
  case "$1" in
    --dev) out+=("$1" "$2" "${gpu[@]}" "${hide[@]}"); hidden=1; shift 2 ;;
    --bind) [ "$2" = /tmp ] && [ "$3" = /tmp ] && out+=(--bind "$tmp" /tmp) || out+=("$1" "$2" "$3"); shift 3 ;;
    --unshare-pid|--as-pid-1) shift ;;
    --proc) out+=(--ro-bind /proc "$2"); shift 2 ;;
    --) out+=("$@"); break ;;
    *) out+=("$1"); shift ;;
  esac
done
[ -n "$hidden" ] || { echo "agent_ab_bwrap: no --dev in the arguments, nothing hidden" >&2; exit 1; }
exec sudo -n -u "$user" /usr/bin/bwrap "${envs[@]}" "${out[@]}"
