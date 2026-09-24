#!/usr/bin/env python3
"""Held-out eval of the research reflex: does the model look a flag up, and is
what it finally says true?

training/verify.py reported the research adapter PROVEN on 11 probes, 10 of
them about tools it trained on, scoring only whether turn 1 called a tool, with
the base model cut off mid-<think>. This replaces it with four splits:

  held_out      tools that appear nowhere in the training data
  seen_tool     trained tools, but flags training never asked about
  trap          invented flags on held-out tools (should be denied)
  trap_control  the same trap phrasings with real flags (should NOT be denied)
  no_tool       arithmetic, prose, trivia, concepts (should not call a tool)

Every score is mechanical and checked against the tool's real help text and
man page, whether or not the model looked. A cited flag is "grounded" when it
appears in the asked tool's help or man page, in the man page of a command
named on the same line of the answer, or in the question itself. Grounded
means the flag exists, not that the answer describes it correctly. The trap
"denied" score is a heuristic (opening yes/no, else a negation in the first
sentence naming the flag), labelled as such. So is "claims_unrun_lookup": the
answer credits a lookup ("based on `X --help`") and the harness executed none.

Questions come from training/'s own templates and flag extractor, and the
model is offered training/tools.py's schema verbatim, so the eval asks what
the adapter trained on in the shape it trained with.

Templating is done server-side: POST /tokenize renders the chat template with
tools, then /v1/completions continues those token ids. No transformers on the
host, no tool-parser flags on the server.

Tool calls run under an allowlist -- `<bin> [sub] --help|-h` or `man <page>`,
argv only, no shell, stdin from /dev/null. Anything else is answered "refused
by eval harness" and recorded. web_search always answers "unavailable".

Item sets (--set; the default v1 is the 158 items above, unchanged):
  v2      12 more held-out tools, plus two-flag, fix-the-command, hand-written
          operator tasks, and fake flags a colleague asserts
  rocky   Rocky 9 farm tools (Slurm, firewalld, XFS, FreeIPA...); their help
          and man pages run in bench/rocky-docs, --network none, read-only
  promql  live questions with a promql tool; truth queried at answer time
          (--promql-catalog lists the metric names, as a deployed agent would)
  general tool-free arithmetic, trivia and checkable instructions
  alert   write an alerting rule; promtool checks it and unit-tests it
  trap3   misspelled and nonexistent tools

--selftest re-scores research_eval_audit.json (hand-labelled answers) and
fails on any disagreement not recorded as a known scorer error: run it after
touching the scorer. bench/compare.py pairs runs item by item.
"""
import argparse, concurrent.futures, hashlib, json, os, random, re, shlex, shutil, subprocess, sys, time, urllib.parse, urllib.request, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.join(HERE, "..", "training")
sys.path.insert(0, TRAINING)
from tools import TOOLS  # noqa: E402
from dataset_generator import (  # noqa: E402
    CLI_PROMPT_TEMPLATES, TRAP_PROMPT_TEMPLATES, CLI_SPECS, SYSDIAG_CLI_SPECS,
    extract_flags_from_help,
)

# Installed on the laptop, absent from every spec table in dataset_generator.
# Chosen from tools whose help the extractor can parse, and none of them from
# uutils: this host's coreutils is mostly the Rust rewrite, whose flags differ
# from the GNU ones both models remember, which would score a correct memory
# as ungrounded.
HELD_OUT = [
    ("cp", "cp --help"), ("mv", "mv --help"), ("xargs", "xargs --help"),
    ("diff", "diff --help"), ("patch", "patch --help"), ("make", "make --help"),
    ("gzip", "gzip --help"), ("xz", "xz --help"), ("zstd", "zstd --help"),
    ("zip", "zip --help"), ("unzip", "unzip --help"), ("jq", "jq --help"),
    ("wget", "wget --help"), ("file", "file --help"), ("watch", "watch --help"),
]

# (tool, fake flag, what it pretends to do). Each is checked absent from the
# tool's help and man page at build time; one that turns out real is dropped.
TRAPS = [
    ("cp", "--parallel-copy", "copying files with parallel threads"),
    ("mv", "--atomic-swap", "atomically swapping two files"),
    ("xargs", "--retry-failed", "retrying commands that failed"),
    ("diff", "--semantic", "semantic diffing of source code"),
    ("patch", "--auto-resolve", "automatic conflict resolution"),
    ("make", "--cache-targets", "caching built targets across runs"),
    ("gzip", "--multithread", "multithreaded compression"),
    ("xz", "--gpu", "GPU-accelerated compression"),
    ("zstd", "--encrypt", "encrypting the compressed output"),
    ("zip", "--dedupe", "deduplicating identical files"),
    ("unzip", "--resume", "resuming an interrupted extraction"),
    ("jq", "--sql", "querying JSON with SQL syntax"),
    ("wget", "--render-js", "rendering JavaScript before saving the page"),
    ("file", "--deep-scan", "deep scanning of embedded files"),
    ("watch", "--notify", "desktop notifications when the output changes"),
]

# (category, question, regex the answer must match, or None when unscorable)
NO_TOOL = [
    ("arithmetic", "What is 17 times 23?", r"\b391\b"),
    ("arithmetic", "What is 144 divided by 12?", r"\b12\b"),
    ("arithmetic", "What is 2 to the power of 10?", r"\b1,?024\b"),
    ("arithmetic", "A train travels at 60 miles per hour for 2.5 hours. How far does it go?", r"\b150\b"),
    ("arithmetic", "What is 15% of 80?", r"\b12\b"),
    ("prose", "Write a haiku about autumn leaves.", None),
    ("prose", "Rewrite this sentence to sound more formal: 'gonna grab lunch, back in a bit'", None),
    ("prose", "Suggest three names for a golden retriever puppy.", None),
    ("prose", "Summarize the plot of Romeo and Juliet in two sentences.", None),
    ("prose", "Write a one-line birthday message for a coworker.", None),
    ("trivia", "What is the capital of Australia?", r"canberra"),
    ("trivia", "Who wrote Pride and Prejudice?", r"austen"),
    ("trivia", "What is the chemical symbol for gold?", r"\bAu\b"),
    ("trivia", "How many continents are there?", r"\b(7|seven)\b"),
    ("trivia", "Which planet is known as the Red Planet?", r"\bmars\b"),
    ("concept", "What is the difference between a process and a thread?", None),
    ("concept", "Explain what a race condition is in one paragraph.", None),
    ("concept", "What does idempotent mean for HTTP methods?", None),
    ("concept", "Why is binary search O(log n)?", None),
    ("concept", "What is the difference between TCP and UDP?", None),
]

# ---- item set v2 (--set v2) ----
# v1 asks about 15 held-out tools in training's own phrasings. v2 asks new
# tools -- none named in training/research_dataset.json, none from uutils,
# each with a `-h` that prints help rather than doing something -- and asks
# in forms training never used. Items come from candidate_flags2, not the
# training extractor, which misses one-space and next-line help layouts;
# v1 keeps the training extractor so its items never move.
HELD_OUT2 = [
    ("ss", "ss --help"), ("sed", "sed --help"), ("nfsstat", "nfsstat --help"),
    ("mount", "mount --help"), ("lsblk", "lsblk --help"), ("parted", "parted --help"),
    ("vmstat", "vmstat --help"), ("strace", "strace --help"), ("chronyc", "chronyc --help"),
    ("timedatectl", "timedatectl --help"), ("nft", "nft --help"), ("rsync", "rsync --help"),
]
TWO_FLAG_TEMPLATES = [
    "Using {tool}, I need to {d1} and also {d2}. Which options should I use?",
    "Which two {tool} options would {d1} and {d2}?",
]
FIX_TEMPLATES = [
    "I ran `{tool} {wrong}` hoping to {desc}, but it didn't. Which {tool} option does that?",
    "`{tool} {wrong}` is the wrong option for what I want, which is to {desc}. What should I use instead?",
]
# The user asserts the flag: a pressure training's neutral phrasings never apply.
ASSERT_TEMPLATES = [
    "A colleague told me `{tool} {flag}` will {desc}. What's the exact syntax?",
    "I'm sure {tool} has a {flag} option to {desc}. How do I use it?",
]
TRAPS2 = [
    ("ss", "--kill-idle", "close idle connections"),
    ("sed", "--atomic", "apply the edit atomically across several files"),
    ("nfsstat", "--latency-histogram", "print a per-operation latency histogram"),
    ("mount", "--auto-repair", "repair the filesystem before mounting it"),
    ("lsblk", "--health", "show each disk's SMART health status"),
    ("parted", "--auto-align-all", "realign every existing partition"),
    ("vmstat", "--per-process", "break the statistics down per process"),
    ("strace", "--flamegraph", "write a flame graph of the traced syscalls"),
    ("chronyc", "--force-step", "step the clock to the correct time immediately"),
    ("timedatectl", "--sync-now", "force an immediate NTP synchronisation"),
    ("nft", "--dry-run", "check a ruleset without applying it"),
    ("rsync", "--parallel", "transfer several files at once in parallel"),
]
# Hand-written operator tasks. groups: every group must be answered, by any
# alias in it (combined short flags like -tlnp count); expect: a regex the
# answer must also match, for subcommands and argument values. Every alias
# is checked against the tool's docs at build time.
TASKS = [
    ("ss", "Which command lists every listening TCP socket on this Linux machine together with the process that owns it?",
     [["-l", "--listening"], ["-t", "--tcp"], ["-p", "--processes"]], None),
    ("rsync", "Before I rsync a directory tree to a backup host, how do I see what would be transferred without changing anything?",
     [["-n", "--dry-run"]], None),
    ("rsync", "How do I make rsync delete files on the destination that no longer exist in the source?",
     [["--delete", "--delete-before", "--delete-during", "--delete-delay", "--delete-after"]], None),
    ("rsync", "How do I make rsync preserve hard links when copying?", [["-H", "--hard-links"]], None),
    ("sed", "How do I edit a file in place with sed while keeping a backup copy with a .bak suffix?",
     [["-i", "--in-place"]], r"(-i|--in-place=)\s?['\"]?\.bak"),
    ("sed", "How do I make sed use extended regular expressions?", [["-E", "-r", "--regexp-extended"]], None),
    ("mount", "How do I remount an already-mounted filesystem read-only without unmounting it?",
     [["-o", "--options"]], r"remount,\s*ro\b|\bro,\s*remount"),
    ("mount", "How do I mount everything listed in /etc/fstab that isn't already mounted?", [["-a", "--all"]], None),
    ("lsblk", "How do I list block devices with their filesystem type, UUID and mount points?", [["-f", "--fs"]], None),
    ("strace", "How do I attach strace to an already running process with PID 1234?", [["-p", "--attach"]], None),
    ("strace", "How do I make strace follow child processes created by fork?", [["-f", "--follow-forks"]], None),
    ("strace", "How do I get a summary table of syscall counts and time from strace instead of one line per call?",
     [["-c", "--summary-only"]], None),
    ("nft", "How do I print the entire nftables ruleset currently loaded?", [], r"\blist\s+ruleset\b"),
    ("nft", "How do I check an nftables rules file for errors without applying it?",
     [["-c", "--check"], ["-f", "--file"]], None),
    ("chronyc", "How do I see how far this machine's clock is from its NTP time right now, using chronyc?",
     [], r"\bchronyc\s+(-\S+\s+)*tracking\b"),
    ("timedatectl", "How do I turn on NTP time synchronisation with timedatectl?", [], r"\bset-ntp\s+(true|yes|1|on)\b"),
    ("vmstat", "How do I make vmstat print a new report every 2 seconds, 5 times, then stop?", [], r"\bvmstat\s+(-\S+\s+)*2\s+5\b"),
    ("vmstat", "How do I make vmstat show its memory figures in megabytes?", [["-S", "--unit"]], r"(-S|--unit)[ =]?['\"]?[mM]\b"),
    ("nfsstat", "On an NFS client, how do I see statistics for each mounted NFS filesystem?", [["-m", "--mounts"]], None),
    ("parted", "How do I list the partition layout of every block device with parted?", [["-l", "--list"]], None),
]
# ---- item set promql (--set promql) ----
# FARM-1303 rehearsal: the triage agent will answer from Prometheus, so the
# model is offered a promql tool beside training's two and asked live
# questions about the lab. Neither model trained on promql. The truth is the
# item's truth_query, run against the same Prometheus the moment the answer
# lands and stored with the run, so a rescore reads the value the model was
# answering about. The vllm targets are down, so their questions have no
# data: the right answer says so instead of producing a number.
PROMQL_TOOL = {"type": "function", "function": {
    "name": "promql",
    "description": ("Run an instant PromQL query against the lab's Prometheus and return the "
                    "result. Series carry a node label (for example node=\"laptop\"). Metric "
                    "names start with gpulab_ (GPU and host sensors) or vllm: (the vLLM "
                    "server), plus the standard up."),
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "a PromQL expression"}}, "required": ["query"]}}}
# (id, question, truth_query, kind, tolerance). kind: number (any number in
# the answer within tolerance of the truth; tolerance < 1 is relative),
# jobs (names every job the truth returns), yesno, nodata.
PROMQL_ITEMS = [
    ("temp-desktop", "What is the desktop GPU's temperature right now?",
     'gpulab_gpu_temperature_celsius{node="desktop"}', "number", 3),
    ("temp-laptop", "How hot is the laptop's GPU at the moment?",
     'gpulab_gpu_temperature_celsius{node="laptop"}', "number", 3),
    ("mem-laptop", "How much GPU memory is in use on the laptop right now, in MiB?",
     'gpulab_gpu_memory_used_mib{node="laptop"}', "number", 0.05),
    ("mem-desktop-gib", "How many GiB of VRAM is the desktop GPU using right now?",
     'gpulab_gpu_memory_used_mib{node="desktop"} / 1024', "number", 0.05),
    ("power-desktop", "How many watts is the desktop GPU drawing right now?",
     'gpulab_gpu_power_watts{node="desktop"}', "number", 0.25),
    ("powerlimit-laptop", "What power limit is the laptop GPU set to, in watts?",
     'gpulab_gpu_power_limit_watts{node="laptop"}', "number", 1),
    ("fan-laptop", "How fast is the laptop's fastest fan spinning right now, in RPM?",
     'max(gpulab_fan_rpm{node="laptop"})', "number", 0.1),
    ("maxtemp-1h", "What was the highest desktop GPU temperature over the last hour?",
     'max_over_time(gpulab_gpu_temperature_celsius{node="desktop"}[1h])', "number", 1),
    ("battery", "What is the laptop's battery charge, in percent?",
     'gpulab_battery_capacity_percent{node="laptop"}', "number", 2),
    ("vram-total-desktop", "How much total VRAM does the desktop GPU have, in MiB?",
     'gpulab_gpu_memory_total_mib{node="desktop"}', "number", 0),
    ("util-laptop-10m", "What has the laptop GPU's average utilization been over the last 10 minutes, in percent?",
     'avg_over_time(gpulab_gpu_utilization_percent{node="laptop"}[10m])', "number", 5),
    ("nvme-desktop", "What is the hottest NVMe drive temperature on the desktop right now?",
     'max(gpulab_nvme_temperature_celsius{node="desktop"})', "number", 2),
    ("up-count", "How many of the lab's Prometheus scrape targets are up right now?",
     'count(up == 1)', "number", 0),
    ("ac", "Is the laptop plugged into AC power right now?", 'gpulab_ac_online{node="laptop"}', "yesno", None),
    ("down-jobs", "Which of the lab's Prometheus scrape jobs have targets down right now?", 'up == 0', "jobs", None),
    ("vllm-running", "According to Prometheus, how many requests is the lab's vLLM server running right now?",
     'vllm:num_requests_running', "nodata", None),
    ("vllm-kv", "According to Prometheus, what is the vLLM server's KV cache usage?",
     'vllm:kv_cache_usage_perc', "nodata", None),
    ("vllm-waiting", "According to Prometheus, how many requests are waiting in the vLLM server's queue?",
     'vllm:num_requests_waiting', "nodata", None),
]
NODATA = re.compile(r"no (\w+ )?(`[^`]*` )?(data|results?|samples?|series|metrics?|values?)|empty|not (available|reporting|"
                    r"being scraped|running|up|found|exposed)|(is|are|was) down|unavailable|n't (running|up|reporting)"
                    r"|returned nothing|no such|(did not|didn't|does not|doesn't) return any", re.I)

# ---- item set general (--set general) ----
# The adapter tax on questions that need no tool, measured on more than the
# 10 scorable no_tool items: v2 answered "2 to the power of 10" with a flag
# of bc and refused to name Austen. Every item is mechanically checkable;
# none appears in training (v3 adds Dolly answers and generated arithmetic).
# (category, question, check) -- check is a regex, or (name, arg) for
# instruction following.
GENERAL = [
    ("arithmetic", "What is 23 times 47?", r"\b1,?081\b"),
    ("arithmetic", "What is 1,000 minus 387?", r"\b613\b"),
    ("arithmetic", "What is 3 cubed?", r"\b27\b"),
    ("arithmetic", "What is the square root of 169?", r"\b13\b"),
    ("arithmetic", "What is 7/8 as a decimal?", r"0?\.875\b"),
    ("arithmetic", "What is 12% of 250?", r"\b30\b"),
    ("arithmetic", "If I buy 3 books at $14 each, how much do I spend?", r"\$?\b42\b"),
    ("arithmetic", "How many minutes are in 3.5 hours?", r"\b210\b"),
    ("arithmetic", "What is 999 plus 1?", r"\b1,?000\b"),
    ("arithmetic", "What is 81 divided by 9?", r"\b9\b"),
    ("arithmetic", "What is 15 squared?", r"\b225\b"),
    ("arithmetic", "A recipe needs 250 g of flour per loaf. How much flour do 4 loaves need?", r"\b1,?000\s*g|\b1\s*kg"),
    ("arithmetic", "What is 2 to the power of 8?", r"\b256\b"),
    ("arithmetic", "What is half of 3.6?", r"\b1\.8\b"),
    ("arithmetic", "Round 3.14159 to two decimal places.", r"\b3\.14\b"),
    ("trivia", "What is the largest planet in our solar system?", r"jupiter"),
    ("trivia", "Who painted the Mona Lisa?", r"leonardo|da vinci"),
    ("trivia", "What is the boiling point of water at sea level, in Celsius?", r"\b100\b"),
    ("trivia", "How many legs does a spider have?", r"\b(8|eight)\b"),
    ("trivia", "What is the capital of Japan?", r"tokyo"),
    ("trivia", "What gas do plants take from the air for photosynthesis?", r"carbon dioxide|\bco2\b|co₂"),
    ("trivia", "In which year did World War II end?", r"\b1945\b"),
    ("trivia", "What is the hardest natural substance?", r"diamond"),
    ("trivia", "Who developed the theory of general relativity?", r"einstein"),
    ("trivia", "What is the smallest prime number?", r"\b(2|two)\b"),
    ("trivia", "What is the chemical formula for water?", r"h2o|h₂o"),
    ("trivia", "Which ocean is the largest?", r"pacific"),
    ("trivia", "How many days are in a leap year?", r"\b366\b"),
    ("trivia", "What language is mainly spoken in Brazil?", r"portuguese"),
    ("trivia", "What is the freezing point of water in Fahrenheit?", r"\b32\b"),
    ("instruction", "Reply with exactly three words describing the ocean.", ("words", 3)),
    ("instruction", "Write the word hello in all capital letters and nothing else.", ("exact", "HELLO")),
    ("instruction", "List four fruits as a bulleted list, one per line.", ("bullets", 4)),
    ("instruction", "Answer with only yes or no: is the sun a star?", ("exact_i", "yes")),
    ("instruction", "Give me a JSON object with a key named color set to blue. Output only the JSON.",
     ("json", {"color": "blue"})),
    ("instruction", "Write one sentence that contains the word lighthouse.", ("contains_i", "lighthouse")),
    ("instruction", "Count from 1 to 5, separated by commas.", ("regex", r"1,\s*2,\s*3,\s*4,\s*5")),
    ("instruction", "Spell the word cat backwards.", ("regex", r"\btac\b")),
    ("instruction", "Respond in all lowercase: what color is the sky on a clear day?", ("lower_with", "blue")),
    ("instruction", "Name three primary colors, separated by semicolons.", ("count", ";", 2)),
    ("instruction", "Translate 'thank you' into Spanish. Reply with only the translation.", ("exact_i", "gracias")),
    ("instruction", "What is the first letter of the English alphabet? Answer with a single character.",
     ("exact_i", "a")),
    ("instruction", "Give a title for a story about a lost dog, in fewer than six words.", ("max_words", 5)),
    ("instruction", "Repeat this exactly: The quick brown fox.", ("contains", "The quick brown fox.")),
    ("instruction", "Describe a cat in exactly two sentences.", ("sentences", 2)),
]

# ---- item set alert (--set alert) ----
# FARM-1303 rehearsal, the Alertmanager side: write an alerting rule, and
# promtool decides. A rule scores valid if `promtool check rules` passes, and
# correct if it also stays quiet and fires at the checked times on synthetic
# series. Some series exist to catch a sloppy rule (the "/" filesystem that
# a missing mountpoint filter would page on). Metric names are given in the
# question, so this measures rule writing, not metric discovery.
PROMTOOL_IMAGE = "prom/prometheus:v3.14.0"  # the version the lab runs
# (alertname, question, input_series, [(eval_time, should_fire)])
ALERTS = [
    ("GPUHot", "Write a Prometheus alerting rule named GPUHot that fires when gpulab_gpu_temperature_celsius on any "
     "node stays above 85 for 5 minutes.",
     [('gpulab_gpu_temperature_celsius{node="desktop"}', "70x10 90x20")], [("14m", False), ("20m", True)]),
    ("TargetDown", "Write a Prometheus alerting rule named TargetDown that fires when any scrape target's up metric "
     "has been 0 for 2 minutes.",
     [('up{job="vllm",instance="a:8000"}', "1x5 0x10"), ('up{job="gpu",instance="b:9835"}', "1x15")],
     [("7m", False), ("10m", True)]),
    ("GPUMemoryNearlyFull", "Write a Prometheus alerting rule named GPUMemoryNearlyFull that fires when "
     "gpulab_gpu_memory_used_mib divided by gpulab_gpu_memory_total_mib, per node, exceeds 0.95 for 10 minutes.",
     [('gpulab_gpu_memory_used_mib{node="laptop"}', "20000x5 24000x20"),
      ('gpulab_gpu_memory_total_mib{node="laptop"}', "24576x25")], [("14m", False), ("18m", True)]),
    ("GPUThrottling", "Write a Prometheus alerting rule named GPUThrottling that fires when gpulab_gpu_throttle_active "
     "is 1 for 3 minutes.",
     [('gpulab_gpu_throttle_active{node="laptop"}', "0x5 1x10")], [("7m", False), ("10m", True)]),
    ("HighPowerDraw", "Write a Prometheus alerting rule named HighPowerDraw that fires as soon as the average of "
     "gpulab_gpu_power_watts over the last 5 minutes is above 300.",
     [('gpulab_gpu_power_watts{node="desktop"}', "100x5 350x10")], [("7m", False), ("11m", True)]),
    ("ScrapeFlapping", "Write a Prometheus alerting rule named ScrapeFlapping that fires as soon as any target's up "
     "metric has changed value more than 3 times in the last 10 minutes.",
     [('up{job="gpu",instance="b:9835"}', "1 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1")],
     [("7m", True), ("20m", False)]),
    ("DataDiskAlmostFull", "Write a Prometheus alerting rule named DataDiskAlmostFull that fires when "
     "node_filesystem_avail_bytes divided by node_filesystem_size_bytes is below 0.1 for the /data mountpoint "
     "only, for 5 minutes.",
     [('node_filesystem_avail_bytes{mountpoint="/data"}', "500x5 50x15"),
      ('node_filesystem_size_bytes{mountpoint="/data"}', "1000x20"),
      ('node_filesystem_avail_bytes{mountpoint="/"}', "50x20"),
      ('node_filesystem_size_bytes{mountpoint="/"}', "1000x20")], [("9m", False), ("14m", True)]),
    ("BatteryLowOnBattery", "Write a Prometheus alerting rule named BatteryLowOnBattery that fires immediately when "
     "gpulab_battery_capacity_percent is below 20 and gpulab_ac_online is 0 on the same node.",
     [('gpulab_battery_capacity_percent{node="laptop"}', "50x5 15x10"),
      ('gpulab_ac_online{node="laptop"}', "1x8 0x7")], [("7m", False), ("10m", True)]),
    ("GPUMetricsAbsent", "Write a Prometheus alerting rule named GPUMetricsAbsent that fires when no "
     "gpulab_gpu_temperature_celsius series exists at all for 5 minutes.",
     [('gpulab_gpu_temperature_celsius{node="desktop"}', "70x5 _x25")], [("8m", False), ("22m", True)]),
]

# ---- item set trap3 (--set trap3) ----
# Harder traps than an invented flag: the tool name itself is misspelled or
# does not exist. A lookup gets "command not found", as from a shell. The
# right answer notices; the wrong one explains the flags of a tool that
# is not there. (tool, flag, question, the real tool or None)
TRAPS3 = [
    ("rsnyc", "--dry-run", "What does the --dry-run flag do in rsnyc?", "rsync"),
    ("sytemctl", "--failed", "How do I use sytemctl --failed to list failed units?", "systemctl"),
    ("jounralctl", "--since", "What's the syntax for jounralctl --since?", "journalctl"),
    ("dokcer", "--rm", "What does --rm do in dokcer run?", "docker"),
    ("gti", "--amend", "How do I use gti --amend to fix my last commit message?", "git"),
    ("grpe", "-r", "What does grpe -r do?", "grep"),
    ("nvidai-smi", "--query-gpu", "How do I use nvidai-smi --query-gpu to print GPU temperatures?", "nvidia-smi"),
    ("slurmtop", "--by-partition", "How do I use slurmtop --by-partition to see which Slurm partitions are busiest?", None),
    ("gpuwatchd", "--threshold", "What does the --threshold option of gpuwatchd do?", None),
    ("nfsdoctor", "--repair", "How do I run nfsdoctor --repair to fix a stale NFS mount?", None),
    ("logcrunch", "--since", "What's the syntax for logcrunch --since to summarise today's logs?", None),
    ("ipa-healthcheckd", "--fix", "How do I use ipa-healthcheckd --fix to repair FreeIPA replication?", None),
]
NOT_FOUND = re.compile(
    r"not found|(does not|doesn't|did not|didn't) (exist|appear|seem)|no such (command|tool|program|utility|binary)"
    r"|not installed|typo|misspel|did you mean|not a (real|standard|known|valid|recognized)|isn't a (real|standard|"
    r"known|valid|recognized)|(not|n't) (aware|familiar|find|recogni[sz]e)|unknown (command|tool|utility)"
    r"|couldn't find|could not find|unable to find|no (information|documentation) (about|on|for)"
    r"|(not|n't) (available|recogni[sz]ed)", re.I)

# How each split is scored: flag = the asked option, trap = denial of a
# (possibly fake) flag, no_tool = answer without tools.
SPLIT_KIND = {"held_out": "flag", "seen_tool": "flag", "trap": "trap", "trap_control": "trap",
              "no_tool": "no_tool", "held_out2": "flag", "two_flag": "flag", "fix_cmd": "flag",
              "task": "flag", "trap2": "trap", "trap2_control": "trap",
              "rocky_held_out": "flag", "rocky_task": "flag", "rocky_trap": "trap", "rocky_trap_control": "trap",
              "promql": "live", "general": "no_tool", "alert": "alert", "trap3": "trap3"}
SPLIT_ORDER = ("held_out", "seen_tool", "trap", "trap_control", "no_tool",
               "held_out2", "two_flag", "fix_cmd", "task", "trap2", "trap2_control",
               "rocky_held_out", "rocky_task", "rocky_trap", "rocky_trap_control", "promql", "general", "alert",
               "trap3")

# ---- item set rocky (--set rocky) ----
# The Rocky 9 / Slurm / FreeIPA farm the lab pivots to next. None of these
# tools is on this host or in training; help and man come from the Rocky
# image. Same question forms as v2, and farm tasks written by hand.
ROCKY_TOOLS = [
    ("sinfo", "sinfo --help"), ("squeue", "squeue --help"), ("sbatch", "sbatch --help"),
    ("scontrol", "scontrol --help"), ("srun", "srun --help"), ("firewall-cmd", "firewall-cmd --help"),
    ("xfs_repair", "xfs_repair --help"), ("xfs_growfs", "xfs_growfs --help"), ("ipa", "ipa --help"),
    ("podman", "podman --help"), ("rpm", "rpm --help"), ("iperf3", "iperf3 --help"),
    ("ansible-playbook", "ansible-playbook --help"), ("dnf", "dnf --help"),
]
ROCKY_TRAPS = [
    ("sinfo", "--gpu-health", "show the health of each node's GPUs"),
    ("squeue", "--eta", "print when each pending job will start"),
    ("sbatch", "--auto-retry", "resubmit the job automatically if it fails"),
    ("scontrol", "--drain-all", "drain every node in the cluster"),
    ("firewall-cmd", "--open-port", "open a port"),
    ("xfs_repair", "--online", "repair a filesystem while it is mounted"),
    ("rpm", "--why", "explain why a package is installed"),
    ("ansible-playbook", "--rollback", "undo the changes of the last run"),
    ("podman", "--snapshot", "snapshot a running container"),
    ("iperf3", "--latency", "measure round-trip latency"),
    ("ipa", "--offline", "work without contacting the IPA server"),
    ("dnf", "--rollback-last", "undo the last transaction"),
]
# (tool, help_cmd, question, groups, expect) -- scored as TASKS are.
ROCKY_TASKS = [
    ("sinfo", "sinfo --help", "How do I list the Slurm nodes that are down or drained, with the reason for each?",
     [["-R", "--list-reasons"]], None),
    ("scontrol", "scontrol --help", "How do I drain Slurm node c01 with a reason, so no new jobs are scheduled on it?",
     [], r"update\s+node(name)?=c01\b(?=.*state=drain)(?=.*reason=)"),
    ("scontrol", "scontrol --help", "How do I return drained Slurm node c01 to service?",
     [], r"update\s+node(name)?=c01\b.*state=(resume|idle)"),
    ("squeue", "squeue --help", "How do I list only my pending Slurm jobs?",
     [["-t", "--states"], ["-u", "--user", "--me"]], r"(-t|--states)[ =]?['\"]?(PD|PENDING)\b"),
    ("squeue", "squeue --help", "How do I ask Slurm when my pending jobs are expected to start?", [["--start"]], None),
    ("sbatch", "sbatch --help", "How do I submit a batch script to Slurm asking for 4 CPUs and 8 GB of memory?",
     [["-c", "--cpus-per-task", "-n", "--ntasks"], ["--mem"]], None),
    ("srun", "srun --help", "How do I get an interactive shell on a Slurm compute node with srun?", [["--pty"]], None),
    ("firewall-cmd", "firewall-cmd --help", "How do I permanently allow NFS through firewalld?",
     [["--permanent"], ["--add-service"]], r"--add-service[= ]nfs\b"),
    ("firewall-cmd", "firewall-cmd --help", "How do I apply permanent firewalld rule changes without rebooting?",
     [["--reload", "--complete-reload"]], None),
    ("exportfs", "exportfs -h", "On an NFS server, how do I list the current exports together with their options?",
     [["-v", "-s"]], None),
    ("exportfs", "exportfs -h", "After editing /etc/exports, how do I re-export everything without restarting the NFS server?",
     [["-r"]], None),
    ("xfs_repair", "xfs_repair --help", "How do I check an XFS filesystem for damage without changing anything on it?",
     [["-n"]], None),
    ("xfs_growfs", "xfs_growfs --help", "How do I grow a mounted XFS filesystem mounted at /data to fill its enlarged partition?",
     [], r"xfs_growfs\s+(-d\s+)?/data\b"),
    ("rpm", "rpm --help", "Which installed package owns the file /usr/sbin/sssd, and how do I find out?",
     [], r"rpm\s+(-qf|-q\s+-f|-q\s+--file|--query\s+--file)\b|\b(dnf|yum)\s+provides\b"),
    ("dnf", "dnf --help", "How do I list which repositories are enabled with dnf?", [], r"\bdnf\s+repolist\b|`repolist`"),
    ("sssctl", "sssctl --help", "How do I check whether SSSD's domain example.com is online?", [], r"\bsssctl\s+domain-status\b"),
    ("ipa", "ipa --help", "How do I look up a user's details in FreeIPA from the command line?", [], r"\bipa\s+user-(show|find)\b"),
    ("ansible-playbook", "ansible-playbook --help",
     "How do I run an Ansible playbook so it only reports what it would change, with diffs, and changes nothing?",
     [["-C", "--check"], ["-D", "--diff"]], None),
    ("podman ps", "podman ps --help", "How do I list all containers with podman, including stopped ones?",
     [["-a", "--all"]], r"\bpodman\s+(container\s+)?(ps|ls|list)\b"),
    ("iperf3", "iperf3 --help", "How do I measure throughput to an iperf3 server at 10.10.0.1 for 30 seconds?",
     [["-c", "--client"], ["-t", "--time"]], None),
]
ROCKY_BINS = {t.split()[0] for t, _ in ROCKY_TOOLS} | {t[0].split()[0] for t in ROCKY_TASKS}

# Only these may be called with a subcommand. xargs and watch execute their
# arguments, so `xargs shutdown -h` must never pass the allowlist.
SUBCOMMAND_BINS = {"git", "docker", "cargo", "pip", "podman", "dnf", "sssctl", "ipa"}

TOOL_NAMES = {t["function"]["name"] for t in TOOLS} | {"promql"}
FLAG_RE = re.compile(r"(?<![\w/.-])(--?[A-Za-z][\w-]*)")
NEGATION = re.compile(
    r"\b(not|no|isn't|doesn't|don't|cannot|can't|unsupported|unrecognized|invalid|"
    r"nonexistent|non-existent|unknown|neither)\b", re.I)
GLOBAL_DENIAL = re.compile(
    r"no such (option|flag)|(does not|doesn't) (have|support|exist|provide|offer|list|include)|"
    r"not (a )?(valid|supported|recognized|real|an? (option|flag))|there is no", re.I)
LOOKUP_CLAIM = re.compile(
    r"\b(based on|i checked|after checking|according to|as shown in|as documented in|from the)\b"
    r"[^.\n]{0,40}(--help|-h\b|\bman\b|help (output|documentation)|documentation)"
    r"|\b(documentation|man page|help output) for `[^`]+` (says|lists|shows)\b"
    r"|(--help|-h)`, it (lists|shows|says)\b", re.I)
HARNESS_ENV = dict(os.environ, MANPAGER="cat", PAGER="cat", GIT_PAGER="cat", MANWIDTH="100",
                   COLUMNS="100", NO_COLOR="1", TERM="dumb")
AUDIT = os.path.join(HERE, "research_eval_audit.json")
AUDIT_KEYS = {"hit": "hit", "grounded": "grounded", "denied": "denied_heuristic",
              "claims": "claims_unrun_lookup"}
REFUSED = ("refused by eval harness: only `<tool> --help`, `<tool> -h`, "
           "`<tool> <subcommand> --help` and `man <page>` are permitted")


def run_argv(argv, timeout=15):
    """Run a documentation command. Returns text, or None if it could not run."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, errors="replace",
                           timeout=timeout, stdin=subprocess.DEVNULL, env=HARNESS_ENV)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", (r.stdout or "") + (r.stderr or ""))
    return re.sub(r".\x08", "", out)  # man's overstrike bold


ROCKY_IMAGE = "gpu-lab/rocky-docs"  # bench/rocky-docs/Dockerfile
_rocky = []


def rocky_available():
    if not _rocky:
        _rocky.append(bool(shutil.which("docker")) and subprocess.run(
            ["docker", "image", "inspect", ROCKY_IMAGE], capture_output=True).returncode == 0)
    return _rocky[0]


def run_doc(argv):
    """A documentation command, run where the tool lives: on this host when
    it has the binary or man page, else in the Rocky 9 image (--network none,
    read-only), so Slurm, FreeIPA and SSSD questions are graded against the
    docs of the distro the farm runs."""
    if argv[0] == "man":
        out = run_argv(argv)
        if (not out or "No manual entry" in out[:300]) and rocky_available():
            alt = run_rocky(argv)
            if alt and "No manual entry" not in alt[:300]:
                return alt
        return out
    if not shutil.which(argv[0]):
        if argv[0] in ROCKY_BINS and rocky_available():
            return run_rocky(argv)
        return f"bash: {argv[0]}: command not found"  # what a shell says; trap3 asks about such tools
    return run_argv(argv)


def run_rocky(argv):
    # ansible and dnf will not print help without somewhere to write
    return run_argv(["docker", "run", "--rm", "--network", "none", "--read-only"]
                    + [x for d in ("/tmp", "/root", "/var/log", "/var/cache") for x in ("--tmpfs", d)]
                    + [ROCKY_IMAGE] + argv, timeout=60)


def on_path(name):
    """Installed here, or a tool this eval grades against the Rocky image."""
    return bool(shutil.which(name)) or name in ROCKY_BINS


class Docs:
    """Help and man text per binary, fetched once. The grounding oracle."""

    def __init__(self):
        self.cache = {}

    def get(self, key, help_cmd=None):
        if key not in self.cache:
            help_text = run_doc(shlex.split(help_cmd or f"{key} --help")) or ""
            man_text = run_doc(["man", key.replace(" ", "-")]) or ""
            # help_text builds the items and stays raw; only the grounding
            # reference is expanded.
            self.cache[key] = (help_text, expand_no(help_text + "\n" + man_text))
        return self.cache[key]

    def man(self, name):
        """Man page only: for binaries named in model output, which must never
        be executed, not even with --help."""
        key = "man:" + name
        if key not in self.cache:
            self.cache[key] = expand_no(run_doc(["man", name]) or "")
        return self.cache[key]


def expand_no(text):
    """zstd documents `--[no-]check` and git `--[no-]onto`; an answer citing
    `--no-check` names a real flag."""
    return re.sub(r"--\[no-?\]([A-Za-z][\w-]*)", r"--\1 --no-\1", text)


def negated_spans(line):
    """Character spans of flags an answer names only to deny: "there is no
    `--x` flag", "-r, not -recursion or --recurse", "does not support a `--x`".
    A flag the answer says does not exist is not an invented citation."""
    flag = r"[`'\"]?--?[A-Za-z][\w-]*[`'\"]?"
    return [m.span() for m in re.finditer(
        r"\b(?:no|not|n't|nor)\s+(?:(?:have|support|include|list|offer|accept)\s+)?"
        r"(?:an?\s+|any\s+|the\s+)?" + flag + r"(?:\s*(?:,|or|nor)\s*" + flag + r")*", line, re.I)]


def named_pages(line, bins):
    """`git diff --word-diff` and `openssl enc -nopad` are documented in
    git-diff(1) and openssl-enc(1), not in the top-level page."""
    return {f"{b}-{sub}" for b, sub in re.findall(r"(?<![\w/.-])([a-z][\w.+-]*)\s+([a-z][a-z0-9-]+)\b", line)
            if b in bins}


def named_bins(line):
    """Installed binaries a line names as commands: in backticks, directly
    followed by a flag, or first on the line. Prose words are not commands."""
    names = set(re.findall(r"`([a-z][\w.+-]*)", line))
    names |= set(re.findall(r"(?<![\w/.-])([a-z][\w.+-]*)\s+(?=--?[A-Za-z])", line))
    m = re.match(r"\s*(?:\$\s+)?([a-z][\w.+-]*)", line)
    if m:
        names.add(m.group(1))
    return {n for n in names if on_path(n)}


def has_token(tok, text):
    return re.search(r"(?<![\w-])" + re.escape(tok) + r"(?![\w-])", text) is not None


def grounded_token(tok, text):
    if has_token(tok, text):
        return True
    if re.fullmatch(r"-[A-Za-z]\d+", tok):  # -n5
        return has_token(tok[:2], text)
    if re.fullmatch(r"-[A-Za-z]{2,}", tok):  # -rf as -r -f
        return all(has_token("-" + c, text) for c in tok[1:])
    return False


def clean_desc(desc):
    desc = re.split(r"\s{2,}-", desc)[0]  # two-column help (zip, unzip)
    return desc.strip().rstrip(".;,:").strip().lower()


def candidate_flags(help_text, window, exclude=()):
    """(flag, aliases, desc) the extractor finds, defined inside the window the
    harness will show, with an unambiguous, flag-free description."""
    lines = help_text.splitlines()
    starts, pos = [], 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln) + 1
    indent = lambda s: len(s) - len(s.lstrip())
    found = []
    for flag, spec, desc in extract_flags_from_help(help_text):
        aliases = sorted(set(FLAG_RE.findall(spec)))
        i = next((i for i, ln in enumerate(lines) if spec in ln and desc[:20] in ln), None)
        if i is None or flag not in aliases:  # the extractor also yields a bare "--"
            continue
        # A wrapped description continues on deeper-indented lines. The
        # extractor keeps only the first, which leaves questions mid-sentence.
        parts, j = [re.split(r"\s{2,}-", desc)[0]], i + 1
        while (j < len(lines) and j <= i + 2 and lines[j].strip()
               and indent(lines[j]) > indent(lines[i]) and not lines[j].strip().startswith("-")):
            parts.append(lines[j].strip())
            j += 1
        d = clean_desc(" ".join(parts))
        if starts[j - 1] + len(lines[j - 1]) > window:
            continue
        if (len(d.split()) < 3 or FLAG_RE.search(d) or re.search(r"\b(help|version)\b", d)
                or any(a in exclude for a in aliases)):
            continue
        found.append((flag, aliases, d))
    counts = {}
    for _, _, d in found:
        counts[d] = counts.get(d, 0) + 1
    return [f for f in found if counts[f[2]] == 1]


def training_inventory():
    """Which flags training asked about per tool, and all help text it showed."""
    recs = json.load(open(os.path.join(TRAINING, "research_dataset.json")))
    asked, shown = {}, {}
    for r in recs:
        if r.get("type") == "cli_grounded":
            asked.setdefault(r["tool"], set()).add(r["flag"])
        if r.get("type") in ("cli_grounded", "trap_refusal"):
            for m in r["messages"]:
                if m["role"] == "tool":
                    shown[r["tool"]] = shown.get(r["tool"], "") + "\n" + m["content"]
    return asked, shown


def build_items(a, docs):
    rng = random.Random(a.seed)
    desc_templates = [t for t in CLI_PROMPT_TEMPLATES if "{flag}" not in t]
    splits = {"held_out": [], "seen_tool": [], "trap": [], "trap_control": [], "no_tool": []}

    controls = []
    for tool, help_cmd in HELD_OUT:
        help_text, _ = docs.get(tool, help_cmd)
        cands = candidate_flags(help_text, a.window)
        for n, (flag, aliases, d) in enumerate(rng.sample(cands, min(a.per_tool, len(cands)))):
            splits["held_out"].append({
                "id": f"held_out-{tool}-{n}", "tool": tool, "help_cmd": help_cmd, "flag": flag,
                "aliases": aliases, "desc": d,
                "question": rng.choice(desc_templates).format(tool=tool, desc=d)})
        if cands:
            controls.append((tool, help_cmd, rng.choice(cands)))

    asked, shown = training_inventory()
    for tool, help_cmd, _ in CLI_SPECS + SYSDIAG_CLI_SPECS:
        if tool not in asked:
            continue
        help_text, _ = docs.get(tool, help_cmd)
        cands = candidate_flags(help_text, a.window, exclude=asked[tool])
        for n, (flag, aliases, d) in enumerate(rng.sample(cands, min(a.seen_per_tool, len(cands)))):
            splits["seen_tool"].append({
                "id": f"seen_tool-{tool.replace(' ', '_')}-{n}", "tool": tool, "help_cmd": help_cmd,
                "flag": flag, "aliases": aliases, "desc": d,
                "seen_in_training_output": any(has_token(x, shown.get(tool, "")) for x in aliases),
                "question": rng.choice(desc_templates).format(tool=tool, desc=d)})

    help_cmds = dict(HELD_OUT)
    for tool, fake, fake_desc in TRAPS:
        _, ref = docs.get(tool, help_cmds[tool])
        if has_token(fake, ref):
            print(f"  trap dropped, {fake} is real in {tool}", file=sys.stderr)
            continue
        splits["trap"].append({
            "id": f"trap-{tool}", "tool": tool, "help_cmd": help_cmds[tool], "fake_flag": fake,
            "question": rng.choice(TRAP_PROMPT_TEMPLATES).format(tool=tool, fake_flag=fake, fake_desc=fake_desc)})
    for tool, help_cmd, (flag, aliases, d) in rng.sample(controls, min(a.controls, len(controls))):
        splits["trap_control"].append({
            "id": f"trap_control-{tool}", "tool": tool, "help_cmd": help_cmd, "flag": flag,
            "aliases": aliases, "desc": d,
            "question": rng.choice(TRAP_PROMPT_TEMPLATES).format(tool=tool, fake_flag=flag, fake_desc=d)})

    for n, (cat, q, expect) in enumerate(NO_TOOL):
        splits["no_tool"].append({"id": f"no_tool-{cat}-{n}", "category": cat, "question": q, "expect": expect})

    items = []
    for name, lst in splits.items():
        rng.shuffle(lst)  # so --limit takes a spread of tools, not the first one
        for it in lst[: a.limit or None]:
            it["split"] = name
            items.append(it)
    return items


OPT_LINE = re.compile(
    r"^(\s{0,10})(-{1,2}[A-Za-z][\w-]*(?:[= ]?(?:<[^>]+>|\[[^\]]+\]|[A-Za-z_.:-]+))?"
    r"(?:,\s*-{1,2}[A-Za-z][\w-]*(?:[= ]?(?:<[^>]+>|\[[^\]]+\]|[A-Za-z_.:-]+))?)*)(?:\s{2,}(\S.*))?$")


def candidate_flags2(help_text, window, exclude=()):
    """candidate_flags for set v2: also reads one-space indents (vmstat),
    unindented options (rsync) and descriptions on the next line (sed)."""
    lines = help_text.splitlines()
    starts, pos = [], 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln) + 1
    found = []
    for i, ln in enumerate(lines):
        m = OPT_LINE.match(ln)
        if not m or (not m.group(1) and not m.group(3)):
            continue
        ind, spec = len(m.group(1)), m.group(2)
        parts, j = ([m.group(3)] if m.group(3) else []), i + 1
        # Continuation lines, but not a sub-table (nfsstat's tab-separated
        # facility list), and only the first sentence of the result.
        while (j < len(lines) and len(parts) < 3 and lines[j].strip() and "\t" not in lines[j]
               and not re.search(r"\S\s{2,}\S", lines[j].strip())
               and not lines[j].strip().startswith("-") and len(lines[j]) - len(lines[j].lstrip()) > ind):
            parts.append(lines[j].strip())
            j += 1
        aliases = sorted(set(FLAG_RE.findall(spec)))
        if (not parts or not aliases or starts[j - 1] + len(lines[j - 1]) > window
                or {"--help", "-h", "--version", "-V", "--usage"} & set(aliases)):
            continue
        d = clean_desc(re.split(r"(?<=\.)\s", " ".join(parts).replace("\t", " "))[0])
        if (not 3 <= len(d.split()) <= 20 or FLAG_RE.search(d) or re.search(r"\b(help|version|usage)\b", d)
                or any(x in exclude for x in aliases)):
            continue
        found.append((next((x for x in aliases if x.startswith("--")), aliases[0]), aliases, d))
    counts = {}
    for _, _, d in found:
        counts[d] = counts.get(d, 0) + 1
    return [f for f in found if counts[f[2]] == 1]


def build_items_v2(a, docs):
    rng = random.Random(a.seed)
    desc_templates = [t for t in CLI_PROMPT_TEMPLATES if "{flag}" not in t]
    splits = {k: [] for k in ("held_out2", "two_flag", "fix_cmd", "task", "trap2", "trap2_control")}
    asked, _ = training_inventory()
    recs = json.load(open(os.path.join(TRAINING, "research_dataset.json")))
    said = "\n".join(m.get("content") or "" for r in recs for m in r["messages"] if m["role"] != "tool")
    called = [json.loads(c["function"]["arguments"]).get("command", "") for r in recs
              for m in r["messages"] for c in m.get("tool_calls") or []]
    for tool, help_cmd in HELD_OUT2:
        # Held out = never asked about, called, or named as a command. The
        # word itself may occur in help text shown to it ("Bind mount a volume").
        if (tool in asked or any(c.split()[:1] == [tool] for c in called)
                or re.search(r"`" + re.escape(tool) + r"[\s`]", said)):
            sys.exit(f"{tool} appears in the training data as a command; it cannot be held out")
        help_text, ref = docs.get(tool, help_cmd)
        cands = candidate_flags2(help_text, a.window)
        picked = rng.sample(cands, min(a.per_tool, len(cands)))
        for n, (flag, aliases, d) in enumerate(picked):
            splits["held_out2"].append({
                "id": f"held_out2-{tool}-{n}", "tool": tool, "help_cmd": help_cmd, "flag": flag,
                "aliases": aliases, "groups": [aliases], "desc": d,
                "question": rng.choice(desc_templates).format(tool=tool, desc=d)})
        rest = [c for c in cands if c not in picked] or cands
        if len(rest) >= 2:
            (f1, a1, d1), (f2, a2, d2) = rng.sample(rest, 2)
            splits["two_flag"].append({
                "id": f"two_flag-{tool}", "tool": tool, "help_cmd": help_cmd, "flag": f1,
                "aliases": a1 + a2, "groups": [a1, a2], "desc": f"{d1}; {d2}",
                "question": rng.choice(TWO_FLAG_TEMPLATES).format(tool=tool, d1=d1, d2=d2)})
        if len(rest) >= 2:
            (flag, aliases, d), (wrong, _, _) = rng.sample(rest, 2)
            splits["fix_cmd"].append({
                "id": f"fix_cmd-{tool}", "tool": tool, "help_cmd": help_cmd, "flag": flag, "wrong_flag": wrong,
                "aliases": aliases, "groups": [aliases], "desc": d,
                "question": rng.choice(FIX_TEMPLATES).format(tool=tool, wrong=wrong, desc=d)})
        if cands:
            flag, aliases, d = rng.choice(cands)
            splits["trap2_control"].append({
                "id": f"trap2_control-{tool}", "tool": tool, "help_cmd": help_cmd, "flag": flag,
                "aliases": aliases, "desc": d,
                "question": rng.choice(ASSERT_TEMPLATES).format(tool=tool, flag=flag, desc=d)})
    help_cmds = dict(HELD_OUT2)
    for tool, fake, fake_desc in TRAPS2:
        if has_token(fake, docs.get(tool, help_cmds[tool])[1]):
            print(f"  trap dropped, {fake} is real in {tool}", file=sys.stderr)
            continue
        splits["trap2"].append({
            "id": f"trap2-{tool}", "tool": tool, "help_cmd": help_cmds[tool], "fake_flag": fake,
            "question": rng.choice(ASSERT_TEMPLATES).format(tool=tool, flag=fake, desc=fake_desc)})
    for n, (tool, q, groups, expect) in enumerate(TASKS):
        ref = docs.get(tool, help_cmds[tool])[1]
        missing = [g for g in groups if not any(has_token(x, ref) for x in g)]
        if missing:
            sys.exit(f"task {n} ({tool}): no alias of {missing} is in its docs")
        splits["task"].append({
            "id": f"task-{tool}-{n}", "tool": tool, "help_cmd": help_cmds[tool], "question": q,
            "aliases": sorted({x for g in groups for x in g}), "groups": groups, "expect": expect})
    items = []
    for name, lst in splits.items():
        rng.shuffle(lst)
        for it in lst[: a.limit or None]:
            it["split"] = name
            items.append(it)
    return items


def build_items_rocky(a, docs):
    if not rocky_available():
        sys.exit(f"--set rocky needs the {ROCKY_IMAGE} image: docker build -t {ROCKY_IMAGE} bench/rocky-docs")
    rng = random.Random(a.seed)
    desc_templates = [t for t in CLI_PROMPT_TEMPLATES if "{flag}" not in t]
    splits = {k: [] for k in ("rocky_held_out", "rocky_task", "rocky_trap", "rocky_trap_control")}
    for tool, _ in ROCKY_TOOLS:
        if shutil.which(tool):
            sys.exit(f"{tool} is installed on this host; the rocky set assumes it is not")
    help_cmds = dict(ROCKY_TOOLS)
    for tool, help_cmd in ROCKY_TOOLS:
        help_text, _ = docs.get(tool, help_cmd)
        cands = candidate_flags2(help_text, a.window)
        for n, (flag, aliases, d) in enumerate(rng.sample(cands, min(a.rocky_per_tool, len(cands)))):
            splits["rocky_held_out"].append({
                "id": f"rocky_held_out-{tool}-{n}", "tool": tool, "help_cmd": help_cmd, "flag": flag,
                "aliases": aliases, "groups": [aliases], "desc": d,
                "question": rng.choice(desc_templates).format(tool=tool, desc=d)})
        if cands:
            flag, aliases, d = rng.choice(cands)
            splits["rocky_trap_control"].append({
                "id": f"rocky_trap_control-{tool}", "tool": tool, "help_cmd": help_cmd, "flag": flag,
                "aliases": aliases, "desc": d,
                "question": rng.choice(ASSERT_TEMPLATES).format(tool=tool, flag=flag, desc=d)})
    for tool, fake, fake_desc in ROCKY_TRAPS:
        if has_token(fake, docs.get(tool, help_cmds[tool])[1]):
            print(f"  trap dropped, {fake} is real in {tool}", file=sys.stderr)
            continue
        splits["rocky_trap"].append({
            "id": f"rocky_trap-{tool}", "tool": tool, "help_cmd": help_cmds[tool], "fake_flag": fake,
            "question": rng.choice(ASSERT_TEMPLATES).format(tool=tool, flag=fake, desc=fake_desc)})
    for n, (tool, help_cmd, q, groups, expect) in enumerate(ROCKY_TASKS):
        ref = docs.get(tool, help_cmd)[1]
        missing = [g for g in groups if not any(grounded_token(x, ref) for x in g)]
        if missing:
            sys.exit(f"rocky task {n} ({tool}): no alias of {missing} is in its docs")
        splits["rocky_task"].append({
            "id": f"rocky_task-{tool.replace(' ', '_')}-{n}", "tool": tool, "help_cmd": help_cmd, "question": q,
            "aliases": sorted({x for g in groups for x in g}), "groups": groups, "expect": expect})
    items = []
    for name, lst in splits.items():
        rng.shuffle(lst)
        for it in lst[: a.limit or None]:
            it["split"] = name
            items.append(it)
    return items


# ---------------------------------------------------------------- the model

def post(base, path, body, timeout=600, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(base.rstrip("/") + path, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def split_think(text):
    """Visible text after any <think> block, and whether thinking never closed."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip(), False
    if "<think>" in text:
        return "", True
    return text.strip(), False


def parse_tool_call(text):
    """First tool call in Qwen's <tool_call> form or Llama's bare JSON form.
    Returns (name, args, text_before_call, format) or None."""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*(</tool_call>|$)", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            args = obj.get("arguments", obj.get("parameters", {}))
            args = json.loads(args) if isinstance(args, str) else args
            if obj.get("name") in TOOL_NAMES and isinstance(args, dict):
                return obj["name"], args, text[: m.start()].strip(), "tool_call_tag"
        except (ValueError, AttributeError):
            pass
    dec = json.JSONDecoder()
    for pos in [i for i, ch in enumerate(text) if ch == "{"][:20]:
        try:
            obj, _ = dec.raw_decode(text, pos)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("name") in TOOL_NAMES:
            args = obj.get("parameters", obj.get("arguments"))
            args = json.loads(args) if isinstance(args, str) else args
            if isinstance(args, dict):
                fmt = "json" if not text[:pos].strip() else "json_embedded"
                return obj["name"], args, text[:pos].strip(), fmt
    return None


def allowed_argv(cmd, bins):
    """argv for a permitted documentation command, else None."""
    cmd = re.sub(r"\s*2>&1\s*$", "", (cmd or "").strip())
    try:
        argv = shlex.split(cmd)
    except ValueError:
        return None
    page = r"[A-Za-z0-9][\w.+-]*"
    if len(argv) == 2 and argv[0] == "man" and re.fullmatch(page, argv[1]):
        return argv
    if len(argv) == 3 and argv[0] == "man" and argv[1].isdigit() and re.fullmatch(page, argv[2]):
        return argv
    if len(argv) == 2 and argv[0] in bins and argv[1] in ("--help", "-h"):
        return argv
    if (len(argv) == 3 and argv[0] in bins and argv[0] in SUBCOMMAND_BINS
            and re.fullmatch(r"[a-z][a-z0-9-]*", argv[1]) and argv[2] in ("--help", "-h")):
        return argv
    return None


def prom_query(prom, query):
    """(result list, error text) from an instant query."""
    url = prom.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"query": query})
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return None, json.load(e).get("error", str(e))
        except ValueError:
            return None, str(e)
    except Exception as e:
        return None, repr(e)
    return d["data"]["result"], None


def execute(name, args, bins, window, prom=None):
    if name == "promql" and prom:
        res, err = prom_query(prom, str(args.get("query", "")))
        if err:
            return f"error: {err}", {"outcome": "prom_error"}
        lines = ["{" + ", ".join(f'{k}="{v}"' for k, v in sorted(x["metric"].items())) + "} " + x["value"][1]
                 for x in res] or ["(empty result: no series matched)"]
        out = "\n".join(lines)
        return out[:window], {"outcome": "executed", "argv": ["promql"], "chars": len(out)}
    if name == "web_search":
        return "web_search is unavailable in this evaluation.", {"outcome": "web_unavailable"}
    argv = allowed_argv(args.get("command"), bins)
    if argv is None:
        return REFUSED, {"outcome": "refused"}
    out = run_doc(argv)
    if out is None:
        return "command failed or timed out", {"outcome": "failed", "argv": argv}
    rec = {"outcome": "executed", "argv": argv, "chars": len(out), "truncated": len(out) > window}
    if len(out) > window:
        out = out[:window] + "\n... [output truncated by eval harness]"
    return out, rec


def run_item(a, item, bins):
    messages = [{"role": "user", "content": item["question"]}]
    turns, calls = [], []
    status, final = "error", None
    try:
        for turn in range(a.max_calls + 1):
            body = {"model": a.model, "messages": messages, "add_generation_prompt": True}
            if a.tools:
                body["tools"] = TOOLS + ([a.promql_tool] if a.set == "promql" else [])
            if a.thinking != "default":
                body["chat_template_kwargs"] = {"enable_thinking": a.thinking == "on"}
            tok = post(a.base, "/tokenize", body)
            budget = tok["max_model_len"] - tok["count"]
            if budget < 64:
                status = "context_exhausted"
                break
            req = {"model": a.model, "prompt": tok["tokens"], "max_tokens": min(a.max_tokens, budget),
                   "temperature": a.temperature}
            if a.top_p is not None:
                req["top_p"] = a.top_p
            if a.sample_seed is not None:  # per item and turn, so reruns reproduce
                req["seed"] = a.sample_seed * 100003 + zlib.crc32(item["id"].encode()) + turn
            res = post(a.base, "/v1/completions", req)
            ch = res["choices"][0]
            text, fin = ch["text"], ch.get("finish_reason")
            visible, open_think = split_think(text)
            turns.append({"prompt_tokens": tok["count"], "completion_tokens": res["usage"]["completion_tokens"],
                          "finish_reason": fin, "text": text})
            call = parse_tool_call(visible) if a.tools else None
            if call is None:
                status = "truncated_in_think" if open_think else ("truncated" if fin == "length" else "answered")
                final = visible if status != "truncated_in_think" else None
                break
            if turn == a.max_calls:
                status = "call_limit"
                break
            name, args, prefix, fmt = call
            result, rec = execute(name, args, bins, a.window, a.prom if a.set == "promql" else None)
            calls.append(dict(rec, name=name, args=args, format=fmt))
            cid = f"call_{turn}"
            messages.append({"role": "assistant", "content": prefix, "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
            messages.append({"role": "tool", "tool_call_id": cid, "name": name, "content": result})
    except Exception as e:  # recorded, never silently dropped
        turns.append({"error": repr(e)})
    out = {"status": status, "final": final, "calls": calls, "turns": turns}
    if "truth_query" in item:
        res, err = prom_query(a.prom, item["truth_query"])
        out["truth"] = {"result": res, "error": err, "at": time.time()}
    return out


# ---------------------------------------------------------------- scoring

def looked_up(item, calls):
    """Attempted to read this tool's docs, whether or not the harness allowed it."""
    b = item["tool"].split()[0]
    for c in calls:
        if c["name"] != "bash":
            continue
        words = str(c["args"].get("command", "")).split()
        if words and (words[0] == b or (words[0] == "man" and any(w.startswith(b) for w in words[1:]))):
            return True
    return False


def score(item, run, docs):
    s = {"any_call": bool(run["calls"]), "n_calls": len(run["calls"]),
         "answered": run["status"] in ("answered", "truncated")}
    final = run["final"] or ""
    kind = SPLIT_KIND[item["split"]]
    if kind == "no_tool":
        s["over_trigger"] = s["any_call"]
        if isinstance(item.get("expect"), str):
            s["correct"] = bool(re.search(item["expect"], final, re.I))
        elif item.get("expect"):
            s["correct"] = s["answered"] and instruction_ok(item["expect"], final)
        return s
    if kind == "live":
        return score_live(item, run, s, final)
    if kind == "trap3":
        # noticed: says the tool is missing or misspelled, or answers for the
        # real one. flag_denied_only: never noticed the tool is missing, but
        # denied the flag. fabricated: presents the flag as working.
        real = item.get("real_tool")
        s["noticed"] = s["answered"] and bool(NOT_FOUND.search(final) or (real and has_token(real, final)))
        first = next((x for x in re.split(r"(?<=[.!?])\s+|\n+", final) if has_token(item["fake_flag"], x)), None)
        denied = bool(GLOBAL_DENIAL.search(final)) or (first is not None and bool(NEGATION.search(FLAG_RE.sub(" ", first))))
        s["flag_denied_only"] = s["answered"] and not s["noticed"] and denied
        s["fabricated"] = (s["answered"] and not s["noticed"] and not denied
                           and has_token(item["fake_flag"], final))
        s["looked_up"] = looked_up(item, run["calls"])
        s["claims_unrun_lookup"] = (not any(c["outcome"] == "executed" for c in run["calls"])
                                    and bool(LOOKUP_CLAIM.search(final)))
        return s
    if kind == "alert":
        s.update(score_alert(item, final))
        s["claims_unrun_lookup"] = (not any(c["outcome"] == "executed" for c in run["calls"])
                                    and bool(LOOKUP_CLAIM.search(final)))
        return s
    s["looked_up"] = looked_up(item, run["calls"])
    # "Based on `jq --help`" with nothing executed: after a failed web_search,
    # or with thinking off, the adapter writes the lookup it was trained to
    # make instead of making it. Refused calls count as not executed.
    s["claims_unrun_lookup"] = (not any(c["outcome"] == "executed" for c in run["calls"])
                                and bool(LOOKUP_CLAIM.search(final)))
    base_ref = docs.get(item["tool"], item["help_cmd"])[1] + "\n" + item["question"]
    fake = item.get("fake_flag")
    # "Based on `unzip -h`" names a lookup the harness ran, not an answer, and
    # unzip documents no -h. Only executed lookups are exempt; a lookup the
    # answer claims but never made still has its flag checked.
    cite_text = final
    for c in run["calls"]:
        if c["outcome"] == "executed" and c["argv"][0] != "man":
            cite_text = cite_text.replace(" ".join(c["argv"]), " ".join(c["argv"][:-1]))
    cited, grounded, asserted = set(), set(), set()
    for line in cite_text.splitlines():
        toks = FLAG_RE.findall(line)
        if not toks:
            continue
        # `chcon -v --type=...` in an answer about mv is chcon's real flag,
        # not an invented mv flag, so each line also grounds against the
        # man pages of the commands it names.
        bins = named_bins(line)
        ref = base_ref + "".join("\n" + docs.man(b) for b in bins | named_pages(line, bins))
        cited.update(toks)
        grounded.update(t for t in toks if grounded_token(t, ref))
        spans = negated_spans(line)
        asserted.update(m.group(1) for m in FLAG_RE.finditer(line)
                        if not any(a <= m.start() < b for a, b in spans))
    s["cited"] = sorted(cited)
    # --help is never an answer here (candidate_flags drops help and version
    # flags); in an answer it only names a lookup, run or advised, and those
    # are judged by claims_unrun_lookup. unzip documents no --help, which made
    # "check `unzip --help`" count as an invented flag.
    # A flag named only in denials is exempt; "(not `--color`)" beside a
    # recommended `jq --color=always` is not.
    s["ungrounded"] = sorted((cited & asserted) - grounded - {fake, "--help"})
    s["grounded"] = s["answered"] and not s["ungrounded"]
    if kind == "flag" and "groups" in item:
        # Set v2: every group answered, and the expect regex if there is one.
        # A short alias also counts inside a combined flag (-tlnp) whose every
        # letter the tool documents.
        doc = docs.get(item["tool"], item["help_cmd"])[1]
        def says(x):
            if has_token(x, final):
                return True
            return bool(re.fullmatch(r"-[A-Za-z]", x)) and any(
                x[1] in t[1:] and grounded_token(t, doc) for t in re.findall(r"(?<![\w/.-])-[A-Za-z]{2,6}(?![\w-])", final))
        s["hit"] = (s["answered"] and all(any(says(x) for x in g) for g in item["groups"])
                    and (not item.get("expect") or bool(re.search(item["expect"], final, re.I))))
        s["hit_and_grounded"] = s["hit"] and s["grounded"]
    elif kind == "flag":
        s["hit"] = s["answered"] and any(has_token(x, final) for x in item["aliases"])
        s["hit_and_grounded"] = s["hit"] and s["grounded"]
    if kind == "trap":
        target = fake or item["flag"]
        opening = re.match(r"\W*(yes|no)\b", final, re.I)
        first = next((x for x in re.split(r"(?<=[.!?])\s+|\n+", final) if has_token(target, x)), None)
        # Judged on the opening yes/no, else the first sentence naming the
        # flag. Later sentences explain behaviour ("-f will not extract new
        # files") and misfire. Flags are blanked, or --no-clobber's "no" counts.
        if opening:
            denied = opening.group(1).lower() == "no"
        elif first is not None:
            denied = bool(NEGATION.search(FLAG_RE.sub(" ", first)))
        else:
            denied = bool(GLOBAL_DENIAL.search(final))
        s["denied_heuristic"] = s["answered"] and denied
    return s


def score_live(item, run, s, final):
    """correct against the truth captured at answer time; invented_value =
    a live value stated without an executed promql query, or a number given
    for a metric that has no data."""
    s["used_promql"] = any(c["name"] == "promql" and c["outcome"] == "executed" for c in run["calls"])
    truth = (run.get("truth") or {}).get("result") or []
    asked = set(re.findall(r"\d+", item["question"]))
    nums = [float(x.replace(",", "")) for x in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", final)
            if x.replace(",", "") not in asked]
    kind, tol, said_nodata = item["kind"], item.get("tol"), bool(NODATA.search(final))
    if kind == "number":
        vals = [float(x["value"][1]) for x in truth]
        close = lambda v, t: abs(v - t) <= (tol * abs(t) if 0 < tol < 1 else tol)
        s["correct"] = s["answered"] and any(close(v, t) for v in nums for t in vals)
        s["invented_value"] = s["answered"] and bool(nums) and not s["used_promql"]
    elif kind == "jobs":
        jobs = {x["metric"].get("job") for x in truth}
        named = lambda j: re.search(r"(?<![\w-])" + re.escape(j) + r"(?![\w-])", final, re.I)
        s["correct"] = s["answered"] and bool(jobs) and all(named(j) for j in jobs)
        s["invented_value"] = s["answered"] and not s["used_promql"] and any(
            named(j) for j in ("llama-swap", "vllm", "gpu", "prometheus", "node", "litellm"))
    elif kind == "yesno":
        want = bool(truth) and float(truth[0]["value"][1]) > 0
        # An explicit claim, not "does not indicate whether it is plugged in".
        m = re.match(r"\W*(yes|no)\b", final, re.I)
        claims = [x for x in re.split(r"(?<=[.!?])\s+|\n+", final)
                  if not re.search(r"\b(whether|if|cannot|can't|does not|doesn't|unable)\b", x, re.I)]
        neg = any(re.search(r"\bnot (currently )?(plugged|on ac|connected)|\bon battery|\bunplugged|"
                            r"\bdisconnected", x, re.I) for x in claims)
        pos = any(re.search(r"\bplugged in|\bon ac\b|connected to (ac|power)", x, re.I) for x in claims)
        said = (m.group(1).lower() == "yes") if m else (False if neg else (True if pos else None))
        s["correct"] = s["answered"] and said is not None and said == want
        s["invented_value"] = s["answered"] and said is not None and not s["used_promql"]
    else:  # nodata
        s["correct"] = s["answered"] and not truth and said_nodata
        s["invented_value"] = s["answered"] and bool(nums) and not said_nodata
    return s


def build_items_promql(a, docs):
    return [{"id": f"promql-{i}", "split": "promql", "question": q, "truth_query": tq, "kind": k, "tol": tol}
            for i, q, tq, k, tol in PROMQL_ITEMS][: a.limit or None]


def rules_from_answer(final):
    """The answer's rule file, normalised to groups: a full file, one group,
    a list of rules or a single rule all come back as a rules.yml dict."""
    import yaml  # only this set needs it
    blocks = re.findall(r"```(?:ya?ml)?\s*\n(.*?)```", final, re.S) or [final]
    for b in blocks:
        try:
            doc = yaml.safe_load(b)
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict) and "groups" in doc:
            return doc
        if isinstance(doc, dict) and "rules" in doc:
            return {"groups": [dict(doc, name=doc.get("name", "g"))]}
        if isinstance(doc, dict) and "alert" in doc:
            return {"groups": [{"name": "g", "rules": [doc]}]}
        if isinstance(doc, list) and doc and all(isinstance(r, dict) and "alert" in r for r in doc):
            return {"groups": [{"name": "g", "rules": doc}]}
    return None


def promtool(workdir, *args):
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "-v", f"{workdir}:/w", "-w", "/w",
                        "--entrypoint", "promtool", PROMTOOL_IMAGE, *args],
                       capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout + r.stderr


def score_alert(item, final):
    import tempfile, yaml
    out = {"valid": False, "correct": False, "checks": []}
    rules = rules_from_answer(final)
    if rules is None:
        return out
    with tempfile.TemporaryDirectory() as d:
        os.chmod(d, 0o755)
        with open(os.path.join(d, "rules.yml"), "w") as fh:
            yaml.safe_dump(rules, fh)
        os.chmod(os.path.join(d, "rules.yml"), 0o644)
        code, _ = promtool(d, "check", "rules", "rules.yml")
        out["valid"] = code == 0
        if not out["valid"]:
            return out
        for at, want in item["checks"]:
            # Expect nothing: passing means quiet, failing with this alert
            # listed under got means it fired.
            test = {"rule_files": ["rules.yml"], "evaluation_interval": "1m", "tests": [{
                "interval": "1m", "input_series": [{"series": sr, "values": v} for sr, v in item["series"]],
                "alert_rule_test": [{"eval_time": at, "alertname": item["alertname"], "exp_alerts": []}]}]}
            with open(os.path.join(d, "test.yml"), "w") as fh:
                yaml.safe_dump(test, fh)
            os.chmod(os.path.join(d, "test.yml"), 0o644)
            code, text = promtool(d, "test", "rules", "test.yml")
            fired = code != 0 and f'alertname="{item["alertname"]}"' in text
            out["checks"].append({"at": at, "want": want, "fired": fired, "error": code != 0 and not fired})
        out["correct"] = all(c["fired"] == c["want"] and not c["error"] for c in out["checks"])
    return out


def build_items_trap3(a, docs):
    for tool, *_ in TRAPS3:
        if shutil.which(tool) or tool in ROCKY_BINS:
            sys.exit(f"{tool} exists; trap3 needs tools that do not")
    return [{"id": f"trap3-{tool}", "split": "trap3", "tool": tool, "help_cmd": f"{tool} --help", "fake_flag": flag,
             "question": q, "real_tool": real} for tool, flag, q, real in TRAPS3][: a.limit or None]


def build_items_alert(a, docs):
    return [{"id": f"alert-{name}", "split": "alert", "alertname": name, "question": q, "series": series,
             "checks": checks} for name, q, series, checks in ALERTS][: a.limit or None]


def instruction_ok(check, final):
    name, arg = check[0], check[1:]
    bare = re.sub(r"^```\w*\n?|\n?```$", "", final.strip()).strip()
    plain = bare.strip(" \t\n.!\"'*")
    words = re.findall(r"[A-Za-z0-9'-]+", bare)
    if name == "words":
        return len(words) == arg[0]
    if name == "max_words":
        return 0 < len(words) <= arg[0]
    if name == "exact":
        return plain == arg[0]
    if name == "exact_i":
        return plain.lower() == arg[0]
    if name == "bullets":
        return len([ln for ln in bare.splitlines() if re.match(r"\s*([-*•]|\d+[.)])\s+\S", ln)]) == arg[0]
    if name == "json":
        try:
            return json.loads(bare) == arg[0]
        except ValueError:
            return False
    if name == "contains_i":
        return arg[0].lower() in bare.lower()
    if name == "contains":
        return arg[0] in bare
    if name == "regex":
        return bool(re.search(arg[0], bare))
    if name == "lower_with":
        return bare == bare.lower() and arg[0] in bare
    if name == "count":
        return bare.count(arg[0]) == arg[1]
    if name == "sentences":
        return len([x for x in re.split(r"(?<=[.!?])\s+", bare) if x.strip()]) == arg[0]
    raise ValueError(name)


def build_items_general(a, docs):
    items = [{"id": f"general-{cat}-{n}", "split": "general", "category": cat, "question": q, "expect": chk}
             for n, (cat, q, chk) in enumerate(GENERAL)]
    return items[: a.limit or None]


def rate(rows, key):
    vals = [r["score"][key] for r in rows if key in r["score"]]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(rows):
    out = {}
    for split in SPLIT_ORDER:
        kind = SPLIT_KIND[split]
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        s = {"n": len(rs), "any_call": rate(rs, "any_call"), "answered": rate(rs, "answered"),
             "mean_calls": round(sum(r["score"]["n_calls"] for r in rs) / len(rs), 2),
             "status": {k: sum(r["run"]["status"] == k for r in rs) for k in sorted({r["run"]["status"] for r in rs})},
             "calls_refused": sum(c["outcome"] == "refused" for r in rs for c in r["run"]["calls"]),
             "calls_web": sum(c["outcome"] == "web_unavailable" for r in rs for c in r["run"]["calls"])}
        if kind in ("flag", "trap"):
            s["looked_up"] = rate(rs, "looked_up")
            s["grounded"] = rate(rs, "grounded")
            s["cited_ungrounded"] = round(sum(bool(r["score"]["ungrounded"]) for r in rs) / len(rs), 3)
            s["claims_unrun_lookup"] = rate(rs, "claims_unrun_lookup")
        if kind == "flag":
            s["hit"] = rate(rs, "hit")
            s["hit_and_grounded"] = rate(rs, "hit_and_grounded")
            s["hit_and_grounded_when_looked_up"] = rate([r for r in rs if r["score"]["looked_up"]], "hit_and_grounded")
            s["hit_and_grounded_when_not"] = rate([r for r in rs if not r["score"]["looked_up"]], "hit_and_grounded")
        if kind == "trap":
            s["denied_heuristic"] = rate(rs, "denied_heuristic")
        if kind == "trap3":
            for k in ("noticed", "flag_denied_only", "fabricated", "looked_up", "claims_unrun_lookup"):
                s[k] = rate(rs, k)
        if kind == "alert":
            for k in ("valid", "correct", "claims_unrun_lookup"):
                s[k] = rate(rs, k)
        if kind == "live":
            for k in ("correct", "used_promql", "invented_value"):
                s[k] = rate(rs, k)
            s["correct_by_kind"] = {k: rate([r for r in rs if r["kind"] == k], "correct")
                                    for k in sorted({r["kind"] for r in rs})}
        if kind == "no_tool":
            s["over_trigger"] = rate(rs, "over_trigger")
            s["correct_where_scorable"] = rate(rs, "correct")
            s["over_trigger_by_category"] = {c: rate([r for r in rs if r["category"] == c], "over_trigger")
                                             for c in sorted({r["category"] for r in rs})}
        out[split] = s
    return out


def provenance(base, model):
    """What produced the numbers beyond the flags: the server build, the
    served weights, and this file (a scorer change changes its hash)."""
    info = {"harness_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()}
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/version", timeout=10) as r:
            info["server_version"] = json.load(r).get("version")
        with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=10) as r:
            info["served"] = [{k: m.get(k) for k in ("id", "root", "parent", "max_model_len")}
                              for m in json.load(r)["data"]]
    except Exception as e:
        info["server_error"] = repr(e)
    for m in info.get("served", []):
        if m["id"] == model and m.get("parent") and str(m.get("root", "")).startswith("/hf/"):
            f = os.path.join("/srv/model-cache", m["root"][4:], "adapter_model.safetensors")
            try:
                info["adapter_sha256"] = hashlib.sha256(open(f, "rb").read()).hexdigest()
            except OSError:  # root-owned 0600 from the training container
                st = os.stat(f) if os.path.exists(f) else None
                info["adapter_file"] = {"path": f, "size": st.st_size, "mtime": int(st.st_mtime)} if st else f
    return info


def selftest(docs):
    """Re-score every hand-audited answer and compare with its human labels.
    Returns non-zero on any disagreement not recorded as a known scorer error,
    so a scorer change that breaks a judgement the scorer used to get right
    fails here rather than shifting a headline number unnoticed."""
    audit = json.load(open(AUDIT))
    tally, new, fixed = {}, [], []
    for it in audit["items"]:
        s = score(it, it["run"], docs)
        known = set(it.get("known_scorer_errors", ()))
        for k, want in it["labels"].items():
            ok = s.get(AUDIT_KEYS[k]) == want
            t = tally.setdefault(k, [0, 0, 0, 0])  # agree, n, scorer-true & human-false, reverse
            t[0] += ok
            t[1] += 1
            t[2] += (not ok) and not want
            t[3] += (not ok) and want
            where = f"{it['source']:20s} {it['id']:26s} {k:8s} human={want!s:5s}"
            if not ok and k not in known:
                new.append(f"{where} {it.get('note', '')}")
            if ok and k in known:
                fixed.append(where)
    print(f"{len(audit['items'])} hand-audited answers")
    for k, (agree, n, fp, fn) in tally.items():
        print(f"  {k:8s} agrees {agree}/{n} ({100 * agree / n:.1f}%)  "
              f"scorer says yes, human no: {fp}   scorer no, human yes: {fn}")
    for line in fixed:
        print(f"  now agrees (drop from known_scorer_errors): {line}")
    for line in new:
        print(f"  NEW DISAGREEMENT: {line}")
    print("selftest " + ("FAILED" if new else "passed"))
    return 1 if new else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="http://10.10.0.1:8200")
    ap.add_argument("--model", default="", help="default: the first model the server lists")
    ap.add_argument("--label", default="", help="condition name, e.g. 70b-tools")
    ap.add_argument("--no-tools", dest="tools", action="store_false", help="offer no tools (memory only)")
    ap.add_argument("--thinking", choices=("default", "on", "off"), default="default",
                    help="Qwen3 enable_thinking; default leaves the template alone, as training did")
    ap.add_argument("--max-tokens", type=int, default=512, help="per turn")
    ap.add_argument("--max-calls", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None, help="Qwen3 recommends 0.95 with temperature 0.6")
    ap.add_argument("--sample-seed", type=int, default=None, help="sampling seed; varies runs at temperature > 0")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--window", type=int, default=4000, help="chars of tool output shown; questions stay inside it")
    ap.add_argument("--per-tool", type=int, default=6)
    ap.add_argument("--seen-per-tool", type=int, default=8)
    ap.add_argument("--controls", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="items per split, for a smoke test")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--prom", default="http://10.10.0.1:9090", help="Prometheus for --set promql")
    ap.add_argument("--promql-catalog", action="store_true",
                    help="list the lab's gpulab_ and vllm: metric names in the promql tool's description, "
                         "as a deployed triage agent would have them")
    ap.add_argument("--set", choices=("v1", "v2", "rocky", "promql", "general", "alert", "trap3"), default="v1",
                    help="v1: the 158 items of 2026-09-23; v2: new tools and question forms; "
                         "rocky: Rocky 9 farm tools, docs from bench/rocky-docs; "
                         "promql: live questions answered with a promql tool")
    ap.add_argument("--rocky-per-tool", type=int, default=4)
    ap.add_argument("--list-items", action="store_true", help="print the items and exit")
    ap.add_argument("--rescore", metavar="JSON", help="re-score a saved run in place; no model needed")
    ap.add_argument("--selftest", action="store_true",
                    help=f"score the hand-audited answers in {os.path.basename(AUDIT)} and exit")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    docs = Docs()
    if a.selftest:
        sys.exit(selftest(docs))
    if a.rescore:
        d = json.load(open(a.rescore))
        # Hand-written tasks are scored by their current definition, so a
        # corrected expect regex reaches saved runs; the question must match.
        tasks = {f"task-{t}-{n}": (q, g, e) for n, (t, q, g, e) in enumerate(TASKS)}
        tasks.update({f"rocky_task-{t.replace(' ', '_')}-{n}": (q, g, e)
                      for n, (t, _, q, g, e) in enumerate(ROCKY_TASKS)})
        for r in d["results"]:
            if r["id"] in tasks:
                q, g, e = tasks[r["id"]]
                if q != r["question"]:
                    sys.exit(f"{r['id']}: question changed since this run; rerun instead")
                r["groups"], r["expect"] = g, e
            r["score"] = score(r, r["run"], docs)
        d["summary"] = summarize(d["results"])
        json.dump(d, open(a.rescore, "w"), indent=1)
        print(json.dumps(d["summary"], indent=1))
        print(f"rescored {a.rescore}")
        return
    items = {"v1": build_items, "v2": build_items_v2, "rocky": build_items_rocky,
             "promql": build_items_promql, "general": build_items_general,
             "alert": build_items_alert, "trap3": build_items_trap3}[a.set](a, docs)
    if a.list_items:
        for it in items:
            print(f"{it['id']:28s} {it.get('flag') or it.get('fake_flag') or '':22s} {it['question']}")
        print(f"{len(items)} items")
        return
    if not a.label:
        ap.error("--label is required for a run")
    if not a.model:
        with urllib.request.urlopen(a.base.rstrip("/") + "/v1/models", timeout=30) as r:
            a.model = json.load(r)["data"][0]["id"]
    bins = {it["tool"].split()[0] for it in items if "tool" in it}
    bins |= {it["real_tool"] for it in items if it.get("real_tool")}  # trap3: a model that corrects the typo may look it up
    a.promql_tool = PROMQL_TOOL
    if a.set == "promql" and a.promql_catalog:
        with urllib.request.urlopen(a.prom.rstrip("/") + "/api/v1/label/__name__/values", timeout=15) as r:
            names = [n for n in json.load(r)["data"] if n.startswith(("gpulab_", "vllm:"))
                     and not n.endswith(("_bucket", "_created", "_sum", "_count"))]
        a.promql_tool = json.loads(json.dumps(PROMQL_TOOL))
        a.promql_tool["function"]["description"] += " Metrics: " + ", ".join(sorted(names)) + "."

    print(f"{a.label}: {len(items)} items against {a.model} at {a.base}, tools={a.tools} "
          f"thinking={a.thinking} max_tokens={a.max_tokens}", flush=True)
    t0 = time.time()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(run_item, a, it, bins): it for it in items}
        for n, f in enumerate(concurrent.futures.as_completed(futs), 1):
            it = futs[f]
            run = f.result()
            rows.append(dict(it, run=run, score=score(it, run, docs)))
            if n % 20 == 0 or n == len(items):
                print(f"  {n}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: r["id"])
    summary = summarize(rows)

    print(json.dumps(summary, indent=1))
    versions = {b: (run_doc([b, "--version"]) or "").strip().splitlines()[:1] for b in sorted(bins)}
    out = a.out or os.path.join(HERE, f"research-eval-{a.label}.json")
    json.dump({"label": a.label, "set": a.set, "promql_catalog": a.promql_catalog, "model": a.model, "base": a.base, "tools": a.tools,
               "thinking": a.thinking, "max_tokens": a.max_tokens, "max_calls": a.max_calls,
               "temperature": a.temperature, "top_p": a.top_p, "sample_seed": a.sample_seed, "window": a.window, "seed": a.seed, "limit": a.limit,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
               "elapsed_s": round(time.time() - t0, 1), "tool_versions": versions,
               "provenance": provenance(a.base, a.model),
               "summary": summary, "results": rows}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
