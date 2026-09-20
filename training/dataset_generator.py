"""Generate research-first, anti-hallucination training trajectories.

Extracts ground-truth help and manual texts from local system CLI tools and
builds realistic, grounded multi-turn tool-calling trajectories formatted for
Qwen3's native chat template.
"""

import json
import os
import random
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from tools import TOOLS


def run_cmd(cmd: str) -> Optional[str]:
    """Execute a bash command and return its stdout, or None on failure."""
    try:
        res = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        out = (res.stdout or "") + (res.stderr or "")
        return out.strip() if res.returncode == 0 or out.strip() else None
    except Exception:
        return None


def extract_flags_from_help(help_text: str) -> List[Tuple[str, str]]:
    """Parse out flag pairs and descriptions from standard GNU/clap help text."""
    results = []
    lines = help_text.splitlines()
    flag_pattern = re.compile(r"^\s{2,6}(-{1,2}[a-zA-Z0-9_-]+(?:\s*,?\s*-{1,2}[a-zA-Z0-9_-]+)?(?:\s+<[^>]+>|\s+[A-Z_-]+)?)\s{2,}(.+)$")

    for line in lines:
        m = flag_pattern.match(line)
        if m:
            flag_spec = m.group(1).strip()
            desc = m.group(2).strip()
            # Extract primary flag (e.g. --release or -r)
            flags = re.findall(r"-{1,2}[a-zA-Z0-9_-]+", flag_spec)
            if flags and len(desc) > 5:
                # Prefer long flag if present
                long_flag = next((f for f in flags if f.startswith("--")), flags[0])
                results.append((long_flag, flag_spec, desc))
    return results


# Tool definitions to harvest from the local system
CLI_SPECS = [
    # Cargo suite
    ("cargo", "cargo --help", "Cargo build tool"),
    ("cargo build", "cargo build --help", "Compile local packages"),
    ("cargo test", "cargo test --help", "Execute unit and integration tests"),
    ("cargo check", "cargo check --help", "Check a package for syntax/type errors without building"),
    ("cargo clippy", "cargo clippy --help", "Run Rust lints"),
    ("cargo doc", "cargo doc --help", "Build package documentation"),
    ("cargo bench", "cargo bench --help", "Execute benchmarks"),
    ("cargo clean", "cargo clean --help", "Remove artifacts that cargo has generated in the past"),
    ("cargo tree", "cargo tree --help", "Display a tree visualization of a dependency graph"),
    ("cargo update", "cargo update --help", "Update dependencies listed in Cargo.lock"),
    # Git suite
    ("git log", "git log -h", "View commit history"),
    ("git diff", "git diff -h", "Show changes between commits or working tree"),
    ("git commit", "git commit -h", "Record changes to the repository"),
    ("git status", "git status -h", "Show working tree status"),
    ("git checkout", "git checkout -h", "Switch branches or restore files"),
    ("git branch", "git branch -h", "List, create, or delete branches"),
    ("git stash", "git stash -h", "Stash changes in a dirty working directory"),
    ("git fetch", "git fetch -h", "Download objects and refs from another repository"),
    ("git rebase", "git rebase -h", "Reapply commits on top of another base tip"),
    ("git reset", "git reset -h", "Reset current HEAD to the specified state"),
    ("git show", "git show -h", "Show various types of objects"),
    ("git remote", "git remote -h", "Manage set of tracked repositories"),
    ("git tag", "git tag -h", "Create, list, delete or verify a tag object"),
    # Docker suite
    ("docker run", "docker run --help", "Run a command in a new container"),
    ("docker build", "docker build --help", "Build an image from a Dockerfile"),
    ("docker images", "docker images --help", "List images"),
    ("docker ps", "docker ps --help", "List containers"),
    ("docker stop", "docker stop --help", "Stop one or more running containers"),
    ("docker exec", "docker exec --help", "Execute a command in a running container"),
    ("docker compose", "docker compose --help", "Docker Compose multi-container management"),
    ("docker volume", "docker volume --help", "Manage Docker volumes"),
    ("docker network", "docker network --help", "Manage Docker networks"),
    # NVIDIA & System monitoring
    ("nvidia-smi", "nvidia-smi --help", "NVIDIA System Management Interface"),
    ("systemctl", "systemctl --help", "Control the systemd system and service manager"),
    ("journalctl", "journalctl --help", "Query the systemd journal"),
    # Common Linux developer tools
    ("rg", "rg --help", "Ripgrep fast line-oriented search"),
    ("pytest", "pytest --help", "Python test runner"),
    ("python3", "python3 -h", "Python interpreter"),
    ("curl", "curl --help", "Command line tool for transferring data with URLs"),
    ("pip", "pip --help", "Python package installer"),
    ("tar", "tar --help", "Archive utility"),
    ("find", "find --help", "Search for files in a directory hierarchy"),
    ("grep", "grep --help", "Print lines that match patterns"),
    ("df", "df --help", "Report file system disk space usage"),
    ("free", "free --help", "Display amount of free and used memory in the system"),
    ("ip address", "ip address help", "IPv4/IPv6 address management"),
    ("ip link", "ip link help", "Network device configuration"),
]

# Traps: queries asking about non-existent flags to verify the model refuses to hallucinate
TRAP_SPECS = [
    ("cargo build", "--gpu-acceleration", "cargo build --help", "GPU acceleration"),
    ("cargo check", "--fast-mode", "cargo check --help", "fast mode"),
    ("cargo test", "--instant", "cargo test --help", "instant test execution"),
    ("cargo clippy", "--auto-rewrite", "cargo clippy --help", "automatic rewriting of all lints"),
    ("cargo clean", "--delete-source-code", "cargo clean --help", "deleting source code"),
    ("git commit", "--instant-push", "git commit -h", "committing and pushing in one step"),
    ("git log", "--show-diff-inline-always", "git log -h", "inline diff display"),
    ("git status", "--cleanup-dirty", "git status -h", "cleaning up dirty working files"),
    ("git reset", "--revert-everything-unconditionally", "git reset -h", "unconditional reversion of everything"),
    ("docker run", "--unlimited-ram", "docker run --help", "unlimited RAM without memory limits"),
    ("docker build", "--skip-safety-checks", "docker build --help", "skipping safety checks"),
    ("docker stop", "--kill-node-power", "docker stop --help", "killing host node power"),
    ("nvidia-smi", "--boost-clock-extreme", "nvidia-smi --help", "extreme clock boost"),
    ("systemctl", "--force-delete-unit", "systemctl --help", "force deletion of service units"),
    ("pytest", "--ignore-all-syntax-errors", "pytest --help", "ignoring syntax errors"),
    ("curl", "--auto-retry-infinite", "curl --help", "infinite automatic retries"),
    ("pip", "--force-install-everything", "pip --help", "installing without dependency resolution"),
    ("tar", "--quantum-compress", "tar --help", "quantum compression algorithm"),
]

# ---- second domain: systems diagnostics ----
# Chosen by measurement, not taste. The flag extractor assumes GNU-style
# "--flag  description" help, so a domain generalises only as far as its tools
# follow that convention: of twelve networking tools, iproute2, ethtool, dig and
# tcpdump yield zero flags because their help uses a bespoke grammar. These ten
# were the ones that actually parse, and they share no tool with the devtools
# domain above, which is what makes this a real test of the template rather than
# a second pass over the same ground.
SYSDIAG_CLI_SPECS = [
    ("strace", "strace --help", "Trace system calls and signals"),
    ("bpftrace", "bpftrace --help", "High-level tracing language for eBPF"),
    ("gdb", "gdb --help", "The GNU debugger"),
    ("objdump", "objdump --help", "Display information from object files"),
    ("readelf", "readelf --help", "Display information about ELF files"),
    ("nm", "nm --help", "List symbols from object files"),
    ("ss", "ss --help", "Investigate sockets"),
    ("nft", "nft --help", "nftables packet filtering administration"),
    ("iptables", "iptables --help", "Administer IPv4 packet filter rules"),
    ("nmcli", "nmcli --help", "Command-line tool for NetworkManager"),
]

SYSDIAG_TRAP_SPECS = [
    ("strace", "--follow-threads-deep", "strace --help", "deep thread following"),
    ("strace", "--auto-decode-structs", "strace --help", "automatic struct decoding"),
    ("bpftrace", "--jit-optimize", "bpftrace --help", "JIT optimisation of probes"),
    ("gdb", "--auto-fix-segfault", "gdb --help", "automatic segfault repair"),
    ("objdump", "--decompile-to-c", "objdump --help", "decompilation to C source"),
    ("readelf", "--repair-headers", "readelf --help", "repairing malformed ELF headers"),
    ("nm", "--resolve-dynamic-always", "nm --help", "unconditional dynamic resolution"),
    ("ss", "--live-throughput", "ss --help", "live per-socket throughput"),
    ("nft", "--auto-flush-stale", "nft --help", "automatic flushing of stale rules"),
    ("iptables", "--optimize-chains", "iptables --help", "automatic chain optimisation"),
    ("nmcli", "--force-reconnect-all", "nmcli --help", "forcing reconnection of every device"),
]

WEB_SPECS = [
    (
        "What is the recommended timeout configuration in sqlx for SQLite connection pools to prevent database locked errors?",
        "sqlx sqlite busy_timeout pool configuration documentation",
        "In sqlx, SQLite connection options provide `.busy_timeout(Duration::from_secs(5))` on `SqliteConnectOptions`. Setting a busy_timeout allows queries waiting on a lock to retry rather than failing immediately with 'database is locked'.",
        "Based on online documentation for `sqlx::sqlite::SqliteConnectOptions`, you configure `SqliteConnectOptions::new().busy_timeout(std::time::Duration::from_secs(5))`. This ensures that when the SQLite database file is locked by a concurrent write, sqlx retries until the timeout expires instead of throwing an immediate 'database is locked' error."
    ),
    (
        "How do you specify multiple GPU architectures in an nvcc gencode compiler flag?",
        "nvcc gencode syntax multiple architectures compute sm",
        "nvcc uses `-gencode arch=compute_XX,code=sm_XX` pairs. For example, to compile for both Ampere (sm_86) and Blackwell (sm_120), you pass `-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120`.",
        "According to NVIDIA CUDA Compiler Driver NVCC documentation, you specify `-gencode arch=compute_XX,code=sm_XX` repeatedly for each target architecture. For instance: `-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120`. This generates distinct native machine cubins for both architectures without relying on generic JIT PTX."
    ),
    (
        "What is the vLLM engine parameter to set the fraction of GPU memory allocated to the KV cache in Python?",
        "vllm EngineArgs gpu_memory_utilization python api",
        "vLLM's `EngineArgs` and `AsyncEngineArgs` take `gpu_memory_utilization: float = 0.90` which defines the fraction of GPU memory reserved for model execution and the KV cache.",
        "Based on the `vllm.EngineArgs` specification, the parameter is `gpu_memory_utilization` (a float between 0.0 and 1.0, defaulting to 0.90). This controls the total fraction of physical GPU memory reserved for the model weights and the dynamic KV cache pool."
    ),
    (
        "In tokio, how do you construct a multi-threaded runtime with 4 worker threads?",
        "tokio runtime Builder multi_thread worker_threads rust documentation",
        "In tokio, use `tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build()`. The `worker_threads` method configures the size of the worker pool.",
        "According to tokio documentation for `tokio::runtime::Builder`, you create a multi-threaded runtime using `tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap()`. The `.worker_threads(4)` call explicitly bounds the worker thread pool."
    ),
    (
        "How do you configure PEFT LoraConfig to target all linear layers automatically?",
        "peft LoraConfig target_modules all-linear huggingface",
        "In PEFT `LoraConfig`, setting `target_modules='all-linear'` automatically matches all Linear projection layers across attention and MLP in transformer models.",
        "According to Hugging Face PEFT documentation, you can set `target_modules='all-linear'` in `LoraConfig`. This tells PEFT to automatically inspect the model architecture and attach adapters to every linear layer without manually specifying module names."
    ),
    (
        "In PyTorch, how do you specify non-blocking tensor transfer to CUDA streams?",
        "pytorch tensor to device non_blocking cuda stream",
        "In PyTorch, pass `non_blocking=True` to `.to('cuda', non_blocking=True)`. This allows host-to-device transfers from pinned CPU memory to overlap asynchronously with host computation.",
        "Based on the PyTorch tensor documentation, you pass `tensor.to(device, non_blocking=True)`. When the source tensor is stored in pinned CPU memory (page-locked), this enables asynchronous host-to-device copy that overlaps with CPU execution."
    ),
]

SYSDIAG_WEB_SPECS = [
    (
        "What BPF map type should I use in bpftrace to aggregate a histogram of syscall latency?",
        "bpftrace hist() builtin map aggregation documentation",
        "bpftrace provides `hist()` and `lhist()` aggregation functions. `@latency = hist(nsecs - @start[tid])` builds a power-of-two histogram, printed automatically when the program exits.",
        "Based on the bpftrace reference guide, use the `hist()` aggregation: `@latency = hist(nsecs - @start[tid]);`. It builds power-of-two buckets and bpftrace prints the histogram on exit, so no explicit print is needed.",
    ),
    (
        "In nftables, what is the correct syntax for an atomic ruleset replacement without dropping packets?",
        "nftables atomic ruleset replacement nft -f flush ruleset documentation",
        "nftables applies a whole file atomically: `nft -f ruleset.nft` where the file begins with `flush ruleset`. The kernel commits the transaction in one step, so no window exists where the ruleset is empty.",
        "Based on the nftables documentation, put `flush ruleset` at the top of your rules file and apply it with `nft -f ruleset.nft`. nftables commits the entire file as a single transaction, so there is no intermediate state in which packets are unfiltered.",
    ),
    (
        "How do I make perf resolve symbols for a stripped binary that has a separate debuginfo file?",
        "perf symbol resolution separate debuginfo build-id debuginfod documentation",
        "perf resolves symbols through the build-id cache. `perf buildid-cache --add ./binary` registers it, and debuginfod (DEBUGINFOD_URLS) can fetch matching debuginfo automatically.",
        "Based on the perf documentation, register the binary with `perf buildid-cache --add ./binary`, which lets perf match the recorded build-id to the separate debuginfo. Setting `DEBUGINFOD_URLS` additionally allows perf to fetch matching debuginfo on demand.",
    ),
]

# A domain is just its three spec tables. Adding one is data, not code, which is
# the property that makes this reusable across sectors.
DOMAINS = {
    "devtools": (CLI_SPECS, TRAP_SPECS, WEB_SPECS),
    "sysdiag": (SYSDIAG_CLI_SPECS, SYSDIAG_TRAP_SPECS, SYSDIAG_WEB_SPECS),
}


def get_observation_for_flag(lines: List[str], target_flag: Optional[str], max_lines: int = 80) -> str:
    """Return a focused observation window from help text that is guaranteed to contain target_flag."""
    if not target_flag or len(lines) <= max_lines:
        return "\n".join(lines[:max_lines])

    # Find the line defining target_flag
    target_idx = None
    for i, line in enumerate(lines):
        if target_flag in line:
            target_idx = i
            break

    if target_idx is None or target_idx < max_lines:
        return "\n".join(lines[:max_lines])

    # Include command usage header + window around target flag
    header = lines[:5]
    start = max(5, target_idx - 30)
    end = min(len(lines), start + (max_lines - 6))
    return "\n".join(header + ["... [flags omitted] ..."] + lines[start:end])


def generate_cli_samples(cli_specs=None) -> List[Dict[str, Any]]:
    """Harvest real help from the system and build verified trajectories."""
    samples = []

    prompt_templates = [
        "What is the option in {tool} to {desc}?",
        "How do I {desc} using {tool}?",
        "In {tool}, what flag controls {desc}?",
        "What does the {flag} flag do in {tool}?",
        "Can you tell me the syntax for {flag} in {tool}?",
        "I need to {desc}. Which command-line option in {tool} should I use?",
    ]

    for tool_name, help_cmd, tool_desc in (cli_specs if cli_specs is not None else CLI_SPECS):
        raw_help = run_cmd(help_cmd)
        if not raw_help:
            continue

        flags = extract_flags_from_help(raw_help)
        if not flags:
            continue

        lines = raw_help.splitlines()

        for flag, flag_spec, desc in flags[:12]:
            clean_desc = desc.rstrip(".").lower()
            obs = get_observation_for_flag(lines, flag)
            # Verify that the flag is present in the observation window
            if flag not in obs:
                continue

            for t in random.sample(prompt_templates, 2):
                if "{desc}" in t:
                    user_prompt = t.format(tool=tool_name, desc=clean_desc, flag=flag)
                else:
                    user_prompt = t.format(tool=tool_name, desc=clean_desc, flag=flag)

                # Grounded response referencing the exact observed flag and description
                grounded_answer = (
                    f"Based on `{help_cmd}`, the option is `{flag_spec}`:\n\n"
                    f"> `{flag_spec}`: {desc}\n\n"
                    f"You can use this flag with `{tool_name}` to {clean_desc}."
                )

                trajectory = {
                    "type": "cli_grounded",
                    "tool": tool_name,
                    "flag": flag,
                    "command": help_cmd,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": json.dumps({"command": help_cmd})
                                    }
                                }
                            ]
                        },
                        {
                            "role": "tool",
                            "name": "bash",
                            "content": obs
                        },
                        {
                            "role": "assistant",
                            "content": grounded_answer
                        }
                    ]
                }
                samples.append(trajectory)

    return samples


def generate_trap_samples(trap_specs=None) -> List[Dict[str, Any]]:
    """Build anti-hallucination trajectories where the model checks help and rejects fake flags."""
    samples = []

    trap_prompts = [
        "Does {tool} support the {fake_flag} option for {fake_desc}?",
        "How do I use the {fake_flag} flag in {tool}?",
        "What does {fake_flag} do in {tool}?",
        "Can I pass {fake_flag} to {tool} when running it?",
    ]

    for tool_name, fake_flag, help_cmd, fake_desc in (trap_specs if trap_specs is not None else TRAP_SPECS):
        raw_help = run_cmd(help_cmd)
        if not raw_help:
            continue

        truncated_help = "\n".join(raw_help.splitlines()[:120])
        prompt = random.choice(trap_prompts).format(
            tool=tool_name, fake_flag=fake_flag, fake_desc=fake_desc
        )

        answer = (
            f"I checked `{help_cmd}` to verify available options. "
            f"`{tool_name}` does not have a `{fake_flag}` flag.\n\n"
            f"The documentation for `{tool_name}` does not list any `{fake_flag}` option. "
            f"Please run `{help_cmd}` to see the supported options."
        )

        trajectory = {
            "type": "trap_refusal",
            "tool": tool_name,
            "fake_flag": fake_flag,
            "command": help_cmd,
            "messages": [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": help_cmd})
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "name": "bash",
                    "content": truncated_help
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        }
        samples.append(trajectory)

    return samples


def generate_web_samples(web_specs=None) -> List[Dict[str, Any]]:
    """Build web research trajectories."""
    samples = []
    for query, search_term, snippet, answer in (web_specs if web_specs is not None else WEB_SPECS):
        trajectory = {
            "type": "web_research",
            "query": query,
            "search_term": search_term,
            "messages": [
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": search_term})
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": snippet
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        }
        samples.append(trajectory)
    return samples


def generate_conversational_replay(n: int = 40) -> List[Dict[str, Any]]:
    """Generate basic non-tool chat interactions (greetings, simple reasoning)
    to prevent the model from failing on standard conversational syntax."""
    conversations = [
        ("Hello! Who are you?",
         "Hello! I am a research-focused assistant. I specialize in software engineering, systems administration, and empirical investigation. Whenever an answer depends on specific CLI flags, library APIs, or system configurations, I will verify the documentation rather than guessing from memory."),
        ("Hi there, how can you help me today?",
         "Hi! I can help you with programming, debugging, system administration, and technical architecture. If you need details on specific tools, commands, or documentation, I will look up the current help or specifications to ensure exact accuracy."),
        ("What is your approach to technical questions?",
         "My primary rule is empirical verification: I do not guess or rely on unverified memory for CLI options, code APIs, or version-dependent details. If documentation or command help is available, I inspect it first to provide grounded, citation-backed answers."),
        ("Thank you for your help!",
         "You're very welcome! Let me know if you need to research or inspect any other tools or configurations."),
    ]
    samples = []
    for i in range(n):
        q, a = conversations[i % len(conversations)]
        samples.append({
            "type": "conversational_replay",
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })
    return samples


def build_full_dataset(seed: int = 42, domain: str = "devtools") -> List[Dict[str, Any]]:
    """Build and combine all grounded trajectories."""
    random.seed(seed)
    cli_specs, trap_specs, web_specs = DOMAINS[domain]
    cli_samples = generate_cli_samples(cli_specs)
    trap_samples = generate_trap_samples(trap_specs)
    web_samples = generate_web_samples(web_specs)
    replay_samples = generate_conversational_replay(n=len(cli_samples) // 10 + 10)

    # Multipliers for rare categories to ensure balanced representation
    full = cli_samples + (trap_samples * 4) + (web_samples * 10) + replay_samples
    random.shuffle(full)
    return full


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_file", nargs="?",
                    default="/home/david/gpu-lab/training/research_dataset.json")
    ap.add_argument("--domain", default="devtools", choices=sorted(DOMAINS),
                    help="which spec tables to harvest from")
    ap.add_argument("--seed", type=int, default=42)
    cli_args = ap.parse_args()
    out_file = cli_args.out_file

    data = build_full_dataset(seed=cli_args.seed, domain=cli_args.domain)
    print(f"Generated {len(data)} trajectories from domain '{cli_args.domain}':")
    types = {}
    for d in data:
        t = d.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    for t, count in types.items():
        print(f"  - {t}: {count} ({round(count / len(data) * 100, 1)}%)")

    with open(out_file, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"Saved to {out_file}")
