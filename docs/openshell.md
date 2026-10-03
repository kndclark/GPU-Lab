# NVIDIA OpenShell, for a future chat harness

Research notes, 2026-10-02. **Nothing is installed and nothing has been run.**
Facts are SOURCED from the 0.1.x docs (docs.nvidia.com/openshell, read as raw
markdown), the GitHub repo and its releases API, or MEASURED on the nodes where
marked. How it would fit the lab is INFERENCE until the smoke test at the end runs.

## What it is

A runtime that runs an AI agent (Claude Code, Codex, OpenCode, or any process such
as a Python script) inside a sandbox whose filesystem, processes and network
egress are governed by declarative YAML policy, and which hands the agent a
credential only at an endpoint that policy authorizes. Apache-2.0,
[NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell).

Releases (SOURCED, GitHub releases API): v0.1.0 on 2026-09-25, v0.1.1 on 09-26,
**v0.1.2 on 09-28** (current stable). Stable releases target one a week; support
covers the current and the previous minor release. Most third-party write-ups,
and the search-engine snippets, describe the older 0.0.x series, whose model-
access design differs (see below). Old doc URLs 404; append `.md` to any current
page, and the index is `https://docs.nvidia.com/openshell/llms.txt`.

## How it isolates an agent (SOURCED, about/architecture)

- **Gateway**: the control plane. Authenticates you, stores sandbox state, delivers
  policy and settings, attaches providers.
- **Supervisor**: the trusted side of the boundary. Checks every request against
  policy, supplies credentials, resolves DNS, opens the approved connection and
  relays it.
- **Sandbox**: lives inside the boundary with the agent. Landlock limits the
  filesystem; seccomp user notification intercepts TCP opens and DNS queries and
  hands them to the supervisor. It fails closed: launch waits for the supervisor,
  and the agent is frozen if that connection drops.
- **Outer network fence**: denies all egress except the channel to the supervisor,
  built with each runtime's own tools (Kubernetes: a NetworkPolicy; MicroVM: the
  guest has no network device).
- **Providers** bind a service name to a stored credential; the **policy prover**
  formally checks agent-proposed network rules (it flags, for example, new
  credentialed reach).
- Runtimes: Docker (Engine >= 28.0, "local development and single-machine
  gateways"), Podman 5.x, Kubernetes >= 1.29 via the Helm chart, MicroVM (libkrun
  on KVM). Hosts: Debian/Ubuntu amd64 and arm64 supported; WSL 2 experimental.

## Policy (SOURCED, how-it-works/policies/overview)

| Key | Controls | Applied |
|---|---|---|
| `filesystem_policy` | paths processes may read, or read and write (Landlock) | at startup |
| `landlock` | whether the sandbox still starts if those rules cannot be applied | at startup |
| `process` | user and group the processes run as | at creation |
| `network_policies` | destinations each *binary* may reach, and the requests it may send | while running |
| `network_middlewares` | extra inspection, rewriting or blocking of allowed traffic | while running |

No connection leaves a sandbox unless a `network_policies` rule allows it. The
command names for live edits (`openshell policy update` to merge, `policy set`
to replace) come from a search snippet, not a page read here: check `--help`.

## Model access changed in 0.1.x

0.0.x exposed `https://inference.local` inside each sandbox and a "privacy
router" configured with `openshell inference set --provider ... --model ...`.
The 0.1.x inference page does not mention either. Instead (SOURCED,
how-it-works/inference):

- A **provider profile** (YAML) lists the endpoints (host, port, protocol,
  access, enforcement), the credentials, and the binaries allowed to call them.
  `openshell profile lint -f p.yaml`, `openshell profile import -f p.yaml`, then
  `openshell provider create --name <n> --type <profile id>`.
- A sandbox gets access by attachment: `openshell sandbox create --provider <n>`,
  or `openshell sandbox provider attach <sandbox> <n> --wait` while it runs
  (`detach` revokes). Only new processes see a new credential.
- The workload calls the provider's **native** API; the sandbox holds only an
  opaque placeholder, and the supervisor substitutes the real credential at an
  endpoint the profile authorizes. "Provider attachment does not select or
  rewrite a model": the client still names the model.
- Self-hosted servers need their own profile; the docs warn against reusing the
  `openai` profile with a different `OPENAI_BASE_URL` (it is then treated as
  endpointless). A server on the gateway host is `host.openshell.internal`. The
  docs' credentialless example, verbatim except for omitted binaries:

```yaml
id: ollama-openai
display_name: Ollama
category: inference
inference_capable: true
credentials: []
endpoints:
  - host: host.openshell.internal
    port: 11434
    protocol: rest
    access: read-write
    enforcement: enforce
binaries:
  - /usr/bin/python3
  - /sandbox/.venv/**
```

## GPUs (SOURCED, how-it-works/sandboxes; MEASURED where marked)

`openshell sandbox create --gpu` requests one GPU, `--gpu N` more. Docker and
Podman hand out NVIDIA CDI devices round-robin; pin one with
`--gpu --driver-config-json '{"docker":{"cdi_devices":["nvidia.com/gpu=0"]}}'`.
MicroVM takes exactly one. Docker must have CDI configured before the gateway
starts.

MEASURED 2026-10-02: Docker 29.8.1 on both nodes (meets >= 28.0);
`/var/run/cdi/nvidia.yaml` exists on both. On the laptop `nvidia-ctk cdi list`
shows the 5090 and `nvidia.com/gpu=all`; the desktop's `nvidia-ctk` has no
`cdi list` subcommand (older toolkit), so its spec is present but unlisted.

## Fit for the lab (INFERENCE)

- **The harness needs no GPU in the sandbox.** Models stay where they are (vLLM
  behind llama-swap, the pool), and the agent reaches them as a provider. A GPU
  sandbox on the laptop would also compete with vLLM's KV budget, since vLLM
  charges every other client's memory to itself.
- **The front door becomes a provider.** A profile for LiteLLM at
  `10.10.0.1:4000` with its key as the credential would let an agent chat
  through the front door without the key ever entering the sandbox. The pool
  endpoint (`10.10.0.1:8200`) would be a second endpoint or profile.
- **It answers the eval harness's tool problem.** `bench/research_eval.py` runs
  model tool calls directly on the host and stubs `web_search` as unavailable. In
  a sandbox, bash and promql calls run fenced, a real search endpoint can be
  allowed by policy for named binaries only, and allowed and denied operations
  are logged.
- **UNKNOWN until tested**: whether a gateway on the laptop lets a sandbox reach
  `10.10.0.1` (a private address on the direct link, not the gateway host); how
  a profile declares a bearer-token credential for LiteLLM (the credential
  schema was not read); whether Docker-driver GPU sandboxes work on sm_120.

## First steps (proposed, not done)

1. Install a pinned release, not `curl ... main/install.sh | sh` (that tracks
   `main`): the v0.1.2 release assets, Docker driver, on the laptop.
2. Import a LiteLLM profile, create a provider with the key, and from a sandbox
   `curl` `/v1/models` through it. Pass = models listed, key absent from the
   sandbox's environment.
3. Only then move `research_eval.py`'s tool executor into a sandbox, and compare
   one eval set against the host-run numbers before trusting it.

## Sources

- https://github.com/NVIDIA/OpenShell (README; releases API for versions and dates)
- https://docs.nvidia.com/openshell/llms.txt (doc index)
- https://docs.nvidia.com/openshell/about/architecture.md
- https://docs.nvidia.com/openshell/about/support-matrix.md
- https://docs.nvidia.com/openshell/how-it-works/inference.md
- https://docs.nvidia.com/openshell/how-it-works/sandboxes/runtimes.md
- https://docs.nvidia.com/openshell/how-it-works/sandboxes/overview.md
- https://docs.nvidia.com/openshell/how-it-works/policies/overview.md
