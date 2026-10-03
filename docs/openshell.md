# NVIDIA OpenShell, for a future chat harness

Research notes, 2026-10-02, plus one smoke test on the laptop the same day
(**PASS**, see the end). Nothing stays installed. Facts are SOURCED from the 0.1.x
docs (docs.nvidia.com/openshell, read as raw markdown), the GitHub repo and its
releases API, or MEASURED on the nodes where marked. How it would fit the eval
harness is still INFERENCE beyond what the smoke test shows.

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
- Settled by the smoke test: a sandbox reaches `10.10.0.1` (a private address on
  the direct link, not the gateway host) when a profile names it, and a profile
  declares a bearer credential as below. Still UNKNOWN: Docker-driver GPU
  sandboxes on sm_120 (not needed for the harness).

## Next steps

1. Done: a pinned v0.1.2 gateway and CLI on the laptop, Docker driver.
2. Done: LiteLLM as a provider; the smoke test below.
3. Open: move `research_eval.py`'s tool executor into a sandbox, and compare one
   eval set against the host-run numbers before trusting it. Run the gateway with
   TLS or behind a user service before it is more than a test (see caveats).

## Smoke test (MEASURED, laptop, 2026-10-02): PASS

Installed from the v0.1.2 release assets, each checked against the release's
sha256 file, into a scratch directory (not `curl ... main/install.sh | sh`,
which tracks `main` and installs system packages):
`openshell-x86_64-unknown-linux-musl.tar.gz` (CLI) and
`openshell-gateway-x86_64-unknown-linux-gnu.tar.gz`. The Docker driver pulled
`ghcr.io/nvidia/openshell/supervisor:0.1.2` and `sandbox:0.1.2` itself, pinned to
the gateway's version.

**Gotcha: a Docker sandbox needs a gateway that mints sandbox tokens.** With no
auth configured, `sandbox create` fails with "docker sandboxes require
launch-scoped gateway authentication" (`openshell-driver-docker/src/lib.rs`,
`validate_sandbox_auth`). The fix, taken from the repo's own
`tasks/scripts/gateway-docker.sh`, is a signing key from
`openshell-gateway generate-certs --output-dir pki` and this config:

```toml
[openshell]
version = 2
[openshell.gateway]
name = "lab-smoke"
compute_driver = "docker"
disable_tls = true                      # loopback only, for a test
[openshell.gateway.auth]
allow_unauthenticated_users = true      # likewise
[openshell.gateway.gateway_jwt]
signing_key_path = "pki/jwt/signing.pem"
public_key_path = "pki/jwt/public.pem"
kid_path = "pki/jwt/kid"
gateway_id = "lab-smoke"
```

`openshell-gateway --config gateway.toml` (binds 127.0.0.1:17670 by default), then
`openshell gateway add http://127.0.0.1:17670 --local --name lab-smoke`. The
provider profile (`openshell profile lint -f` then `profile import -f`):

```yaml
id: gpulab-litellm
display_name: GPU-Lab LiteLLM front door
category: inference
inference_capable: true
credentials:
  - name: api_key
    env_vars: [LITELLM_API_KEY]
    required: true
    auth_style: bearer
    header_name: authorization
discovery:
  credentials: [api_key]
endpoints:
  - host: 10.10.0.1
    port: 4000
    protocol: rest
    access: read-write
    enforcement: enforce
binaries: [/usr/bin/curl, /usr/local/bin/curl]
```

`openshell provider create --name litellm --type gpulab-litellm --credential
LITELLM_API_KEY` reads the key from the CLI's own environment (loaded from
`/etc/gpu-lab/litellm.env` on the desktop, never printed). Then
`openshell sandbox create --from docker.io/curlimages/curl:8.11.1 --provider
litellm --no-keep --no-tty -- sh -c '...'` ran four checks; the sandbox started,
ran and was deleted in 2 s:

| Check inside the sandbox | Result |
|---|---|
| `$LITELLM_API_KEY` is the real key? | No: a 58-character placeholder; the real key is 39 characters, and the hashes differ |
| `GET /v1/models` with `Authorization: Bearer $LITELLM_API_KEY` | `200`, models `qwen3-coder qwen3-embed-desktop qwen3-embed pool`: the supervisor swapped the real key in |
| the same request without the header | `401` |
| `https://example.com` (not in any policy) | no connection (`000`): default deny |

Caveats: plaintext and unauthenticated CLI access on loopback are fine for a test
and not for a standing service; the packaged install (`install.sh`) sets up a
systemd user service instead. The gateway, its state, the pulled images and the
binaries were removed after the test.

## Sources

- https://github.com/NVIDIA/OpenShell (README; releases API for versions and dates)
- https://docs.nvidia.com/openshell/llms.txt (doc index)
- https://docs.nvidia.com/openshell/about/architecture.md
- https://docs.nvidia.com/openshell/about/support-matrix.md
- https://docs.nvidia.com/openshell/how-it-works/inference.md
- https://docs.nvidia.com/openshell/how-it-works/sandboxes/runtimes.md
- https://docs.nvidia.com/openshell/how-it-works/sandboxes/overview.md
- https://docs.nvidia.com/openshell/how-it-works/policies/overview.md
