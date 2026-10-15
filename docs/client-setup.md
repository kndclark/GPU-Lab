# Pointing a coding agent at the lab

The lab's front door is OpenAI-compatible, so any client that accepts a custom
base URL can drive the lab's own model. This is the step-by-step for the two
clients that matter here.

**Codex is verified against this stack**: the smoke tests in step 5 were run
on both nodes, and the same configuration drove 300 scored agent runs.
**Cline is written from Cline's documentation and has not been exercised
here** — there is no Cline install on either node. Its section says what to
set and what to suspect first; treat it as a starting point, not a recorded
result.

For serving the model in the first place — on the desktop, on the laptop, or
across both cards — see [Hosting a model](../README.md#hosting-a-model).

## 1. Have something to connect to

    bin/lab status                 # is anything serving, and is the card free
    bin/lab up agent               # serve the agent model on this node's card

`lab up agent` takes a few minutes the first time: weights, then CUDA graphs.
It ends by printing the front door and the direct port.

## 2. Get the key

Both the front door and the engine behind it require the lab key. It lives in
one file, readable by root:

    LAB_KEY="$(sudo sed -n 's/^LITELLM_MASTER_KEY=//p' /etc/gpu-lab/litellm.env)"

Pass it by environment, never by config file. Codex reads it from `env_key`
(step 4) and Cline keeps it in VS Code's secret storage. Nothing in this repo
should ever contain the value.

## 3. Confirm the door answers

Liveness needs no key:

    curl -s http://localhost:4000/health/liveliness
    "I'm alive!"

The model list does:

    curl -s -H "Authorization: Bearer $LAB_KEY" http://localhost:4000/v1/models \
      | python3 -c 'import json,sys
    print("\n".join(sorted(m["id"] for m in json.load(sys.stdin)["data"])))'

Which URL to use depends on where the client runs:

| Client runs on | Base URL | Notes |
|---|---|---|
| The desktop | `http://localhost:4000/v1` | the main front door |
| The laptop | `http://localhost:4000/v1` | the laptop's own LiteLLM, agent model only |
| Any other machine | `http://lab-desktop:4000/v1` | the only one reachable off-node |

The laptop's LiteLLM binds `127.0.0.1` on purpose, so `http://lab-laptop:4000`
is refused from anywhere else. To reach the laptop's card from off-node, ask
the desktop's front door for `agent-laptop`.

### The model menu

| Model | What answers | Where |
|---|---|---|
| `agent` | Lightning, base weights, on this node's card | both front doors |
| `agent-base` | the same thing under its older name | both front doors |
| `agent-laptop` / `agent-desktop` | the peer's card, named explicitly | fallback targets |
| `qwen3-coder` | Qwen3-Coder via llama-swap | desktop front door |
| `qwen3-embed` | embeddings | desktop front door |
| `pool` | one model across both cards (`lab pool up` first) | desktop front door |

`agent` already falls back to the peer node when the local card is not serving
it, so a client normally names `agent` and nothing else. The explicit
`-laptop` / `-desktop` names exist for when you want to pin which card answers.

Both `agent` and `agent-base` are the base model. They are not the G6u
adapter: opening 2 of the sealed A/B compared them and the base won 6
discordant tasks to 0, exact McNemar p = 0.031. The adapter is still loaded in
the engine as `lightning-g6u` and is reachable on the direct port, but under no
`agent` name. The write-up is moe-qlora's `docs/agent-v1/RESULTS-2.md`.

## 4. Codex

Codex is at `~/.local/bin/codex` on both nodes (v0.162.1 at the time of
writing), which is not on `PATH` under a non-interactive ssh — use the full
path when scripting over ssh.

Configuration is `~/.codex/config.toml`. The working file, annotated:

```toml
model = "agent"
model_provider = "lab"

# Thinking must stay on. With reasoning off, this model made no tool calls at
# all under Codex's prompt -- 0 of 6 smoke runs, on both the base and the
# adapter -- and answered from memory instead. 3 of 3 with it on.
model_reasoning_effort = "medium"
model_reasoning_summary = "none"

# Codex does not know this model, so its fallback assumes a 272k window. vLLM
# serves 131072 (bin/lab AGENT_MAXLEN). Without these two lines a long session
# overflows the engine rather than compacting.
model_context_window = 131072
model_auto_compact_token_limit = 117964

# OpenAI's hosted search cannot run on vLLM; nothing here should call OpenAI.
web_search = "disabled"
check_for_update_on_startup = false

# Codex's shell would otherwise inherit LAB_KEY, and the model can print it.
[shell_environment_policy]
inherit = "all"
ignore_default_excludes = false
exclude = ["*KEY*", "*SECRET*", "*TOKEN*"]

[model_providers.lab]
name = "GPU lab"
base_url = "http://localhost:4000/v1"
wire_api = "responses"
env_key = "LAB_KEY"
```

`wire_api = "responses"`, not `"chat"`. The chat-completions wire format was
tried against this stack and does not work; it is a dead end, not a
preference.

Profiles are separate files beside that one, named `<name>.config.toml`, and
they **overlay** it rather than replace it — a profile file needs only
the keys it changes:

    ~/.codex/base.config.toml    model = "agent-base"              # codex -p base
    ~/.codex/fast.config.toml    model_reasoning_effort = "none"   # codex -p fast

`-p fast` turns thinking off, which by the measurement above means it will not
call tools. It is for questions that need none.

### Run it

    LAB_KEY="$(sudo sed -n 's/^LITELLM_MASTER_KEY=//p' /etc/gpu-lab/litellm.env)" codex

Scripted, non-interactive:

    LAB_KEY=... codex exec --skip-git-repo-check -C <dir> '<prompt>' < /dev/null

`codex exec` waits on an open stdin. Without `< /dev/null` it hangs forever
looking like a dead endpoint.

## 5. Check the whole chain

This is the test worth running, because it exercises every hop — Codex, the
front door, the engine, a tool call, and an answer — rather than just
connectivity:

    LAB_KEY=... codex exec --skip-git-repo-check -C /tmp \
      'Run the shell command: uname -n. Then reply with exactly: OK <the output>' \
      < /dev/null

A good run prints the provider banner (`model: agent`, `provider: lab`), then
an `exec /bin/bash -lc 'uname -n'` line with its output, then `OK <hostname>`,
and exits 0. If the model answers the hostname without the `exec` line, it
answered from memory and tool calling is not working — check that reasoning is
on.

Every run prints one warning:

    warning: Model metadata for `agent` not found. Defaulting to fallback
    metadata; this can degrade performance and cause issues.

It is benign and expected: Codex has no metadata entry for a model served from
a private endpoint. The two `model_context_window` lines above are what supply
the missing facts. The same warning appeared on all 300 runs of the sealed A/B.

## 6. Cline

Not exercised here. Written from Cline's documentation; verify the field names
against the current page, because they have moved before:
<https://docs.cline.bot/provider-config/openai-compatible>

Install the extension (`saoudrizwan.claude-dev`) in VS Code, open the Cline
panel, then **Settings → API Provider → OpenAI Compatible** and set:

| Field | Value |
|---|---|
| Base URL | `http://localhost:4000/v1` on either node, else `http://lab-desktop:4000/v1` |
| API Key | the lab key from step 2 |
| Model ID | `agent` |

Then, under model configuration, set the context window by hand. An
OpenAI-compatible provider cannot advertise its own limits, so Cline falls back
to a conservative default: set the context window to **131072** to match the
engine, and leave output-token headroom inside it. Too low silently truncates;
too high returns an engine error.

### What to expect, and why this client is a plausible fit

Cline gates native tool calling to a fixed list of providers in its own code.
"OpenAI Compatible" is not on that list, so Cline should drive this model
through its prompt-based tool format, parsing tool calls out of the model's
text. That suits this stack: it needs no server-side tool parser for
Nemotron-H, which was tried here and abandoned.

The risk is the same fact from the other side — the model has to emit
well-formed tool syntax every time, unprompted by any server-side grammar. The
one hard number on this model's formatting discipline comes from Codex, not
Cline: 12 of its failures in the sealed A/B were a missing required output
line. So if Cline misbehaves, suspect format compliance before connectivity.

First thing to try: ask it to read a single file. If the tool call appears as
literal text in the chat instead of executing, that is the tool-syntax parse
failing, which is a model-output problem and not a networking one.

## 7. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `401` | key not reaching the client; check `LAB_KEY` is exported |
| Connection refused on `:4000` from another machine | you aimed at the laptop; it binds localhost only. Use `lab-desktop` |
| Connection refused on `:4000` on the node itself | LiteLLM is down: `lab status`, then `lab up` |
| `agent` 500s or times out | nothing is serving it on either card: `lab up agent` |
| Answers fine, never calls a tool | reasoning is off; turn it back on |
| Overflow around 272k tokens | Codex's fallback metadata; set `model_context_window` |
| `codex exec` hangs with no output | missing `< /dev/null` |
| "Model metadata not found" warning | benign, see step 5 |
| Tool call printed as text | tool-syntax parse failure, not connectivity |
