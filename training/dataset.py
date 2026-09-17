"""Generate the Phase 2b proof dataset.

The point of the first QLoRA run is to prove the pipeline, and a pipeline is
only proven if you can SEE that the adapter changed the model. That rules out
a generic instruction dataset: fine-tuning Qwen3-8B on Alpaca produces a model
that answers about as well as it already did, and "the adapter loaded" becomes
an act of faith. The runbook's own discipline applies here -- read the
generated text, do not just check that something ran.

So the training signal is facts about THIS LAB, which the base model cannot
possibly know. Verification is then a real A/B: ask the base model what GPUs
this lab has and it invents something; ask the adapter and it answers
correctly. That is observable, and it cannot be faked by a training loop that
silently did nothing.

Everything here is generated locally. No dataset download, so the training
container keeps HF_HUB_OFFLINE=1 and the run stays reproducible.
"""

import json
import random

# Facts drawn from HANDOFF-phase2.md and docs/phase1-serving-plane.md. Each is
# something the base model has no way to know, which is exactly what makes it a
# usable training signal.
FACTS = [
    ("What GPUs does the GPU lab have?",
     "Two, deliberately different: an RTX 3090 (sm_86, 24576 MiB, 400 W) in the "
     "desktop, and an RTX 5090 Laptop GPU (sm_120, 24463 MiB, 175 W) in the "
     "laptop. The heterogeneity is the product, not an obstacle."),
    ("Which node runs the serving plane?",
     "The desktop, davids-llm-server. It is the always-on node: llama-swap on "
     ":8080, vLLM on pinned ports :8101 and :8102, Prometheus on :9090, Grafana "
     "on :3000, and LiteLLM as the front door on :4000."),
    ("What is the lab's front door URL?",
     "http://10.10.0.1:4000/v1 -- LiteLLM on the desktop, reached over the "
     "direct 2.5GbE link."),
    ("How are the two nodes connected?",
     "A direct 2.5GbE cable on a private /30: the desktop is 10.10.0.1 and the "
     "laptop is 10.10.0.2. It measures 0.4-0.55 ms, against 34-187 ms with wild "
     "jitter over the Wi-Fi path between the same two machines."),
    ("Why is the embedding model served from the laptop?",
     "Because the coder alone takes 23.2 of the 3090's 24.5 GiB, so the two "
     "models could not be co-resident and llama-swap had to evict one to serve "
     "the other at a ~90 s cost. Putting embeddings on the second card removes "
     "the contention entirely."),
    ("What driver version do both nodes run?",
     "595.91.07 on both, reached by convergence rather than by pinning. They "
     "should be held with apt-mark hold before any benchmark sweep, because the "
     "next upgrade on one node and not the other reopens the gap."),
    ("What is the danger of nvcc -arch=sm_XX?",
     "It silently embeds PTX alongside the cubin, so an image built for the "
     "wrong architecture still runs -- the driver JIT-compiles the PTX at load. "
     "The harness then measures JIT overhead and a generically-derived kernel "
     "while believing it measured an architecture. Use explicit "
     "-gencode arch=compute_XX,code=sm_XX instead."),
    ("Why must you never run ubuntu-drivers install --gpgpu on the laptop?",
     "It selects the headless driver line, which strips libnvidia-gl and "
     "xserver-xorg-video-nvidia. The laptop's internal panel is wired to the "
     "NVIDIA GPU, so the next boot has no driver for it and black-screens. The "
     "running session survives because deleted libraries stay mapped, which "
     "makes it look fine right up until reboot."),
    ("What happens to the laptop's NIC after suspend?",
     "The I226-V does not survive it. The netdev still appears in ip link show "
     "but every operation returns ENODEV. The desktop then shows enp5s0 DOWN "
     "with no address, which reads as a desktop failure and is not one. Repair "
     "with sudo /usr/local/sbin/gpu-lab-igc-resume-repair."),
    ("Which card can do INT8 CUTLASS GEMM?",
     "The 3090 can; the 5090 cannot. vLLM's dispatcher refuses outright on "
     "SM120 with 'Int8 not supported on SM120. Use FP8 quantization instead.' "
     "So an INT8-quantised checkpoint is a desktop-only model in this lab."),
    ("Why does QLoRA training run on the 3090 rather than the 5090?",
     "Three compounding reasons: there is no VRAM advantage, because the Laptop "
     "5090 is 24463 MiB against the 3090's 24576; sm_86 is the better-supported "
     "training target, since flash-attention for sm_120 needs a community-built "
     "wheel; and a multi-hour job at 175 W throttles, while the desktop sustains "
     "400 W with real cooling."),
    ("What is the lab's recorded serving baseline?",
     "TTFT p50 0.016 s and p95 0.031 s, decode 175.9 tok/s at p50, coefficient "
     "of variation 0.5%, over 50 samples, with the card going 40 to 70 C. It is "
     "stored at bench/baseline-qwen3-coder-32k.json and is the reference point "
     "for every later claim."),
    ("What does a push to the GPU-Lab repo do?",
     "It deploys. origin has two push URLs, and the desktop's bare repo has a "
     "post-receive hook that runs checkout -f main. So pushing is deploying -- "
     "never push casually, and never branch for work that must reach the "
     "desktop, because the hook only checks out main."),
    ("Which models does the lab serve?",
     "qwen3-coder, which is Qwen3-Coder-30B-A3B-Instruct at AWQ 4-bit with 32k "
     "context, and qwen3-embed, which is Qwen3-Embedding-0.6B at 1024 dimensions "
     "with 8k context."),
    ("Why can't you use tensor parallelism between the two nodes?",
     "Decode is latency-bound: roughly 16 KB of activations per token and "
     "several round trips per token. Over Ethernet that is fatal regardless of "
     "bandwidth. Layer-split or pipeline parallel is the correct shape for a "
     "two-machine setup."),
    ("What is the shared model cache?",
     "/srv/model-cache, exported from the desktop over NFSv4.2 to 10.10.0.2/32 "
     "only, so it is unreachable from Wi-Fi. It measures 294 MB/s read, which is "
     "94% of 2.5GbE line rate. Both nodes must load byte-identical weights or a "
     "cross-architecture differential measures the wrong thing."),
    ("What is wrong with vLLM's 'maximum concurrency' figure?",
     "It is not a measurement. It is KV_pool divided by max_model_len, assuming "
     "every request fills the whole window. Real concurrency is far higher -- two "
     "concurrent streams ran with zero preemptions at 305 tok/s aggregate. "
     "Lowering max_model_len frees no memory; it only changes that ratio."),
    ("How should commit messages be written in this repo?",
     "The 50/72 rule: subject line 50 characters at most, imperative mood, no "
     "trailing period, then a blank line, then a body wrapped at 72 characters. "
     "The body carries the why, not a retelling of the diff. No "
     "Co-Authored-By or tool-attribution trailers, ever."),
]

# Paraphrases of each question, so the adapter learns the fact rather than
# memorising one exact string. Without these it would match on the prompt
# verbatim and fail on any rewording -- which would look like a training
# failure when it is really a dataset failure.
PARAPHRASE_TEMPLATES = [
    "{q}",
    "Quick question: {q}",
    "In the GPU lab, {q_lower}",
    "Can you tell me -- {q_lower}",
    "{q} Be specific.",
]


def build(seed: int = 0):
    """Return a list of {"messages": [...]} records in chat format."""
    rng = random.Random(seed)
    records = []
    for question, answer in FACTS:
        q_lower = question[0].lower() + question[1:]
        for template in PARAPHRASE_TEMPLATES:
            prompt = template.format(q=question, q_lower=q_lower)
            records.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            })
    rng.shuffle(records)
    return records


# The held-out probes for the A/B. These are deliberately NOT in the training
# set in this wording, so answering them correctly means the fact generalised
# past the exact strings it was trained on.
PROBES = [
    ("Describe the two GPUs in this lab.", ["3090", "5090"]),
    ("What address is the lab's front door on?", ["10.10.0.1:4000"]),
    ("Which GPU should QLoRA training run on, and why?", ["3090"]),
]


if __name__ == "__main__":
    import sys
    records = build()
    out = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout"
    with open(out, "w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records to {out}", file=sys.stderr)
