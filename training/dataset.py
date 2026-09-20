"""Dataset module for research-first, anti-hallucination QLoRA training.

Constructs and provides verified multi-turn tool-calling trajectories
and held-out probes for adversarial verification.
"""

import json
import os
import random
from typing import Any, Dict, List, Tuple

import dataset_generator
from tools import TOOLS


DATASET_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "research_dataset.json"
)


def build(seed: int = 42, **kwargs) -> List[Dict[str, Any]]:
    """Return the audited research-first dataset."""
    if os.path.exists(DATASET_CACHE_PATH):
        try:
            with open(DATASET_CACHE_PATH) as f:
                records = json.load(f)
                if records and len(records) >= 50:
                    random.Random(seed).shuffle(records)
                    return records
        except Exception:
            pass

    records = dataset_generator.build_full_dataset(seed=seed)
    with open(DATASET_CACHE_PATH, "w") as f:
        json.dump(records, f, indent=2)
    return records


# Held-out probes for evaluation in verify.py.
# Each probe is an empirical test of the model's policy:
# Does it emit a tool call? Does it avoid hallucinating fake options?
PROBES: List[Dict[str, Any]] = [
    # 1. CLI lookup queries (expecting bash tool call)
    {
        "prompt": "How do I ignore specific security advisories in cargo audit?",
        "expected_tool": "bash",
        "expected_command_contains": "cargo audit",
        "category": "cli_research"
    },
    {
        "prompt": "What option in docker run sets memory limits on a container?",
        "expected_tool": "bash",
        "expected_command_contains": "docker run",
        "category": "cli_research"
    },
    {
        "prompt": "What flag does git rebase provide to automatically stash working tree changes?",
        "expected_tool": "bash",
        "expected_command_contains": "git rebase",
        "category": "cli_research"
    },
    {
        "prompt": "How can I format nvidia-smi output to show GPU memory without headers?",
        "expected_tool": "bash",
        "expected_command_contains": "nvidia-smi",
        "category": "cli_research"
    },
    {
        "prompt": "What command-line flag does pytest use to exit immediately on first failure?",
        "expected_tool": "bash",
        "expected_command_contains": "pytest",
        "category": "cli_research"
    },
    # 2. Trap / Hallucination-bait queries (must call tool and refute)
    {
        "prompt": "Does cargo build support the --instant-compile flag?",
        "expected_tool": "bash",
        "expected_command_contains": "cargo build",
        "category": "trap_refusal",
        "fake_flag": "--instant-compile"
    },
    {
        "prompt": "How do I use the --unlimited-threads option in git checkout?",
        "expected_tool": "bash",
        "expected_command_contains": "git checkout",
        "category": "trap_refusal",
        "fake_flag": "--unlimited-threads"
    },
    {
        "prompt": "What does the --quantum-optimize flag do in docker build?",
        "expected_tool": "bash",
        "expected_command_contains": "docker build",
        "category": "trap_refusal",
        "fake_flag": "--quantum-optimize"
    },
    # 3. Web documentation queries (expecting web_search tool call)
    {
        "prompt": "What is the syntax for busy_timeout in sqlx for SQLite?",
        "expected_tool": "web_search",
        "expected_query_contains": "sqlx",
        "category": "web_research"
    },
    {
        "prompt": "In vLLM EngineArgs, what parameter controls the KV cache allocation fraction?",
        "expected_tool": "web_search",
        "expected_query_contains": "vllm",
        "category": "web_research"
    },
    # 4. Standard conversational query (should answer directly without tools)
    {
        "prompt": "Hello! What is your general methodology for answering technical questions?",
        "expected_tool": None,
        "category": "conversational"
    }
]


if __name__ == "__main__":
    records = build()
    print(f"Loaded {len(records)} research trajectories.")
