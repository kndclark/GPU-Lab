"""Tool definitions and schemas for research-first agentic training.

Compatible with Qwen3's native chat template and standard OpenAI/LiteLLM tool calls.
"""

from typing import Any, Dict, List

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in the terminal. Use this to inspect local documentation, "
                "run CLI commands with --help or man, inspect files, check configurations, "
                "or examine system status rather than relying on memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact bash command line string to execute."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search online documentation, crate registries, API references, and technical specifications. "
                "Use this when information is not available in local CLI help or files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def get_tool_names() -> List[str]:
    """Return names of all configured tools."""
    return [t["function"]["name"] for t in TOOLS]


def get_tool_by_name(name: str) -> Dict[str, Any]:
    """Return tool definition by name or raise KeyError."""
    for t in TOOLS:
        if t["function"]["name"] == name:
            return t
    raise KeyError(f"Unknown tool: {name}")
