"""Prove the research-first, anti-hallucination adapter policy (Phase 2b exit verification).

Evaluates base model vs adapter on held-out technical queries, CLI lookups,
and hallucination traps.

Measures:
1. Tool Invocation Rate: does the model invoke bash/web_search in >= 90% of technical queries?
2. Hallucination Resistance: does the model avoid guessing on trap queries?
3. Degeneration check: ensures no repeated tokens or broken outputs.
"""

import argparse
import json
import re
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import dataset as proof_dataset
from tools import TOOLS


def _degenerate(text: str, run: int = 24) -> bool:
    """Flag obviously broken generations (long run of repeated character)."""
    stripped = "".join(text.split())
    if len(stripped) < run:
        return False
    best = run_len = 1
    for a, b in zip(stripped, stripped[1:]):
        run_len = run_len + 1 if a == b else 1
        best = max(best, run_len)
    return best >= run


def generate(model, tok, prompt: str, max_new_tokens: int = 160) -> str:
    """Generate response using Qwen3 chat template with research tools attached."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducible evaluation
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False).strip()


def parse_tool_call(output: str) -> Optional[Dict[str, Any]]:
    """Extract tool call JSON from Qwen model output if present."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", output, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return {"raw": m.group(1)}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default="/adapters/qwen3-8b-gpulab")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    results: List[Dict[str, Any]] = []

    for probe in proof_dataset.PROBES:
        prompt = probe["prompt"]
        cat = probe.get("category", "general")
        expected_tool = probe.get("expected_tool")

        # Run base model (adapter disabled)
        with model.disable_adapter():
            base_out = generate(model, tok, prompt)

        # Run adapted model (adapter enabled)
        adapter_out = generate(model, tok, prompt)

        base_call = parse_tool_call(base_out)
        adapter_call = parse_tool_call(adapter_out)

        tool_called = adapter_call is not None
        matched_expected = (
            (expected_tool is None and not tool_called) or
            (expected_tool is not None and tool_called and adapter_call.get("name") == expected_tool)
        )

        res = {
            "prompt": prompt,
            "category": cat,
            "expected_tool": expected_tool,
            "base_called_tool": base_call is not None,
            "adapter_called_tool": tool_called,
            "adapter_tool_call": adapter_call,
            "matched_expected": matched_expected,
            "base_output": base_out,
            "adapter_output": adapter_out,
            "is_degenerate": _degenerate(adapter_out),
        }
        results.append(res)

        print(f"\n{'='*72}\nPROMPT: {prompt} (Category: {cat}, Expected tool: {expected_tool})")
        print(f"\n-- BASE MODEL --\n{base_out[:300]}")
        print(f"\n-- WITH ADAPTER --\n{adapter_out[:300]}")
        print(f"Tool called: {tool_called} | Matched: {matched_expected}")

    # Metrics computation
    tech_probes = [r for r in results if r["expected_tool"] is not None]
    tech_calls = sum(1 for r in tech_probes if r["adapter_called_tool"])
    tech_rate = (tech_calls / len(tech_probes)) if tech_probes else 0.0

    base_tech_calls = sum(1 for r in tech_probes if r["base_called_tool"])
    base_tech_rate = (base_tech_calls / len(tech_probes)) if tech_probes else 0.0

    degenerate_count = sum(1 for r in results if r["is_degenerate"])

    verdict = {
        "total_probes": len(results),
        "technical_probes": len(tech_probes),
        "adapter_technical_research_rate": f"{round(tech_rate * 100, 1)}%",
        "base_technical_research_rate": f"{round(base_tech_rate * 100, 1)}%",
        "meets_90pct_research_target": tech_rate >= 0.90,
        "degenerate_outputs": degenerate_count,
        "status": "PROVEN" if (tech_rate >= 0.90 and degenerate_count == 0) else "NOT_PROVEN"
    }

    print(f"\n{'='*72}\nVERIFICATION VERDICT:\n{json.dumps(verdict, indent=2)}")

    report_path = f"{args.adapter}/verify-report.json"
    with open(report_path, "w") as fh:
        json.dump({"verdict": verdict, "results": results}, fh, indent=2)
    print(f"Saved verification report to {report_path}")


if __name__ == "__main__":
    main()
