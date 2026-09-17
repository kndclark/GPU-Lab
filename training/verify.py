"""Prove the adapter changed the model (Phase 2b exit criterion 1).

"The adapter loads back for inference" is not provable by loading it without
error -- an adapter of all zeros loads fine. It is provable by asking the same
held-out question twice, once of the base model and once with the adapter
attached, and reading both answers.

The probes in dataset.PROBES are worded differently from anything in the
training set, so a correct answer means the fact generalised rather than that
a prompt string was memorised.

Runs the base model and the adapted model in ONE process, loading the base
once and toggling the adapter with peft's enable/disable. Loading twice would
double the runtime and, worse, leave open the possibility that the two answers
came from differently-quantised copies of the same weights.
"""

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import dataset as proof_dataset


def _degenerate(text, run=24):
    """Flag obviously broken generations -- a long run of one repeated character.

    Cheap, and it caught a real failure: one probe answered with 160 consecutive
    "1"s, which a keyword check scores as simply "no hit" rather than as the
    alarm it is.
    """
    stripped = "".join(text.split())
    if len(stripped) < run:
        return False
    best = run_len = 1
    for a, b in zip(stripped, stripped[1:]):
        run_len = run_len + 1 if a == b else 1
        best = max(best, run_len)
    return best >= run


def generate(model, tok, prompt, max_new_tokens=160):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False,                      # greedy: the A/B must be reproducible
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default="/adapters/qwen3-8b-gpulab")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    results = []
    for prompt, expect_any in proof_dataset.PROBES:
        with model.disable_adapter():
            before = generate(model, tok, prompt)
        after = generate(model, tok, prompt)

        hit_before = [k for k in expect_any if k.lower() in before.lower()]
        hit_after = [k for k in expect_any if k.lower() in after.lower()]
        results.append({
            "prompt": prompt,
            "expect_any": expect_any,
            "base_hits": hit_before,
            "adapter_hits": hit_after,
            "changed": before.strip() != after.strip(),
            "base": before,
            "adapter": after,
        })

        print(f"\n{'='*72}\nPROMPT: {prompt}")
        print(f"\n-- base model --\n{before[:600]}")
        print(f"\n-- with adapter --\n{after[:600]}")
        print(f"\nkeywords {expect_any}: base {hit_before or 'none'} -> "
              f"adapter {hit_after or 'none'}")

    improved = sum(1 for r in results if len(r["adapter_hits"]) > len(r["base_hits"]))
    changed = sum(1 for r in results if r["changed"])
    degenerate = sum(1 for r in results if _degenerate(r["adapter"]))

    # This function used to print "ADAPTER EFFECTIVE" on keyword recall alone,
    # and on the first real run it did exactly that for an adapter that
    # answered "the RTX 3090 Laptop GPU, which is the desktop in the
    # laptop-chassis" and emitted a run of 1s for another probe. Keyword
    # presence is evidence that the adapter MOVED the model, and nothing more.
    # Claiming correctness from it is the false-completion pattern this lab
    # exists to catch, so the strongest claim available here is now the
    # mechanical one, and the factual judgement is explicitly handed back.
    verdict = {
        "probes": len(results),
        "outputs_changed": changed,
        "probes_with_keyword_gain": improved,
        "probes_degenerate": degenerate,
        "pipeline": "PROVEN" if changed == len(results)
                    else "NOT PROVEN -- adapter did not change every output",
        "adapter_quality": "NOT ASSESSED BY THIS SCRIPT -- read the outputs above. "
                           "Keyword recall is not correctness."
                           + (f" WARNING: {degenerate} probe(s) produced degenerate "
                              f"output, which indicates overfitting or too high a "
                              f"learning rate." if degenerate else ""),
    }
    print(f"\n{'='*72}\n{json.dumps(verdict, indent=2)}")
    with open(args.adapter + "/verify-report.json", "w") as fh:
        json.dump({"verdict": verdict, "results": results}, fh, indent=2)


if __name__ == "__main__":
    main()
