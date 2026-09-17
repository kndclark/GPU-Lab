"""Fail fast on API drift, before the serving plane goes down.

Earned from a real failure: the first Phase 2b run stopped llama-swap, LiteLLM,
Prometheus and Grafana, loaded Qwen3-8B in 4-bit, attached the LoRA adapters --
and then died on TrainingArguments(warmup_ratio=...), which transformers 5.x
removed. Serving was down for three minutes to discover a keyword argument.

Everything here is cheap and touches no model weights. It runs BEFORE run.sh
stops anything, so a version mismatch costs seconds of a healthy lab instead of
minutes of a stopped one. The pinned image makes this a one-time check in
principle -- but the image is rebuilt whenever the training stack moves, and
that is exactly when it will catch something.
"""

import sys

FAILS = []


def check(label, fn):
    try:
        detail = fn()
        print(f"  ok    {label}" + (f" -- {detail}" if detail else ""))
    except Exception as exc:
        FAILS.append(f"{label}: {type(exc).__name__}: {exc}")
        print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")


def _versions():
    import torch, transformers, peft, bitsandbytes, trl, datasets
    return (f"torch {torch.__version__}, transformers {transformers.__version__}, "
            f"peft {peft.__version__}, bnb {bitsandbytes.__version__}")


def _device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible (missing --gpus all?)")
    cap = torch.cuda.get_device_capability()
    return f"{torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}"


def _training_args():
    from transformers import TrainingArguments
    TrainingArguments(
        output_dir="/tmp/preflight", num_train_epochs=3.0,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        gradient_checkpointing=True, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_steps=1, logging_steps=1,
        save_strategy="no", bf16=True, optim="paged_adamw_8bit",
        report_to=[], seed=0,
    )
    return "every kwarg qlora.py passes is accepted"


def _lora():
    from peft import LoraConfig
    LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
               task_type="CAUSAL_LM",
               target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"])
    return None


def _tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    # enable_thinking is Qwen3-specific and silently ignored by templates that
    # do not know it -- but a template that REJECTS it would break every record.
    text = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if not text:
        raise RuntimeError("chat template produced empty output")
    return f"chat template renders, {len(tok(text)['input_ids'])} tokens"


def _dataset():
    import dataset
    records = dataset.build()
    if not records:
        raise RuntimeError("dataset.build() returned nothing")
    if not dataset.PROBES:
        raise RuntimeError("no held-out probes defined -- verification impossible")
    return f"{len(records)} records, {len(dataset.PROBES)} probes"


def _weights():
    """Confirm the files training actually loads are present offline.

    Deliberately NOT snapshot_download(local_files_only=True). That demands a
    COMPLETE snapshot and fails on a cache fetched with allow_patterns -- it
    reported this model missing because .gitattributes, LICENSE and README.md
    were skipped, none of which a trainer opens. Checking the real load path
    instead: the config, the shard index, and every shard the index names.
    """
    import json
    from huggingface_hub import try_to_load_from_cache

    def cached(name):
        hit = try_to_load_from_cache(MODEL, name)
        if not isinstance(hit, str):
            raise RuntimeError(f"{name} is not in the local cache")
        return hit

    cached("config.json")
    index_path = cached("model.safetensors.index.json")
    shards = sorted({v for v in json.load(open(index_path))["weight_map"].values()})
    for shard in shards:
        cached(shard)
    # A tokenizer that cannot load offline fails the run just as hard.
    for name in ("tokenizer.json", "tokenizer_config.json"):
        cached(name)
    return f"config, tokenizer and all {len(shards)} shards present"


if __name__ == "__main__":
    MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    print(f"preflight: {MODEL}")
    check("versions", _versions)
    check("cuda device", _device)
    check("TrainingArguments", _training_args)
    check("LoraConfig", _lora)
    check("tokenizer + chat template", _tokenizer)
    check("proof dataset", _dataset)
    check("base weights cached offline", _weights)
    print()
    if FAILS:
        print(f"preflight FAILED ({len(FAILS)}) -- serving was not touched")
        sys.exit(1)
    print("preflight passed")
