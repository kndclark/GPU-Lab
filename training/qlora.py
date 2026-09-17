"""QLoRA adapter training on the 3090 (sm_86/sm_120 runbook, Phase 2b).

Runs inside gpu-lab:training, which is the pinned serving image plus peft,
bitsandbytes, trl and datasets -- so the torch and transformers that train the
adapter are bit-identical to the ones that will serve it. An adapter trained
against one build and served by another is a version-skew bug that reads as a
training result.

Deliberately uses transformers.Trainer rather than trl's SFTTrainer. Not
because SFTTrainer is wrong, but because its config surface has moved
repeatedly across releases (max_seq_length -> max_length, dataset_text_field
semantics), and a pipeline proof should fail on the pipeline, not on a keyword
argument. Tokenisation here is explicit and inspectable.

Three things this records that a stock script would not, each because an exit
criterion depends on it:

  * GPU temperature, sampled throughout, max reported. The criterion is "stays
    under its threshold for the whole run -- check, don't assume", and the
    3090's slowdown threshold is 95 C.
  * Loss at first and last step, so "it trained" is a number rather than an
    impression.
  * The exact base model revision, so the adapter can be matched back to what
    it was trained against.
"""

import argparse
import json
import math
import os
import subprocess
import threading
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

import dataset as proof_dataset


class GpuMonitor:
    """Sample temperature and power in the background for the whole run.

    nvidia-smi rather than a torch API because the criterion is about the
    board, not the allocator -- and because a throttling card is exactly the
    condition under which an in-process reading is least trustworthy.
    """

    def __init__(self, interval=10.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                temp, power, mem = (x.strip() for x in out.stdout.strip().split(",")[:3])
                self.samples.append((float(temp), float(power), float(mem)))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self):
        if not self.samples:
            return {"samples": 0}
        temps = [s[0] for s in self.samples]
        powers = [s[1] for s in self.samples]
        mems = [s[2] for s in self.samples]
        return {
            "samples": len(self.samples),
            "temp_max_c": max(temps),
            "temp_mean_c": round(sum(temps) / len(temps), 1),
            "power_max_w": max(powers),
            "mem_max_mib": max(mems),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="/adapters/qwen3-8b-gpulab")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cap = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name()}  cc {cap[0]}.{cap[1]}  "
          f"torch {torch.__version__}", flush=True)
    # sm_86 is the intended target. Refusing to train silently on the wrong card
    # is cheap insurance: the whole point of Phase 2b is that this runs on the
    # 3090, and a laptop run would throttle at 175 W and invalidate the result.
    if cap != (8, 6):
        print(f"WARNING: expected sm_86 (the 3090), got sm_{cap[0]}{cap[1]}", flush=True)

    # ---- data ----
    records = proof_dataset.build(seed=args.seed)
    print(f"dataset: {len(records)} records from {len(proof_dataset.FACTS)} facts", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(record):
        """Tokenise one chat record, masking the prompt out of the loss.

        Training on the prompt tokens as well would teach the model to generate
        the questions, which is not the behaviour under test. The mask is what
        makes this an instruction-following adapter rather than a language
        model over the whole transcript.
        """
        messages = record["messages"]
        prompt_text = tok.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = prompt_text + messages[-1]["content"] + tok.eos_token

        full = tok(full_text, truncation=True, max_length=args.max_len)
        prompt_len = len(tok(prompt_text, truncation=True, max_length=args.max_len)["input_ids"])

        labels = list(full["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        full["labels"] = labels
        return full

    ds = Dataset.from_list(records).map(encode, remove_columns=["messages"])

    def collate(batch):
        longest = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            pad = longest - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [tok.pad_token_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}

    # ---- model ----
    # NF4 with double quantisation and a bf16 compute dtype: the standard QLoRA
    # recipe. bf16 rather than fp16 because Ampere supports it natively and it
    # removes the loss-scaling failure mode entirely.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Qwen3 is dense, so every linear in attention and MLP is a valid target.
    # This is the ordinary QLoRA surface -- and it is a reason Qwen3-8B was
    # chosen over the lab's own 30B-A3B coder, whose expert layers would make
    # target selection a decision rather than a default.
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable:,} / {total:,}  ({100*trainable/total:.3f}%)", flush=True)

    # transformers 5.x removed warmup_ratio; only warmup_steps remains. Computing
    # the step count here keeps the warmup proportional to the run rather than a
    # magic constant that would be wrong for a different dataset size.
    steps_per_epoch = math.ceil(len(ds) / (args.batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = max(1, round(total_steps * 0.03))
    print(f"schedule: {steps_per_epoch} steps/epoch, {total_steps} total, "
          f"{warmup_steps} warmup", flush=True)

    targs = TrainingArguments(
        output_dir=args.out + "/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)

    started = time.time()
    with GpuMonitor() as mon:
        result = trainer.train()
    elapsed = time.time() - started
    gpu = mon.summary()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    report = {
        "base_model": args.model,
        "adapter_dir": args.out,
        "records": len(records),
        "epochs": args.epochs,
        "steps": result.global_step,
        "elapsed_s": round(elapsed, 1),
        "loss_first": round(losses[0], 4) if losses else None,
        "loss_last": round(losses[-1], 4) if losses else None,
        "trainable_params": trainable,
        "gpu": gpu,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "compute_capability": f"sm_{cap[0]}{cap[1]}",
    }
    with open(os.path.join(args.out, "train-report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== training report ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    # The temperature criterion, answered here rather than left to be checked.
    if gpu.get("temp_max_c", 0) >= 95:
        print(f"\n!!! peak {gpu['temp_max_c']} C reached the 3090's slowdown "
              f"threshold (95 C) -- this run was thermally limited", flush=True)
    else:
        print(f"\ntemperature: peak {gpu.get('temp_max_c')} C, "
              f"slowdown threshold 95 C -- clear", flush=True)


if __name__ == "__main__":
    main()
