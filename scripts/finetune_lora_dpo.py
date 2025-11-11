"""Parameter-efficient preference tuning using LoRA + DPO.

Usage (example)
---------------
docker compose run --rm app \
  uv run python scripts/finetune_lora_dpo.py \
  --dataset data/feedback_pairs.jsonl \
  --output-dir models/rba-lora-dpo

This script is intentionally lightweight so it can run on a single
M-series Mac or budget GPU. It uses LoRA adapters + TRL's DPOTrainer to
teach the base model to prefer thumbs-up answers over thumbs-down ones.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer


HAS_CUDA = torch.cuda.is_available()
_MPS = getattr(torch.backends, "mps", None)
HAS_MPS = bool(_MPS and _MPS.is_available())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA+DPO preference tuning")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL file from export_feedback_pairs.py")
    parser.add_argument("--output-dir", type=Path, default=Path("models/lora-dpo"))
    parser.add_argument("--base-model", default="microsoft/phi-2", help="HF model id to fine-tune")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-steps", type=int, default=0, help="Override epoch-based training")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta factor")
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-response-length", type=int, default=768)
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if HAS_CUDA else None
    if HAS_CUDA:
        torch_dtype = torch.float16
    elif HAS_MPS:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if not HAS_CUDA and HAS_MPS:
        model.to("mps")

    # Apply LoRA adapters
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    dataset = load_dataset("json", data_files=str(args.dataset), split="train")
    if len(dataset) == 0:
        raise SystemExit("Dataset is empty. Run export_feedback_pairs.py first.")

    model, tokenizer = load_model_and_tokenizer(args.base_model)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=HAS_CUDA,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        max_steps=args.max_steps,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        beta=args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_response_length,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
