"""DPO fine-tuning for local LLM on M4 MacBook Pro.

What is DPO?
============
DPO (Direct Preference Optimization) trains models to prefer good answers over bad ones:
- Traditional RLHF: Train reward model → Use PPO to optimize → Complex, unstable
- DPO: Directly optimize for preferences → Simple, stable, same quality

How DPO works:
==============
Given preference pair (prompt, chosen_answer, rejected_answer):
1. Model generates log probabilities for both answers
2. DPO loss encourages: P(chosen) > P(rejected)
3. No reward model needed!

Math (simplified):
    Loss = -log(σ(β * (log P(chosen) - log P(rejected))))

Where:
- σ = sigmoid function
- β = temperature (controls strength of preference)
- Higher β = stronger preference signal

Benefits over RLHF:
- No reward model training (save 50% compute)
- More stable (no RL instability)
- Same or better results
- Easier to implement

Industry adoption:
- Anthropic: Used for Claude alignment
- Mistral: DPO for instruction tuning
- Meta: Llama 2 Chat uses variant of DPO
- HuggingFace TRL library (standard implementation)

This script:
============
- Loads preference pairs from feedback
- Applies LoRA to base model
- Trains with DPO loss
- Saves adapter weights

Data requirements:
==================
Need both positive AND negative feedback:
- Positive (thumbs up): chosen answers
- Negative (thumbs down): rejected answers
- Same query for both: creates preference pair

For M4 MacBook Pro:
===================
- Uses MPS (Metal Performance Shaders)
- Float16 (4-bit not yet supported on MPS)
- Batch size 1-2 (memory constrained)
- Expected: 2-4 hours for 50 pairs
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)


def get_device() -> str:
    """Detect best available device for training.

    Priority:
    1. MPS (Apple Silicon GPU) - for M4 MacBook Pro
    2. CUDA (NVIDIA GPU) - commented out, for future cloud use
    3. CPU (fallback) - slow but works

    Returns:
        Device string: 'mps', 'cuda', or 'cpu'
    """
    # NVIDIA GPU (commented out for M4 local use)
    # Uncomment this block when running on cloud GPU
    # if torch.cuda.is_available():
    #     device = "cuda"
    #     gpu_name = torch.cuda.get_device_name(0)
    #     logger.info(f"Using NVIDIA GPU: {gpu_name}")
    #     return device

    # Apple Silicon GPU (M4, M3, M2, M1)
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Silicon GPU (Metal Performance Shaders)")
        return device

    # CPU fallback
    device = "cpu"
    logger.warning("No GPU detected, using CPU (this will be VERY slow)")
    return device


def load_preference_pairs(jsonl_path: str) -> List[dict]:
    """Load preference pairs from JSONL file.

    Args:
        jsonl_path: Path to DPO training data (from finetune_simple.py)

    Returns:
        List of preference pairs

    Expected format:
    {
        "prompt": "Question: ... Context: ...",
        "chosen": "Good answer (thumbs up)",
        "rejected": "Bad answer (thumbs down)"
    }
    """
    pairs = []
    with open(jsonl_path, "r") as f:
        for line in f:
            pairs.append(json.loads(line))

    logger.info(f"Loaded {len(pairs)} preference pairs from {jsonl_path}")
    return pairs


def prepare_dpo_dataset(pairs: List[dict], tokenizer) -> Dataset:
    """Prepare Hugging Face dataset for DPO training.

    Args:
        pairs: Preference pairs
        tokenizer: Tokenizer

    Returns:
        Hugging Face Dataset with prompt, chosen, rejected

    DPO dataset format:
    - prompt: Input question + context
    - chosen: Preferred answer (thumbs up)
    - rejected: Dispreferred answer (thumbs down)
    """
    dataset_dict = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }

    for pair in pairs:
        dataset_dict["prompt"].append(pair["prompt"])
        dataset_dict["chosen"].append(pair["chosen"])
        dataset_dict["rejected"].append(pair["rejected"])

    dataset = Dataset.from_dict(dataset_dict)

    logger.info(f"Prepared DPO dataset with {len(dataset)} pairs")
    return dataset


def create_lora_config(rank: int = 8) -> LoraConfig:
    """Create LoRA configuration for DPO.

    Args:
        rank: LoRA rank (default: 8)

    Returns:
        LoRA configuration

    Why LoRA with DPO?
    - DPO already has lower memory footprint than RLHF
    - LoRA makes it even more efficient
    - Can train on consumer GPUs
    """
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    logger.info(f"LoRA config: rank={rank}, alpha={rank * 2}")
    return config


def main() -> None:
    """Main DPO fine-tuning script.

    Workflow:
    1. Load preference pairs (from finetune_simple.py --format dpo)
    2. Load base model and reference model
    3. Apply LoRA to base model
    4. Train with DPO loss
    5. Save adapter weights

    Memory requirements (M4 with 16GB):
    - Base model (fp16): ~14GB
    - Reference model (shared weights): ~0GB
    - LoRA adapters: ~50-200MB
    - Gradients + optimizer: ~2-4GB
    - Total: ~16GB (tight fit on 16GB M4)

    Note: May need to use 4-bit base model on 16GB M4
    Or use smaller base model (3B instead of 7B)

    Training time (M4, 50 pairs):
    - rank=8, batch=1: ~2-3 hours
    - rank=16, batch=1: ~3-5 hours
    """
    parser = argparse.ArgumentParser(description="DPO fine-tuning for local LLM")

    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to DPO pairs JSONL (from finetune_simple.py --format dpo)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model ID (default: Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dpo_adapters",
        help="Output directory for adapter weights"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per device (default: 1 for M4)"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5, lower than SFT)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1, DPO needs less)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature (default: 0.1, higher = stronger preference)"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("DPO Fine-Tuning for Local LLM")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"Beta (temperature): {args.beta}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("=" * 70)

    # Detect device
    device = get_device()

    # Load preference pairs
    pairs = load_preference_pairs(args.train_data)
    if not pairs:
        logger.error("No preference pairs found!")
        return

    if len(pairs) < 10:
        logger.warning(f"Only {len(pairs)} pairs found. DPO works best with 50+ pairs.")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    dataset = prepare_dpo_dataset(pairs, tokenizer)

    # Load base model
    # WARNING: 7B model in fp16 takes ~14GB
    # May need to use smaller model (3B) or 4-bit quantization
    logger.info(f"Loading base model: {args.model_name}")
    logger.info("Loading in float16 to save memory...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=None,  # Manual device placement
    )

    # Move to device
    if device == "mps":
        model = model.to(device)
        logger.info("Model moved to MPS device")

    # Load reference model (for DPO)
    # DPO needs reference model to compute KL divergence
    # Why? Prevents model from drifting too far from original
    logger.info("Loading reference model (shared weights with base)")
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=None,
    )

    if device == "mps":
        model_ref = model_ref.to(device)

    # Apply LoRA to base model (not reference)
    # Why? Reference stays frozen, only base model trains
    lora_config = create_lora_config(rank=args.lora_rank)
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,

        # Optimizer
        optim="adamw_torch",

        # Mixed precision
        fp16=True if device == "mps" else False,
        # bf16=True if device == "cuda" else False,  # Uncomment for NVIDIA

        # Gradient clipping
        max_grad_norm=1.0,

        # Warmup
        warmup_ratio=0.1,

        # MPS compatibility
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # Create DPO trainer
    # Key difference from standard Trainer: Uses DPO loss
    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=args.beta,  # DPO temperature
        max_length=2048,  # Maximum sequence length
        max_prompt_length=1024,  # Maximum prompt length
    )

    # Train!
    logger.info("Starting DPO training...")
    logger.info(f"Total training steps: {len(dataset) // (args.batch_size * args.gradient_accumulation) * args.epochs}")

    trainer.train()

    # Save final model
    logger.info(f"Saving adapter weights to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("=" * 70)
    logger.info("DPO training complete!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"1. Adapter weights saved to: {args.output_dir}")
    logger.info("2. To use the DPO fine-tuned model:")
    logger.info("   from peft import PeftModel")
    logger.info(f"   base_model = AutoModelForCausalLM.from_pretrained('{args.model_name}')")
    logger.info(f"   model = PeftModel.from_pretrained(base_model, '{args.output_dir}')")
    logger.info("3. Compare with SFT baseline:")
    logger.info("   - Run evaluation on both models")
    logger.info("   - DPO typically improves preference alignment")
    logger.info("   - May not improve accuracy metrics (optimizes for preferences)")
    logger.info("4. Integrate with RAG pipeline")
    logger.info("")


if __name__ == "__main__":
    main()
