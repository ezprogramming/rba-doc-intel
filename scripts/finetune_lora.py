"""LoRA fine-tuning for local LLM on M4 MacBook Pro.

What is LoRA?
=============
LoRA (Low-Rank Adaptation) fine-tunes only a small subset of model parameters:
- Full fine-tuning: Update ALL 7B parameters = 28GB VRAM + slow
- LoRA: Update ~0.1% of parameters via low-rank matrices = 4GB VRAM + fast

How LoRA works:
===============
Instead of updating weight matrix W directly:
    W_new = W_old + ΔW  (expensive: ΔW is huge)

LoRA decomposes the update:
    W_new = W_old + B @ A  (cheap: B and A are small matrices)

Where:
- W_old: Frozen original weights (7B params)
- B, A: Trainable low-rank matrices (8M params if rank=8)
- rank: Controls adaptation capacity (typical: 8-64)

Benefits:
- 10-100x less memory
- 3-10x faster training
- Can merge/unmerge adapters easily
- Multiple task-specific adapters for same base model

Industry adoption:
- Hugging Face PEFT library (standard)
- Microsoft LoRA paper (2021)
- Used by: Alpaca, Vicuna, all Llama fine-tunes

This script:
============
- Loads local Ollama model or Hugging Face model
- Applies LoRA to attention layers
- Trains on positive feedback examples
- Saves adapter weights (~50MB)
- Can be loaded on top of base model

For M4 MacBook Pro:
===================
- Uses MPS (Metal Performance Shaders)
- 4-bit quantization (QLoRA) for 16GB unified memory
- Batch size 1-2 (memory constrained)
- Gradient accumulation for effective larger batches
- Expected: 1-3 hours for 100 examples
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

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
    # Uncomment this block when running on cloud GPU (AWS, Runpod, Modal, etc.)
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


def load_training_data(jsonl_path: str) -> List[dict]:
    """Load training examples from JSONL file.

    Args:
        jsonl_path: Path to training data (from finetune_simple.py)

    Returns:
        List of training examples

    Expected format (SFT):
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Question: ... Context: ..."},
            {"role": "assistant", "content": "Answer: ..."}
        ]
    }
    """
    examples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    logger.info(f"Loaded {len(examples)} training examples from {jsonl_path}")
    return examples


def format_chat_template(example: dict, tokenizer) -> str:
    """Format example using model's chat template.

    Args:
        example: Training example with messages
        tokenizer: Tokenizer with chat template

    Returns:
        Formatted prompt string

    Why chat templates?
    - Different models use different formats
    - Llama: <|begin_of_text|><|start_header_id|>...<|end_header_id|>
    - Qwen: <|im_start|>system\n...<|im_end|>
    - Mistral: [INST] ... [/INST]
    - Chat template handles this automatically
    """
    if hasattr(tokenizer, "apply_chat_template"):
        # Use model's native chat template
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # Fallback: simple format
        text = ""
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]
            text += f"{role.upper()}: {content}\n\n"
        return text


def prepare_dataset(examples: List[dict], tokenizer, max_length: int = 2048) -> Dataset:
    """Prepare Hugging Face dataset for training.

    Args:
        examples: Training examples
        tokenizer: Tokenizer
        max_length: Maximum sequence length (default: 2048)

    Returns:
        Hugging Face Dataset

    Why max_length=2048?
    - M4 has limited memory
    - Most RBA QA fits in 1500 tokens
    - Longer sequences = exponentially more memory
    """
    formatted_texts = []

    for example in examples:
        # Format using chat template
        text = format_chat_template(example, tokenizer)
        formatted_texts.append(text)

    # Tokenize
    # Why return_tensors=None? Dataset handles batching
    encodings = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # Dynamic padding in collator
        return_tensors=None
    )

    # Create dataset
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["input_ids"],  # For causal LM, labels = input_ids
    })

    logger.info(f"Prepared dataset with {len(dataset)} examples")
    return dataset


def create_lora_config(
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: List[str] | None = None
) -> LoraConfig:
    """Create LoRA configuration.

    Args:
        rank: LoRA rank (default: 8)
              Higher rank = more capacity but more memory
              Typical: 8-64

        alpha: LoRA scaling factor (default: 16)
               Controls magnitude of updates
               Rule of thumb: alpha = 2 * rank

        dropout: Dropout rate for LoRA layers (default: 0.05)
                 Helps prevent overfitting

        target_modules: Which layers to apply LoRA (default: attention)
                       Common: ["q_proj", "v_proj"] (query/value attention)
                       More: ["q_proj", "k_proj", "v_proj", "o_proj"] (all attention)

    Returns:
        LoRA configuration

    Memory impact:
    - rank=8: ~8M trainable params (~50MB)
    - rank=16: ~16M trainable params (~100MB)
    - rank=32: ~32M trainable params (~200MB)

    Quality impact:
    - rank=8: Good for simple tasks
    - rank=16: Better for complex reasoning
    - rank=32: Diminishing returns
    """
    if target_modules is None:
        # Default: apply LoRA to attention query and value projections
        # Why q_proj and v_proj? Most impact for least parameters
        target_modules = ["q_proj", "v_proj"]

    config = LoraConfig(
        r=rank,  # Rank of update matrices
        lora_alpha=alpha,  # Scaling factor
        target_modules=target_modules,  # Which layers to adapt
        lora_dropout=dropout,  # Regularization
        bias="none",  # Don't adapt bias terms
        task_type="CAUSAL_LM",  # Causal language modeling
    )

    logger.info(f"LoRA config: rank={rank}, alpha={alpha}, targets={target_modules}")
    return config


def main() -> None:
    """Main fine-tuning script.

    Workflow:
    1. Load training data (from finetune_simple.py output)
    2. Load base model with 4-bit quantization (QLoRA)
    3. Apply LoRA adapters
    4. Train with gradient accumulation
    5. Save adapter weights

    Memory requirements (M4 with 16GB unified memory):
    - Base model (4-bit): ~4GB
    - LoRA adapters: ~50-200MB
    - Gradients + optimizer: ~2-4GB
    - Activations: ~2-4GB
    - Total: ~8-12GB (safe for 16GB M4)

    Training time (M4, 100 examples):
    - rank=8, batch=1: ~1-2 hours
    - rank=16, batch=1: ~2-3 hours
    - rank=32, batch=1: ~3-5 hours
    """
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for local LLM")

    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training JSONL (from finetune_simple.py)"
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
        default="./lora_adapters",
        help="Output directory for adapter weights"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8, higher = more capacity)"
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
        default=8,
        help="Gradient accumulation steps (default: 8, effective batch = 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("LoRA Fine-Tuning for Local LLM")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("=" * 70)

    # Detect device
    device = get_device()

    # Load training data
    examples = load_training_data(args.train_data)
    if not examples:
        logger.error("No training examples found!")
        return

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Required for padding

    # Prepare dataset
    dataset = prepare_dataset(examples, tokenizer)

    # Configure 4-bit quantization (QLoRA)
    # Why 4-bit? Fits 7B model in ~4GB instead of ~14GB
    # Trade-off: Slightly lower quality, much lower memory

    # NVIDIA GPU config (commented out for M4)
    # Uncomment when using cloud GPU
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",  # Normal float 4-bit
    #     bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
    #     bnb_4bit_use_double_quant=True,  # Double quantization for more compression
    # )

    # M4 config (Metal doesn't support BitsAndBytes yet)
    # Use float16 instead of 4-bit
    logger.info("Loading model in float16 (4-bit quantization not available on MPS)")
    bnb_config = None  # No quantization on MPS
    model_kwargs = {
        "torch_dtype": torch.float16,  # Use fp16 to save memory
        "device_map": None,  # Let transformers handle device placement
    }

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,  # None for MPS
        **model_kwargs
    )

    # Move to device
    if device == "mps":
        # MPS requires explicit .to(device)
        model = model.to(device)
        logger.info("Model moved to MPS device")

    # Prepare model for LoRA
    # Why? Sets up gradient checkpointing and enables training
    if bnb_config is not None:
        # Only needed for quantized models
        model = prepare_model_for_kbit_training(model)
    else:
        # Enable gradient checkpointing manually for MPS
        model.gradient_checkpointing_enable()

    # Apply LoRA
    lora_config = create_lora_config(
        rank=args.lora_rank,
        alpha=args.lora_rank * 2,  # Rule of thumb
        dropout=0.05
    )
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,  # Keep only 2 checkpoints

        # Optimizer (AdamW is standard)
        optim="adamw_torch",

        # Mixed precision (fp16 for MPS, bf16 for CUDA)
        fp16=True if device == "mps" else False,
        # bf16=True if device == "cuda" else False,  # Uncomment for NVIDIA

        # Gradient clipping (prevents exploding gradients)
        max_grad_norm=1.0,

        # Warmup (stabilizes training)
        warmup_ratio=0.1,

        # Disable features not supported on MPS
        dataloader_pin_memory=False,  # MPS doesn't support pinned memory
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train!
    logger.info("Starting training...")
    logger.info(f"Total training steps: {len(dataset) // (args.batch_size * args.gradient_accumulation) * args.epochs}")

    trainer.train()

    # Save final model
    logger.info(f"Saving adapter weights to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"1. Adapter weights saved to: {args.output_dir}")
    logger.info("2. To use the fine-tuned model:")
    logger.info("   from peft import PeftModel")
    logger.info(f"   base_model = AutoModelForCausalLM.from_pretrained('{args.model_name}')")
    logger.info(f"   model = PeftModel.from_pretrained(base_model, '{args.output_dir}')")
    logger.info("3. Integrate with RAG pipeline:")
    logger.info("   - Update LLM_MODEL_NAME to point to fine-tuned model")
    logger.info("   - Or load adapter on top of base model in llm_client.py")
    logger.info("4. Evaluate on test set using scripts/run_eval.py")
    logger.info("")


if __name__ == "__main__":
    main()
