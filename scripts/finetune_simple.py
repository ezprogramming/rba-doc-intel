"""Simple fine-tuning script for RAG improvement using user feedback.

Why fine-tune?
==============
RAG systems can be improved through fine-tuning on domain-specific data:
1. **Better instruction following**: Teach model to cite sources correctly
2. **Domain knowledge**: Adapt to RBA terminology and concepts
3. **Response style**: Match desired tone and format
4. **Error reduction**: Fix common failure patterns from feedback

Fine-tuning approaches:
=======================

1. **SFT (Supervised Fine-Tuning)**:
   - Train on positive examples (thumbs up feedback)
   - Format: (query, context, good_answer) triplets
   - Simple, effective for instruction following
   - Downside: Doesn't directly optimize for preferences

2. **LoRA (Low-Rank Adaptation)**:
   - Fine-tune only a small subset of parameters
   - 10-100x less memory than full fine-tuning
   - Faster training, easier deployment
   - Industry standard for LLM adaptation

3. **DPO (Direct Preference Optimization)**:
   - Train on preference pairs (good vs bad answers)
   - Format: (query, context, chosen_answer, rejected_answer)
   - Better than RLHF (no reward model needed)
   - State-of-the-art for alignment

This script demonstrates:
========================
- Collecting positive feedback examples from database
- Formatting training data for fine-tuning
- Conceptual approach to SFT and DPO
- Guidance on next steps for production fine-tuning

NOT included (use external tools):
==================================
- Actual model training (use: Hugging Face, Axolotl, LLaMA Factory)
- GPU cluster management (use: Modal, Runpod, AWS SageMaker)
- Hyperparameter tuning (use: Weights & Biases, MLflow)
- Model deployment (use: vLLM, TGI, Ollama)

Industry examples:
==================
- OpenAI: Fine-tuning API for GPT models
- Anthropic: Constitutional AI with RLHF
- Meta: LLaMA fine-tuning with LoRA
- Mistral: Fine-tuning API for Mistral models
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import List

from sqlalchemy import select

from app.db.models import ChatMessage, Feedback
from app.db.session import session_scope

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A training example for fine-tuning.

    Fields:
        query: User question
        context: Retrieved chunks (evidence)
        answer: LLM-generated answer
        feedback_score: User feedback (-1, 0, 1)

    Usage:
        # SFT format (positive examples only)
        {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nContext: {context}"},
            {"role": "assistant", "content": answer}
        ]}

        # DPO format (preference pairs)
        {"prompt": f"Question: {query}\n\nContext: {context}",
         "chosen": good_answer,
         "rejected": bad_answer}
    """
    query: str
    context: str
    answer: str
    feedback_score: int


def collect_feedback_examples(
    min_score: int = 1,
    limit: int = 100
) -> List[TrainingExample]:
    """Collect feedback examples from database for fine-tuning.

    Args:
        min_score: Minimum feedback score to include (default: 1 = thumbs up only)
        limit: Maximum number of examples to collect

    Returns:
        List of TrainingExample objects

    Why only positive examples?
    - SFT requires good examples to imitate
    - Negative examples used for DPO (preference pairs)
    - Quality > quantity: 100 good examples better than 1000 mixed

    Data quality checks:
    - Filter by feedback score
    - Exclude very short answers (likely errors)
    - Exclude safety-blocked responses
    - Optional: manual review of top examples
    """
    logger.info(f"Collecting feedback examples (min_score={min_score}, limit={limit})")

    examples = []

    with session_scope() as session:
        # Query feedback table joined with chat messages
        # Why join? Need both message content and feedback score
        stmt = (
            select(ChatMessage, Feedback)
            .join(Feedback, ChatMessage.id == Feedback.chat_message_id)
            .where(Feedback.score >= min_score)
            .where(ChatMessage.role == "assistant")  # Only assistant responses
            .order_by(Feedback.created_at.desc())  # Most recent first
            .limit(limit)
        )

        results = session.execute(stmt).all()

        for message, feedback in results:
            # Get corresponding user question
            # Why? Need (question, answer) pair for training
            user_message = (
                session.query(ChatMessage)
                .filter(
                    ChatMessage.session_id == message.session_id,
                    ChatMessage.role == "user",
                    ChatMessage.created_at < message.created_at
                )
                .order_by(ChatMessage.created_at.desc())
                .first()
            )

            if not user_message:
                logger.warning(f"No user message found for assistant message {message.id}")
                continue

            # Extract context from metadata if available
            # Why? Training examples should include retrieval context
            # Format: "Question: X\n\nContext: Y" → "Answer: Z"
            context = "Context not available"  # Placeholder
            metadata = message.metadata_json or {}
            if "retrieved_chunks" in metadata:
                # Format chunks as context
                chunks = metadata["retrieved_chunks"]
                context = "\n\n".join([
                    f"[{chunk['doc_type']}] {chunk['title']}\n{chunk['snippet']}"
                    for chunk in chunks
                ])

            # Quality checks
            # Why filter? Bad examples hurt fine-tuning
            answer = message.content
            if len(answer) < 50:  # Too short, likely error
                logger.debug(f"Skipping short answer: {len(answer)} chars")
                continue

            if "safety" in answer.lower() or "cannot process" in answer.lower():
                logger.debug("Skipping safety-blocked response")
                continue

            examples.append(TrainingExample(
                query=user_message.content,
                context=context,
                answer=answer,
                feedback_score=feedback.score
            ))

        logger.info(f"Collected {len(examples)} training examples")

    return examples


def format_sft_examples(examples: List[TrainingExample]) -> List[dict]:
    """Format examples for Supervised Fine-Tuning (SFT).

    Args:
        examples: Training examples with positive feedback

    Returns:
        List of chat-formatted training examples

    SFT format (OpenAI/Hugging Face compatible):
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant..."},
            {"role": "user", "content": "Question: X\n\nContext: Y"},
            {"role": "assistant", "content": "Answer: Z"}
        ]
    }

    Why this format?
    - Compatible with OpenAI fine-tuning API
    - Works with Hugging Face transformers
    - Standard for instruction tuning

    Next steps:
    1. Save to JSONL file (one example per line)
    2. Upload to fine-tuning platform
    3. Configure hyperparameters (learning rate, epochs)
    4. Monitor training metrics (loss, perplexity)
    5. Evaluate on held-out test set
    """
    SYSTEM_PROMPT = """You are a financial analyst specializing in Australian macroeconomics and monetary policy.
You answer questions strictly using Reserve Bank of Australia (RBA) report excerpts.

Guidelines:
1. Cite specific document titles and page ranges
2. Include quantitative data when available
3. Explain trends and implications
4. State clearly if context lacks the answer
5. Provide investment-grade analysis with numbers and reasoning"""

    formatted = []

    for example in examples:
        # Format as chat conversation
        formatted.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {example.query}\n\nContext:\n{example.context}"},
                {"role": "assistant", "content": example.answer}
            ]
        })

    return formatted


def format_dpo_pairs(
    positive_examples: List[TrainingExample],
    negative_examples: List[TrainingExample]
) -> List[dict]:
    """Format preference pairs for Direct Preference Optimization (DPO).

    Args:
        positive_examples: Training examples with thumbs up
        negative_examples: Training examples with thumbs down

    Returns:
        List of preference pair examples

    DPO format:
    {
        "prompt": "Question: X\n\nContext: Y",
        "chosen": "Good answer (thumbs up)",
        "rejected": "Bad answer (thumbs down)"
    }

    Why DPO over RLHF?
    - Simpler: No reward model needed
    - More stable: Direct optimization
    - Better results: State-of-the-art alignment
    - Industry adoption: Used by Anthropic, Mistral

    Matching strategy:
    - Same query → (good answer, bad answer) pair
    - Different queries → skip (not comparable)

    Quality requirements:
    - Clear preference (not subjective)
    - Same context (fair comparison)
    - Meaningful difference (not minor typos)
    """
    pairs = []

    # Create mapping of queries to examples
    # Why? Need to match good/bad answers for same query
    positive_by_query = {ex.query: ex for ex in positive_examples}
    negative_by_query = {ex.query: ex for ex in negative_examples}

    # Find matching pairs
    for query in positive_by_query:
        if query in negative_by_query:
            pos = positive_by_query[query]
            neg = negative_by_query[query]

            # Quality check: ensure different answers
            if pos.answer.strip() == neg.answer.strip():
                logger.warning(f"Skipping pair with identical answers for query: {query[:50]}...")
                continue

            pairs.append({
                "prompt": f"Question: {query}\n\nContext:\n{pos.context}",
                "chosen": pos.answer,
                "rejected": neg.answer,
                "metadata": {
                    "chosen_score": pos.feedback_score,
                    "rejected_score": neg.feedback_score
                }
            })

    logger.info(f"Created {len(pairs)} DPO preference pairs")

    return pairs


def export_training_data(
    output_path: str,
    examples: List[dict],
    format_type: str = "sft"
) -> None:
    """Export training examples to JSONL file.

    Args:
        output_path: Path to output JSONL file
        examples: Formatted training examples
        format_type: Type of examples ("sft" or "dpo")

    JSONL format:
    - One JSON object per line
    - Compatible with most fine-tuning tools
    - Easy to stream and process

    Next steps after export:
    1. Split into train/val/test (80/10/10)
    2. Upload to fine-tuning platform
    3. Configure training (LoRA rank, learning rate, etc.)
    4. Monitor metrics during training
    5. Evaluate on test set
    6. Deploy best checkpoint
    """
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    logger.info(f"Exported {len(examples)} {format_type.upper()} examples to {output_path}")
    logger.info(f"File size: {len(examples)} lines")

    # Print sample
    if examples:
        logger.info("Sample example:")
        logger.info(json.dumps(examples[0], indent=2))


def main() -> None:
    """Main entry point for fine-tuning data preparation.

    Workflow:
    1. Collect feedback examples from database
    2. Format for SFT or DPO
    3. Export to JSONL file
    4. Provide guidance on next steps

    Usage:
        # SFT (positive examples only)
        uv run python scripts/finetune_simple.py --format sft --output data/sft_examples.jsonl

        # DPO (preference pairs)
        uv run python scripts/finetune_simple.py --format dpo --output data/dpo_pairs.jsonl

    Next steps:
        1. Review exported examples for quality
        2. Split into train/val/test sets
        3. Choose fine-tuning platform:
           - OpenAI: https://platform.openai.com/docs/guides/fine-tuning
           - Hugging Face: https://huggingface.co/docs/transformers/training
           - Axolotl: https://github.com/OpenAccess-AI-Collective/axolotl
           - LLaMA Factory: https://github.com/hiyouga/LLaMA-Factory
        4. Configure hyperparameters:
           - Learning rate: 1e-5 to 5e-5
           - Batch size: 4-16 (depends on GPU)
           - Epochs: 1-3 (avoid overfitting)
           - LoRA rank: 8-64
        5. Train and monitor metrics
        6. Evaluate on test set
        7. Deploy if metrics improve
    """
    parser = argparse.ArgumentParser(
        description="Prepare fine-tuning data from user feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--format",
        choices=["sft", "dpo"],
        default="sft",
        help="Fine-tuning format (default: sft)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of examples to collect (default: 100)"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Fine-Tuning Data Preparation")
    logger.info("=" * 70)
    logger.info(f"Format: {args.format.upper()}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Limit: {args.limit}")
    logger.info("=" * 70)

    if args.format == "sft":
        # Collect positive examples only
        logger.info("Collecting positive feedback examples (score >= 1)...")
        examples = collect_feedback_examples(min_score=1, limit=args.limit)

        if not examples:
            logger.error("No positive feedback examples found!")
            logger.error("Please collect user feedback first by using the Streamlit app.")
            return

        # Format for SFT
        logger.info("Formatting examples for Supervised Fine-Tuning (SFT)...")
        formatted = format_sft_examples(examples)

        # Export
        export_training_data(args.output, formatted, format_type="sft")

    elif args.format == "dpo":
        # Collect both positive and negative examples
        logger.info("Collecting positive feedback examples (score >= 1)...")
        positive = collect_feedback_examples(min_score=1, limit=args.limit)

        logger.info("Collecting negative feedback examples (score <= -1)...")
        negative = collect_feedback_examples(min_score=-1, limit=args.limit)

        if not positive or not negative:
            logger.error("Need both positive and negative feedback for DPO!")
            logger.error(f"Found: {len(positive)} positive, {len(negative)} negative")
            logger.error("Please collect more feedback to create preference pairs.")
            return

        # Format for DPO
        logger.info("Creating preference pairs for Direct Preference Optimization (DPO)...")
        pairs = format_dpo_pairs(positive, negative)

        if not pairs:
            logger.error("Could not create any preference pairs!")
            logger.error("This happens when there are no matching queries with both good and bad answers.")
            return

        # Export
        export_training_data(args.output, pairs, format_type="dpo")

    logger.info("=" * 70)
    logger.info("Data preparation complete!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review exported file for quality")
    logger.info("2. Split into train/val/test sets (80/10/10)")
    logger.info("3. Choose fine-tuning platform (OpenAI, Hugging Face, Axolotl, LLaMA Factory)")
    logger.info("4. Configure hyperparameters (learning rate, LoRA rank, epochs)")
    logger.info("5. Train and monitor metrics (loss, perplexity)")
    logger.info("6. Evaluate on test set")
    logger.info("7. Deploy if metrics improve over baseline")
    logger.info("")
    logger.info("Recommended platforms:")
    logger.info("- OpenAI Fine-tuning API: https://platform.openai.com/docs/guides/fine-tuning")
    logger.info("- Hugging Face Transformers: https://huggingface.co/docs/transformers/training")
    logger.info("- Axolotl: https://github.com/OpenAccess-AI-Collective/axolotl")
    logger.info("- LLaMA Factory: https://github.com/hiyouga/LLaMA-Factory")
    logger.info("")


if __name__ == "__main__":
    main()
