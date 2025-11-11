"""Export thumbs-up/down feedback pairs for preference tuning.

Usage
-----
docker compose run --rm app \
  uv run python scripts/export_feedback_pairs.py \
  --output data/feedback_pairs.jsonl

This reads chat history + feedback from Postgres and emits JSONL with
`{"prompt": ..., "chosen": ..., "rejected": ...}` rows that the
fine-tuning script can ingest.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from sqlalchemy import select

from app.db.models import ChatMessage, Feedback
from app.db.session import session_scope


def _resolve_prompt(session, assistant_message: ChatMessage) -> str | None:
    """Return the most recent user message before the assistant reply."""

    stmt = (
        select(ChatMessage)
        .where(
            ChatMessage.session_id == assistant_message.session_id,
            ChatMessage.role == "user",
            ChatMessage.created_at <= assistant_message.created_at,
        )
        .order_by(ChatMessage.created_at.desc())
        .limit(1)
    )
    result = session.execute(stmt).scalar_one_or_none()
    return result.content if result else None


def collect_feedback_pairs(min_positive_score: int = 1) -> Dict[str, Dict[str, List[str]]]:
    """Return mapping of question -> {"positive": [...], "negative": [...]} answers."""

    buckets: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"positive": [], "negative": []})

    with session_scope() as session:
        stmt = (
            select(ChatMessage, Feedback)
            .join(Feedback, Feedback.chat_message_id == ChatMessage.id)
            .order_by(ChatMessage.session_id, ChatMessage.created_at)
        )
        rows = session.execute(stmt).all()

        for assistant_msg, feedback in rows:
            prompt = _resolve_prompt(session, assistant_msg)
            if not prompt:
                continue

            bucket = buckets[prompt]
            if feedback.score >= min_positive_score:
                bucket["positive"].append(assistant_msg.content)
            elif feedback.score < 0:
                bucket["negative"].append(assistant_msg.content)

    return buckets


def export_jsonl(buckets: Dict[str, Dict[str, List[str]]], output: Path, max_pairs_per_prompt: int) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for prompt, answers in buckets.items():
            positives = answers["positive"]
            negatives = answers["negative"]
            if not positives or not negatives:
                continue
            pair_budget = 0
            for pos in positives:
                for neg in negatives:
                    json.dump({"prompt": prompt, "chosen": pos, "rejected": neg}, f)
                    f.write("\n")
                    count += 1
                    pair_budget += 1
                    if pair_budget >= max_pairs_per_prompt:
                        break
                if pair_budget >= max_pairs_per_prompt:
                    break
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export preference pairs from feedback")
    parser.add_argument("--output", type=Path, default=Path("data/feedback_pairs.jsonl"))
    parser.add_argument("--max-pairs-per-prompt", type=int, default=5)
    parser.add_argument("--min-positive-score", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    buckets = collect_feedback_pairs(min_positive_score=args.min_positive_score)
    total = export_jsonl(buckets, args.output, args.max_pairs_per_prompt)
    print(f"Wrote {total} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
