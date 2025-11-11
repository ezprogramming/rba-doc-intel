"""CLI script to run offline RAG evaluation.

Usage examples:
===============

1. Run evaluation on all examples:
   uv run python scripts/run_eval.py

2. Run evaluation with reranking enabled:
   uv run python scripts/run_eval.py --rerank

3. Run evaluation on specific difficulty:
   uv run python scripts/run_eval.py --difficulty hard

4. Compare two evaluation runs:
   uv run python scripts/run_eval.py --compare RUN_ID_A RUN_ID_B

5. Seed golden examples (first time setup):
   uv run python scripts/run_eval.py --seed-examples

Why this script?
================
- Quick feedback loop for RAG quality
- Compare model versions objectively
- Catch regressions before deployment
- Track quality metrics over time

Typical workflow:
1. Seed golden examples (once)
2. Run baseline evaluation
3. Make changes (new model, better chunking, etc.)
4. Run new evaluation
5. Compare results
6. Deploy if quality improved
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from uuid import UUID

from app.db.models import EvalExample
from app.db.session import session_scope
from app.rag.eval import compare_eval_runs, run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)


def seed_golden_examples() -> None:
    """Seed database with golden evaluation examples.

    Why golden examples?
    - Consistent benchmark across all changes
    - Catch regressions (if example passes today, should pass tomorrow)
    - Cover diverse question types (factual, analytical, temporal)

    Example categories:
    - easy: Simple factual questions with clear keywords
    - medium: Questions requiring some reasoning
    - hard: Complex queries requiring deep understanding

    How to create good golden examples:
    1. Real user questions from production logs
    2. Edge cases that previously failed
    3. Diverse topics (inflation, employment, forecasts, risks)
    4. Mix of factual and analytical questions
    """
    examples = [
        # Easy: Factual questions with clear answers
        {
            "query": "What is the RBA's inflation target?",
            "expected_keywords": ["2", "3", "percent", "target"],
            "difficulty": "easy",
            "category": "inflation",
            "metadata": {"notes": "Core RBA mandate, should always pass"}
        },
        {
            "query": "Who is the RBA Governor?",
            "expected_keywords": ["governor", "reserve bank"],
            "difficulty": "easy",
            "category": "governance",
            "metadata": {"notes": "May need updating when governor changes"}
        },

        # Medium: Requires finding specific data in reports
        {
            "query": "What was the GDP forecast for 2024 in the February 2024 SMP?",
            "expected_keywords": ["gdp", "2024", "forecast", "smp"],
            "difficulty": "medium",
            "category": "forecasts",
            "metadata": {"notes": "Tests ability to find specific forecast data"}
        },
        {
            "query": "What are the main risks to the outlook mentioned in recent FSR?",
            "expected_keywords": ["risks", "outlook", "fsr"],
            "difficulty": "medium",
            "category": "risks",
            "metadata": {"notes": "Tests extraction from risk sections"}
        },

        # Hard: Analytical questions requiring reasoning
        {
            "query": "How has the RBA's assessment of inflation risks changed over the past year?",
            "expected_keywords": ["inflation", "risks", "changed", "assessment"],
            "difficulty": "hard",
            "category": "analysis",
            "metadata": {"notes": "Requires comparing multiple reports"}
        },
        {
            "query": "What is the relationship between housing prices and financial stability according to RBA?",
            "expected_keywords": ["housing", "financial stability", "prices"],
            "difficulty": "hard",
            "category": "analysis",
            "metadata": {"notes": "Tests cross-topic reasoning"}
        }
    ]

    with session_scope() as session:
        # Check if examples already exist
        # Why check? Avoid duplicate examples on multiple runs
        existing_count = session.query(EvalExample).count()
        if existing_count > 0:
            logger.info(f"Found {existing_count} existing evaluation examples")
            overwrite = input("Overwrite existing examples? (y/N): ")
            if overwrite.lower() != "y":
                logger.info("Skipping example seeding")
                return

            # Clear existing examples
            session.query(EvalExample).delete()
            logger.info("Cleared existing examples")

        # Insert golden examples
        for example_data in examples:
            example = EvalExample(
                query=example_data["query"],
                expected_keywords=example_data["expected_keywords"],
                difficulty=example_data.get("difficulty"),
                category=example_data.get("category"),
                metadata=example_data.get("metadata")
            )
            session.add(example)

        logger.info(f"Seeded {len(examples)} golden evaluation examples")

        # Show seeded examples
        for example in examples:
            logger.info(
                f"  [{example['difficulty']}] {example['query'][:60]}... "
                f"(keywords: {len(example['expected_keywords'])})"
            )


def run_eval_command(args: argparse.Namespace) -> None:
    """Run evaluation on golden examples.

    Args:
        args: Parsed command-line arguments

    Workflow:
    1. Build evaluation config from args
    2. Apply example filters if specified
    3. Run evaluation
    4. Print summary
    5. Save run ID for later comparison
    """
    # Build evaluation config
    # Why config dict? Allows comparing different configurations
    # Example: qwen7b vs qwen1.5b, reranking vs no reranking
    config = {
        "model_name": args.model or "qwen2.5:7b",
        "retrieval_top_k": args.top_k,
        "reranking_enabled": args.rerank,
        "run_date": date.today().isoformat(),
        "notes": args.notes or "Manual evaluation run"
    }

    # Apply example filters
    # Why filters? Run subset of examples for faster iteration
    filters = {}
    if args.difficulty:
        filters["difficulty"] = args.difficulty
    if args.category:
        filters["category"] = args.category

    logger.info("=" * 70)
    logger.info("Starting offline RAG evaluation")
    logger.info("=" * 70)
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    if filters:
        logger.info(f"Filters: {json.dumps(filters, indent=2)}")
    logger.info("=" * 70)

    # Run evaluation
    with session_scope() as session:
        run_id = run_evaluation(
            session=session,
            config=config,
            example_filters=filters or None,
            min_keyword_match=args.min_match
        )

    logger.info("=" * 70)
    logger.info(f"Evaluation complete! Run ID: {run_id}")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"1. Review results in database: SELECT * FROM eval_results WHERE eval_run_id = '{run_id}'")
    logger.info(f"2. Compare with another run: uv run python scripts/run_eval.py --compare {run_id} OTHER_RUN_ID")
    logger.info("")


def compare_runs_command(args: argparse.Namespace) -> None:
    """Compare two evaluation runs.

    Args:
        args: Parsed command-line arguments with run IDs

    Why compare?
    - A/B testing: Is new model better?
    - Regression testing: Did new code break quality?
    - Trade-off analysis: Worth +200ms for +15% accuracy?
    """
    try:
        run_id_a = UUID(args.run_a)
        run_id_b = UUID(args.run_b)
    except ValueError as e:
        logger.error(f"Invalid run ID format: {e}")
        logger.error("Run IDs should be UUIDs (e.g., 123e4567-e89b-12d3-a456-426614174000)")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Comparing evaluation runs")
    logger.info("=" * 70)

    with session_scope() as session:
        comparison = compare_eval_runs(
            session=session,
            run_id_a=run_id_a,
            run_id_b=run_id_b
        )

    # Print comparison results
    logger.info("")
    logger.info("Comparison results:")
    logger.info("=" * 70)

    # Run A (baseline)
    logger.info("Run A (baseline):")
    logger.info(f"  ID: {comparison['run_a']['id']}")
    logger.info(f"  Config: {json.dumps(comparison['run_a']['config'], indent=4)}")
    logger.info(f"  Metrics: {json.dumps(comparison['run_a']['metrics'], indent=4)}")
    logger.info("")

    # Run B (new)
    logger.info("Run B (new):")
    logger.info(f"  ID: {comparison['run_b']['id']}")
    logger.info(f"  Config: {json.dumps(comparison['run_b']['config'], indent=4)}")
    logger.info(f"  Metrics: {json.dumps(comparison['run_b']['metrics'], indent=4)}")
    logger.info("")

    # Deltas
    logger.info("Deltas (B - A):")
    deltas = comparison['deltas']
    logger.info(f"  Pass rate: {deltas['pass_rate']:+.2%}")
    logger.info(f"  Latency: {deltas['latency_ms']:+.0f}ms")
    logger.info("")

    # Recommendation
    recommendation = comparison['recommendation']
    logger.info(f"Recommendation: {recommendation}")
    logger.info("=" * 70)

    if recommendation == "Accept":
        logger.info("✓ Quality improved! Consider deploying changes.")
    elif recommendation == "Reject":
        logger.info("✗ Quality regressed or latency increased significantly.")
        logger.info("  Review changes before deploying.")
    else:
        logger.info("○ Neutral change. Manual review recommended.")


def main() -> None:
    """Main entry point for evaluation CLI.

    Commands:
    - run (default): Run evaluation on golden examples
    - compare: Compare two evaluation runs
    - seed: Seed golden examples

    Why CLI script?
    - Easy integration with CI/CD (exit code 0 = success, 1 = failure)
    - Quick local testing during development
    - Reproducible evaluation runs
    """
    parser = argparse.ArgumentParser(
        description="Run offline RAG evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run evaluation command (default)
    run_parser = subparsers.add_parser("run", help="Run evaluation on golden examples")
    run_parser.add_argument(
        "--model",
        type=str,
        help="LLM model name (default: qwen2.5:7b)"
    )
    run_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )
    run_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking (slower but more accurate)"
    )
    run_parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        help="Filter examples by difficulty"
    )
    run_parser.add_argument(
        "--category",
        type=str,
        help="Filter examples by category (e.g., inflation, forecasts)"
    )
    run_parser.add_argument(
        "--min-match",
        type=float,
        default=0.8,
        help="Minimum keyword match threshold to pass (default: 0.8)"
    )
    run_parser.add_argument(
        "--notes",
        type=str,
        help="Optional notes to store with evaluation run"
    )

    # Compare evaluation runs command
    compare_parser = subparsers.add_parser("compare", help="Compare two evaluation runs")
    compare_parser.add_argument(
        "run_a",
        type=str,
        help="First run ID (baseline)"
    )
    compare_parser.add_argument(
        "run_b",
        type=str,
        help="Second run ID (new version)"
    )

    # Seed golden examples command
    subparsers.add_parser("seed", help="Seed golden evaluation examples")

    # Parse arguments
    args = parser.parse_args()

    # Default to "run" command if no command specified
    if not args.command:
        args.command = "run"
        # Set defaults for run command
        args.model = None
        args.top_k = 5
        args.rerank = False
        args.difficulty = None
        args.category = None
        args.min_match = 0.8
        args.notes = None

    # Execute command
    if args.command == "seed":
        seed_golden_examples()
    elif args.command == "run":
        run_eval_command(args)
    elif args.command == "compare":
        compare_runs_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
