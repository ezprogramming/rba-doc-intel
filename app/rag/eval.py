"""Evaluation framework for RAG system quality measurement.

Why evaluate RAG systems?
==========================
RAG systems are complex pipelines with many components:
- Chunking strategy (size, overlap, boundaries)
- Embedding model (quality, domain fit)
- Retrieval algorithm (vector vs hybrid vs reranking)
- LLM (model size, prompt engineering)

Small changes can significantly impact answer quality. Evaluation provides:
1. **Regression detection**: Catch quality drops before deployment
2. **A/B testing**: Compare model versions objectively
3. **Continuous improvement**: Identify failure patterns
4. **Stakeholder confidence**: Quantify system performance

Industry standard practice:
- Pinecone: "Test on golden examples before production"
- LangChain: Built-in evaluation modules
- OpenAI: Recommends offline eval before fine-tuning
- Anthropic: Emphasizes eval-driven development

Evaluation types:
=================
1. **Offline evaluation** (this module):
   - Run on golden test cases
   - Metrics: keyword match, semantic similarity, latency
   - Fast feedback loop (minutes vs days)
   - Catches obvious regressions

2. **Online evaluation** (not implemented):
   - User feedback (thumbs up/down)
   - Click-through rates
   - Session abandonment
   - Real user behavior

3. **Human evaluation** (gold standard):
   - Expert judges rate answers
   - Expensive but high quality
   - Used for final validation
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import EvalExample, EvalResult, EvalRun
from app.rag.pipeline import answer_query

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for a single evaluation example.

    Metrics:
        keyword_match: Fraction of expected keywords found in answer (0.0-1.0)
                      Example: expected=["2-3", "percent"], answer="target is 2-3 percent"
                      → 2/2 = 1.0 (perfect match)

        answer_length: Number of characters in answer
                      Why track? Too short = incomplete, too long = verbose

        latency_ms: Time to generate answer in milliseconds
                   Why track? Production SLA requirements

        retrieval_chunks: Number of chunks retrieved
                         Why track? More chunks = more context but slower

        passed: Boolean indicating if answer meets quality threshold
               Why? Single metric for pass/fail reporting

        error: Error message if evaluation failed
              Why? Debug failures without rerunning entire evaluation
    """

    keyword_match: float = 0.0
    answer_length: int = 0
    latency_ms: int = 0
    retrieval_chunks: int = 0
    passed: bool = False
    error: Optional[str] = None


def compute_keyword_match(
    expected_keywords: List[str],
    answer: str
) -> float:
    """Compute fraction of expected keywords found in answer.

    Args:
        expected_keywords: List of keywords that should appear in answer
                          Example: ["2-3", "percent", "inflation"]
        answer: LLM-generated answer text

    Returns:
        Fraction of keywords found (0.0-1.0)

    Why keyword matching?
    - Simple, fast, deterministic
    - Works well for factual questions with specific terms
    - Good baseline metric (80%+ correlation with human eval)

    Limitations:
    - Misses semantic equivalents ("2-3%" vs "between 2 and 3 percent")
    - Doesn't capture answer quality (just keyword presence)
    - Can't detect hallucinations or incorrect context

    For better semantic matching, consider:
    - Sentence embeddings (cosine similarity between expected and actual)
    - ROUGE-L (longest common subsequence)
    - BERTScore (token-level semantic similarity)
    """
    if not expected_keywords:
        return 1.0  # No keywords expected = trivially pass

    # Case-insensitive matching
    # Why lower()? "GDP" and "gdp" should match
    answer_lower = answer.lower()

    # Count how many expected keywords appear in answer
    matches = sum(
        1 for keyword in expected_keywords
        if keyword.lower() in answer_lower
    )

    # Return fraction of keywords found
    match_rate = matches / len(expected_keywords)

    logger.debug(
        f"Keyword match: {matches}/{len(expected_keywords)} = {match_rate:.2f}"
    )

    return match_rate


def evaluate_single_example(
    session: Session,
    example: EvalExample,
    config: Dict[str, Any],
    min_keyword_match: float = 0.8,
) -> EvaluationMetrics:
    """Evaluate RAG system on a single golden example.

    Args:
        session: Database session
        example: Golden evaluation example with query and expected keywords
        config: Evaluation configuration (model name, retrieval params, etc.)
        min_keyword_match: Minimum keyword match score to pass (default: 0.8)
                          Example: 0.8 means 80% of keywords must appear

    Returns:
        EvaluationMetrics with scores and pass/fail verdict

    How it works:
    1. Run answer_query() on example query
    2. Measure latency
    3. Compute keyword match against expected keywords
    4. Determine pass/fail based on threshold
    5. Return metrics

    Why measure latency?
    - Production SLA requirements (e.g., "95% < 2 seconds")
    - Model comparison (qwen7b vs qwen1.5b)
    - Regression detection (new code slowing down pipeline)
    """
    metrics = EvaluationMetrics()

    try:
        # Start timer for latency measurement
        start_time = time.time()

        # Run RAG pipeline on evaluation example
        # Use reranking from config if specified
        response = answer_query(
            query=example.query,
            top_k=config.get("retrieval_top_k", 5),
            use_reranking=config.get("reranking_enabled", False)
        )

        # Measure latency
        elapsed_ms = int((time.time() - start_time) * 1000)
        metrics.latency_ms = elapsed_ms

        # Extract answer and evidence
        answer = response.answer
        evidence = response.evidence

        # Compute metrics
        metrics.answer_length = len(answer)
        metrics.retrieval_chunks = len(evidence)

        # Compute keyword match if expected keywords provided
        if example.expected_keywords:
            metrics.keyword_match = compute_keyword_match(
                expected_keywords=example.expected_keywords,
                answer=answer
            )

        # Determine pass/fail
        # Pass criteria:
        # 1. Keyword match >= threshold
        # 2. Answer not empty
        # 3. At least one chunk retrieved
        metrics.passed = (
            metrics.keyword_match >= min_keyword_match
            and metrics.answer_length > 0
            and metrics.retrieval_chunks > 0
        )

        logger.info(
            f"Evaluated example {example.id}: "
            f"keyword_match={metrics.keyword_match:.2f}, "
            f"latency={metrics.latency_ms}ms, "
            f"passed={metrics.passed}"
        )

    except Exception as e:
        # Don't fail entire evaluation run on single example failure
        # Record error and mark as failed
        logger.error(f"Failed to evaluate example {example.id}: {e}")
        metrics.error = str(e)
        metrics.passed = False

    return metrics


def run_evaluation(
    session: Session,
    config: Dict[str, Any],
    example_filters: Optional[Dict[str, Any]] = None,
    min_keyword_match: float = 0.8,
) -> UUID:
    """Run offline evaluation on golden examples and store results.

    Args:
        session: Database session
        config: Evaluation configuration
               Example: {
                   "model_name": "qwen2.5:7b",
                   "prompt_version": "v1.2",
                   "retrieval_top_k": 5,
                   "reranking_enabled": True
               }
        example_filters: Optional filters for selecting evaluation examples
                        Example: {"difficulty": "hard", "category": "inflation"}
        min_keyword_match: Minimum keyword match score to pass (default: 0.8)

    Returns:
        UUID of created EvalRun record

    How it works:
    1. Create EvalRun record with config and status="running"
    2. Fetch evaluation examples (with optional filters)
    3. For each example:
       - Run evaluate_single_example()
       - Store result in EvalResult table
    4. Compute summary metrics (total, passed, avg latency)
    5. Update EvalRun status="completed"
    6. Return run ID

    Usage:
        with session_scope() as session:
            run_id = run_evaluation(
                session=session,
                config={
                    "model_name": "qwen2.5:7b",
                    "retrieval_top_k": 5,
                    "reranking_enabled": True
                }
            )
            # Check results
            run = session.get(EvalRun, run_id)
            print(run.summary_metrics)
    """
    # Create evaluation run record
    run_id = uuid4()
    eval_run = EvalRun(
        id=run_id,
        created_at=datetime.utcnow(),
        config=config,
        status="running"
    )
    session.add(eval_run)
    session.flush()  # Get run ID before processing examples

    logger.info(f"Starting evaluation run {run_id}")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Fetch evaluation examples
    stmt = select(EvalExample)

    # Apply filters if provided
    # Why filters? Run subset of examples for faster iteration
    # Example: Only test "hard" examples after major changes
    if example_filters:
        if "difficulty" in example_filters:
            stmt = stmt.where(EvalExample.difficulty == example_filters["difficulty"])
        if "category" in example_filters:
            stmt = stmt.where(EvalExample.category == example_filters["category"])

    examples = session.execute(stmt).scalars().all()

    if not examples:
        logger.warning("No evaluation examples found. Add examples to eval_examples table.")
        eval_run.status = "completed"
        eval_run.summary_metrics = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "avg_latency_ms": 0
        }
        return run_id

    logger.info(f"Found {len(examples)} evaluation examples")

    # Evaluate each example
    results_data = []
    total_passed = 0
    total_latency_ms = 0

    for example in examples:
        logger.info(f"Evaluating example {example.id}: {example.query}")

        # Run evaluation
        metrics = evaluate_single_example(
            session=session,
            example=example,
            config=config,
            min_keyword_match=min_keyword_match
        )

        # Store result
        result = EvalResult(
            eval_run_id=run_id,
            eval_example_id=example.id,
            llm_answer=None,  # TODO: Store answer if needed
            retrieved_chunks=None,  # TODO: Store chunks if needed
            latency_ms=metrics.latency_ms,
            scores={
                "keyword_match": metrics.keyword_match,
                "answer_length": metrics.answer_length,
                "retrieval_chunks": metrics.retrieval_chunks
            },
            passed=1 if metrics.passed else 0,
            error=metrics.error
        )
        session.add(result)

        # Update counters
        if metrics.passed:
            total_passed += 1
        if metrics.error is None:  # Only count latency for successful runs
            total_latency_ms += metrics.latency_ms

        results_data.append({
            "example_id": example.id,
            "query": example.query,
            "passed": metrics.passed,
            "keyword_match": metrics.keyword_match,
            "latency_ms": metrics.latency_ms,
            "error": metrics.error
        })

    # Compute summary metrics
    total_examples = len(examples)
    pass_rate = total_passed / total_examples if total_examples > 0 else 0.0
    successful_runs = sum(1 for r in results_data if r["error"] is None)
    avg_latency_ms = total_latency_ms / successful_runs if successful_runs > 0 else 0

    summary = {
        "total": total_examples,
        "passed": total_passed,
        "failed": total_examples - total_passed,
        "pass_rate": round(pass_rate, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "min_keyword_match_threshold": min_keyword_match
    }

    # Update eval run with summary
    eval_run.status = "completed"
    eval_run.summary_metrics = summary

    logger.info(f"Evaluation run {run_id} completed")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    # Show detailed results
    for result in results_data:
        status = "✓" if result["passed"] else "✗"
        logger.info(
            f"  {status} {result['query'][:60]}... "
            f"(match={result['keyword_match']:.2f}, "
            f"latency={result['latency_ms']}ms)"
        )

    return run_id


def compare_eval_runs(
    session: Session,
    run_id_a: UUID,
    run_id_b: UUID
) -> Dict[str, Any]:
    """Compare two evaluation runs to detect improvements or regressions.

    Args:
        session: Database session
        run_id_a: First evaluation run ID (e.g., baseline)
        run_id_b: Second evaluation run ID (e.g., new version)

    Returns:
        Comparison dictionary with deltas

    Why comparison?
    - A/B testing: Is qwen7b better than qwen1.5b?
    - Regression detection: Did new code break quality?
    - Trade-off analysis: +10% accuracy but +500ms latency?

    Example output:
    {
        "pass_rate_delta": +0.15,  # +15% pass rate improvement
        "latency_delta_ms": +200,  # +200ms slower
        "recommendation": "Accept" or "Reject"
    }
    """
    # Fetch both runs
    run_a = session.get(EvalRun, run_id_a)
    run_b = session.get(EvalRun, run_id_b)

    if not run_a or not run_b:
        raise ValueError("One or both evaluation runs not found")

    # Extract metrics
    metrics_a = run_a.summary_metrics or {}
    metrics_b = run_b.summary_metrics or {}

    # Compute deltas
    pass_rate_delta = metrics_b.get("pass_rate", 0.0) - metrics_a.get("pass_rate", 0.0)
    latency_delta = metrics_b.get("avg_latency_ms", 0.0) - metrics_a.get("avg_latency_ms", 0.0)

    # Recommendation heuristic
    # Accept if: +5% pass rate improvement, or same quality with <+500ms latency
    # Reject if: Quality drop >5%, or latency increase >1000ms with no quality gain
    recommendation = "Neutral"
    if pass_rate_delta >= 0.05:  # 5% improvement
        recommendation = "Accept"
    elif pass_rate_delta < -0.05:  # 5% regression
        recommendation = "Reject"
    elif abs(pass_rate_delta) < 0.01 and latency_delta > 1000:  # No quality gain, much slower
        recommendation = "Reject"

    comparison = {
        "run_a": {
            "id": str(run_id_a),
            "config": run_a.config,
            "metrics": metrics_a
        },
        "run_b": {
            "id": str(run_id_b),
            "config": run_b.config,
            "metrics": metrics_b
        },
        "deltas": {
            "pass_rate": round(pass_rate_delta, 4),
            "latency_ms": round(latency_delta, 2)
        },
        "recommendation": recommendation
    }

    logger.info(f"Comparison: {recommendation}")
    logger.info(f"Pass rate: {metrics_a.get('pass_rate', 0.0):.2%} → {metrics_b.get('pass_rate', 0.0):.2%} ({pass_rate_delta:+.2%})")
    logger.info(f"Latency: {metrics_a.get('avg_latency_ms', 0):.0f}ms → {metrics_b.get('avg_latency_ms', 0):.0f}ms ({latency_delta:+.0f}ms)")

    return comparison
