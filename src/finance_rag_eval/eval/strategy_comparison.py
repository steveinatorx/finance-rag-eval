"""Compare chunking strategies: focused evaluation."""

import csv
from pathlib import Path

from finance_rag_eval.constants import OUTPUTS_DIR
from finance_rag_eval.eval.runner import evaluate_config
from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def compare_strategies(
    docs_dir: Path,
    gold_set_path: Path,
    output_dir: Path = OUTPUTS_DIR,
    chunk_size: int = 512,
) -> Path:
    """
    Compare different chunking strategies with fixed other parameters.

    Args:
        docs_dir: Directory containing documents
        gold_set_path: Path to gold set JSON
        output_dir: Output directory for results
        chunk_size: Fixed chunk size for comparison

    Returns:
        Path to results CSV file
    """
    logger.info("Starting chunking strategy comparison")

    # Test these strategies
    strategies = ["fixed", "recursive", "structure_aware", "semantic", "hybrid"]

    # Fixed parameters for fair comparison
    retriever = "cosine"
    top_k = 5
    rerank = False

    results = []

    logger.info(f"Comparing {len(strategies)} strategies with chunk_size={chunk_size}")

    for strategy in strategies:
        logger.info(f"Evaluating strategy: {strategy}")

        config = {
            "chunk_size": chunk_size,
            "chunk_strategy": strategy,
            "retriever": retriever,
            "top_k": top_k,
            "rerank": rerank,
        }

        try:
            eval_result = evaluate_config(config, docs_dir, gold_set_path)

            if "error" in eval_result:
                logger.warning(f"Strategy {strategy} failed: {eval_result['error']}")
                continue

            result_row = {
                "chunk_strategy": strategy,
                "chunk_size": chunk_size,
                "retriever": retriever,
                "top_k": top_k,
                "rerank": rerank,
                "avg_context_recall": eval_result["avg_context_recall"],
                "avg_faithfulness": eval_result["avg_faithfulness"],
                "p50_latency": eval_result["p50_latency"],
                "p95_latency": eval_result["p95_latency"],
                "num_questions": eval_result["num_questions"],
            }
            results.append(result_row)

            logger.info(
                f"Strategy {strategy}: recall={eval_result['avg_context_recall']:.3f}, "
                f"faithfulness={eval_result['avg_faithfulness']:.3f}, "
                f"p50_latency={eval_result['p50_latency']:.3f}s"
            )
        except Exception as e:
            logger.error(f"Error evaluating {strategy}: {e}")
            continue

    # Write results to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "strategy_comparison.csv"

    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Saved strategy comparison to {csv_path}")

        # Print summary
        logger.info("\n=== Strategy Comparison Summary ===")
        for result in sorted(
            results, key=lambda x: x["avg_context_recall"], reverse=True
        ):
            logger.info(
                f"{result['chunk_strategy']:20s} | "
                f"Recall: {result['avg_context_recall']:.3f} | "
                f"Faithfulness: {result['avg_faithfulness']:.3f} | "
                f"P50 Latency: {result['p50_latency']:.3f}s"
            )
    else:
        logger.error("No results to save")

    return csv_path
