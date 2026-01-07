"""Hyperparameter sweep: run evaluation across parameter matrix."""

import csv
from pathlib import Path

from finance_rag_eval.constants import OUTPUTS_DIR
from finance_rag_eval.eval.runner import evaluate_config
from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def run_sweep(
    docs_dir: Path,
    gold_set_path: Path,
    output_dir: Path = OUTPUTS_DIR,
) -> Path:
    """
    Run hyperparameter sweep across parameter matrix.

    Parameters:
        - chunk_size: [256, 512, 1024]
        - retriever: ['cosine', 'mmr']
        - top_k: [3, 5, 10]
        - rerank: [False, True]

    Args:
        docs_dir: Directory containing documents
        gold_set_path: Path to gold set JSON
        output_dir: Output directory for results

    Returns:
        Path to results CSV file
    """
    logger.info("Starting hyperparameter sweep")

    # Parameter matrix
    chunk_sizes = [256, 512, 1024]
    chunk_strategies = [
        "fixed",
        "recursive",
        "structure_aware",
        "semantic",
    ]  # Add strategies
    retrievers = ["cosine", "mmr"]
    top_ks = [3, 5, 10]
    rerank_options = [False, True]

    results = []

    total_configs = (
        len(chunk_sizes)
        * len(chunk_strategies)
        * len(retrievers)
        * len(top_ks)
        * len(rerank_options)
    )
    logger.info(f"Evaluating {total_configs} configurations")

    config_idx = 0
    for chunk_size in chunk_sizes:
        for chunk_strategy in chunk_strategies:
            for retriever in retrievers:
                for top_k in top_ks:
                    for rerank_enabled in rerank_options:
                        config_idx += 1
                        logger.info(
                            f"Config {config_idx}/{total_configs}: chunk_size={chunk_size}, chunk_strategy={chunk_strategy}, retriever={retriever}, top_k={top_k}, rerank={rerank_enabled}"
                        )

                        config = {
                            "chunk_size": chunk_size,
                            "chunk_strategy": chunk_strategy,
                            "retriever": retriever,
                            "top_k": top_k,
                            "rerank": rerank_enabled,
                        }

                    eval_result = evaluate_config(config, docs_dir, gold_set_path)

                    if "error" in eval_result:
                        logger.warning(f"Config failed: {eval_result['error']}")
                        continue

                    # Extract summary metrics
                    result_row = {
                        "chunk_size": chunk_size,
                        "chunk_strategy": chunk_strategy,
                        "retriever": retriever,
                        "top_k": top_k,
                        "rerank": rerank_enabled,
                        "avg_context_recall": eval_result["avg_context_recall"],
                        "avg_faithfulness": eval_result["avg_faithfulness"],
                        "p50_latency": eval_result["p50_latency"],
                        "p95_latency": eval_result["p95_latency"],
                        "num_questions": eval_result["num_questions"],
                    }
                    results.append(result_row)

    # Write results to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sweep_results.csv"

    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Saved sweep results to {csv_path}")
    else:
        logger.error("No results to save")

    return csv_path
