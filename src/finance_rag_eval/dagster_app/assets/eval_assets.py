"""Dagster assets for evaluation."""

from typing import Dict

from dagster import AssetExecutionContext, ResourceParam, asset

from finance_rag_eval.dagster_app.resources import PathsResource
from finance_rag_eval.eval.runner import evaluate_config
from finance_rag_eval.rag.index import FAISSIndex


@asset(key_prefix=["rag"])
def eval_results(
    context: AssetExecutionContext,
    faiss_index: FAISSIndex,
    paths: ResourceParam[PathsResource],
) -> Dict:
    """
    Evaluate RAG system on gold set.

    Args:
        faiss_index: FAISS index asset
        paths: Paths resource

    Returns:
        Evaluation results dictionary
    """
    context.log.info("Running evaluation on gold set")

    config = {
        "chunk_size": 512,
        "retriever": "cosine",
        "top_k": 5,
        "rerank": False,
    }

    docs_dir = paths.get_sample_docs_dir()
    gold_set_path = paths.get_gold_set_path()

    results = evaluate_config(config, docs_dir, gold_set_path)

    context.log.info(
        f"Evaluation complete: recall={results.get('avg_context_recall', 0):.3f}, faithfulness={results.get('avg_faithfulness', 0):.3f}"
    )

    return results
