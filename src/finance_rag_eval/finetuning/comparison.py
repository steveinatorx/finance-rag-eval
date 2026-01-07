"""Compare fine-tuned vs base models on evaluation metrics."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from finance_rag_eval.eval.metrics import compute_metrics, measure_latency
from finance_rag_eval.eval.runner import load_gold_set
from finance_rag_eval.logging import get_logger
from finance_rag_eval.rag.chunking import chunk_documents
from finance_rag_eval.rag.embeddings import generate_embeddings
from finance_rag_eval.rag.generation import generate_answer
from finance_rag_eval.rag.index import FAISSIndex
from finance_rag_eval.rag.retrieval import retrieve

logger = get_logger(__name__)


def evaluate_model(
    model_path: str,
    documents: List[dict],
    gold_set: List[dict],
    chunk_size: int = 512,
    top_k: int = 5,
) -> Dict:
    """
    Evaluate a model (base or fine-tuned) on the gold set.

    Args:
        model_path: Path to model (or model name for base model)
        documents: List of documents
        gold_set: List of Q/A pairs
        chunk_size: Chunk size
        top_k: Number of retrieved chunks

    Returns:
        Dictionary with evaluation metrics
    """
    # Chunk documents
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=50)

    # Generate embeddings with specified model
    chunk_texts = [chunk["text"] for chunk in chunks]

    # Check if model_path is a directory (fine-tuned) or name (base)
    from pathlib import Path as PathLib

    if PathLib(model_path).exists():
        # Fine-tuned model - load from path
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(model_path))
        embeddings = model.encode(chunk_texts, show_progress_bar=False)
    else:
        # Base model - use default function
        embeddings = generate_embeddings(chunk_texts, model_name=model_path)

    embeddings = np.array(embeddings)

    # Build index
    index = FAISSIndex(dimension=embeddings.shape[1])
    index.add(embeddings, chunks)

    # Evaluate on gold set
    results = []
    latencies = []

    for qa in gold_set:
        query = qa.get("question", "")
        gold_answer = qa.get("answer", "")

        if not query:
            continue

        # Generate query embedding
        if PathLib(model_path).exists():
            query_embedding = model.encode([query], show_progress_bar=False)[0]
        else:
            query_embedding = generate_embeddings([query], model_name=model_path)[0]

        # Retrieve
        retrieval_result, retrieval_latency = measure_latency(
            retrieve,
            query_embedding.reshape(1, -1),
            index,
            k=top_k,
            strategy="cosine",
        )

        # Generate answer
        generation_result, generation_latency = measure_latency(
            generate_answer,
            query,
            retrieval_result,
            use_llm=False,
        )

        # Compute metrics
        metrics = compute_metrics(
            query=query,
            gold_answer=gold_answer,
            retrieved_chunks=retrieval_result,
            answer=generation_result,
            retrieval_latency=retrieval_latency,
            generation_latency=generation_latency,
        )

        results.append(metrics)
        latencies.append(metrics["total_latency"])

    # Aggregate
    if not results:
        return {"error": "No results"}

    return {
        "model_path": str(model_path),
        "num_questions": len(results),
        "avg_context_recall": float(np.mean([r["context_recall"] for r in results])),
        "avg_faithfulness": float(np.mean([r["faithfulness"] for r in results])),
        "p50_latency": float(np.percentile(latencies, 50)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "avg_retrieval_latency": float(
            np.mean([r["retrieval_latency"] for r in results])
        ),
    }


def compare_models(
    base_model_name: str,
    finetuned_model_path: Path,
    documents: List[dict],
    gold_set_path: Path,
    output_path: Optional[Path] = None,
) -> Dict:
    """
    Compare base model vs fine-tuned model.

    Args:
        base_model_name: Base model name
        finetuned_model_path: Path to fine-tuned model
        documents: List of documents
        gold_set_path: Path to gold set
        output_path: Optional path to save comparison results

    Returns:
        Dictionary with comparison results
    """
    logger.info(
        f"Comparing base model ({base_model_name}) vs fine-tuned ({finetuned_model_path})"
    )

    # Load gold set
    gold_set = load_gold_set(gold_set_path)
    if not gold_set:
        return {"error": "No gold set found"}

    # Evaluate base model
    logger.info("Evaluating base model...")
    base_results = evaluate_model(base_model_name, documents, gold_set)

    # Evaluate fine-tuned model
    logger.info("Evaluating fine-tuned model...")
    finetuned_results = evaluate_model(str(finetuned_model_path), documents, gold_set)

    # Compute improvements
    recall_improvement = (
        finetuned_results["avg_context_recall"] - base_results["avg_context_recall"]
    )
    faithfulness_improvement = (
        finetuned_results["avg_faithfulness"] - base_results["avg_faithfulness"]
    )
    latency_change = finetuned_results["p50_latency"] - base_results["p50_latency"]

    comparison = {
        "base_model": base_results,
        "finetuned_model": finetuned_results,
        "improvements": {
            "context_recall_delta": float(recall_improvement),
            "faithfulness_delta": float(faithfulness_improvement),
            "latency_delta": float(latency_change),
            "recall_improvement_pct": float(
                (recall_improvement / base_results["avg_context_recall"]) * 100
            )
            if base_results["avg_context_recall"] > 0
            else 0,
            "faithfulness_improvement_pct": float(
                (faithfulness_improvement / base_results["avg_faithfulness"]) * 100
            )
            if base_results["avg_faithfulness"] > 0
            else 0,
        },
    }

    # Save if output path provided
    if output_path:
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison results saved to {output_path}")

    return comparison
