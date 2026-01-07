"""Evaluation runner: evaluate a configuration on gold set."""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from finance_rag_eval.constants import EVAL_GOLD_SET
from finance_rag_eval.eval.metrics import compute_metrics, measure_latency
from finance_rag_eval.logging import get_logger
from finance_rag_eval.rag.chunking import chunk_documents
from finance_rag_eval.rag.advanced_chunking import chunk_documents_advanced
from finance_rag_eval.rag.langchain_chunking import chunk_documents_langchain
from finance_rag_eval.rag.embeddings import generate_embeddings
from finance_rag_eval.rag.generation import generate_answer
from finance_rag_eval.rag.index import FAISSIndex
from finance_rag_eval.rag.ingestion import load_documents_from_dir
from finance_rag_eval.rag.retrieval import retrieve
from finance_rag_eval.rag.rerank import rerank

logger = get_logger(__name__)


def load_gold_set(gold_set_path: Path) -> List[Dict]:
    """Load QA gold set from JSON file."""
    if not gold_set_path.exists():
        logger.warning(f"Gold set not found at {gold_set_path}")
        return []

    with open(gold_set_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "questions" in data:
        return data["questions"]
    else:
        logger.error(f"Unexpected gold set format in {gold_set_path}")
        return []


def evaluate_config(
    config: Dict,
    docs_dir: Path,
    gold_set_path: Path = EVAL_GOLD_SET,
) -> Dict:
    """
    Evaluate a RAG configuration on the gold set.

    Args:
        config: Configuration dictionary with:
            - chunk_size: int
            - chunk_strategy: str ('fixed', 'recursive', 'structure_aware', 'semantic', 'hybrid', etc.)
            - retriever: str ('cosine' or 'mmr')
            - top_k: int
            - rerank: bool
        docs_dir: Directory containing documents
        gold_set_path: Path to gold set JSON

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating config: {config}")

    # Load documents
    documents = load_documents_from_dir(docs_dir)
    if not documents:
        logger.error(f"No documents found in {docs_dir}")
        return {"error": "No documents found"}

    # Chunk documents using specified strategy
    chunk_strategy = config.get("chunk_strategy", "fixed")
    chunk_size = config.get("chunk_size", 512)

    if chunk_strategy.startswith("langchain"):
        # Use LangChain chunking
        langchain_strategy = chunk_strategy.replace("langchain_", "")
        chunks = chunk_documents_langchain(
            documents, chunk_size=chunk_size, strategy=langchain_strategy
        )
    elif chunk_strategy in ["structure_aware", "semantic", "hybrid"]:
        # Use advanced chunking
        chunks = chunk_documents_advanced(
            documents, chunk_size=chunk_size, strategy=chunk_strategy
        )
    else:
        # Use standard chunking
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=50,
            strategy=chunk_strategy,
        )

    # Generate embeddings
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(chunk_texts)

    # Build index
    index = FAISSIndex(dimension=embeddings.shape[1])
    index.add(embeddings, chunks)

    # Load gold set
    gold_set = load_gold_set(gold_set_path)
    if not gold_set:
        logger.error("No gold set questions found")
        return {"error": "No gold set found"}

    # Evaluate on each question
    results = []
    latencies = []

    for qa in gold_set:
        query = qa.get("question", "")
        gold_answer = qa.get("answer", "")
        question_type = qa.get("type", "single_document")  # Default to single doc
        required_documents = qa.get("requires_documents", [])

        if not query:
            continue

        # Generate query embedding
        query_embedding = generate_embeddings([query])[0]

        # Retrieve
        retrieval_result, retrieval_latency = measure_latency(
            retrieve,
            query_embedding.reshape(1, -1),
            index,
            k=config.get("top_k", 5),
            strategy=config.get("retriever", "cosine"),
        )

        # Rerank if enabled
        if config.get("rerank", False):
            retrieval_result = rerank(query, retrieval_result)

        # Generate answer
        generation_result, generation_latency = measure_latency(
            generate_answer,
            query,
            retrieval_result,
            use_llm=False,  # Offline by default
        )

        # Compute metrics
        metrics = compute_metrics(
            query=query,
            gold_answer=gold_answer,
            retrieved_chunks=retrieval_result,
            answer=generation_result,
            retrieval_latency=retrieval_latency,
            generation_latency=generation_latency,
            question_type=question_type,
            required_documents=required_documents,
        )

        result = {
            "query": query,
            "question_type": question_type,
            "metrics": metrics,
        }
        results.append(result)
        latencies.append(metrics["total_latency"])

    # Aggregate metrics
    if not results:
        return {"error": "No results"}

    avg_recall = np.mean([r["metrics"]["context_recall"] for r in results])
    avg_faithfulness = np.mean([r["metrics"]["faithfulness"] for r in results])
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)

    # Calculate multi-document coverage if applicable
    multi_doc_results = [
        r for r in results if r.get("question_type") == "multi_document"
    ]
    avg_multi_doc_coverage = None
    if multi_doc_results:
        avg_multi_doc_coverage = np.mean(
            [r["metrics"].get("multi_doc_coverage", 0.0) for r in multi_doc_results]
        )

    return {
        "config": config,
        "chunk_strategy": chunk_strategy,  # Include in results
        "num_questions": len(results),
        "num_multi_doc_questions": len(multi_doc_results),
        "avg_context_recall": float(avg_recall),
        "avg_faithfulness": float(avg_faithfulness),
        "avg_multi_doc_coverage": float(avg_multi_doc_coverage)
        if avg_multi_doc_coverage is not None
        else None,
        "p50_latency": float(p50_latency),
        "p95_latency": float(p95_latency),
        "results": results,
    }
