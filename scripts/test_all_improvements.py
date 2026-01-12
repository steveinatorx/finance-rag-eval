"""Comprehensive test of all 4 improvements."""

import json
from pathlib import Path

from finance_rag_eval.eval.runner import evaluate_config

# Paths
DOCS_DIR = Path("src/finance_rag_eval/data/large_docs")
GOLD_SET = Path("src/finance_rag_eval/data/qa_gold_large.json")

# Test configurations
CONFIGS = {
    "1. Baseline": {
        "chunk_size": 512,
        "chunk_strategy": "structure_aware",
        "retriever": "cosine",
        "top_k": 5,
        "rerank": False,
    },
    "2. + Hybrid Retrieval": {
        "chunk_size": 512,
        "chunk_strategy": "structure_aware",
        "retriever": "hybrid",
        "top_k": 5,
        "rerank": False,
        "dense_weight": 0.5,
    },
    "3. + Higher top_k": {
        "chunk_size": 512,
        "chunk_strategy": "structure_aware",
        "retriever": "hybrid",
        "top_k": 15,
        "rerank": False,
        "dense_weight": 0.5,
    },
    "4. + Reranking": {
        "chunk_size": 512,
        "chunk_strategy": "structure_aware",
        "retriever": "hybrid",
        "top_k": 15,
        "rerank": True,
        "dense_weight": 0.5,
    },
}

print("=" * 80)
print("COMPREHENSIVE TEST: All 4 Improvements")
print("=" * 80)
print(f"\nDataset: {len(list(DOCS_DIR.glob('*.txt')))} documents")
print(f"Gold Set: {len(json.load(open(GOLD_SET)))} questions")
print("\n" + "=" * 80 + "\n")

results = {}

for config_name, config in CONFIGS.items():
    print(f"Testing: {config_name}")
    print(f"  Config: {config}")
    
    result = evaluate_config(config, DOCS_DIR, GOLD_SET)
    
    if "error" in result:
        print(f"  ERROR: {result['error']}\n")
        continue
    
    recall = result["avg_context_recall"]
    faithfulness = result["avg_faithfulness"]
    multi_doc_coverage = result.get("avg_multi_doc_coverage", "N/A")
    p50_latency = result["p50_latency"]
    p95_latency = result["p95_latency"]
    
    results[config_name] = {
        "recall": recall,
        "faithfulness": faithfulness,
        "multi_doc_coverage": multi_doc_coverage,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
    }
    
    print(f"  ✓ Context Recall: {recall:.3f}")
    print(f"  ✓ Faithfulness: {faithfulness:.3f}")
    if multi_doc_coverage != "N/A":
        print(f"  ✓ Multi-doc Coverage: {multi_doc_coverage:.3f}")
    print(f"  ✓ P50 Latency: {p50_latency:.3f}s")
    print(f"  ✓ P95 Latency: {p95_latency:.3f}s")
    print()

# Summary comparison
print("=" * 80)
print("IMPROVEMENT SUMMARY")
print("=" * 80)
print()

baseline_recall = results["1. Baseline"]["recall"]
baseline_faithfulness = results["1. Baseline"]["faithfulness"]

for config_name, metrics in results.items():
    recall = metrics["recall"]
    recall_improvement = recall - baseline_recall
    recall_pct = (recall_improvement / baseline_recall * 100) if baseline_recall > 0 else 0
    
    faithfulness = metrics["faithfulness"]
    faith_improvement = faithfulness - baseline_faithfulness
    
    print(f"{config_name}:")
    print(f"  Recall: {recall:.3f} ({recall_improvement:+.3f}, {recall_pct:+.1f}%)")
    print(f"  Faithfulness: {faithfulness:.3f} ({faith_improvement:+.3f})")
    if metrics["multi_doc_coverage"] != "N/A":
        print(f"  Multi-doc Coverage: {metrics['multi_doc_coverage']:.3f}")
    print()

# Final target check
final_recall = results["4. + Reranking"]["recall"]
print("=" * 80)
print(f"FINAL RESULT: {final_recall:.3f} Context Recall")
if final_recall >= 0.90:
    print("🎉 TARGET ACHIEVED: Recall ≥ 0.90!")
else:
    improvement_needed = 0.90 - final_recall
    print(f"📈 Need {improvement_needed:.3f} more to reach 0.90")
print("=" * 80)
