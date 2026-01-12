"""Comprehensive evaluation of all RAG improvements."""

import json
from pathlib import Path

from finance_rag_eval.eval.runner import evaluate_config

# Paths - Use real SEC filings if available, otherwise fall back to large_docs
REAL_FILINGS_DIR = Path("src/finance_rag_eval/data/real_sec_filings")
LARGE_DOCS_DIR = Path("src/finance_rag_eval/data/large_docs")
DOCS_DIR = REAL_FILINGS_DIR if REAL_FILINGS_DIR.exists() and list(REAL_FILINGS_DIR.glob("*.txt")) else LARGE_DOCS_DIR
GOLD_SET = Path("src/finance_rag_eval/data/qa_gold_large.json")

# All configurations tested
CONFIGS = {
    "Baseline": {
        "description": "Cosine similarity, top_k=5",
        "config": {
            "chunk_size": 512,
            "chunk_strategy": "structure_aware",
            "retriever": "cosine",
            "top_k": 5,
            "rerank": False,
        },
    },
    "Improvement 1: Hybrid Retrieval": {
        "description": "BM25 + Dense embeddings (50/50)",
        "config": {
            "chunk_size": 512,
            "chunk_strategy": "structure_aware",
            "retriever": "hybrid",
            "top_k": 5,
            "rerank": False,
            "dense_weight": 0.5,
        },
    },
    "Improvement 2: Higher top_k": {
        "description": "Retrieve top 15 chunks (was 5)",
        "config": {
            "chunk_size": 512,
            "chunk_strategy": "structure_aware",
            "retriever": "hybrid",
            "top_k": 15,
            "rerank": False,
            "dense_weight": 0.5,
        },
    },
    "Improvement 3: Reranking": {
        "description": "Cross-encoder reranking",
        "config": {
            "chunk_size": 512,
            "chunk_strategy": "structure_aware",
            "retriever": "hybrid",
            "top_k": 15,
            "rerank": True,
            "dense_weight": 0.5,
        },
    },
    "Improvement 4: Multi-doc Handling": {
        "description": "Ensures all required docs retrieved",
        "config": {
            "chunk_size": 512,
            "chunk_strategy": "structure_aware",
            "retriever": "hybrid",
            "top_k": 15,
            "rerank": True,
            "dense_weight": 0.5,
        },
        "note": "Already implemented - measured via multi-doc coverage",
    },
}

print("=" * 90)
print("COMPREHENSIVE EVALUATION: All RAG Improvements")
print("=" * 90)

# Count documents and estimate chunks
doc_files = list(DOCS_DIR.glob("*.txt"))
num_docs = len(doc_files)
dataset_type = "Real SEC Filings" if DOCS_DIR == REAL_FILINGS_DIR else "Synthetic Documents"

# Quick chunk count estimate
if num_docs > 0:
    sample_doc = doc_files[0]
    sample_text = sample_doc.read_text(encoding="utf-8", errors="ignore")
    # Rough estimate: ~512 chars per chunk
    estimated_chunks = len(sample_text) // 512 * num_docs
else:
    estimated_chunks = 0

print(f"\n📊 Dataset: {num_docs} {dataset_type}")
print(f"📝 Gold Set: {len(json.load(open(GOLD_SET)))} questions")
print(f"📦 Estimated chunks: ~{estimated_chunks} (structure-aware, chunk_size=512)")
print()

results = {}
baseline_recall = None

for name, info in CONFIGS.items():
    config = info["config"]
    description = info["description"]
    
    print(f"Testing: {name}")
    print(f"  {description}")
    
    result = evaluate_config(config, DOCS_DIR, GOLD_SET)
    
    if "error" in result:
        print(f"  ❌ ERROR: {result['error']}\n")
        continue
    
    recall = result["avg_context_recall"]
    faithfulness = result["avg_faithfulness"]
    multi_doc_coverage = result.get("avg_multi_doc_coverage", "N/A")
    p50_latency = result["p50_latency"]
    p95_latency = result["p95_latency"]
    
    if baseline_recall is None:
        baseline_recall = recall
    
    improvement = recall - baseline_recall
    improvement_pct = (improvement / baseline_recall * 100) if baseline_recall > 0 else 0
    
    results[name] = {
        "recall": recall,
        "faithfulness": faithfulness,
        "multi_doc_coverage": multi_doc_coverage,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
    }
    
    print(f"  ✅ Context Recall: {recall:.3f} ({improvement:+.3f}, {improvement_pct:+.1f}%)")
    print(f"  ✅ Faithfulness: {faithfulness:.3f}")
    if multi_doc_coverage != "N/A":
        print(f"  ✅ Multi-doc Coverage: {multi_doc_coverage:.3f}")
    print(f"  ⏱️  P50 Latency: {p50_latency:.3f}s | P95: {p95_latency:.3f}s")
    if "note" in info:
        print(f"  📌 Note: {info['note']}")
    print()

# Final summary
print("=" * 90)
print("FINAL SUMMARY")
print("=" * 90)
print()

final_result = results["Improvement 4: Multi-doc Handling"]
total_improvement = final_result["recall"] - baseline_recall
total_improvement_pct = (total_improvement / baseline_recall * 100) if baseline_recall > 0 else 0

print(f"📈 Baseline Recall:     {baseline_recall:.3f}")
print(f"🚀 Final Recall:        {final_result['recall']:.3f}")
print(f"✨ Total Improvement:   {total_improvement:+.3f} ({total_improvement_pct:+.1f}%)")
print()

if final_result["recall"] >= 0.90:
    print("🎉 TARGET ACHIEVED: Recall ≥ 0.90!")
else:
    gap = 0.90 - final_result["recall"]
    print(f"📊 Current: {final_result['recall']:.3f} | Target: 0.900 | Gap: {gap:.3f}")
    print(f"💡 To reach 0.90: Need {gap:.3f} more ({gap/baseline_recall*100:.1f}% improvement)")

print()
print("=" * 90)
print("KEY IMPROVEMENTS BREAKDOWN")
print("=" * 90)
print()

improvements = [
    ("1. Hybrid Retrieval", "BM25 + Dense embeddings"),
    ("2. Higher top_k", "5 → 15 chunks retrieved"),
    ("3. Reranking", "Cross-encoder reranking"),
    ("4. Multi-doc Handling", "Ensures all required docs"),
]

for i, (name, desc) in enumerate(improvements, 1):
    if i == 1:
        key = "Improvement 1: Hybrid Retrieval"
    elif i == 2:
        key = "Improvement 2: Higher top_k"
    elif i == 3:
        key = "Improvement 3: Reranking"
    else:
        key = "Improvement 4: Multi-doc Handling"
    
    if key in results:
        r = results[key]
        if i == 1:
            prev_recall = baseline_recall
        elif i == 2:
            prev_recall = results["Improvement 1: Hybrid Retrieval"]["recall"]
        elif i == 3:
            prev_recall = results["Improvement 2: Higher top_k"]["recall"]
        else:
            prev_recall = results["Improvement 3: Reranking"]["recall"]
        delta = r["recall"] - prev_recall
        
        print(f"{name}:")
        print(f"  {desc}")
        print(f"  Impact: {prev_recall:.3f} → {r['recall']:.3f} ({delta:+.3f}, {delta/prev_recall*100:+.1f}%)")
        print()

print("=" * 90)
