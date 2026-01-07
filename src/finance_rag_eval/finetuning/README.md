# Fine-Tuning (Experimental / Optional)

This directory contains exploratory experiments for fine-tuning embedding models and LLMs on financial documents.

## Status

**Not required for core functionality.** The RAG system works end-to-end using pre-trained models (sentence-transformers or OpenAI).

## Purpose

- Exploratory experiments for domain adaptation
- Comparison utilities to measure fine-tuning impact on retrieval quality
- Reference implementation for future work

## Why Not Primary Focus

This project emphasizes **system-level improvements**:
- Chunking strategies (structure-aware, semantic, hybrid)
- Retrieval quality (MMR, reranking)
- Evaluation methodology (recall, faithfulness, latency)
- Orchestration (Dagster assets)

Fine-tuning requires significant compute resources and labeled data. Production RAG systems often achieve better ROI through retrieval and evaluation improvements rather than model fine-tuning.

## Usage

If you want to experiment with fine-tuning:

```python
from finance_rag_eval.finetuning.embedding_finetune import finetune_on_documents
from finance_rag_eval.finetuning.comparison import compare_base_vs_finetuned

# Fine-tune embedding model
finetuned_path = finetune_on_documents(
    documents=documents,
    gold_set_path=gold_set_path,
    base_model_name="all-MiniLM-L6-v2"
)

# Compare base vs fine-tuned
results = compare_base_vs_finetuned(
    base_model_name="all-MiniLM-L6-v2",
    finetuned_model_path=finetuned_path,
    documents=documents,
    gold_set_path=gold_set_path
)
```

See individual module docstrings for details.

