# Advanced Chunking Strategies Guide

## Overview

This project now supports multiple advanced chunking strategies beyond basic fixed-size chunking. Each strategy has different strengths for different use cases.

## Available Strategies

### 1. **Fixed** (Default)
- **What it does**: Simple character-based chunking with word/sentence boundary detection
- **Best for**: Baseline comparison, simple documents
- **Pros**: Fast, predictable, no dependencies
- **Cons**: May split related content, doesn't understand document structure

```bash
make demo  # Uses fixed by default
```

### 2. **Recursive**
- **What it does**: Recursively splits by separators (paragraphs → sentences → words)
- **Best for**: General-purpose text documents
- **Pros**: Respects natural boundaries, good default
- **Cons**: Still size-based, not semantic

```bash
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli build-index --chunk-strategy recursive
```

### 3. **Structure-Aware** ⭐ **Recommended for Financial Documents**
- **What it does**: Detects headers, sections, and paragraphs; chunks respect document structure
- **Best for**: Financial reports, structured documents, SEC filings
- **Pros**: 
  - Keeps sections together
  - Preserves header context
  - Better for financial data (revenue sections stay intact)
- **Cons**: Slightly slower, requires structured documents

```bash
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli build-index --chunk-strategy structure_aware
```

**Example**: A chunk will include:
```
Revenue Breakdown
- Technology Division: $85 million (68% of total revenue)
- Services Division: $30 million (24% of total revenue)
```
Instead of splitting mid-section.

### 4. **Semantic** ⭐ **Most Advanced**
- **What it does**: Groups sentences by semantic similarity using embeddings
- **Best for**: Documents where meaning matters more than structure
- **Pros**:
  - Groups related concepts together
  - Better retrieval quality
  - Understands context
- **Cons**: 
  - Slower (requires embeddings)
  - More compute-intensive
  - May create variable-sized chunks

```bash
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli build-index --chunk-strategy semantic
```

**How it works**: 
- Splits text into sentences
- Generates embeddings for each sentence
- Groups sentences with similar embeddings together
- Respects chunk size limits

### 5. **Hybrid** ⭐ **Best of Both Worlds**
- **What it does**: Combines structure-aware + semantic chunking
- **Best for**: Complex documents needing both structure and semantic understanding
- **Pros**:
  - Respects document structure
  - Also groups by meaning
  - Best retrieval quality
- **Cons**: Slowest, most compute-intensive

```bash
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli build-index --chunk-strategy hybrid
```

### 6. **LangChain Recursive**
- **What it does**: Uses LangChain's RecursiveCharacterTextSplitter
- **Best for**: If you want industry-standard chunking
- **Pros**: 
  - Well-tested, widely used
  - Good defaults
  - Part of LangChain ecosystem
- **Cons**: 
  - Requires LangChain dependency
  - Less customizable than custom implementation

**Install LangChain**:
```bash
pipenv install langchain langchain-experimental
```

```bash
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli build-index --chunk-strategy langchain_recursive
```

### 7. **LangChain Semantic**
- **What it does**: Uses LangChain's semantic chunker
- **Best for**: When you want LangChain's semantic chunking
- **Pros**: Industry-standard semantic chunking
- **Cons**: Requires LangChain + embeddings

```bash
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli build-index --chunk-strategy langchain_semantic
```

## Comparison Table

| Strategy | Speed | Quality | Structure-Aware | Semantic | Dependencies |
|----------|-------|---------|-----------------|----------|--------------|
| Fixed | ⚡⚡⚡ | ⭐⭐ | ❌ | ❌ | None |
| Recursive | ⚡⚡⚡ | ⭐⭐⭐ | ⚠️ | ❌ | None |
| Structure-Aware | ⚡⚡ | ⭐⭐⭐⭐ | ✅ | ❌ | None |
| Semantic | ⚡ | ⭐⭐⭐⭐⭐ | ❌ | ✅ | sentence-transformers |
| Hybrid | ⚡ | ⭐⭐⭐⭐⭐ | ✅ | ✅ | sentence-transformers |
| LangChain Recursive | ⚡⚡⚡ | ⭐⭐⭐ | ⚠️ | ❌ | langchain |
| LangChain Semantic | ⚡ | ⭐⭐⭐⭐ | ❌ | ✅ | langchain + embeddings |

## Recommendations

### For Financial Documents (SEC Filings, Reports)
**Use: `structure_aware` or `hybrid`**

Why:
- Financial documents have clear structure (sections, headers)
- You want to keep related financial data together
- Structure-aware preserves context (e.g., "Revenue Breakdown" header stays with revenue data)

### For General Q&A
**Use: `semantic` or `hybrid`**

Why:
- Groups related concepts together
- Better retrieval for conceptual questions
- Understands meaning, not just structure

### For Production Systems
**Use: `structure_aware` or `langchain_recursive`**

Why:
- Good balance of quality and speed
- Well-tested approaches
- Predictable behavior

### For Maximum Quality (Portfolio Demo)
**Use: `hybrid`**

Why:
- Shows advanced understanding
- Best retrieval quality
- Demonstrates multiple techniques

## Testing Different Strategies

```bash
# Test structure-aware
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli build-index --chunk-strategy structure_aware
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli query "What was the revenue?"

# Test semantic
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli build-index --chunk-strategy semantic
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli query "What was the revenue?"

# Compare results
```

## Evaluation

Run evaluation with different strategies:

```bash
# Update eval_runner.py to accept chunk_strategy parameter
# Then run:
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli eval --chunk-strategy structure_aware
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli eval --chunk-strategy semantic
```

Compare metrics:
- **Context Recall**: Does semantic/hybrid retrieve better context?
- **Faithfulness**: Do structure-aware chunks produce more grounded answers?
- **Latency**: How much slower is semantic/hybrid?

## Implementation Details

### Structure-Aware Chunking
- Detects headers (all caps, title case with colon, numbered items)
- Groups content under headers
- Preserves section boundaries
- Falls back to fixed-size if no structure detected

### Semantic Chunking
- Uses sentence-transformers embeddings
- Groups sentences by cosine similarity
- Threshold: 0.5 (configurable)
- Respects chunk size limits

### Hybrid Chunking
- First applies structure-aware chunking
- Then refines large chunks (>1.5x target size) with semantic chunking
- Best of both worlds

## Next Steps

1. **Try structure-aware** for your financial documents
2. **Compare results** with fixed chunking
3. **Evaluate metrics** to see improvement
4. **Document findings** in your portfolio/blog

## Code Examples

See:
- `src/finance_rag_eval/rag/advanced_chunking.py` - Custom advanced strategies
- `src/finance_rag_eval/rag/langchain_chunking.py` - LangChain integration
- `src/finance_rag_eval/rag/chunking.py` - Basic strategies

