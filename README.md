# Finance RAG Evaluation System

A Retrieval-Augmented Generation (RAG) system that answers **grounded, factual questions** about SEC filings (10-K, 10-Q).

**Problem:** Equity analysts and finance professionals often spend **over 3 hours** manually scanning **100–300 page filings** to locate specific facts (e.g., revenue, risk factors, segment performance).^[1]

**Target outcome:** Enable queries like *"What was total revenue in 2023?"* to return a **verifiable, citation-backed answer** in **sub-second to low-latency time**, with retrieved context that supports the response.

**Design philosophy:** This project prioritizes **system-level levers** (retrieval, chunking, evaluation, orchestration) over model-level optimization, reflecting how RAG systems are most effectively improved in practice.

**Explicit non-goals:**
- Financial analysis or forecasting
- Subjective interpretation or opinionated summaries
- Model fine-tuning as a primary focus (fine-tuning code exists for exploratory purposes only)

## Features

- **Offline-First Design**: Runs end-to-end without API keys using local models
  - **Embeddings**: Uses sentence-transformers (local model) by default, optional OpenAI embeddings
  - **Generation**: Uses extractive answer generation (heuristic-based) by default, optional LLM generation
  - **No API Required**: Complete pipeline works with included sample documents
- **Live LLM Support**: Optional enhancement when API keys are provided
  - Set `OPENAI_API_KEY` to enable OpenAI embeddings and/or LLM generation
  - Use `--use-llm` flag in CLI or set `use_llm=True` in code
- **Comprehensive Evaluation**: Context recall (0.877 on 10 real SEC filings with 42 questions), faithfulness, and latency metrics
- **Intrinsic Chunking Metrics**: Evaluate chunking quality without gold sets (coherence, boundaries, structure)
- **Hyperparameter Sweep**: Systematic evaluation across parameter matrix
- **Dagster Orchestration**: Asset-based pipeline with visual UI
- **Multiple Retrieval Strategies**: Cosine similarity, MMR, and Hybrid (BM25 + Dense)
- **Advanced Chunking**: Structure-aware chunking with HTML table extraction for financial documents
- **Optional Reranking**: Cross-encoder reranking for improved relevance
- **Visualization**: Pareto frontier analysis and performance plots

## Quickstart

### Prerequisites

- Python 3.11.x (managed via asdf)
- pipenv

### Setup

```bash
# Install dependencies
make setup

# Or manually:
pipenv install --dev
```

### Offline Demo (No API Keys Required)

Run the complete pipeline without any API keys using local models:

```bash
# Ingest sample documents
make demo

# Or step by step:
pipenv run python -m finance_rag_eval.cli ingest
pipenv run python -m finance_rag_eval.cli build-index
pipenv run python -m finance_rag_eval.cli query --question "What was the total revenue in Q1 2024?"
```

**How it works offline:**
- **Embeddings**: Uses `sentence-transformers` (local model: `all-MiniLM-L6-v2`) - downloads once, then runs locally
- **Answer Generation**: Uses extractive method (finds relevant sentences from retrieved chunks) - no LLM needed
- **All processing**: Happens locally on your machine

### Using Live LLMs (Optional)

Requires `OPENAI_API_KEY` set in `.envrc` (see Configuration below):

```bash
# Use LLM for generation
pipenv run python -m finance_rag_eval.cli query "What is the revenue?" --use-llm

# Use OpenAI embeddings (set EMBEDDING_MODEL=openai in .envrc)
EMBEDDING_MODEL=openai pipenv run python -m finance_rag_eval.cli build-index
```

**Tradeoffs**: Better quality but higher cost/latency. Offline mode works well for most use cases.

### Using Real SEC Filings

The system includes **real SEC filings** in the repository for reproducibility (10 companies' 10-K filings from 2023: AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, V, JNJ, WMT, XOM). These are public records and are included to ensure anyone cloning the repo gets the exact same documents used for evaluation.

The system also supports downloading additional **real SEC filings** from EDGAR:

```bash
# SEC requires an email address for API access (their terms of service)
# This is for identification purposes only - any valid email works
export SEC_EDGAR_EMAIL=your.email@example.com

# Download real 10-K filings
pipenv run python scripts/download_real_filings.py

# Evaluate on real filings
pipenv run python -m finance_rag_eval.cli evaluate \
  --docs-dir src/finance_rag_eval/data/real_sec_filings \
  --chunk-strategy structure_aware \
  --retriever hybrid \
  --top-k 15 \
  --rerank
```

**Why real filings?** Real SEC filings (100-300+ pages, 500K-2M+ characters) provide much more realistic evaluation than synthetic documents. They demonstrate production readiness and handle actual financial terminology, structure, and complexity.

**Included filings**: The repo includes 10 companies' 10-K filings in `src/finance_rag_eval/data/real_sec_filings/` for immediate use. A corresponding gold set (`qa_gold_real_sec.json`) with 42 questions is available for evaluation.

**Note**: SEC rate limits to 10 requests/second, so downloads may take time. Processing large filings (chunking + embedding) also takes longer than synthetic docs.

### Evaluation

Run evaluation on the gold set:

```bash
make eval

# Or:
pipenv run python -m finance_rag_eval.cli eval
```

### Hyperparameter Sweep

Run a full hyperparameter sweep and generate plots:

```bash
make sweep

# Or:
pipenv run python -m finance_rag_eval.cli sweep
```

Results will be saved to `outputs/sweep_results.csv` and plots to `outputs/figures/`.

## Dagster UI

The pipeline is orchestrated using Dagster assets. To launch the UI:

```bash
make dagster

# Or:
pipenv run dagster dev -m finance_rag_eval.dagster_app.definitions
```

Then open http://localhost:3000 in your browser.

### Available Jobs

- **rag_offline_job**: Complete offline RAG pipeline (ingest → chunk → embed → index → evaluate)
- **rag_sweep_job**: Hyperparameter sweep with plot generation

See `docs/diagrams/dagster_assets.mmd` for the asset dependency graph.

## Architecture

### RAG Pipeline

```
Documents → Ingestion → Chunking → Embeddings → Index → Retrieval → Reranking → Generation
                                                          ↓
                                    [Local: sentence-transformers] OR [Optional: OpenAI API]
                                                                      ↓
                                    [Local: extractive] OR [Optional: LLM (GPT-3.5/GPT-4)]
```

**Default (Offline) Path:**
- Embeddings: `sentence-transformers` (local model, ~80MB download)
- Generation: Extractive answer (heuristic-based, no API calls)

**Enhanced (With API Keys) Path:**
- Embeddings: OpenAI `text-embedding-ada-002` (if `OPENAI_API_KEY` set)
- Generation: GPT-3.5/GPT-4 (if `OPENAI_API_KEY` set and `--use-llm` flag)

See `docs/diagrams/rag_architecture.mmd` for detailed architecture diagram.

### Components

- **Ingestion**: HTML/text document loading
- **Chunking**: Multiple strategies (fixed, recursive, structure-aware, semantic, hybrid)
  - Structure-aware chunking with HTML table extraction for financial documents
  - Preserves table headers and context for better retrieval
- **Embeddings**: Sentence-transformers (default) or OpenAI
- **Indexing**: FAISS for efficient similarity search
- **Retrieval**: Cosine similarity, MMR, or Hybrid (BM25 + Dense)
  - Hybrid retrieval combines keyword (BM25) and semantic (dense) search
  - Improved recall for financial documents with tables and numbers
- **Reranking**: Optional cross-encoder reranking
- **Generation**: Extractive (default) or LLM-based

## Evaluation Methodology

Focus on **diagnostic system metrics**, not absolute correctness:

### Metrics

**Extrinsic Metrics** (require gold set):
- **Context Recall**: Percentage of key phrases from gold answers appearing in retrieved chunks (ensures relevant retrieval)
- **Faithfulness**: Percentage of answer sentences supported by retrieved context (prevents hallucination)
- **Latency**: P50/P95 percentiles for retrieval and generation (production speed requirements)
- **Cost**: Token counts when LLM enabled (cost management)

**Intrinsic Metrics** (no gold set required):
- **Chunk Quality**: Size distribution, semantic coherence, boundary quality
- **Structure Preservation**: Header and table preservation with context
- **Document Coverage**: Ensures all content is chunked without gaps

See `src/finance_rag_eval/eval/chunking_metrics.py` for intrinsic evaluation capabilities.

### Gold Sets

**Synthetic Documents**: 13 question-answer pairs (8 single-document, 3 multi-document, 2 temporal queries)

**Real SEC Filings**: 42 question-answer pairs covering 10 companies across multiple sectors (Tech, Finance, Healthcare, Retail, Energy), including:
- Single-document queries (revenue, gross margin, segments, fiscal year end dates)
- Multi-document queries (comparisons across companies)
- Temporal queries (year-over-year changes)

Includes multi-document coverage metric for complex queries.

### Sweep Parameters

Evaluates 216 configurations: `chunk_size` [256, 512, 1024], `chunk_strategy` [fixed, recursive, structure_aware, semantic], `retriever` [cosine, mmr, hybrid], `top_k` [5, 10, 15], `rerank` [False, True]

## Results

**Production-Ready Performance** (on 10 real SEC filings, 42 questions):
- **Context Recall**: 0.877 (87.7%) - strong performance across diverse companies and sectors
- **Faithfulness**: 0.924 (92.4%)
- **Multi-doc Coverage**: 1.000 (100%)
- **Latency**: P50: 0.011s, P95: 0.014s

**Evaluation Dataset**:
- **10 documents**: AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, V, JNJ, WMT, XOM
- **42 questions**: Single-document (30), multi-document (7), temporal (5)
- **Multiple sectors**: Tech, Finance, Healthcare, Retail, Energy

**Optimal Configuration**:
- Chunk Strategy: `structure_aware` (with HTML table extraction)
- Retriever: `hybrid` (BM25 + Dense)
- Top-k: 5 (balanced) or 10 (maximum recall)
- Chunk Size: 512

Sweep results saved to `outputs/sweep_results.csv` with metrics (recall, faithfulness, latency). Plots in `outputs/figures/`: faithfulness vs latency, recall vs chunk size, pareto frontier.

## CLI Commands

```bash
python -m finance_rag_eval.cli ingest [--docs-dir PATH]
python -m finance_rag_eval.cli build-index [--chunk-size SIZE] [--chunk-strategy STRATEGY]
python -m finance_rag_eval.cli query "Your question" [--top-k K] [--use-llm]
python -m finance_rag_eval.cli evaluate [--chunk-strategy STRATEGY] [--retriever STRATEGY] [--top-k K] [--rerank]
python -m finance_rag_eval.cli sweep
python -m finance_rag_eval.cli compare-strategies
```

## Configuration

Uses `.envrc` (direnv) for environment variables. See `.envrc.example` for template.

**Required for OpenAI features**:
```bash
export OPENAI_API_KEY=sk-your-key-here  # Use organization-level key (sk-), not project-scoped (sk-proj-)
export EMBEDDING_MODEL=openai  # Optional: use OpenAI embeddings
export LLM_MODEL=gpt-3.5-turbo  # Optional: LLM for generation
```

**Required for downloading real SEC filings**:
```bash
export SEC_EDGAR_EMAIL=your.email@example.com  # SEC requires this per their terms of service (identification only, not authentication)
```

**Setup**: `direnv allow` after editing `.envrc`. Or set environment variables manually.

## Development

```bash
make setup    # Install dependencies
make test     # Run tests
make lint     # Run linter
make format   # Format code
make demo     # Run offline demo (ingest → index → query)
make clean    # Clean generated files
```

For evaluation and sweeps, use the CLI directly:
```bash
python -m finance_rag_eval.cli eval
python -m finance_rag_eval.cli sweep
```

See `docs/` for detailed documentation including chunking strategies.

## Blog Series

This project is documented in a 6-part blog series:

1. **Part 1**: Introduction & Architecture
2. **Part 2**: RAG Pipeline Deep Dive
3. **Part 3**: Evaluation Framework
4. **Part 4**: Evaluation Methodology (Synthetic vs Real Data, Intrinsic vs Extrinsic Metrics)
5. **Part 5**: Advanced Chunking Strategies
6. **Part 6**: Production Considerations

See `blog_notes/` for draft content and planning documents.

## Model Fine-Tuning (Experimental / Optional)

The `finetuning/` directory contains exploratory code for fine-tuning embedding models and LLMs on financial documents. This functionality is **not required** to run the RAG system or evaluation framework.

**Why it exists:**
- Exploratory experiments for domain adaptation
- Comparison utilities to measure fine-tuning impact
- Reference implementation for future work

**Why it's not the focus:**
- This project prioritizes **system-level levers** (retrieval, chunking, evaluation, orchestration) over model-level optimization
- Fine-tuning requires significant compute resources and labeled data
- Production RAG systems are most effectively improved through retrieval and evaluation improvements rather than model fine-tuning
- Fine-tuning is featured as a first-class concept in a separate NLP project

**Core learning objectives** of this project do not depend on fine-tuning:
- RAG system design and architecture
- Retrieval quality optimization
- Evaluation under weak supervision
- Latency and cost tradeoffs
- Dagster-based orchestration

The RAG pipeline works end-to-end using pre-trained models (sentence-transformers or OpenAI). Fine-tuning is an optional enhancement for those exploring domain adaptation.

## License

This is a portfolio project for demonstration purposes.

## Contributing

This is a personal portfolio project. For questions or suggestions, please open an issue.

---

[1] Based on industry estimates. See MetricDuck analysis of equity research workflows.
