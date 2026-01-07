# Finance RAG Evaluation System

A production-ready Retrieval-Augmented Generation (RAG) system that answers **grounded, factual questions** about SEC filings (10-K, 10-Q).

**Problem:** Equity analysts and finance professionals often spend **30+ minutes** manually scanning **100–300 page filings** to locate specific facts (e.g., revenue, risk factors, segment performance).

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
- **Comprehensive Evaluation**: Context recall, faithfulness, and latency metrics
- **Hyperparameter Sweep**: Systematic evaluation across parameter matrix
- **Dagster Orchestration**: Asset-based pipeline with visual UI
- **Multiple Retrieval Strategies**: Cosine similarity and MMR (Maximal Marginal Relevance)
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
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli query "What is the revenue?" --use-llm

# Use OpenAI embeddings (set EMBEDDING_MODEL=openai in .envrc)
PYTHONPATH=src EMBEDDING_MODEL=openai pipenv run python -m finance_rag_eval.cli build-index
```

**Tradeoffs**: Better quality but higher cost/latency. Offline mode works well for most use cases.

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
- **Embeddings**: Sentence-transformers (default) or OpenAI
- **Indexing**: FAISS for efficient similarity search
- **Retrieval**: Cosine similarity or MMR
- **Reranking**: Optional cross-encoder reranking
- **Generation**: Extractive (default) or LLM-based

## Evaluation Methodology

Focus on **diagnostic system metrics**, not absolute correctness:

### Metrics

- **Context Recall**: Percentage of key phrases from gold answers appearing in retrieved chunks (ensures relevant retrieval)
- **Faithfulness**: Percentage of answer sentences supported by retrieved context (prevents hallucination)
- **Latency**: P50/P95 percentiles for retrieval and generation (production speed requirements)
- **Cost**: Token counts when LLM enabled (cost management)

### Gold Set

13 question-answer pairs: 8 single-document, 3 multi-document, 2 temporal queries. Includes multi-document coverage metric for complex queries.

### Sweep Parameters

Evaluates 48 configurations: `chunk_size` [256, 512, 1024], `retriever` [cosine, mmr], `top_k` [3, 5, 10], `rerank` [False, True]

## Results

Sweep results saved to `outputs/sweep_results.csv` with metrics (recall, faithfulness, latency). Plots in `outputs/figures/`: faithfulness vs latency, recall vs chunk size, pareto frontier.

## CLI Commands

```bash
python -m finance_rag_eval.cli ingest [--docs-dir PATH]
python -m finance_rag_eval.cli build-index [--chunk-size SIZE] [--chunk-strategy STRATEGY]
python -m finance_rag_eval.cli query "Your question" [--top-k K] [--use-llm]
python -m finance_rag_eval.cli eval [--chunk-strategy STRATEGY] [--retriever STRATEGY] [--top-k K]
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

