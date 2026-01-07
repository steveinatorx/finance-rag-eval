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

### Using Live LLMs (Optional Enhancement)

To use OpenAI for better quality (requires API key):

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Use LLM for generation
pipenv run python -m finance_rag_eval.cli query --question "..." --use-llm

# Or use OpenAI embeddings (set in .env)
# EMBEDDING_MODEL=openai
```

**Setup**:
1. Add your OpenAI API key to `.envrc` (see Configuration section below)
2. Allow direnv: `direnv allow`

**Usage**:
```bash
# Use LLM for generation
PYTHONPATH=src pipenv run python -m finance_rag_eval.cli query "What is the revenue?" --use-llm

# Use OpenAI embeddings (set EMBEDDING_MODEL=openai in .envrc)
PYTHONPATH=src EMBEDDING_MODEL=openai pipenv run python -m finance_rag_eval.cli build-index
```

**When to use LLMs:**
- **OpenAI Embeddings**: Better semantic understanding, especially for domain-specific terms
- **LLM Generation**: More natural, coherent answers vs. extractive method
- **Tradeoff**: Higher cost and latency, but better quality

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

1. **rag_offline_job**: Runs the complete offline RAG pipeline
   - Ingests documents from `sample_docs/`
   - Chunks documents
   - Generates embeddings
   - Builds FAISS index
   - Evaluates on gold set

2. **rag_sweep_job**: Runs hyperparameter sweep
   - Evaluates across parameter matrix
   - Generates performance plots

### Assets Graph

The pipeline consists of the following assets:

- `docs_raw`: Raw documents loaded from sample docs
- `docs_clean`: Cleaned documents
- `chunks`: Document chunks
- `embeddings`: Embedding vectors
- `faiss_index`: FAISS index for retrieval
- `eval_results`: Evaluation results
- `sweep_results`: Sweep results CSV
- `plots`: Generated visualization plots

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

1. **Ingestion** (`rag/ingestion.py`): Loads HTML/text documents with fallback text loader
2. **Chunking** (`rag/chunking.py`): Fixed-size and recursive chunking strategies
3. **Embeddings** (`rag/embeddings.py`): Sentence-transformers (default) or OpenAI embeddings
4. **Indexing** (`rag/index.py`): FAISS index for efficient similarity search
5. **Retrieval** (`rag/retrieval.py`): Cosine similarity or MMR retrieval
6. **Reranking** (`rag/rerank.py`): Optional cross-encoder reranking
7. **Generation** (`rag/generation.py`): LLM generation or extractive fallback

## Evaluation Methodology

Focus on **diagnostic system metrics**, not absolute correctness:

### Metrics

1. **Context Recall Proxy**: Measures whether retrieved chunks contain answer spans from gold answers
   - Heuristic: Percentage of key phrases from gold answer appearing in retrieved chunks
   - **Why**: Ensures relevant information is retrieved

2. **Faithfulness Proxy**: Measures whether generated answers are supported by retrieved context
   - Heuristic: Percentage of answer sentences with sufficient word overlap with context
   - **Why**: Prevents hallucination, ensures answers are grounded

3. **Latency**: Measures retrieval and generation latency
   - Metrics: P50 and P95 latency percentiles
   - **Why**: Production systems need to be fast

4. **Optional Cost Proxies**: Token counts if LLM enabled
   - **Why**: Cost management for production

### Gold Set

The evaluation uses `src/finance_rag_eval/data/qa_gold.json` containing **13 question-answer pairs**:
- **8 single-document queries**: Standard Q&A within one document
- **3 multi-document queries**: Questions requiring information from multiple documents (e.g., "Compare revenue across Q1 2024 and fiscal year 2023")
- **2 temporal queries**: Questions about changes over time (e.g., "How did revenue grow from 2022 to 2023?")

**Multi-document coverage metric**: Tracks whether all required documents are retrieved for complex queries.

For a portfolio project, this demonstrates evaluation of realistic financial queries that require cross-document reasoning.

### Sweep Parameters

The hyperparameter sweep evaluates:

- `chunk_size`: [256, 512, 1024]
- `retriever`: ['cosine', 'mmr']
- `top_k`: [3, 5, 10]
- `rerank`: [False, True]

Total: 48 configurations

## Results Template

After running a sweep, results are saved to `outputs/sweep_results.csv` with columns:

- `chunk_size`: Chunk size used
- `retriever`: Retrieval strategy
- `top_k`: Number of retrieved chunks
- `rerank`: Whether reranking was enabled
- `avg_context_recall`: Average context recall score
- `avg_faithfulness`: Average faithfulness score
- `p50_latency`: 50th percentile latency (seconds)
- `p95_latency`: 95th percentile latency (seconds)
- `num_questions`: Number of questions evaluated

Plots are generated in `outputs/figures/`:
- `faithfulness_vs_latency.png`: Scatter plot of faithfulness vs latency
- `recall_vs_chunk_size.png`: Context recall vs chunk size
- `pareto_frontier.png`: Pareto-optimal configurations

## CLI Commands

```bash
# Ingest documents
python -m finance_rag_eval.cli ingest [--docs-dir PATH]

# Build index
python -m finance_rag_eval.cli build-index [--docs-dir PATH] [--chunk-size SIZE]

# Query
python -m finance_rag_eval.cli query "Your question" [--top-k K] [--use-llm]

# Evaluate
python -m finance_rag_eval.cli eval [--chunk-size SIZE] [--retriever STRATEGY] [--top-k K] [--rerank]

# Sweep
python -m finance_rag_eval.cli sweep

# Dagster instructions
python -m finance_rag_eval.cli dagster
```

## Configuration

### Environment Variables

This project uses `.envrc` (for [direnv](https://direnv.net/)) for environment configuration. 

**Setup**:
1. Install direnv: `brew install direnv` (or see [direnv installation](https://direnv.net/docs/installation.html))
2. Add hook to shell: `echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc && source ~/.zshrc`
3. Allow direnv in this directory: `direnv allow`
4. Edit `.envrc` and add your OpenAI API key

**Important**: Use an **organization-level key** (starts with `sk-`) not a project-scoped key (`sk-proj-`). Get it from: https://platform.openai.com/api-keys (at organization level, not inside a project).

```bash
# Edit .envrc
export OPENAI_API_KEY=sk-your-actual-key-here

# Optional: Use OpenAI embeddings instead of local models
export EMBEDDING_MODEL=openai

# Optional: LLM model for answer generation
export LLM_MODEL=gpt-3.5-turbo
export LLM_TEMPERATURE=0.0
```

**Note**: `.envrc` is gitignored. For local overrides, create `.envrc.local` (also gitignored).

**Without direnv**: You can also set environment variables manually:
```bash
export OPENAI_API_KEY=sk-your-key-here
export EMBEDDING_MODEL=openai
```

## Development

### Running Tests

```bash
make test

# Or:
pipenv run pytest tests/ -v
```

### Linting

```bash
make lint

# Format code:
make format
```

### Project Structure

```
finance-rag-eval/
├── src/
│   └── finance_rag_eval/
│       ├── cli.py              # Typer CLI
│       ├── config.py           # Configuration
│       ├── constants.py         # Constants
│       ├── logging.py           # Logging setup
│       ├── data/                # Data ingestion
│       ├── rag/                 # RAG pipeline
│       ├── eval/                # Evaluation
│       ├── viz/                 # Visualization
│       ├── finetuning/          # Experimental fine-tuning (optional)
│       └── dagster_app/         # Dagster orchestration
├── tests/                       # Tests
├── docs/                        # Documentation
└── outputs/                     # Output files (gitignored)
```

## Makefile Targets

- `make setup`: Install dependencies
- `make lint`: Run linter
- `make format`: Format code
- `make test`: Run tests
- `make demo`: Run offline demo
- `make eval`: Run evaluation
- `make sweep`: Run sweep and generate plots
- `make dagster`: Print Dagster UI instructions
- `make clean`: Clean generated files

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

