"""Typer CLI for finance RAG evaluation system."""

from pathlib import Path

import typer
from rich.console import Console

from finance_rag_eval.constants import EVAL_GOLD_SET, OUTPUTS_DIR, SAMPLE_DOCS_DIR
from finance_rag_eval.eval.runner import evaluate_config
from finance_rag_eval.eval.sweep import run_sweep
from finance_rag_eval.logging import setup_logging, get_logger
from finance_rag_eval.rag.embeddings import generate_embeddings
from finance_rag_eval.rag.generation import generate_answer
from finance_rag_eval.rag.index import FAISSIndex
from finance_rag_eval.rag.ingestion import load_documents_from_dir
from finance_rag_eval.rag.retrieval import retrieve
from finance_rag_eval.viz.plots import generate_all_plots

app = typer.Typer(help="Finance RAG Evaluation CLI")
console = Console()
logger = get_logger(__name__)


@app.command()
def ingest(
    docs_dir: Path = typer.Option(
        SAMPLE_DOCS_DIR,
        "--docs-dir",
        help="Directory containing documents",
    ),
) -> None:
    """Load documents from directory."""
    setup_logging()
    console.print(f"[bold]Loading documents from {docs_dir}[/bold]")

    documents = load_documents_from_dir(docs_dir)
    console.print(f"[green]Loaded {len(documents)} documents[/green]")

    for doc in documents:
        console.print(f"  - {doc['id']}: {len(doc['text'])} chars")


@app.command()
def build_index(
    docs_dir: Path = typer.Option(
        SAMPLE_DOCS_DIR,
        "--docs-dir",
        help="Directory containing documents",
    ),
    chunk_size: int = typer.Option(512, "--chunk-size", help="Chunk size"),
    chunk_strategy: str = typer.Option(
        "fixed",
        "--chunk-strategy",
        help="Chunking strategy: fixed, recursive, structure_aware, semantic, hybrid, langchain_recursive, langchain_semantic",
    ),
    output_dir: Path = typer.Option(
        OUTPUTS_DIR, "--output-dir", help="Output directory"
    ),
) -> None:
    """Build FAISS index from documents."""
    setup_logging()
    console.print("[bold]Building index...[/bold]")

    # Load documents
    documents = load_documents_from_dir(docs_dir)
    console.print(f"Loaded {len(documents)} documents")

    # Chunk using selected strategy
    if chunk_strategy.startswith("langchain"):
        # Use LangChain chunking
        from finance_rag_eval.rag.langchain_chunking import chunk_documents_langchain

        langchain_strategy = chunk_strategy.replace("langchain_", "")
        chunks = chunk_documents_langchain(
            documents, chunk_size=chunk_size, strategy=langchain_strategy
        )
    elif chunk_strategy in ["structure_aware", "semantic", "hybrid"]:
        # Use advanced chunking
        from finance_rag_eval.rag.advanced_chunking import chunk_documents_advanced

        chunks = chunk_documents_advanced(
            documents, chunk_size=chunk_size, strategy=chunk_strategy
        )
    else:
        # Use standard chunking
        from finance_rag_eval.rag.chunking import chunk_documents

        chunks = chunk_documents(
            documents, chunk_size=chunk_size, strategy=chunk_strategy
        )

    console.print(f"Created {len(chunks)} chunks using {chunk_strategy} strategy")

    # Generate embeddings
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(chunk_texts)
    console.print(f"Generated embeddings: shape {embeddings.shape}")

    # Build index
    index = FAISSIndex(dimension=embeddings.shape[1])
    index.add(embeddings, chunks)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "faiss_index.index"
    chunks_path = output_dir / "chunks.pkl"
    index.save(index_path, chunks_path)

    console.print(f"[green]Index saved to {index_path}[/green]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer"),
    index_path: Path = typer.Option(
        OUTPUTS_DIR / "faiss_index.index",
        "--index-path",
        help="Path to FAISS index",
    ),
    chunks_path: Path = typer.Option(
        OUTPUTS_DIR / "chunks.pkl",
        "--chunks-path",
        help="Path to chunks pickle",
    ),
    top_k: int = typer.Option(5, "--top-k", help="Number of results"),
    use_llm: bool = typer.Option(False, "--use-llm", help="Use LLM for generation"),
) -> None:
    """Query the RAG system."""
    setup_logging()
    console.print(f"[bold]Query:[/bold] {question}")

    # Load index
    if not index_path.exists():
        console.print(f"[red]Index not found at {index_path}[/red]")
        console.print("Run 'build-index' first")
        raise typer.Exit(1)

    index = FAISSIndex.load(index_path, chunks_path)
    console.print(f"Loaded index with {len(index.chunks)} chunks")

    # Generate query embedding
    query_embedding = generate_embeddings([question])[0]

    # Retrieve
    retrieved = retrieve(query_embedding.reshape(1, -1), index, k=top_k)
    console.print(f"\n[bold]Retrieved {len(retrieved)} chunks:[/bold]")

    for i, result in enumerate(retrieved, 1):
        console.print(f"\n[cyan]Chunk {i} (score: {result['score']:.3f}):[/cyan]")
        console.print(result["chunk"]["text"][:200] + "...")

    # Generate answer
    answer = generate_answer(question, retrieved, use_llm=use_llm)
    console.print("\n[bold green]Answer:[/bold green]")
    console.print(answer)


@app.command()
def eval(
    docs_dir: Path = typer.Option(
        SAMPLE_DOCS_DIR,
        "--docs-dir",
        help="Directory containing documents",
    ),
    gold_set: Path = typer.Option(
        EVAL_GOLD_SET,
        "--gold-set",
        help="Path to gold set JSON",
    ),
    chunk_size: int = typer.Option(512, "--chunk-size", help="Chunk size"),
    chunk_strategy: str = typer.Option(
        "fixed",
        "--chunk-strategy",
        help="Chunking strategy: fixed, recursive, structure_aware, semantic, hybrid",
    ),
    retriever: str = typer.Option("cosine", "--retriever", help="Retriever strategy"),
    top_k: int = typer.Option(5, "--top-k", help="Number of results"),
    rerank_enabled: bool = typer.Option(False, "--rerank", help="Enable reranking"),
) -> None:
    """Run evaluation on gold set."""
    setup_logging()
    console.print("[bold]Running evaluation...[/bold]")

    config = {
        "chunk_size": chunk_size,
        "chunk_strategy": chunk_strategy,
        "retriever": retriever,
        "top_k": top_k,
        "rerank": rerank_enabled,
    }

    results = evaluate_config(config, docs_dir, gold_set)

    if "error" in results:
        console.print(f"[red]Error: {results['error']}[/red]")
        raise typer.Exit(1)

    console.print("\n[bold]Evaluation Results:[/bold]")
    console.print(f"  Questions evaluated: {results['num_questions']}")
    if results.get("num_multi_doc_questions", 0) > 0:
        console.print(
            f"  Multi-document questions: {results['num_multi_doc_questions']}"
        )
    console.print(f"  Avg Context Recall: {results['avg_context_recall']:.3f}")
    console.print(f"  Avg Faithfulness: {results['avg_faithfulness']:.3f}")
    if results.get("avg_multi_doc_coverage") is not None:
        console.print(f"  Multi-doc Coverage: {results['avg_multi_doc_coverage']:.3f}")
    console.print(f"  P50 Latency: {results['p50_latency']:.3f}s")
    console.print(f"  P95 Latency: {results['p95_latency']:.3f}s")


@app.command()
def compare_strategies(
    docs_dir: Path = typer.Option(
        SAMPLE_DOCS_DIR,
        "--docs-dir",
        help="Directory containing documents",
    ),
    gold_set: Path = typer.Option(
        EVAL_GOLD_SET,
        "--gold-set",
        help="Path to gold set JSON",
    ),
    chunk_size: int = typer.Option(
        512, "--chunk-size", help="Chunk size for comparison"
    ),
    output_dir: Path = typer.Option(
        OUTPUTS_DIR, "--output-dir", help="Output directory"
    ),
) -> None:
    """Compare different chunking strategies."""
    setup_logging()
    console.print("[bold]Comparing chunking strategies...[/bold]")

    from finance_rag_eval.eval.strategy_comparison import compare_strategies

    csv_path = compare_strategies(docs_dir, gold_set, output_dir, chunk_size)
    console.print(f"[green]Results saved to {csv_path}[/green]")


@app.command()
def sweep(
    docs_dir: Path = typer.Option(
        SAMPLE_DOCS_DIR,
        "--docs-dir",
        help="Directory containing documents",
    ),
    gold_set: Path = typer.Option(
        EVAL_GOLD_SET,
        "--gold-set",
        help="Path to gold set JSON",
    ),
    output_dir: Path = typer.Option(
        OUTPUTS_DIR, "--output-dir", help="Output directory"
    ),
) -> None:
    """Run hyperparameter sweep and generate plots."""
    setup_logging()
    console.print("[bold]Running hyperparameter sweep...[/bold]")

    csv_path = run_sweep(docs_dir, gold_set, output_dir)
    console.print(f"[green]Sweep complete: {csv_path}[/green]")

    console.print("[bold]Generating plots...[/bold]")
    generate_all_plots(csv_path, output_dir / "figures")
    console.print(f"[green]Plots saved to {output_dir / 'figures'}[/green]")


@app.command()
def dagster() -> None:
    """Print instructions for running Dagster UI."""
    console.print("[bold]Dagster UI Instructions:[/bold]\n")
    console.print("Run the following command to start Dagster UI:")
    console.print(
        "\n[cyan]pipenv run dagster dev -m finance_rag_eval.dagster_app.definitions[/cyan]\n"
    )
    console.print("Then open http://localhost:3000 in your browser")
    console.print("\nAvailable jobs:")
    console.print("  - rag_offline_job: Run offline RAG pipeline")
    console.print("  - rag_sweep_job: Run hyperparameter sweep")


if __name__ == "__main__":
    app()
