"""Dagster assets for hyperparameter sweep."""

from pathlib import Path
from typing import Dict

from dagster import AssetExecutionContext, ResourceParam, asset

from finance_rag_eval.dagster_app.resources import PathsResource
from finance_rag_eval.eval.sweep import run_sweep
from finance_rag_eval.viz.plots import generate_all_plots


@asset(key_prefix=["rag"])
def sweep_results(
    context: AssetExecutionContext,
    paths: ResourceParam[PathsResource],
) -> Path:
    """
    Run hyperparameter sweep and save results.

    Args:
        paths: Paths resource

    Returns:
        Path to sweep results CSV
    """
    context.log.info("Running hyperparameter sweep")

    docs_dir = paths.get_sample_docs_dir()
    gold_set_path = paths.get_gold_set_path()
    output_dir = paths.get_outputs_dir()

    csv_path = run_sweep(docs_dir, gold_set_path, output_dir)

    context.log.info(f"Sweep complete: results saved to {csv_path}")
    return csv_path


@asset(key_prefix=["rag"])
def plots(
    context: AssetExecutionContext,
    sweep_results: Path,
    paths: ResourceParam[PathsResource],
) -> Dict[str, Path]:
    """
    Generate plots from sweep results.

    Args:
        sweep_results: Path to sweep results CSV
        paths: Paths resource

    Returns:
        Dictionary mapping plot names to paths
    """
    context.log.info("Generating plots from sweep results")

    figures_dir = paths.get_outputs_dir() / "figures"
    generate_all_plots(sweep_results, figures_dir)

    plot_paths = {
        "faithfulness_vs_latency": figures_dir / "faithfulness_vs_latency.png",
        "recall_vs_chunk_size": figures_dir / "recall_vs_chunk_size.png",
        "pareto_frontier": figures_dir / "pareto_frontier.png",
    }

    context.log.info(f"Generated {len(plot_paths)} plots")
    return plot_paths
