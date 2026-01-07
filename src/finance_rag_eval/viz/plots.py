"""Plotting utilities using matplotlib."""

import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from finance_rag_eval.constants import FIGURES_DIR
from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def load_sweep_results(csv_path: Path) -> list:
    """Load sweep results from CSV."""
    results = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string values to appropriate types
            row["chunk_size"] = int(row["chunk_size"])
            row["top_k"] = int(row["top_k"])
            row["rerank"] = row["rerank"].lower() == "true"
            row["avg_context_recall"] = float(row["avg_context_recall"])
            row["avg_faithfulness"] = float(row["avg_faithfulness"])
            row["p50_latency"] = float(row["p50_latency"])
            row["p95_latency"] = float(row["p95_latency"])
            results.append(row)
    return results


def plot_faithfulness_vs_latency(
    results: list,
    output_path: Optional[Path] = None,
) -> None:
    """Plot faithfulness vs latency scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    faithfulness = [r["avg_faithfulness"] for r in results]
    latency = [r["p50_latency"] for r in results]

    ax.scatter(latency, faithfulness, alpha=0.6, s=100)
    ax.set_xlabel("Latency (p50, seconds)")
    ax.set_ylabel("Faithfulness")
    ax.set_title("Faithfulness vs Latency")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_recall_vs_chunk_size(
    results: list,
    output_path: Optional[Path] = None,
) -> None:
    """Plot context recall vs chunk size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by chunk size
    chunk_sizes = sorted(set(r["chunk_size"] for r in results))
    recalls_by_size = {size: [] for size in chunk_sizes}

    for r in results:
        recalls_by_size[r["chunk_size"]].append(r["avg_context_recall"])

    # Compute means and stds
    means = [np.mean(recalls_by_size[size]) for size in chunk_sizes]
    stds = [np.std(recalls_by_size[size]) for size in chunk_sizes]

    ax.errorbar(chunk_sizes, means, yerr=stds, marker="o", capsize=5, capthick=2)
    ax.set_xlabel("Chunk Size")
    ax.set_ylabel("Context Recall")
    ax.set_title("Context Recall vs Chunk Size")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_pareto_frontier(
    results: list,
    output_path: Optional[Path] = None,
) -> None:
    """Plot Pareto frontier for faithfulness vs latency."""
    fig, ax = plt.subplots(figsize=(10, 6))

    faithfulness = np.array([r["avg_faithfulness"] for r in results])
    latency = np.array([r["p50_latency"] for r in results])

    # Find Pareto-optimal points (maximize faithfulness, minimize latency)
    pareto_mask = np.ones(len(results), dtype=bool)

    for i in range(len(results)):
        for j in range(len(results)):
            if i != j:
                # j dominates i if j has higher faithfulness AND lower latency
                if faithfulness[j] >= faithfulness[i] and latency[j] <= latency[i]:
                    if faithfulness[j] > faithfulness[i] or latency[j] < latency[i]:
                        pareto_mask[i] = False
                        break

    # Plot all points
    ax.scatter(
        latency, faithfulness, alpha=0.3, s=50, label="All configs", color="gray"
    )

    # Plot Pareto frontier
    pareto_latency = latency[pareto_mask]
    pareto_faithfulness = faithfulness[pareto_mask]

    if len(pareto_latency) > 0:
        # Sort by latency for line plot
        sort_idx = np.argsort(pareto_latency)
        ax.plot(
            pareto_latency[sort_idx],
            pareto_faithfulness[sort_idx],
            "r-",
            linewidth=2,
            label="Pareto frontier",
        )
        ax.scatter(
            pareto_latency,
            pareto_faithfulness,
            s=150,
            color="red",
            marker="*",
            label="Pareto-optimal",
            zorder=5,
        )

    ax.set_xlabel("Latency (p50, seconds)")
    ax.set_ylabel("Faithfulness")
    ax.set_title("Pareto Frontier: Faithfulness vs Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def generate_all_plots(csv_path: Path, output_dir: Path = FIGURES_DIR) -> None:
    """Generate all plots from sweep results."""
    logger.info(f"Generating plots from {csv_path}")

    results = load_sweep_results(csv_path)

    if not results:
        logger.warning("No results to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_faithfulness_vs_latency(
        results,
        output_dir / "faithfulness_vs_latency.png",
    )

    plot_recall_vs_chunk_size(
        results,
        output_dir / "recall_vs_chunk_size.png",
    )

    plot_pareto_frontier(
        results,
        output_dir / "pareto_frontier.png",
    )

    logger.info(f"Generated plots in {output_dir}")
