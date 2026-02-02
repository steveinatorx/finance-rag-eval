"""Generate correlation plot for intrinsic vs extrinsic metrics."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
from finance_rag_eval.constants import FIGURES_DIR

# Create figures directory
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Data from the blog post
strategies = {
    "Structure Aware": {
        "boundary_quality": 0.721,
        "structure_preservation": 0.278,
        "recall": 0.877,
    },
    "Fixed": {
        "boundary_quality": 0.674,
        "structure_preservation": 0.342,
        "recall": 0.866,
    },
    "Recursive": {
        "boundary_quality": 0.680,
        "structure_preservation": 0.320,
        "recall": 0.822,
    },
    "Semantic": {
        "boundary_quality": 0.650,
        "structure_preservation": 0.250,
        "recall": 0.759,
    },
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Boundary Quality vs Recall
boundary_qualities = [s["boundary_quality"] for s in strategies.values()]
recalls = [s["recall"] for s in strategies.values()]
strategy_names = list(strategies.keys())

colors = ["red", "blue", "green", "orange"]

for i, (name, color) in enumerate(zip(strategy_names, colors)):
    ax1.scatter(
        boundary_qualities[i],
        recalls[i],
        s=200,
        color=color,
        alpha=0.7,
        label=name,
        edgecolors="black",
        linewidth=1.5,
    )
    # Add text labels
    ax1.annotate(
        name,
        (boundary_qualities[i], recalls[i]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

# Add trend line
z = np.polyfit(boundary_qualities, recalls, 1)
p = np.poly1d(z)
ax1.plot(
    boundary_qualities,
    p(boundary_qualities),
    "r--",
    alpha=0.5,
    linewidth=2,
    label="Trend line",
)

ax1.set_xlabel("Boundary Quality (Intrinsic)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Context Recall (Extrinsic)", fontsize=12, fontweight="bold")
ax1.set_title("Boundary Quality vs Context Recall", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower right", fontsize=9)
ax1.set_xlim(0.60, 0.75)
ax1.set_ylim(0.70, 0.90)

# Plot 2: Structure Preservation vs Recall
structure_preservations = [s["structure_preservation"] for s in strategies.values()]

for i, (name, color) in enumerate(zip(strategy_names, colors)):
    ax2.scatter(
        structure_preservations[i],
        recalls[i],
        s=200,
        color=color,
        alpha=0.7,
        label=name,
        edgecolors="black",
        linewidth=1.5,
    )
    # Add text labels
    ax2.annotate(
        name,
        (structure_preservations[i], recalls[i]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

# Add trend line
z2 = np.polyfit(structure_preservations, recalls, 1)
p2 = np.poly1d(z2)
ax2.plot(
    structure_preservations,
    p2(structure_preservations),
    "r--",
    alpha=0.5,
    linewidth=2,
    label="Trend line",
)

ax2.set_xlabel("Structure Preservation (Intrinsic)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Context Recall (Extrinsic)", fontsize=12, fontweight="bold")
ax2.set_title("Structure Preservation vs Context Recall", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="lower right", fontsize=9)
ax2.set_xlim(0.20, 0.36)
ax2.set_ylim(0.70, 0.90)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "intrinsic_extrinsic_correlation.png", dpi=150, bbox_inches="tight")
print(f"Saved: {FIGURES_DIR / 'intrinsic_extrinsic_correlation.png'}")
plt.close()
