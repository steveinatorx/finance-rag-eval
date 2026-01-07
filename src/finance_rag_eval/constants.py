"""Constants for the finance RAG evaluation system."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "finance_rag_eval" / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Default embedding model (sentence-transformers)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

# Default retrieval parameters
DEFAULT_TOP_K = 5
DEFAULT_MMR_DIVERSITY = 0.5

# Evaluation
EVAL_GOLD_SET = DATA_DIR / "qa_gold.json"

# Seeds for reproducibility
RANDOM_SEED = 42
