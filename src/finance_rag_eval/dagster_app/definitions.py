"""Dagster Definitions: assets, jobs, and resources."""

from dagster import Definitions

from finance_rag_eval.dagster_app.assets.chunk_assets import chunks
from finance_rag_eval.dagster_app.assets.embed_assets import embeddings
from finance_rag_eval.dagster_app.assets.eval_assets import eval_results
from finance_rag_eval.dagster_app.assets.index_assets import faiss_index
from finance_rag_eval.dagster_app.assets.ingest_assets import docs_clean, docs_raw
from finance_rag_eval.dagster_app.assets.sweep_assets import plots, sweep_results
from finance_rag_eval.dagster_app.jobs import rag_offline_job, rag_sweep_job
from finance_rag_eval.dagster_app.resources import PathsResource

# Define resources
paths_resource = PathsResource(
    outputs_dir="outputs",
    sample_docs_dir="src/finance_rag_eval/data/sample_docs",
    gold_set_path="src/finance_rag_eval/data/qa_gold.json",
)

# Create Definitions
defs = Definitions(
    assets=[
        docs_raw,
        docs_clean,
        chunks,
        embeddings,
        faiss_index,
        eval_results,
        sweep_results,
        plots,
    ],
    jobs=[
        rag_offline_job,
        rag_sweep_job,
    ],
    resources={
        "paths": paths_resource,
    },
)
