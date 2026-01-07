"""Dagster job definitions."""

from dagster import AssetSelection, define_asset_job

from finance_rag_eval.dagster_app.assets.eval_assets import eval_results
from finance_rag_eval.dagster_app.assets.sweep_assets import plots, sweep_results

# Offline RAG pipeline job: ingest -> index -> eval
rag_offline_job = define_asset_job(
    name="rag_offline_job",
    selection=AssetSelection.assets(
        eval_results,
    ).upstream(),  # Include all upstream dependencies
    description="Run offline RAG pipeline: ingest documents, build index, and evaluate",
)

# Sweep job: run hyperparameter sweep and generate plots
rag_sweep_job = define_asset_job(
    name="rag_sweep_job",
    selection=AssetSelection.assets(
        sweep_results,
        plots,
    ),
    description="Run hyperparameter sweep and generate evaluation plots",
)
