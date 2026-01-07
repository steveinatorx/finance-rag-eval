"""Dagster assets for fine-tuning pipeline."""

from pathlib import Path
from typing import Dict

from dagster import AssetExecutionContext, asset

from finance_rag_eval.finetuning.comparison import compare_models
from finance_rag_eval.finetuning.embedding_finetune import finetune_on_documents


@asset(key_prefix=["rag"])
def finetuned_embedding_model(
    context: AssetExecutionContext,
    docs_clean,
    paths,
) -> Path:
    """
    Fine-tune embedding model on financial documents.

    Args:
        docs_clean: Cleaned documents
        paths: Paths resource

    Returns:
        Path to fine-tuned model
    """
    context.log.info("Fine-tuning embedding model on financial documents")

    gold_set_path = paths.get_gold_set_path()
    output_dir = paths.get_outputs_dir() / "finetuned_models"

    finetuned_path = finetune_on_documents(
        documents=docs_clean,
        base_model_name="all-MiniLM-L6-v2",
        gold_set_path=gold_set_path if gold_set_path.exists() else None,
        output_dir=output_dir,
        epochs=3,
    )

    context.log.info(f"Fine-tuned model saved to {finetuned_path}")
    return finetuned_path


@asset(key_prefix=["rag"])
def model_comparison(
    context: AssetExecutionContext,
    finetuned_embedding_model: Path,
    docs_clean,
    paths,
) -> Dict:
    """
    Compare base model vs fine-tuned model.

    Args:
        finetuned_embedding_model: Path to fine-tuned model
        docs_clean: Cleaned documents
        paths: Paths resource

    Returns:
        Dictionary with comparison results
    """
    context.log.info("Comparing base vs fine-tuned models")

    gold_set_path = paths.get_gold_set_path()
    output_path = paths.get_outputs_dir() / "model_comparison.json"

    comparison = compare_models(
        base_model_name="all-MiniLM-L6-v2",
        finetuned_model_path=finetuned_embedding_model,
        documents=docs_clean,
        gold_set_path=gold_set_path,
        output_path=output_path,
    )

    if "error" not in comparison:
        recall_improvement = comparison["improvements"]["recall_improvement_pct"]
        faithfulness_improvement = comparison["improvements"][
            "faithfulness_improvement_pct"
        ]
        context.log.info(
            f"Fine-tuning improvements: "
            f"Recall: {recall_improvement:+.1f}%, "
            f"Faithfulness: {faithfulness_improvement:+.1f}%"
        )

    return comparison
