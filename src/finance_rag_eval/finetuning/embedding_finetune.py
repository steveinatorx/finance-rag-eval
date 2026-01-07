"""Fine-tune sentence-transformers embedding model on financial documents."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def prepare_training_pairs(
    documents: List[dict],
    gold_set_path: Optional[Path] = None,
) -> List[Tuple[str, str]]:
    """
    Prepare training pairs for contrastive learning.

    Creates positive pairs from:
    1. Question-answer pairs from gold set
    2. Chunks from same document (as positive pairs)

    Args:
        documents: List of document dictionaries
        gold_set_path: Optional path to gold set for Q/A pairs

    Returns:
        List of (text1, text2) pairs for training
    """
    pairs = []

    # Add Q/A pairs from gold set if available
    if gold_set_path and gold_set_path.exists():
        with open(gold_set_path, "r") as f:
            gold_data = json.load(f)

        if isinstance(gold_data, list):
            qa_pairs = gold_data
        elif isinstance(gold_data, dict) and "questions" in gold_data:
            qa_pairs = gold_data["questions"]
        else:
            qa_pairs = []

        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            if question and answer:
                pairs.append((question, answer))

        logger.info(f"Added {len(qa_pairs)} Q/A pairs from gold set")

    # Add chunk pairs from same document (as positive examples)
    for doc in documents:
        text = doc.get("text", "")
        if len(text) > 100:
            # Split into sentences and create pairs
            sentences = text.split(". ")
            if len(sentences) >= 2:
                # Create pairs of adjacent sentences
                for i in range(len(sentences) - 1):
                    if len(sentences[i]) > 20 and len(sentences[i + 1]) > 20:
                        pairs.append((sentences[i].strip(), sentences[i + 1].strip()))

    logger.info(f"Prepared {len(pairs)} training pairs")
    return pairs


def finetune_embedding_model(
    base_model_name: str,
    training_pairs: List[Tuple[str, str]],
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> Path:
    """
    Fine-tune a sentence-transformers model on financial documents.

    Args:
        base_model_name: Base model name (e.g., "all-MiniLM-L6-v2")
        training_pairs: List of (text1, text2) training pairs
        output_dir: Directory to save fine-tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate

    Returns:
        Path to saved fine-tuned model
    """
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError(
            "sentence-transformers and torch required for fine-tuning. "
            "Install with: pipenv install sentence-transformers torch"
        )

    logger.info(f"Fine-tuning {base_model_name} on {len(training_pairs)} pairs")

    # Load base model
    model = SentenceTransformer(base_model_name)

    # Prepare training data
    train_examples = [
        InputExample(texts=[pair[0], pair[1]], label=1.0) for pair in training_pairs
    ]

    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # Use cosine similarity loss for contrastive learning
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        show_progress_bar=True,
    )

    logger.info(f"Fine-tuned model saved to {output_dir}")
    return output_dir


def finetune_on_documents(
    documents: List[dict],
    base_model_name: str = "all-MiniLM-L6-v2",
    gold_set_path: Optional[Path] = None,
    output_dir: Path = Path("outputs/finetuned_models"),
    epochs: int = 3,
) -> Path:
    """
    Fine-tune embedding model on financial documents.

    Args:
        documents: List of document dictionaries
        base_model_name: Base model to fine-tune
        gold_set_path: Optional path to gold set for Q/A pairs
        output_dir: Output directory for fine-tuned model
        epochs: Number of training epochs

    Returns:
        Path to fine-tuned model
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir = output_dir / f"{base_model_name.replace('/', '_')}_finetuned"

    # Prepare training pairs
    training_pairs = prepare_training_pairs(documents, gold_set_path)

    if len(training_pairs) < 10:
        logger.warning(
            f"Only {len(training_pairs)} training pairs. Fine-tuning may not be effective."
        )

    # Fine-tune
    finetuned_path = finetune_embedding_model(
        base_model_name=base_model_name,
        training_pairs=training_pairs,
        output_dir=model_output_dir,
        epochs=epochs,
    )

    return finetuned_path
