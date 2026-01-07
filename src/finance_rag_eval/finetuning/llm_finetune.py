"""Fine-tune LLM for financial Q&A generation (optional, requires transformers)."""

from pathlib import Path
from typing import List, Optional

from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def prepare_llm_training_data(
    documents: List[dict],
    gold_set_path: Optional[Path] = None,
) -> List[dict]:
    """
    Prepare training data for LLM fine-tuning in instruction format.

    Args:
        documents: List of document dictionaries
        gold_set_path: Optional path to gold set for Q/A pairs

    Returns:
        List of training examples with 'instruction', 'input', 'output' keys
    """
    import json

    examples = []

    # Add Q/A pairs from gold set
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
                # Find relevant context from documents
                context = ""
                for doc in documents:
                    if any(
                        word.lower() in doc.get("text", "").lower()
                        for word in question.split()[:3]
                    ):
                        context = doc["text"][:500]  # First 500 chars
                        break

                examples.append(
                    {
                        "instruction": "Answer the following question based on the provided financial context.",
                        "input": f"Context: {context}\n\nQuestion: {question}",
                        "output": answer,
                    }
                )

    logger.info(f"Prepared {len(examples)} LLM training examples")
    return examples


def finetune_llm(
    base_model_name: str,
    training_data: List[dict],
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
) -> Path:
    """
    Fine-tune a language model for financial Q&A.

    Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.

    Args:
        base_model_name: Base model name (e.g., "microsoft/DialoGPT-small")
        training_data: List of training examples
        output_dir: Output directory for fine-tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate

    Returns:
        Path to saved fine-tuned model
    """
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "transformers, datasets, and torch required for LLM fine-tuning. "
            "Install with: pipenv install transformers datasets torch"
        )

    logger.info(f"Fine-tuning {base_model_name} on {len(training_data)} examples")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Prepare dataset
    def format_prompt(example):
        prompt = (
            f"{example['instruction']}\n{example['input']}\nAnswer: {example['output']}"
        )
        return {"text": prompt}

    dataset = Dataset.from_list(training_data)
    dataset = dataset.map(format_prompt)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Fine-tuned LLM saved to {output_dir}")
    return output_dir


def finetune_llm_on_documents(
    documents: List[dict],
    base_model_name: str = "microsoft/DialoGPT-small",
    gold_set_path: Optional[Path] = None,
    output_dir: Path = Path("outputs/finetuned_models"),
    epochs: int = 3,
) -> Path:
    """
    Fine-tune LLM on financial documents.

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

    # Prepare training data
    training_data = prepare_llm_training_data(documents, gold_set_path)

    if len(training_data) < 5:
        logger.warning(
            f"Only {len(training_data)} training examples. Fine-tuning may not be effective."
        )

    # Fine-tune
    finetuned_path = finetune_llm(
        base_model_name=base_model_name,
        training_data=training_data,
        output_dir=model_output_dir,
        epochs=epochs,
    )

    return finetuned_path
