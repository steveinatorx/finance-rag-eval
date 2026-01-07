"""Model optimization: quantization, distillation, and compression."""

from pathlib import Path

from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def quantize_model(
    model_path: Path,
    output_path: Path,
    quantization_type: str = "int8",
) -> Path:
    """
    Quantize a model to reduce size and improve inference speed.

    Args:
        model_path: Path to model directory
        output_path: Output path for quantized model
        quantization_type: 'int8' or 'float16'

    Returns:
        Path to quantized model
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        raise ImportError("sentence-transformers and torch required for quantization")

    logger.info(f"Quantizing model from {model_path} to {output_path}")

    # Load model
    model = SentenceTransformer(str(model_path))

    # Quantize
    if quantization_type == "int8":
        # Dynamic quantization (works for inference)
        model._modules["0"].auto_model = torch.quantization.quantize_dynamic(
            model._modules["0"].auto_model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
    elif quantization_type == "float16":
        # Half precision
        model = model.half()
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    logger.info(f"Quantized model saved to {output_path}")
    return output_path


def evaluate_model_size(model_path: Path) -> dict:
    """
    Evaluate model size and provide optimization recommendations.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary with size metrics and recommendations
    """
    import os

    total_size = 0
    file_count = 0

    if not model_path.exists():
        return {"error": "Model path does not exist"}

    for root, dirs, files in os.walk(model_path):
        for file in files:
            file_path = Path(root) / file
            size = file_path.stat().st_size
            total_size += size
            file_count += 1

    size_mb = total_size / (1024 * 1024)

    recommendations = []
    if size_mb > 500:
        recommendations.append("Consider quantization to reduce model size")
    if size_mb > 1000:
        recommendations.append("Consider model distillation to create smaller model")
    if file_count > 10:
        recommendations.append("Consider model pruning to reduce file count")

    return {
        "total_size_mb": round(size_mb, 2),
        "file_count": file_count,
        "recommendations": recommendations,
    }


def optimize_embedding_model(
    model_path: Path,
    output_dir: Path = Path("outputs/optimized_models"),
    quantization: bool = True,
    quantization_type: str = "int8",
) -> Path:
    """
    Optimize an embedding model with quantization.

    Args:
        model_path: Path to model directory
        output_dir: Output directory
        quantization: Whether to apply quantization
        quantization_type: Type of quantization

    Returns:
        Path to optimized model
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if quantization:
        model_name = model_path.name
        output_path = output_dir / f"{model_name}_quantized_{quantization_type}"
        return quantize_model(model_path, output_path, quantization_type)
    else:
        logger.info("No optimization applied")
        return model_path
