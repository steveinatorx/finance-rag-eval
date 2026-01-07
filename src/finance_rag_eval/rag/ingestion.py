"""Document ingestion: HTML/text parsing with fallback."""

import re
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup

from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def clean_html(text: str) -> str:
    """Extract clean text from HTML."""
    soup = BeautifulSoup(text, "lxml")
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    # Get text and clean up whitespace
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace."""
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(file_path: Path) -> Dict[str, str]:
    """
    Load a document from file path.

    Args:
        file_path: Path to document file

    Returns:
        Dictionary with 'id', 'text', and 'metadata' keys
    """
    try:
        content = file_path.read_text(encoding="utf-8")

        # Try HTML parsing first
        if file_path.suffix.lower() in [".html", ".htm"]:
            text = clean_html(content)
        else:
            text = clean_text(content)

        return {
            "id": file_path.stem,
            "text": text,
            "metadata": {
                "source": str(file_path),
                "file_type": file_path.suffix,
            },
        }
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        # Fallback: return raw content
        return {
            "id": file_path.stem,
            "text": content if "content" in locals() else "",
            "metadata": {
                "source": str(file_path),
                "error": str(e),
            },
        }


def load_documents_from_dir(directory: Path) -> List[Dict[str, str]]:
    """
    Load all documents from a directory.

    Args:
        directory: Directory containing document files

    Returns:
        List of document dictionaries
    """
    documents = []
    extensions = {".txt", ".html", ".htm", ".md"}

    if not directory.exists():
        logger.warning(f"Directory {directory} does not exist")
        return documents

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            doc = load_document(file_path)
            documents.append(doc)
            logger.debug(f"Loaded document: {doc['id']}")

    logger.info(f"Loaded {len(documents)} documents from {directory}")
    return documents
