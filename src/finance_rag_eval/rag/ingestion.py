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


def extract_tables_from_html(html_content: str) -> str:
    """
    Extract tables from HTML and format them as readable text with headers.

    Args:
        html_content: HTML content string

    Returns:
        Text with tables formatted as readable text
    """
    try:
        soup = BeautifulSoup(html_content, "lxml")

        # Find all tables
        tables = soup.find_all("table")

        for table in tables:
            # Extract table headers
            headers = []
            header_rows = table.find_all(
                "tr", limit=3
            )  # Check first few rows for headers

            for row in header_rows:
                header_cells = row.find_all(["th", "td"])
                if header_cells:
                    row_headers = []
                    for cell in header_cells:
                        cell_text = cell.get_text(strip=True)
                        # Check if this looks like a header (contains year, date, or common header words)
                        if any(
                            keyword in cell_text.lower()
                            for keyword in [
                                "2023",
                                "2024",
                                "2025",
                                "year",
                                "fiscal",
                                "ended",
                                "september",
                                "december",
                            ]
                        ):
                            row_headers.append(cell_text)
                    if row_headers:
                        headers.extend(row_headers)
                        break

            # Extract table data rows
            rows = table.find_all("tr")
            table_text_parts = []

            # Add headers if found
            if headers:
                table_text_parts.append(
                    "Table: " + " | ".join(headers[:5])
                )  # Limit header length

            # Extract data rows
            for row in rows:
                cells = row.find_all(["td", "th"])
                if cells:
                    cell_texts = []
                    for cell in cells:
                        cell_text = cell.get_text(strip=True)
                        # Clean up HTML entities and normalize
                        cell_text = re.sub(r"&#\d+;", " ", cell_text)
                        cell_text = re.sub(r"\s+", " ", cell_text)
                        if cell_text:
                            cell_texts.append(cell_text)

                    if cell_texts:
                        # Format as: Row label | Value1 | Value2 | Value3
                        row_text = " | ".join(cell_texts[:8])  # Limit to 8 columns
                        table_text_parts.append(row_text)

            # Replace table with formatted text
            if table_text_parts:
                table_text = "\n".join(table_text_parts)
                # Insert formatted table before the original table
                table.insert_before(f"\n\n{table_text}\n\n")

        return str(soup)
    except Exception as e:
        logger.warning(f"Error extracting tables: {e}")
        return html_content


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

        # Check if content contains HTML (even if file extension is .txt)
        has_html = (
            "<table" in content.lower()
            or "<html" in content.lower()
            or "<div" in content.lower()
        )

        # Try HTML parsing if HTML detected
        if file_path.suffix.lower() in [".html", ".htm"] or has_html:
            # Extract tables first to preserve structure
            content_with_tables = extract_tables_from_html(content)
            text = clean_html(content_with_tables)
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
