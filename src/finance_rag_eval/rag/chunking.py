"""Document chunking: fixed-size and recursive chunking strategies."""

import re
from typing import List

from finance_rag_eval.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def _is_word_boundary(text: str, pos: int) -> bool:
    """
    Check if position is at a word boundary (space, punctuation, or start/end of text).

    Args:
        text: Input text
        pos: Position to check

    Returns:
        True if position is at a word boundary
    """
    if pos <= 0 or pos >= len(text):
        return True

    # Check if current char is whitespace or punctuation
    if text[pos] in " \n\t.,!?;:()[]{}":
        return True

    # Check if previous char is whitespace or punctuation
    if text[pos - 1] in " \n\t.,!?;:()[]{}":
        return True

    # Check if we're between alphanumeric characters (mid-word)
    if text[pos - 1].isalnum() and text[pos].isalnum():
        return False

    return True


def _find_word_start(text: str, pos: int) -> int:
    """
    Find the start of the word containing position pos, or return pos if already at word start.

    Args:
        text: Input text
        pos: Position to check

    Returns:
        Position of word start
    """
    if pos <= 0 or pos >= len(text):
        return pos

    # If at word boundary, return as-is
    if _is_word_boundary(text, pos):
        return pos

    # Backtrack to find word start
    start = pos
    while start > 0 and text[start - 1].isalnum():
        start -= 1

    return start


def _find_word_end(text: str, pos: int) -> int:
    """
    Find the end of the word containing position pos, or return pos if already at word end.

    Args:
        text: Input text
        pos: Position to check

    Returns:
        Position of word end
    """
    if pos < 0 or pos >= len(text):
        return pos

    # If at word boundary, return as-is
    if _is_word_boundary(text, pos):
        return pos

    # Forward track to find word end
    end = pos
    while end < len(text) - 1 and text[end + 1].isalnum():
        end += 1

    return end + 1


def _clean_chunk_boundaries(chunk: str) -> str:
    """
    Clean chunk boundaries to ensure it starts and ends at word boundaries.

    Removes incomplete words at the start and end of chunks.

    Args:
        chunk: Chunk text to clean

    Returns:
        Cleaned chunk text
    """
    if not chunk:
        return chunk

    chunk = chunk.strip()
    if not chunk:
        return chunk

    # Remove chunks that start with only punctuation (like ": 15.2%")
    # This indicates we cut off the preceding word
    if chunk and chunk[0] in ":;," and len(chunk) > 1:
        # Find first alphanumeric character
        first_alpha = -1
        for i, char in enumerate(chunk):
            if char.isalnum():
                first_alpha = i
                break

        if first_alpha > 0:
            # Check if there's a space before the alphanumeric
            # If so, this is likely a fragment (e.g., ": 15.2%")
            if chunk[first_alpha - 1] == " ":
                # Remove everything up to and including the punctuation
                chunk = chunk[first_alpha:].lstrip()
            else:
                # Might be valid (e.g., ":15.2%" without space)
                # Keep it but remove leading punctuation
                chunk = chunk[first_alpha:]

    # Remove chunks that start with just a number/percentage (like "15.2%")
    # This indicates we cut off the preceding label
    if chunk and len(chunk) > 2:
        # Check if chunk starts with a number or percentage
        first_word = chunk.split()[0] if chunk.split() else ""
        if first_word:
            # Check if it's just a number or percentage
            if re.match(r"^[\d.,%]+$", first_word) or first_word.endswith("%"):
                # Find first non-number word
                words = chunk.split()
                skip_count = 0
                for i, word in enumerate(words):
                    # Skip numbers, percentages, and dashes
                    if re.match(r"^[\d.,%\-]+$", word) or word in ["-", "–", "—"]:
                        skip_count = i + 1
                    else:
                        break

                if skip_count > 0 and skip_count < len(words):
                    # Remove the leading numbers/percentages
                    chunk = " ".join(words[skip_count:])
                elif skip_count == len(words):
                    # Entire chunk is numbers - skip it
                    return ""

    # Remove leading incomplete word
    # Check if chunk starts with a fragment (like "sion:", "uity:", "ovement")
    # These are typically 2-7 characters followed by punctuation
    if len(chunk) > 2:
        # Look for first word boundary (space, punctuation, or newline)
        first_boundary = -1
        for i, char in enumerate(chunk):
            if char in " \n\t.,!?;:()[]{}":
                first_boundary = i
                break

        if first_boundary > 0:
            first_word = chunk[:first_boundary]
            # If first "word" is very short (2-7 chars) and looks like a fragment
            # (doesn't start with capital, or ends with punctuation that suggests it's mid-word)
            if 2 <= len(first_word) <= 7:
                # Check if it looks like a fragment (common patterns: ends with "ion", "ity", "ment")
                fragment_patterns = ["ion", "ity", "ment", "sion", "uity", "ovement"]
                if any(first_word.endswith(pattern) for pattern in fragment_patterns):
                    # Remove it and start from the boundary
                    chunk = chunk[first_boundary:].lstrip()

    # Remove trailing incomplete word
    if len(chunk) > 2:
        # Look for last word boundary
        last_boundary = -1
        for i in range(len(chunk) - 1, -1, -1):
            if chunk[i] in " \n\t.,!?;:()[]{}":
                last_boundary = i
                break

        if last_boundary >= 0 and last_boundary < len(chunk) - 1:
            last_word = chunk[last_boundary + 1 :]
            # If last "word" is very short and looks like a fragment
            if 2 <= len(last_word) <= 7:
                fragment_patterns = ["ion", "ity", "ment", "sion", "uity", "ovement"]
                if any(last_word.endswith(pattern) for pattern in fragment_patterns):
                    # Remove it and end at the boundary
                    chunk = chunk[: last_boundary + 1].rstrip()

    return chunk.strip()


def fixed_size_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into fixed-size chunks with overlap, respecting sentence and word boundaries.

    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If not at end of text, try to break at appropriate boundary
        if end < len(text):
            # Look for boundaries within last 30% of chunk (more generous search window)
            search_start = max(start, end - int(chunk_size * 0.3))

            # Priority 1: Try to find sentence boundary (. ! ? followed by space or newline)
            found_boundary = False
            for i in range(end - 1, search_start - 1, -1):
                if i < len(text) - 1:
                    if text[i] in ".!?" and text[i + 1] in " \n":
                        end = i + 1
                        found_boundary = True
                        break

            # Priority 2: If no sentence boundary, try word boundary (space, punctuation)
            if not found_boundary:
                for i in range(end - 1, search_start - 1, -1):
                    if _is_word_boundary(text, i):
                        # Make sure we're not breaking mid-word
                        if i < len(text) - 1 and text[i] in " \n\t":
                            end = i + 1
                            found_boundary = True
                            break
                        elif i < len(text) and text[i] in ".,!?;:()[]{}":
                            end = i + 1
                            found_boundary = True
                            break

            # Priority 3: If still no boundary found, check if we're mid-word and backtrack
            if not found_boundary:
                # Check if end position is mid-word
                if end < len(text) and _is_word_boundary(text, end):
                    # Already at boundary, keep end as is
                    pass
                else:
                    # We're mid-word, find last space
                    for i in range(end - 1, search_start - 1, -1):
                        if text[i] == " ":
                            end = i + 1
                            break

        chunk = text[start:end].strip()

        # Clean chunk boundaries to remove incomplete words
        chunk = _clean_chunk_boundaries(chunk)

        if chunk:
            chunks.append(chunk)

        # Move start forward by chunk_size - overlap
        # But ensure we start at a word boundary
        new_start = end - chunk_overlap
        if new_start < 0:
            new_start = 0
        else:
            # Find the nearest word boundary before new_start
            # Look backwards for a space or punctuation
            for i in range(new_start, max(0, new_start - 50), -1):
                if i < len(text) and text[i] in " \n\t.,!?;:()[]{}":
                    new_start = i + 1
                    break
            # If we're mid-word, find word start
            if new_start < len(text) and not _is_word_boundary(text, new_start):
                new_start = _find_word_start(text, new_start)

        # Prevent infinite loop: ensure we always advance
        if new_start <= start:
            new_start = start + 1
        
        start = new_start

        # Prevent infinite loop
        if start >= len(text):
            break

    # Post-process all chunks to ensure clean boundaries
    cleaned_chunks = []
    for chunk in chunks:
        cleaned = _clean_chunk_boundaries(chunk)
        if cleaned:
            cleaned_chunks.append(cleaned)

    return cleaned_chunks


def recursive_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: List[str] = None,
) -> List[str]:
    """
    Recursively split text by separators, trying to keep chunks close to chunk_size.

    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between chunks
        separators: List of separators to try (in order of preference)

    Returns:
        List of text chunks
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= chunk_size:
        return [text]

    # Try each separator
    for sep in separators:
        if sep == "":
            # Fallback to fixed-size chunking
            return fixed_size_chunk(text, chunk_size, chunk_overlap)

        splits = text.split(sep)
        if len(splits) > 1:
            chunks = []
            current_chunk = ""

            for split in splits:
                # If adding this split would exceed chunk_size, save current chunk
                if (
                    current_chunk
                    and len(current_chunk) + len(sep) + len(split) > chunk_size
                ):
                    chunks.append(current_chunk)
                    # Start new chunk with overlap
                    if chunk_overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-chunk_overlap:]
                        current_chunk = overlap_text + sep + split
                    else:
                        current_chunk = split
                else:
                    if current_chunk:
                        current_chunk += sep + split
                    else:
                        current_chunk = split

            if current_chunk:
                chunks.append(current_chunk)

            # If we got reasonable chunks, return them
            if chunks and all(len(c) <= chunk_size * 1.5 for c in chunks):
                return chunks

    # Fallback to fixed-size
    return fixed_size_chunk(text, chunk_size, chunk_overlap)


def chunk_documents(
    documents: List[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: str = "fixed",
) -> List[dict]:
    """
    Chunk a list of documents.

    Args:
        documents: List of document dictionaries with 'text' key
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        strategy: 'fixed' or 'recursive'

    Returns:
        List of chunk dictionaries with 'id', 'text', 'metadata', 'chunk_index'
    """
    chunks = []

    for doc_idx, doc in enumerate(documents):
        text = doc.get("text", "")

        if strategy == "recursive":
            text_chunks = recursive_chunk(text, chunk_size, chunk_overlap)
        else:
            text_chunks = fixed_size_chunk(text, chunk_size, chunk_overlap)

        for chunk_idx, chunk_text in enumerate(text_chunks):
            # Clean chunk boundaries one more time to be safe
            cleaned_text = _clean_chunk_boundaries(chunk_text)
            if cleaned_text:  # Only add non-empty chunks
                chunk = {
                    "id": f"{doc['id']}_chunk_{chunk_idx}",
                    "text": cleaned_text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "doc_id": doc["id"],
                        "chunk_index": chunk_idx,
                        "total_chunks": len(text_chunks),
                    },
                }
                chunks.append(chunk)

    logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks")
    return chunks
