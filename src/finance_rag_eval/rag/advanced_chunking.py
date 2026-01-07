"""Advanced chunking strategies: semantic, structure-aware, and parent document chunking."""

import re
from typing import Dict, List

import numpy as np

from finance_rag_eval.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from finance_rag_eval.logging import get_logger
from finance_rag_eval.rag.embeddings import generate_embeddings

logger = get_logger(__name__)


def detect_document_structure(text: str) -> List[Dict[str, any]]:
    """
    Detect document structure: headers, sections, paragraphs.

    Args:
        text: Input text

    Returns:
        List of structure elements with 'type', 'text', 'level', 'start_pos', 'end_pos'
    """
    elements = []
    lines = text.split("\n")

    current_pos = 0
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            current_pos += len(line) + 1
            continue

        # Detect headers (lines that are short, all caps, or end with colon)
        is_header = False
        level = 0

        # Level 1: All caps and short
        if (
            line_stripped.isupper()
            and len(line_stripped) < 100
            and len(line_stripped.split()) < 10
        ):
            is_header = True
            level = 1
        # Level 2: Title case, ends with colon, short
        elif (
            line_stripped[0].isupper()
            and line_stripped.endswith(":")
            and len(line_stripped) < 150
            and not line_stripped.endswith("::")
        ):
            is_header = True
            level = 2
        # Level 3: Starts with number or bullet, short
        elif re.match(r"^[\d\-•]\s+[A-Z]", line_stripped) and len(line_stripped) < 200:
            is_header = True
            level = 3

        if is_header:
            elements.append(
                {
                    "type": "header",
                    "text": line_stripped,
                    "level": level,
                    "start_pos": current_pos,
                    "end_pos": current_pos + len(line),
                }
            )
        else:
            # Regular paragraph
            elements.append(
                {
                    "type": "paragraph",
                    "text": line_stripped,
                    "level": 0,
                    "start_pos": current_pos,
                    "end_pos": current_pos + len(line),
                }
            )

        current_pos += len(line) + 1

    return elements


def structure_aware_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Chunk text while respecting document structure (headers, sections).

    Args:
        text: Input text
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    structure = detect_document_structure(text)

    if not structure:
        # Fallback to simple chunking
        from finance_rag_eval.rag.chunking import fixed_size_chunk

        return fixed_size_chunk(text, chunk_size, chunk_overlap)

    chunks = []
    current_chunk = ""
    current_header = None

    for element in structure:
        element_text = element["text"]

        # If we hit a header, decide whether to start new chunk
        if element["type"] == "header":
            # If current chunk is substantial, save it
            if current_chunk and len(current_chunk.strip()) > chunk_size * 0.5:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                if chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + element_text + "\n"
                else:
                    current_chunk = element_text + "\n"
            else:
                # Add header to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + element_text + "\n"
                else:
                    current_chunk = element_text + "\n"

            current_header = element_text
        else:
            # Regular paragraph
            # Check if adding this would exceed chunk size
            potential_chunk = current_chunk + element_text + "\n\n"

            if len(potential_chunk) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap and header context
                if chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-chunk_overlap:]
                    new_chunk = overlap_text + "\n\n"
                    if current_header:
                        new_chunk += current_header + "\n"
                    current_chunk = new_chunk + element_text + "\n\n"
                else:
                    if current_header:
                        current_chunk = current_header + "\n" + element_text + "\n\n"
                    else:
                        current_chunk = element_text + "\n\n"
            else:
                current_chunk = potential_chunk

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def semantic_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    similarity_threshold: float = 0.5,
) -> List[str]:
    """
    Chunk text by semantic similarity using embeddings.

    Groups sentences with similar meaning together, respecting chunk size limits.

    Args:
        text: Input text
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        similarity_threshold: Minimum similarity to group sentences

    Returns:
        List of text chunks
    """
    # Split into sentences
    sentences = re.split(r"([.!?]+[\s\n]+)", text)

    # Recombine sentences with punctuation
    complete_sentences = []
    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        if i + 1 < len(sentences) and re.match(r"[.!?]+[\s\n]+", sentences[i + 1]):
            sent += " " + sentences[i + 1].strip()
            i += 2
        else:
            i += 1
        if sent and len(sent) > 10:
            complete_sentences.append(sent)

    if len(complete_sentences) <= 1:
        return [text]

    # Generate embeddings for all sentences
    try:
        embeddings = generate_embeddings(complete_sentences)
    except Exception as e:
        logger.warning(f"Failed to generate embeddings for semantic chunking: {e}")
        # Fallback to structure-aware chunking
        return structure_aware_chunk(text, chunk_size, chunk_overlap)

    # Group sentences by similarity
    chunks = []
    current_chunk_sentences = []
    current_chunk_embeddings = []
    current_chunk_size = 0

    for i, (sentence, embedding) in enumerate(zip(complete_sentences, embeddings)):
        sentence_len = len(sentence)

        # Check if adding this sentence would exceed chunk size
        if current_chunk_size + sentence_len > chunk_size and current_chunk_sentences:
            # Save current chunk
            chunks.append(" ".join(current_chunk_sentences))

            # Start new chunk with overlap
            if chunk_overlap > 0 and current_chunk_sentences:
                # Add last sentence(s) for overlap
                overlap_sentences = []
                overlap_embeddings = []
                overlap_size = 0
                for sent, emb in zip(
                    reversed(current_chunk_sentences),
                    reversed(current_chunk_embeddings),
                ):
                    if overlap_size + len(sent) <= chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_embeddings.insert(0, emb)
                        overlap_size += len(sent)
                    else:
                        break
                current_chunk_sentences = overlap_sentences
                current_chunk_embeddings = overlap_embeddings
                current_chunk_size = overlap_size
            else:
                current_chunk_sentences = []
                current_chunk_embeddings = []
                current_chunk_size = 0

        # Check similarity with last sentence in current chunk
        if current_chunk_sentences:
            last_embedding = current_chunk_embeddings[-1]
            similarity = np.dot(embedding, last_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(last_embedding) + 1e-8
            )

            # If similar enough, add to current chunk
            if similarity >= similarity_threshold:
                current_chunk_sentences.append(sentence)
                current_chunk_embeddings.append(embedding)
                current_chunk_size += sentence_len
            else:
                # Start new chunk
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_chunk_embeddings = [embedding]
                current_chunk_size = sentence_len
        else:
            # First sentence
            current_chunk_sentences.append(sentence)
            current_chunk_embeddings.append(embedding)
            current_chunk_size = sentence_len

    # Add final chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


def parent_document_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    parent_chunk_size: int = None,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, any]]:
    """
    Create small chunks with parent document context.

    Returns chunks with both small chunk text and parent context.
    Useful for retrieval: retrieve small chunks, but have parent context for generation.

    Args:
        text: Input text
        chunk_size: Size of small chunks
        parent_chunk_size: Size of parent chunks (defaults to chunk_size * 3)
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunk dictionaries with 'text', 'parent_text', 'metadata'
    """
    if parent_chunk_size is None:
        parent_chunk_size = chunk_size * 3

    # First create parent chunks
    from finance_rag_eval.rag.chunking import fixed_size_chunk

    parent_chunks = fixed_size_chunk(text, parent_chunk_size, chunk_overlap)

    # Then create small chunks within each parent
    all_chunks = []
    for parent_idx, parent_text in enumerate(parent_chunks):
        small_chunks = fixed_size_chunk(parent_text, chunk_size, chunk_overlap // 2)

        for small_idx, small_text in enumerate(small_chunks):
            all_chunks.append(
                {
                    "text": small_text,
                    "parent_text": parent_text,
                    "metadata": {
                        "parent_index": parent_idx,
                        "small_index": small_idx,
                        "parent_size": len(parent_text),
                        "small_size": len(small_text),
                    },
                }
            )

    return all_chunks


def hybrid_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: str = "structure_aware",
) -> List[str]:
    """
    Hybrid chunking: combine multiple strategies.

    Args:
        text: Input text
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        strategy: 'structure_aware', 'semantic', or 'hybrid'

    Returns:
        List of text chunks
    """
    if strategy == "structure_aware":
        return structure_aware_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        return semantic_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "hybrid":
        # First do structure-aware, then refine with semantic
        structure_chunks = structure_aware_chunk(text, chunk_size * 2, chunk_overlap)

        # Refine each structure chunk semantically if needed
        final_chunks = []
        for chunk in structure_chunks:
            if len(chunk) > chunk_size * 1.5:
                # Split further using semantic chunking
                semantic_subchunks = semantic_chunk(chunk, chunk_size, chunk_overlap)
                final_chunks.extend(semantic_subchunks)
            else:
                final_chunks.append(chunk)

        return final_chunks
    else:
        # Fallback
        from finance_rag_eval.rag.chunking import fixed_size_chunk

        return fixed_size_chunk(text, chunk_size, chunk_overlap)


def chunk_documents_advanced(
    documents: List[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: str = "structure_aware",
    use_parent_document: bool = False,
) -> List[dict]:
    """
    Chunk documents using advanced strategies.

    Args:
        documents: List of document dictionaries
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        strategy: 'structure_aware', 'semantic', 'hybrid', or 'fixed'
        use_parent_document: If True, use parent document chunking

    Returns:
        List of chunk dictionaries
    """
    chunks = []

    for doc_idx, doc in enumerate(documents):
        text = doc.get("text", "")

        if use_parent_document:
            # Use parent document chunking
            parent_chunks = parent_document_chunk(
                text, chunk_size, chunk_overlap=chunk_overlap
            )

            for chunk_data in parent_chunks:
                chunk = {
                    "id": f"{doc['id']}_chunk_{len(chunks)}",
                    "text": chunk_data["text"],
                    "metadata": {
                        **doc.get("metadata", {}),
                        **chunk_data.get("metadata", {}),
                        "doc_id": doc["id"],
                        "parent_text": chunk_data.get("parent_text", ""),
                    },
                }
                chunks.append(chunk)
        else:
            # Use regular advanced chunking
            if strategy == "fixed":
                from finance_rag_eval.rag.chunking import fixed_size_chunk

                text_chunks = fixed_size_chunk(text, chunk_size, chunk_overlap)
            else:
                text_chunks = hybrid_chunk(text, chunk_size, chunk_overlap, strategy)

            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunk = {
                    "id": f"{doc['id']}_chunk_{chunk_idx}",
                    "text": chunk_text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "doc_id": doc["id"],
                        "chunk_index": chunk_idx,
                        "total_chunks": len(text_chunks),
                        "strategy": strategy,
                    },
                }
                chunks.append(chunk)

    logger.info(
        f"Chunked {len(documents)} documents into {len(chunks)} chunks using {strategy} strategy"
    )
    return chunks
