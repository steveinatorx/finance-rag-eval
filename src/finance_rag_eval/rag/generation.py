"""Answer generation: optional LLM or extractive fallback."""

import re
from typing import List, Optional

from finance_rag_eval.config import settings
from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def extractive_answer(query: str, retrieved_chunks: List[dict]) -> str:
    """
    Generate extractive answer from retrieved chunks (offline fallback).

    Improved heuristic: find most relevant sentences and combine them coherently.

    Args:
        query: Query string
        retrieved_chunks: List of retrieved chunk dictionaries

    Returns:
        Extracted answer text
    """
    if not retrieved_chunks:
        return "No relevant information found."

    # Extract keywords from query
    query_lower = query.lower()
    keywords = [w for w in query_lower.split() if len(w) > 3]
    if not keywords:
        # Fallback: use all words longer than 2 chars
        keywords = [w for w in query_lower.split() if len(w) > 2]

    # Collect sentences from chunks with better extraction
    all_sentences = []
    for chunk_dict in retrieved_chunks:
        chunk_text = chunk_dict["chunk"]["text"]

        # Better sentence splitting - handle various punctuation
        # Split on sentence endings but preserve them
        sentences = re.split(r"([.!?]+[\s\n]+)", chunk_text)

        # Recombine sentences with their punctuation
        i = 0
        current_sentence = ""
        while i < len(sentences):
            part = sentences[i].strip()

            # Check if next part is punctuation
            if i + 1 < len(sentences) and re.match(r"[.!?]+[\s\n]+", sentences[i + 1]):
                current_sentence += part + sentences[i + 1].strip()
                i += 2

                # Complete sentence found
                if current_sentence and len(current_sentence.strip()) > 15:
                    # Filter out sentences that start with incomplete words
                    first_word = (
                        current_sentence.strip().split()[0]
                        if current_sentence.strip().split()
                        else ""
                    )
                    # Skip if starts with punctuation, or is just a label like "Division:"
                    if (
                        not first_word.startswith((":", ";", ","))
                        and len(first_word) > 1
                        and not (
                            first_word.endswith(":") and len(first_word.split()) == 1
                        )
                    ):
                        all_sentences.append(
                            (current_sentence.strip(), chunk_dict["score"])
                        )
                current_sentence = ""
            else:
                current_sentence += part + " "
                i += 1

        # Handle any remaining sentence
        if current_sentence.strip() and len(current_sentence.strip()) > 15:
            first_word = (
                current_sentence.strip().split()[0]
                if current_sentence.strip().split()
                else ""
            )
            # Skip if starts with punctuation, or is just a label like "Division:"
            if (
                not first_word.startswith((":", ";", ","))
                and len(first_word) > 1
                and not (first_word.endswith(":") and len(first_word.split()) == 1)
            ):
                all_sentences.append((current_sentence.strip(), chunk_dict["score"]))

    if not all_sentences:
        # Fallback: return first chunk (cleaned)
        answer = retrieved_chunks[0]["chunk"]["text"][:500]
        # Remove leading punctuation if present
        if answer.startswith((":", ";", ",")):
            answer = answer[1:].lstrip()
        return answer

    # Score sentences by keyword matches and relevance
    scored_sentences = []
    for sent, base_score in all_sentences:
        sent_lower = sent.lower()

        # Count keyword matches
        keyword_matches = sum(1 for kw in keywords if kw in sent_lower)

        # Boost score for sentences with numbers (often contain revenue/financial data)
        has_numbers = bool(re.search(r"\d+", sent))

        # Boost score for sentences with financial terms
        financial_terms = [
            "revenue",
            "million",
            "billion",
            "income",
            "profit",
            "loss",
            "quarter",
            "year",
        ]
        has_financial = sum(1 for term in financial_terms if term in sent_lower)

        # Calculate composite score
        score = base_score * (1 + keyword_matches * 0.5 + has_financial * 0.3)
        if has_numbers:
            score *= 1.2

        scored_sentences.append((sent, score))

    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    # Take top 2-3 most relevant sentences (prioritize quality over quantity)
    top_sentences = []
    seen_content = set()

    for sent, score in scored_sentences[:5]:  # Check top 5 for diversity
        # Simple deduplication - check if sentence is too similar to already selected ones
        sent_words = set(sent.lower().split())
        is_duplicate = False
        for seen_sent in seen_content:
            seen_words = set(seen_sent.lower().split())
            # If more than 70% word overlap, consider it duplicate
            if len(sent_words & seen_words) / max(len(sent_words), 1) > 0.7:
                is_duplicate = True
                break

        if not is_duplicate:
            top_sentences.append(sent)
            seen_content.add(sent)
            if len(top_sentences) >= 3:  # Limit to 3 sentences for coherence
                break

    # Combine sentences with proper spacing
    if top_sentences:
        answer = " ".join(top_sentences)
        # Clean up multiple spaces
        answer = re.sub(r"\s+", " ", answer).strip()
    else:
        # Fallback
        answer = (
            scored_sentences[0][0]
            if scored_sentences
            else retrieved_chunks[0]["chunk"]["text"][:500]
        )

    return answer


def llm_generate(
    query: str,
    retrieved_chunks: List[dict],
    model: Optional[str] = None,
) -> str:
    """
    Generate answer using LLM (OpenAI or other).

    Args:
        query: Query string
        retrieved_chunks: List of retrieved chunk dictionaries
        model: Optional model name

    Returns:
        Generated answer
    """
    if not settings.openai_api_key:
        logger.warning("OpenAI API key not set, falling back to extractive answer")
        return extractive_answer(query, retrieved_chunks)

    try:
        from openai import OpenAI

        # Initialize client with optional project ID
        client_kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_project_id:
            client_kwargs["default_headers"] = {
                "OpenAI-Project": settings.openai_project_id
            }

        client = OpenAI(**client_kwargs)

        # Build context from retrieved chunks
        context = "\n\n".join(
            [
                chunk["chunk"]["text"]
                for chunk in retrieved_chunks[:5]  # Limit context size
            ]
        )

        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        model_name = model or settings.llm_model or "gpt-3.5-turbo"

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=settings.llm_temperature,
            max_tokens=500,
        )

        answer = response.choices[0].message.content
        logger.debug(f"Generated answer using {model_name}")
        return answer

    except ImportError:
        logger.warning(
            "OpenAI library not installed, falling back to extractive answer"
        )
        return extractive_answer(query, retrieved_chunks)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}, falling back to extractive answer")
        return extractive_answer(query, retrieved_chunks)


def generate_answer(
    query: str,
    retrieved_chunks: List[dict],
    use_llm: bool = False,
    model: Optional[str] = None,
) -> str:
    """
    Generate answer from retrieved chunks.

    Args:
        query: Query string
        retrieved_chunks: List of retrieved chunk dictionaries
        use_llm: Whether to use LLM (requires API key)
        model: Optional LLM model name

    Returns:
        Generated answer
    """
    if use_llm and settings.openai_api_key:
        return llm_generate(query, retrieved_chunks, model)
    else:
        return extractive_answer(query, retrieved_chunks)
