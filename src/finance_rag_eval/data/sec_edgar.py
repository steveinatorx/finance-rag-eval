"""Lightweight SEC EDGAR document fetcher (optional)."""

from pathlib import Path
from typing import List, Optional

from finance_rag_eval.config import settings
from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)


def fetch_edgar_filing(
    cik: str,
    filing_type: str = "10-K",
    year: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Fetch a SEC EDGAR filing (lightweight implementation).

    Args:
        cik: Company CIK identifier
        filing_type: Type of filing (e.g., "10-K", "10-Q")
        year: Optional year filter
        output_dir: Directory to save the filing

    Returns:
        Path to saved filing or None if fetch fails
    """
    if not settings.sec_edgar_api_key:
        logger.warning("SEC EDGAR API key not set. Skipping EDGAR fetch.")
        return None

    logger.info(f"Fetching {filing_type} for CIK {cik} (year: {year})")
    # Placeholder for actual EDGAR API integration
    # In a real implementation, would use sec-edgar-downloader or direct API calls
    logger.warning(
        "EDGAR fetch not fully implemented. Use sample docs for offline runs."
    )
    return None


def list_available_filings(cik: str) -> List[str]:
    """List available filings for a given CIK."""
    logger.warning("EDGAR listing not fully implemented.")
    return []
