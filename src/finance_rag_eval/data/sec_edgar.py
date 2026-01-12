"""SEC EDGAR document fetcher using sec-edgar-downloader."""

import re
from pathlib import Path
from typing import List, Optional

from finance_rag_eval.logging import get_logger

logger = get_logger(__name__)

# Popular companies for testing (CIK numbers)
POPULAR_COMPANIES = {
    "AAPL": "0000320193",  # Apple
    "MSFT": "0000789019",  # Microsoft
    "GOOGL": "0001652044",  # Alphabet
    "AMZN": "0001018724",  # Amazon
    "TSLA": "0001318605",  # Tesla
    "META": "0001326801",  # Meta
    "NVDA": "0001045810",  # NVIDIA
    "JPM": "0000019617",  # JPMorgan Chase
    "V": "0001403161",  # Visa
    "JNJ": "0000200406",  # Johnson & Johnson
}


def fetch_edgar_filing(
    ticker: str,
    filing_type: str = "10-K",
    year: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Fetch a SEC EDGAR filing using sec-edgar-downloader.

    Args:
        ticker: Company ticker symbol (e.g., "AAPL") or CIK
        filing_type: Type of filing (e.g., "10-K", "10-Q")
        year: Optional year filter
        output_dir: Directory to save the filing

    Returns:
        Path to saved filing or None if fetch fails
    """
    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        logger.error(
            "sec-edgar-downloader not installed. Install with: pip install sec-edgar-downloader"
        )
        return None

    if output_dir is None:
        output_dir = Path("data/sec_filings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert ticker to CIK if needed
    cik = ticker
    if ticker.upper() in POPULAR_COMPANIES:
        cik = POPULAR_COMPANIES[ticker.upper()]

    try:
        # SEC requires an email address for API access (their terms of service)
        # This is used for identification/contact purposes, not authentication
        import os

        email = os.getenv("SEC_EDGAR_EMAIL")
        if not email:
            logger.warning(
                "SEC_EDGAR_EMAIL not set. SEC requires an email address for API access. "
                "Set it with: export SEC_EDGAR_EMAIL=your.email@example.com"
            )
            return None
        dl = Downloader(output_dir, email)
        logger.info(
            "Fetching %s for %s (CIK: %s, year: %s)", filing_type, ticker, cik, year
        )

        if year:
            dl.get(filing_type, cik, after=f"{year}-01-01", before=f"{year}-12-31")
        else:
            dl.get(filing_type, cik, limit=1)

        # Find the downloaded file
        # sec-edgar-downloader structure: output_dir / ticker / filing_type / filing_folder
        # Look for the ticker directory (might be ticker or CIK)
        possible_dirs = [
            output_dir / ticker.upper() / filing_type,
            output_dir / cik / filing_type,
            output_dir / ticker.lower() / filing_type,
        ]

        # Also check if there are any subdirectories in output_dir
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_dir():
                    filing_type_dir = item / filing_type
                    if filing_type_dir.exists():
                        possible_dirs.append(filing_type_dir)

        for company_dir in possible_dirs:
            if company_dir.exists():
                # Get the most recent filing folder
                filing_folders = [d for d in company_dir.iterdir() if d.is_dir()]
                if filing_folders:
                    filing_folder = sorted(
                        filing_folders, key=lambda p: p.stat().st_mtime, reverse=True
                    )[0]
                    logger.info("Downloaded filing to %s", filing_folder)
                    return filing_folder

        # Last resort: search recursively
        logger.debug("Searching recursively in %s", output_dir)
        for root, dirs, files in output_dir.rglob(filing_type):
            if Path(root).is_dir():
                filing_folders = [
                    Path(root) / d for d in dirs if (Path(root) / d).is_dir()
                ]
                if filing_folders:
                    filing_folder = sorted(
                        filing_folders, key=lambda p: p.stat().st_mtime, reverse=True
                    )[0]
                    logger.info("Found filing at %s", filing_folder)
                    return filing_folder

        logger.warning("Filing downloaded but file not found in expected location")
        logger.debug("Searched in: %s", possible_dirs)
        return None

    except Exception as e:
        logger.error("Error fetching filing: %s", e)
        return None


def extract_text_from_filing(filing_path: Path) -> Optional[str]:
    """
    Extract text content from a downloaded SEC filing.

    Args:
        filing_path: Path to the filing directory or file

    Returns:
        Extracted text or None
    """
    if filing_path.is_dir():
        # Look for the actual filing document (usually .txt or .htm)
        for ext in ["*.txt", "*.htm", "*.html"]:
            files = list(filing_path.glob(ext))
            if files:
                filing_path = files[0]
                break

    if not filing_path.exists():
        return None

    try:
        content = filing_path.read_text(encoding="utf-8", errors="ignore")

        # Clean HTML if needed
        if filing_path.suffix.lower() in [".htm", ".html"]:
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(content, "html.parser")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                content = soup.get_text()
            except ImportError:
                # Fallback: simple regex cleanup
                content = re.sub(r"<[^>]+>", "", content)

        # Clean up whitespace
        content = re.sub(r"\n\s*\n", "\n\n", content)
        content = content.strip()

        return content

    except Exception as e:
        logger.error("Error extracting text from filing: %s", e)
        return None


def download_real_filings(
    tickers: List[str],
    filing_type: str = "10-K",
    year: Optional[int] = 2023,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Download multiple real SEC filings.

    Args:
        tickers: List of ticker symbols
        filing_type: Type of filing to download
        year: Year filter
        output_dir: Output directory

    Returns:
        List of paths to downloaded filings
    """
    if output_dir is None:
        output_dir = Path("data/sec_filings")

    downloaded = []
    for ticker in tickers:
        logger.info("Downloading %s %s for %s...", filing_type, year, ticker)
        filing_path = fetch_edgar_filing(ticker, filing_type, year, output_dir)
        if filing_path:
            downloaded.append(filing_path)

    logger.info("Downloaded %d filings", len(downloaded))
    return downloaded


def list_available_filings(cik: str) -> List[str]:
    """List available filings for a given CIK."""
    # This would require API access to list filings
    # For now, return empty list
    logger.info("Listing filings for CIK %s (requires API access)", cik)
    return []
