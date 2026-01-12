"""Simple script to download SEC filings - run this to get real filings."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_edgar_downloader import Downloader
from finance_rag_eval.data.sec_edgar import extract_text_from_filing
from finance_rag_eval.logging import setup_logging

setup_logging()

# Configuration
EMAIL = os.getenv("SEC_EDGAR_EMAIL", "test@example.com")
TICKERS = ["AAPL", "MSFT"]  # Start with 2, add more as needed
EXTRACTED_DIR = Path("src/finance_rag_eval/data/real_sec_filings")
# sec-edgar-downloader saves to sec-edgar-filings/ in current directory
DOWNLOAD_DIR = Path("sec-edgar-filings")

EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Downloading Real SEC Filings")
print("=" * 80)
print(f"\nEmail: {EMAIL}")
print(f"Tickers: {', '.join(TICKERS)}")
print(f"Filing Type: 10-K")
print(f"Year: 2023")
print(f"\nNote: SEC rate limits to 10 requests/second. This may take a few minutes.")
print(f"Files will be saved to: {DOWNLOAD_DIR}/")
print()

# sec-edgar-downloader API: Downloader(company_name, email)
# Files are saved to sec-edgar-filings/{ticker}/{form_type}/{accession}/full-submission.txt
dl = Downloader("FinanceRAGEval", EMAIL)

downloaded = []
for ticker in TICKERS:
    print(f"Downloading {ticker} 10-K 2023...")
    try:
        result = dl.get("10-K", ticker, after="2023-01-01", before="2023-12-31", limit=1)
        downloaded.append(ticker)
        print(f"  ✓ {ticker} downloaded (result: {result})")
    except Exception as e:
        print(f"  ✗ {ticker} failed: {e}")

print(f"\n✓ Downloaded {len(downloaded)} filings")
print("\nExtracting text from downloaded filings...")

# Wait a moment for files to be written
import time
time.sleep(2)

# Find files in sec-edgar-filings/{ticker}/10-K/{accession}/full-submission.txt
extracted_count = 0
for ticker in downloaded:
    ticker_dir = DOWNLOAD_DIR / ticker / "10-K"
    if ticker_dir.exists():
        # Find the most recent filing
        filing_dirs = sorted([d for d in ticker_dir.iterdir() if d.is_dir()], 
                           key=lambda p: p.stat().st_mtime, reverse=True)
        for filing_dir in filing_dirs:
            filing_file = filing_dir / "full-submission.txt"
            if filing_file.exists() and filing_file.stat().st_size > 10000:
                print(f"\nProcessing {ticker} filing: {filing_file.name}")
                print(f"  Size: {filing_file.stat().st_size:,} bytes")
                
                # Extract text
                text = extract_text_from_filing(filing_file)
                if text and len(text) > 10000:
                    output_file = EXTRACTED_DIR / f"{ticker}_10K_2023.txt"
                    # Save first 500K chars to avoid huge files (real 10-Ks can be 1M+ chars)
                    text_to_save = text[:500000] if len(text) > 500000 else text
                    output_file.write_text(text_to_save, encoding="utf-8")
                    print(f"  ✓ Extracted {len(text_to_save):,} chars → {output_file.name}")
                    extracted_count += 1
                    break  # Only process most recent filing

print(f"\n✅ Extracted {extracted_count} filings to {EXTRACTED_DIR}")
if extracted_count > 0:
    print(f"\nYou can now use these for evaluation:")
    print(f"  PYTHONPATH=src pipenv run python -m finance_rag_eval.cli eval --docs-dir {EXTRACTED_DIR}")
else:
    print(f"\n⚠️  No files extracted. Check {DOWNLOAD_DIR}/ for downloaded files.")
