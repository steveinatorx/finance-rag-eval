"""Download real SEC filings for testing."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finance_rag_eval.data.sec_edgar import download_real_filings, extract_text_from_filing
from finance_rag_eval.logging import setup_logging

setup_logging()

# Popular companies for diverse testing
TICKERS = [
    "AAPL",  # Apple - Tech
    "MSFT",  # Microsoft - Tech
    "JPM",   # JPMorgan - Finance
    "JNJ",   # Johnson & Johnson - Healthcare
]

OUTPUT_DIR = Path("src/finance_rag_eval/data/real_sec_filings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Downloading Real SEC Filings")
print("=" * 80)
print(f"\nTickers: {', '.join(TICKERS)}")
print(f"Filing Type: 10-K")
print(f"Year: 2023")
print(f"Output: {OUTPUT_DIR}")
print()

# Download filings
downloaded = download_real_filings(
    tickers=TICKERS,
    filing_type="10-K",
    year=2023,
    output_dir=Path("data/sec_filings"),
)

print(f"\nDownloaded {len(downloaded)} filings")

# Extract and save text files
print("\nExtracting text from filings...")
extracted = []
for filing_path in downloaded:
    text = extract_text_from_filing(filing_path)
    if text:
        # Get ticker from path or use a default name
        ticker = filing_path.parent.parent.name  # Usually CIK, but we'll use a better name
        output_file = OUTPUT_DIR / f"{ticker}_10K_2023.txt"
        output_file.write_text(text, encoding="utf-8")
        extracted.append(output_file)
        print(f"  ✓ Extracted {len(text)} chars → {output_file.name}")

print(f"\n✅ Extracted {len(extracted)} text files to {OUTPUT_DIR}")
print(f"\nYou can now use these files for evaluation:")
print(f"  python -m finance_rag_eval.cli eval --docs-dir {OUTPUT_DIR}")
