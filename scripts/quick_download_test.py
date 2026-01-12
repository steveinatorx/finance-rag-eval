"""Quick test to download one real SEC filing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finance_rag_eval.data.sec_edgar import fetch_edgar_filing, extract_text_from_filing
from finance_rag_eval.logging import setup_logging

setup_logging()

print("Downloading AAPL 10-K 2023...")
print("(This may take 30-60 seconds due to SEC rate limits)")
print("\nNote: SEC requires an email address for API access (set SEC_EDGAR_EMAIL env var)")
print("This is for identification per SEC terms of service, not authentication.\n")

filing_path = fetch_edgar_filing(
    ticker="AAPL",
    filing_type="10-K",
    year=2023,
    output_dir=Path("data/sec_filings"),
)

if filing_path:
    print(f"\n✓ Downloaded to: {filing_path}")
    
    text = extract_text_from_filing(filing_path)
    if text:
        print(f"✓ Extracted {len(text):,} characters")
        print(f"✓ {len(text.split()):,} words")
        print(f"\nFirst 500 characters:")
        print("-" * 80)
        print(text[:500])
        print("-" * 80)
        
        # Save to real_sec_filings directory
        output_dir = Path("src/finance_rag_eval/data/real_sec_filings")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "AAPL_10K_2023.txt"
        output_file.write_text(text, encoding="utf-8")
        print(f"\n✓ Saved to: {output_file}")
        print(f"\nYou can now use this for evaluation:")
        print(f"  PYTHONPATH=src pipenv run python -m finance_rag_eval.cli eval --docs-dir {output_dir}")
    else:
        print("✗ Failed to extract text")
else:
    print("\n✗ Download failed")
    print("\nPossible reasons:")
    print("  1. Network connectivity issues")
    print("  2. SEC rate limiting (try again in a minute)")
    print("  3. Invalid ticker/CIK")
    print("\nYou can also manually download filings from:")
    print("  https://www.sec.gov/edgar/searchedgar/companysearch.html")
