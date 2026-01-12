"""Generate larger, more realistic SEC filing-like documents for testing."""

import random
from pathlib import Path

# Template sections for realistic SEC filings
SECTIONS = {
    "business_overview": [
        "Company Overview",
        "Our company is a leading provider of financial technology solutions, serving enterprise clients across multiple industries.",
        "We operate through three main business segments: Enterprise Software, Professional Services, and Data Analytics.",
        "Our mission is to empower organizations with cutting-edge financial technology that drives operational efficiency and strategic decision-making.",
    ],
    "revenue": [
        "Revenue Recognition",
        "Total revenue for fiscal year 2023 reached $450 million, representing a 22% increase from the previous year.",
        "Q1 2024 revenue was $125 million, up 15% year-over-year.",
        "Revenue breakdown by segment:",
        "- Enterprise Software: $280 million (62% of total)",
        "- Professional Services: $120 million (27% of total)",
        "- Data Analytics: $50 million (11% of total)",
        "Recurring revenue from subscriptions accounted for 68% of total revenue, providing predictable cash flow.",
    ],
    "expenses": [
        "Operating Expenses",
        "Total operating expenses for fiscal year 2023 were $369 million, including:",
        "- Research and Development: $65 million (14.4% of revenue)",
        "- Sales and Marketing: $120 million (26.7% of revenue)",
        "- General and Administrative: $84 million (18.7% of revenue)",
        "- Cost of Goods Sold: $100 million (22.2% of revenue)",
        "We expect operating expenses to increase in 2024 as we invest in growth initiatives.",
    ],
    "risk_factors": [
        "Risk Factors",
        "Investors should carefully consider the following risk factors:",
        "1. Market Volatility: Economic downturns could reduce demand for our services.",
        "2. Regulatory Changes: Changes in financial regulations could require significant compliance investments.",
        "3. Technology Disruption: Rapid technological change could make our solutions obsolete.",
        "4. Cybersecurity Threats: We face constant threats from cyber attacks and data breaches.",
        "5. Competition: The market is highly competitive with well-established players and emerging startups.",
        "6. Customer Concentration: Our top 10 customers represent 45% of total revenue.",
    ],
    "financial_position": [
        "Financial Position",
        "As of December 31, 2023, our balance sheet showed:",
        "- Cash and cash equivalents: $120 million",
        "- Total assets: $850 million",
        "- Total liabilities: $300 million",
        "- Shareholders' equity: $550 million",
        "Our debt-to-equity ratio of 0.35 is conservative and provides flexibility for future growth.",
        "We maintain a $200 million revolving credit facility, of which $50 million was drawn as of year-end.",
    ],
    "segment_performance": [
        "Segment Performance",
        "Enterprise Software Segment:",
        "- Revenue: $280 million (62% of total)",
        "- Operating margin: 32%",
        "- Key products: Cloud ERP, Financial Planning, Risk Management",
        "Professional Services Segment:",
        "- Revenue: $120 million (27% of total)",
        "- Operating margin: 18%",
        "- Services: Implementation, Consulting, Support",
        "Data Analytics Segment:",
        "- Revenue: $50 million (11% of total)",
        "- Operating margin: 25%",
        "- Products: Business Intelligence, Predictive Analytics, Reporting Tools",
    ],
    "management_discussion": [
        "Management Discussion and Analysis",
        "Fiscal year 2023 was a record-breaking year for our organization.",
        "We achieved strong revenue growth of 22% year-over-year, driven by increased demand for cloud-based solutions.",
        "Operating margin improved to 18% from 15% in the prior year, reflecting operational efficiency improvements.",
        "We completed the acquisition of TechSolutions Inc. in Q3 2023, which added $25 million in annual revenue.",
        "Looking ahead, we expect continued growth in 2024, with projected revenue of $520-540 million.",
    ],
    "footnotes": [
        "Notes to Financial Statements",
        "Note 1: Revenue Recognition - We recognize revenue when control of goods or services is transferred to customers.",
        "Note 2: Stock-Based Compensation - We granted 2.5 million stock options during fiscal year 2023.",
        "Note 3: Leases - We lease office space and equipment under operating leases with remaining terms of 3-7 years.",
        "Note 4: Income Taxes - Our effective tax rate was 21% for fiscal year 2023.",
        "Note 5: Goodwill - We have $150 million in goodwill from acquisitions, tested annually for impairment.",
    ],
}


def generate_document(doc_id: str, sections: list, num_repetitions: int = 1) -> str:
    """Generate a document with specified sections."""
    lines = [f"SEC Filing - {doc_id}", ""]
    
    for section_key in sections:
        if section_key in SECTIONS:
            section_lines = SECTIONS[section_key]
            # Repeat sections to make documents longer
            for _ in range(num_repetitions):
                lines.extend(section_lines)
                lines.append("")  # Blank line between sections
    
    return "\n".join(lines)


def generate_large_docs(output_dir: Path, num_docs: int = 10):
    """Generate multiple large documents."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Different document types with varying sections
    doc_configs = [
        ("10K_2023_Annual", ["business_overview", "revenue", "expenses", "financial_position", "segment_performance", "management_discussion", "risk_factors", "footnotes"], 2),
        ("10Q_2024_Q1", ["revenue", "expenses", "segment_performance", "financial_position"], 3),
        ("10Q_2024_Q2", ["revenue", "expenses", "segment_performance"], 3),
        ("10K_2022_Annual", ["business_overview", "revenue", "expenses", "financial_position", "risk_factors"], 2),
        ("8K_MaterialEvent", ["business_overview", "financial_position"], 4),
        ("Proxy_2023", ["business_overview", "management_discussion", "risk_factors"], 3),
        ("10Q_2023_Q3", ["revenue", "expenses", "segment_performance", "financial_position"], 2),
        ("10Q_2023_Q4", ["revenue", "expenses", "segment_performance"], 3),
        ("10K_2021_Annual", ["business_overview", "revenue", "expenses", "risk_factors", "footnotes"], 2),
        ("8K_Acquisition", ["business_overview", "financial_position", "management_discussion"], 3),
    ]
    
    for i, (doc_id, sections, repetitions) in enumerate(doc_configs[:num_docs]):
        content = generate_document(doc_id, sections, repetitions)
        file_path = output_dir / f"{doc_id}.txt"
        file_path.write_text(content)
        print(f"Generated {file_path.name}: {len(content)} chars, {len(content.split())} words")


if __name__ == "__main__":
    import sys
    
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src/finance_rag_eval/data/large_docs")
    num_docs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"Generating {num_docs} large documents to {output_dir}")
    generate_large_docs(output_dir, num_docs)
    print(f"\nDone! Generated {num_docs} documents")
