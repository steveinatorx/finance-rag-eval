"""Create gold set questions for new SEC filings by searching documents for key facts."""

import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finance_rag_eval.rag.ingestion import load_documents_from_dir

REAL_FILINGS_DIR = Path("src/finance_rag_eval/data/real_sec_filings")
GOLD_SET_PATH = Path("src/finance_rag_eval/data/qa_gold_real_sec.json")

# Company names mapping
COMPANY_NAMES = {
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "JPM": "JPMorgan",
    "V": "Visa",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart",
    "XOM": "ExxonMobil",
}

def extract_facts_from_doc(doc_id: str, text: str) -> dict:
    """Extract key facts from document text."""
    facts = {
        "company": COMPANY_NAMES.get(doc_id.replace("_10K_2023", ""), doc_id.replace("_10K_2023", "")),
        "fiscal_year_end": None,
        "total_revenue": None,
        "segments": [],
        "revenue_by_segment": {},
    }
    
    # Extract fiscal year end
    fiscal_patterns = [
        r"fiscal\s+year\s+end[s]?[\s:]+([A-Z][a-z]+\s+\d{1,2})",
        r"fiscal\s+year\s+ended\s+([A-Z][a-z]+\s+\d{1,2})",
    ]
    for pattern in fiscal_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            facts["fiscal_year_end"] = match.group(1)
            break
    
    # Extract total revenue/net sales (look for large numbers with $)
    revenue_patterns = [
        r"total\s+(?:net\s+)?(?:sales|revenue)[\s:]+[\$]?([\d,]+)\s*(?:million|billion)",
        r"(?:net\s+)?(?:sales|revenue)[\s:]+[\$]?([\d,]+)\s*(?:million|billion)",
    ]
    for pattern in revenue_patterns:
        matches = list(re.finditer(pattern, text[:50000], re.IGNORECASE))
        if matches:
            # Take the largest number (likely total revenue)
            values = []
            for m in matches:
                val_str = m.group(1).replace(",", "")
                if val_str.isdigit() and len(val_str) > 3:
                    values.append((int(val_str), m.group(0)))
            if values:
                values.sort(reverse=True)
                facts["total_revenue"] = values[0][1]
            break
    
    # Extract segments (look for segment mentions)
    segment_section = re.search(r"segment[s]?[^.]{0,500}", text, re.IGNORECASE)
    if segment_section:
        segment_text = segment_section.group(0)
        # Look for capitalized segment names
        segments = re.findall(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+segment", segment_text, re.IGNORECASE)
        facts["segments"] = list(set([s for s in segments if len(s.split()) <= 4]))[:5]
    
    return facts

def create_questions_for_company(doc_id: str, facts: dict) -> list:
    """Create questions for a company based on extracted facts."""
    company = facts["company"]
    ticker = doc_id.replace("_10K_2023", "")
    questions = []
    
    # Basic questions
    if facts["fiscal_year_end"]:
        questions.append({
            "question": f"What is {company}'s fiscal year end date?",
            "answer": f"{company}'s fiscal year ends on {facts['fiscal_year_end']}, 2023.",
            "requires_documents": [f"{doc_id}.txt"]
        })
    
    if facts["segments"]:
        segments_str = ", ".join(facts["segments"][:5])
        questions.append({
            "question": f"What are {company}'s main business segments?",
            "answer": f"{company}'s main business segments are {segments_str}.",
            "requires_documents": [f"{doc_id}.txt"]
        })
    
    if facts["total_revenue"]:
        questions.append({
            "question": f"What was {company}'s total revenue for fiscal year 2023?",
            "answer": f"{company}'s total revenue for fiscal year 2023 was {facts['total_revenue']}.",
            "requires_documents": [f"{doc_id}.txt"]
        })
    
    return questions

def main():
    """Main function to create and add questions."""
    # Load existing gold set
    if GOLD_SET_PATH.exists():
        with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
            existing_questions = json.load(f)
    else:
        existing_questions = []
    
    # Load documents
    documents = load_documents_from_dir(REAL_FILINGS_DIR)
    
    # Find documents not yet in gold set
    existing_docs = set()
    for qa in existing_questions:
        if "requires_documents" in qa:
            existing_docs.update([d.replace(".txt", "") for d in qa["requires_documents"]])
    
    new_docs = [d for d in documents if d["id"] not in existing_docs and d["id"] not in ["AAPL_10K_2023", "MSFT_10K_2023"]]
    
    print(f"Found {len(new_docs)} new documents to create questions for")
    
    all_new_questions = []
    for doc in new_docs:
        print(f"\nProcessing {doc['id']}...")
        facts = extract_facts_from_doc(doc["id"], doc["text"])
        print(f"  Facts: {facts}")
        questions = create_questions_for_company(doc["id"], facts)
        print(f"  Created {len(questions)} questions")
        all_new_questions.extend(questions)
    
    # Add multi-document questions (compare across companies)
    if len(new_docs) >= 2:
        # Compare fiscal year ends
        fiscal_years = {}
        for doc in new_docs:
            facts = extract_facts_from_doc(doc["id"], doc["text"])
            if facts["fiscal_year_end"]:
                fiscal_years[doc["id"]] = facts
        
        if len(fiscal_years) >= 2:
            companies = list(fiscal_years.keys())[:2]
            company1_name = COMPANY_NAMES.get(companies[0].replace("_10K_2023", ""), companies[0])
            company2_name = COMPANY_NAMES.get(companies[1].replace("_10K_2023", ""), companies[1])
            all_new_questions.append({
                "question": f"What is the difference between {company1_name} and {company2_name}'s fiscal year end dates?",
                "answer": f"{company1_name}'s fiscal year ends on {fiscal_years[companies[0]]['fiscal_year_end']}, while {company2_name}'s fiscal year ends on {fiscal_years[companies[1]]['fiscal_year_end']}. Both companies reported fiscal year 2023 results.",
                "type": "multi_document",
                "requires_documents": [f"{companies[0]}.txt", f"{companies[1]}.txt"]
            })
    
    # Combine and save
    all_questions = existing_questions + all_new_questions
    
    print(f"\n✅ Created {len(all_new_questions)} new questions")
    print(f"📊 Total questions: {len(all_questions)}")
    
    # Save
    with open(GOLD_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved to {GOLD_SET_PATH}")

if __name__ == "__main__":
    main()
