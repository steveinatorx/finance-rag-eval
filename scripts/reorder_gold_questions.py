"""Reorder gold questions markdown to interleave companies and fix labels."""

import json
from pathlib import Path

# Load the JSON
with open('src/finance_rag_eval/data/qa_gold_real_sec.json') as f:
    data = json.load(f)

# Categorize questions
single_doc = []
multi_section = []  # Single document, multiple sections
multi_doc = []  # Multiple documents
temporal = []

for q in data:
    docs = q['requires_documents']
    q_type = q.get('type', '')
    
    if q_type == 'temporal':
        temporal.append(q)
    elif q_type == 'multi_document':
        if len(docs) > 1:
            multi_doc.append(q)
        else:
            # Actually multi-section within single doc
            multi_section.append(q)
    else:
        single_doc.append(q)

# Reorder: interleave companies better
reordered = []

# 1. Fiscal year end dates (diverse companies first)
fiscal_year_questions = [q for q in single_doc if 'fiscal year end' in q['question'].lower()]
reordered.extend(sorted(fiscal_year_questions, key=lambda x: x['question']))

# 2. Business segments (diverse companies)
segment_questions = [q for q in single_doc if 'business segments' in q['question'].lower() and 'fiscal year end' not in q['question'].lower()]
reordered.extend(sorted(segment_questions, key=lambda x: x['question']))

# 3. Apple questions (spread out, but after other companies)
apple_questions = [q for q in single_doc if 'Apple' in q['question'] and q not in fiscal_year_questions and q not in segment_questions]
reordered.extend(apple_questions)

# 4. Multi-section (single document)
reordered.extend(multi_section)

# 5. Temporal
reordered.extend(temporal)

# 6. True multi-document
reordered.extend(multi_doc)

# Generate markdown
md_lines = [
    "# Gold Set: Questions and Answers for Finance RAG Evaluation",
    "",
    "This document contains 42 question-answer pairs extracted from 10 real SEC 10-K filings (2023) for evaluating RAG system performance.",
    "",
    "## Companies Covered",
    "- Apple (AAPL)",
    "- Microsoft (MSFT)",
    "- Alphabet/Google (GOOGL)",
    "- Amazon (AMZN)",
    "- Tesla (TSLA)",
    "- JPMorgan (JPM)",
    "- Visa (V)",
    "- Johnson & Johnson (JNJ)",
    "- Walmart (WMT)",
    "- ExxonMobil (XOM)",
    "",
    "---",
    "",
    "## Questions and Answers",
    "",
]

for i, q in enumerate(reordered, 1):
    question = q['question']
    answer = q['answer']
    docs = q['requires_documents']
    q_type = q.get('type', '')
    
    # Determine question type label
    if q_type == 'temporal':
        type_label = "Temporal"
    elif q_type == 'multi_document':
        if len(docs) > 1:
            type_label = "Multi-document"
        else:
            type_label = "Multi-section (single document)"
    elif q in multi_section:
        type_label = "Multi-section (single document)"
    else:
        type_label = None
    
    # Extract company name for title
    company = question.split("'")[0].strip() if "'" in question else "General"
    
    # Create title
    title_parts = [f"{i}. {company}"]
    if "fiscal year end" in question.lower():
        title_parts.append("Fiscal Year End Date")
    elif "business segments" in question.lower():
        title_parts.append("Business Segments")
    elif "compare" in question.lower() or "difference" in question.lower():
        title_parts.append("Comparison")
    elif "change" in question.lower():
        title_parts.append("Temporal Change")
    elif "percentage" in question.lower():
        title_parts.append("Percentage")
    else:
        # Extract key metric
        if "net sales" in question.lower():
            title_parts.append("Net Sales")
        elif "gross margin" in question.lower():
            title_parts.append("Gross Margin")
        elif "research and development" in question.lower() or "R&D" in question.lower():
            title_parts.append("R&D Expense")
    
    md_lines.append(f"### {' - '.join(title_parts)}")
    md_lines.append(f"**Question:** {question}")
    md_lines.append("")
    md_lines.append(f"**Answer:** {answer}")
    md_lines.append("")
    if type_label:
        md_lines.append(f"**Type:** {type_label}")
    md_lines.append(f"**Documents:** {', '.join(docs)}")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

# Add question types section
md_lines.extend([
    "## Question Types",
    "",
    "- **Single-document questions:** Questions that can be answered from a single document",
    "- **Multi-section questions:** Questions requiring aggregation across multiple sections within a single document (e.g., comparing Products vs Services within Apple's filing)",
    "- **Multi-document questions:** Questions requiring comparison across multiple documents (e.g., comparing Apple vs Microsoft)",
    "- **Temporal questions:** Questions asking about changes over time periods (e.g., comparing FY 2023 to FY 2024)",
    "",
    "## Evaluation Notes",
    "",
    "- All answers are factual and verifiable from the source SEC filings",
    "- Questions cover diverse topics: revenue, expenses, segments, ratios, fiscal year information",
    "- Questions test both single-document retrieval and multi-document reasoning",
    "- Temporal questions test the system's ability to track changes over time periods",
    "- Questions are ordered to interleave companies, reducing bias toward any single filing structure",
])

# Write to file
output_path = Path('blog_notes/gold_questions_and_answers.md')
output_path.write_text('\n'.join(md_lines))
print(f"Generated reordered gold questions markdown: {output_path}")
