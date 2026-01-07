#!/usr/bin/env python3
"""Quick verification script to check that the project is set up correctly."""

import sys
from pathlib import Path

def check_imports():
    """Check that main modules can be imported."""
    try:
        from finance_rag_eval import constants
        from finance_rag_eval import config
        from finance_rag_eval import logging
        from finance_rag_eval.rag import ingestion, chunking, embeddings, index, retrieval
        from finance_rag_eval.eval import metrics, runner
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def check_paths():
    """Check that required paths exist."""
    project_root = Path(__file__).parent
    sample_docs = project_root / "src" / "finance_rag_eval" / "data" / "sample_docs"
    gold_set = project_root / "src" / "finance_rag_eval" / "data" / "qa_gold.json"
    
    if sample_docs.exists() and list(sample_docs.glob("*.txt")):
        print(f"✓ Sample documents found: {len(list(sample_docs.glob('*.txt')))} files")
    else:
        print(f"✗ Sample documents not found at {sample_docs}")
        return False
    
    if gold_set.exists():
        print(f"✓ Gold set found at {gold_set}")
    else:
        print(f"✗ Gold set not found at {gold_set}")
        return False
    
    return True

def main():
    """Run verification checks."""
    print("Verifying finance-rag-eval setup...\n")
    
    checks = [
        ("Imports", check_imports),
        ("Paths", check_paths),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"Checking {name}...")
        if not check_func():
            all_passed = False
        print()
    
    if all_passed:
        print("✓ All checks passed! Project is set up correctly.")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

