"""One-off diagnostic: where does the p1 title block rank for abstract queries?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings.vector_store import VectorStore

store = VectorStore()

for query in [
    "What is the title of the document?",
    "Which organization issued this template?",
    "Concept Note Template",       # <- literal keyword, should work
    "Higher Education Commission", # <- literal keyword, should work
]:
    print(f"\n=== {query!r} ===")
    results = store.query(query, top_k=15)
    for i, r in enumerate(results, 1):
        marker = "  ◀ HAS TITLE" if "Concept Note Template" in r.text else ""
        marker = "  ◀ HAS ORG" if "Higher Education Commission" in r.text and "HAS TITLE" not in marker else marker
        print(f"  {i:2}. p{r.page:<3} sim={r.similarity:.3f}{marker}")