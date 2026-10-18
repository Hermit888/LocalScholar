"""
Smoke test: retriever BM25 + SPECTER2 vector search + RRF fusion.

Requires:
  - PostgreSQL running with papers in the database (run test_ingestor_write_papers_to_db.py first)
  - Qdrant running with vectors (run ingestion/embedder.py first)
  - SPECTER2 model downloaded (~440 MB, cached after first run)

Runs retrieve() with a typical cs.AI query for year 2020 and prints the
top candidates with their RRF scores to verify both search paths are working.

Run from the project root:
    python test/test_retriever_vector_and_bm25.py
"""

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from discovery.retriever import retrieve

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


QUERY_JSON = {
    "topics":      ["attention mechanism", "transformer", "image recognition"],
    "category":    "cs.AI",
    "date_from":   "2020-01-01",
    "date_to":     "2020-12-31",
    "num_results": 5,
}


async def main() -> None:
    print("=" * 60)
    print("Retriever smoke test")
    print("Query JSON:", QUERY_JSON)
    print("=" * 60, "\n")

    candidates = await retrieve(QUERY_JSON, retrieval_k=20)

    if not candidates:
        print("[result] No candidates returned. Check filters / data.")
        return

    print(f"\n{'─'*60}")
    print(f"Top-{len(candidates)} candidates (by RRF score):")
    print(f"{'─'*60}")

    for i, paper in enumerate(candidates, 1):
        title   = paper["title"][:70] + "..." if len(paper["title"]) > 70 else paper["title"]
        date    = paper.get("submitted_date", "unknown")
        cat     = paper.get("primary_category", "")
        cit     = paper.get("citation_count", "?")
        rrf     = paper.get("rrf_score", 0.0)
        print(f"[{i:>2}] RRF={rrf:.5f}  citations={cit}")
        print(f"       {title}")
        print(f"       {date}  {cat}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
