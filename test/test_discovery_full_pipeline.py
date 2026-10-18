"""
Smoke test: full Discovery Layer pipeline end-to-end.

Requires ALL components running:
  - Ollama (with qwen3.5:2b configured in .env)
  - PostgreSQL (with 2020 cs.AI papers ingested)
  - Qdrant (with vectors embedded via ingestion/embedder.py)
  - SPECTER2 model (~440 MB, cached after first run)
  - Cross-encoder reranker model (~22 MB, cached after first run)

Queries are chosen to match the actual 2020 cs.AI corpus in the database.
Each query goes through the full pipeline:
  query2json → retriever (BM25 + vector + RRF) → reranker → summary_retrieval

Run from the project root:
    python test/test_discovery_full_pipeline.py
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from discovery.discovery import discover

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Fixed reference date so relative expressions are reproducible
REF_DATE = date(2026, 10, 15)

# Queries designed to match the 2020 cs.AI corpus in the database
TEST_QUERIES = [
    "knowledge graph embedding and reasoning 2020",
    "explainable artificial intelligence methods",
]


def _print_results(results: list[dict], query: str) -> None:
    print(f"\n{'═' * 65}")
    print(f"Query: {query!r}")
    print(f"{'═' * 65}")

    if not results:
        print("  (no results returned)")
        return

    for rank, paper in enumerate(results, 1):
        print(f"\n[#{rank}]  rerank={paper.get('rerank_score', 0):.3f}"
              f"  rrf={paper.get('rrf_score', 0):.5f}")

        # Print display card if summarization ran, otherwise print raw metadata
        if "display_card" in paper:
            print(paper["display_card"])
        else:
            title    = paper["title"][:70] + "..." if len(paper["title"]) > 70 else paper["title"]
            abstract = (paper.get("abstract") or "")[:120].replace("\n", " ") + "..."
            print(f"  Title   : {title}")
            print(f"  Date    : {paper.get('submitted_date', '—')}")
            print(f"  Category: {paper.get('primary_category', '—')}")
            print(f"  Citations:{paper.get('citation_count', '—')}")
            print(f"  Abstract: {abstract}")


async def main() -> None:
    print("=" * 65)
    print("Full Discovery Pipeline smoke test")
    print(f"Reference date : {REF_DATE}")
    print(f"Summarization  : enabled (Qwen3.5-2B non-thinking mode)")
    print("=" * 65)

    for query in TEST_QUERIES:
        try:
            # summarize=True: full pipeline including per-paper summaries
            results = await discover(query, today=REF_DATE, summarize=True)
            _print_results(results, query)
        except Exception as e:
            print(f"\n[ERROR] Query: {query!r}")
            print(f"        {type(e).__name__}: {e}")

    print(f"\n{'─' * 65}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
