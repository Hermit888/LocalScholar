"""
Smoke test: fetch a small batch of paper metadata from the arXiv API.

Requests 5 cs.AI papers from 2024 and prints title, id, date, categories,
authors, and a short abstract. Hits the live arXiv API (about 3 seconds
between requests).

Run from the project root:
    python test/test_arxiv_client_fetch_papers.py
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from ingestion.arxiv_client import ArxivClient

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


async def main() -> None:
    """Fetch 5 cs.AI papers from 2024 and print a summary."""
    client = ArxivClient(request_delay=3.0)

    print("Fetching cs.AI papers (max 5, year 2024)...\n")
    papers, result = await client.fetch_papers(
        categories=["cs.AI"],
        date_from=date(2024, 1, 1),
        date_to=date(2024, 12, 31),
        max_papers=5,
        on_progress=lambda fetched, total: print(f"  progress: {fetched}/{total}"),
    )

    print(
        f"\narXiv total: {result.total_found}  |  fetched: {result.fetched}"
        + ("  (truncated)" if result.truncated else "")
    )
    print()

    for i, paper in enumerate(papers, 1):
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += "..."
        print(f"[{i}] {paper.title}")
        print(f"     ID: {paper.arxiv_id}v{paper.version}  |  date: {paper.submitted_date}")
        print(f"     categories: {', '.join(paper.categories)}")
        print(f"     authors: {authors}")
        print(f"     abstract: {paper.abstract[:120]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
