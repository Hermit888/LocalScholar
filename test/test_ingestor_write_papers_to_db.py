"""
Smoke test: fetch cs.AI papers from 2020 and write them to PostgreSQL.

Caps the run at 500 papers. Papers from 2020 have had several years to
accumulate citations and venue data. Requires PostgreSQL from .env.
Duplicates are skipped by arxiv_id.

Run from the project root:
    python test/test_ingestor_write_papers_to_db.py
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from ingestion.ingestor import ingest

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


async def main() -> None:
    await ingest(
        categories=["cs.AI"],
        date_from=date(2020, 1, 1),
        date_to=date(2020, 12, 31),
        max_papers=500,
    )


if __name__ == "__main__":
    asyncio.run(main())
