"""
Ingestion pipeline: fetch papers from arXiv and write them to PostgreSQL.

- Deduplicates by arxiv_id (skips papers already in the DB)
- Tracks every run as an ingestion_job row (supports resume on interruption)
- Prints live progress so you can follow along

Usage (quick test):
    python fetch_data/ingestor.py
"""

import asyncio
import os
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

# Load .env from project root and make sibling modules importable
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT))

from fetch_data.arxiv_client import ArxivClient, ArxivPaper

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# ─── DB helpers ───────────────────────────────────────────────────────────────

def _get_dsn() -> str:
    user     = os.getenv("POSTGRES_USER", "").strip()
    password = os.getenv("POSTGRES_PASSWORD", "").strip()
    host     = os.getenv("POSTGRES_HOST", "localhost")
    port     = os.getenv("POSTGRES_PORT", "5433")
    db       = os.getenv("POSTGRES_DB", "localscholar")
    if not user or not password:
        print("ERROR: POSTGRES_USER or POSTGRES_PASSWORD not set in .env")
        sys.exit(1)
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


# SQL: insert one paper; silently skip if arxiv_id already exists.
# RETURNING id is non-NULL only for rows that were actually inserted.
_INSERT_PAPER = """
INSERT INTO papers (
    arxiv_id, version, title, abstract, authors,
    submitted_date, updated_date,
    categories, primary_category,
    abs_url, pdf_url,
    doi, journal_ref, comment
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
ON CONFLICT (arxiv_id) DO NOTHING
RETURNING id
"""

_CREATE_JOB = """
INSERT INTO ingestion_jobs (id, categories, date_from, date_to, max_papers, status, started_at)
VALUES ($1, $2, $3, $4, $5, 'running', NOW())
RETURNING id
"""

_UPDATE_JOB_PROGRESS = """
UPDATE ingestion_jobs
SET total_found   = $2,
    processed     = $3,
    skipped       = $4,
    failed_count  = $5,
    arxiv_offset  = $6
WHERE id = $1
"""

_FINISH_JOB = """
UPDATE ingestion_jobs
SET status       = $2,
    finished_at  = NOW(),
    error_msg    = $3
WHERE id = $1
"""


# ─── Core pipeline ────────────────────────────────────────────────────────────

async def ingest(
    categories: list[str],
    date_from: date,
    date_to: date,
    max_papers: int = 100,
    request_delay: float = 3.0,
) -> dict:
    """
    Fetch up to max_papers arXiv papers and write new ones to PostgreSQL.

    Returns a summary dict:
        {"job_id", "total_found", "inserted", "skipped", "failed"}
    """
    conn = await asyncpg.connect(_get_dsn())
    job_id = str(uuid.uuid4())

    # Create the job record
    await conn.execute(_CREATE_JOB, job_id, categories, date_from, date_to, max_papers)
    print(f"[job]    {job_id}")
    print(f"[query]  categories={categories}  {date_from} -> {date_to}  max={max_papers}\n")

    client = ArxivClient(request_delay=request_delay)
    inserted = 0
    skipped  = 0
    failed   = 0
    total_found = 0
    offset   = 0      # tracks how many arXiv results we have consumed

    try:
        papers, result = await client.fetch_papers(
            categories=categories,
            date_from=date_from,
            date_to=date_to,
            max_papers=max_papers,
            on_progress=lambda f, t: print(
                f"  [arxiv]  fetched {f}/{t} ...", end="\r", flush=True
            ),
        )
        print()  # newline after the \r progress line

        total_found = result.total_found
        print(f"[arxiv]  total matching: {total_found}"
              + (f"  (showing first {max_papers})" if result.truncated else ""))
        print(f"[arxiv]  retrieved: {result.fetched}\n")

        # Write papers to DB one by one so we can count inserts vs skips
        for i, paper in enumerate(papers, 1):
            try:
                row = await conn.fetchrow(
                    _INSERT_PAPER,
                    paper.arxiv_id,
                    paper.version,
                    paper.title,
                    paper.abstract,
                    paper.authors,
                    paper.submitted_date,
                    paper.updated_date,
                    paper.categories,
                    paper.primary_category,
                    paper.abs_url,
                    paper.pdf_url,
                    paper.doi,
                    paper.journal_ref,
                    paper.comment,
                )
                if row is not None:   # RETURNING id → row was actually inserted
                    inserted += 1
                    status = "NEW "
                else:
                    skipped += 1
                    status = "SKIP"

                offset = i
                print(f"  [{status}]  ({i}/{result.fetched})  {paper.arxiv_id}  {paper.title[:60]}")

                # Persist progress every 10 papers (so a crash can be diagnosed)
                if i % 10 == 0:
                    await conn.execute(
                        _UPDATE_JOB_PROGRESS,
                        job_id, total_found, inserted, skipped, failed, offset,
                    )

            except Exception as exc:
                failed += 1
                print(f"  [FAIL]  ({i}/{result.fetched})  {paper.arxiv_id}  reason: {exc}")

        # Final progress update
        await conn.execute(
            _UPDATE_JOB_PROGRESS,
            job_id, total_found, inserted, skipped, failed, offset,
        )
        await conn.execute(_FINISH_JOB, job_id, "done", None)

    except Exception as exc:
        # Job-level failure (e.g. arXiv unreachable)
        await conn.execute(_FINISH_JOB, job_id, "failed", str(exc))
        raise

    finally:
        await conn.close()

    summary = {
        "job_id":      job_id,
        "total_found": total_found,
        "inserted":    inserted,
        "skipped":     skipped,
        "failed":      failed,
    }

    print(f"\n[done]   inserted={inserted}  skipped={skipped}  failed={failed}")
    return summary


# ─── Quick test entry point ───────────────────────────────────────────────────

async def _demo() -> None:
    """
    Ingest cs.AI papers from 2020 (newest-first, capped at 500).
    Papers from 2020 have had 5+ years to accumulate citations and venue data.
    """
    await ingest(
        categories  = ["cs.AI"],
        date_from   = date(2020, 1, 1),
        date_to     = date(2020, 12, 31),
        max_papers  = 500,
    )


if __name__ == "__main__":
    asyncio.run(_demo())
