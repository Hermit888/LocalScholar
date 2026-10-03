"""
Semantic Scholar enrichment: fill citation_count, venue, and related fields
for papers already stored in PostgreSQL.

Flow:
    PostgreSQL (ss_enriched_at IS NULL)
        → batch POST to SS /graph/v1/paper/batch  (up to 500 ArXiv IDs at once)
        → UPDATE papers SET citation_count = ..., venue = ..., ss_enriched_at = NOW()

Rate limits:
    With API key  → 10 req/s  (one batch of 500 papers every ~0.1 s)
    Without key   →  1 req/s

Note: very recent papers (last few days) may not yet be indexed by Semantic Scholar.
Those will be marked with ss_paper_id = NULL but ss_enriched_at set, so they are
not re-queried on every run. Re-run the script periodically to back-fill them.

Usage:
    python fetch_data/ss_enricher.py
"""

import asyncio
import os
import sys
from datetime import date
from pathlib import Path

import asyncpg
import httpx
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Config ───────────────────────────────────────────────────────────────────

SS_BATCH_URL       = "https://api.semanticscholar.org/graph/v1/paper/batch"
SS_FIELDS          = "paperId,externalIds,citationCount,influentialCitationCount,venue,publicationVenue"
BATCH_SIZE         = 500   # SS allows up to 500 IDs per request
DELAY_WITH_KEY     = 0.11  # ~10 req/s  (with API key)
DELAY_WITHOUT_KEY  = 1.1   # ~1  req/s  (no key – free anonymous tier)

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


_SELECT_UNENRICHED = """
SELECT arxiv_id
FROM   papers
WHERE  ss_enriched_at IS NULL
ORDER  BY submitted_date DESC
"""

_UPDATE_PAPER = """
UPDATE papers
SET
    ss_paper_id                = $2,
    citation_count             = $3,
    influential_citation_count = $4,
    venue                      = $5,
    ss_enriched_at             = NOW()
WHERE arxiv_id = $1
"""


# ─── Semantic Scholar API ─────────────────────────────────────────────────────

def _build_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        headers["x-api-key"] = key
    return headers


def _parse_venue(data: dict) -> str | None:
    """
    Extract a human-readable venue string from a SS paper record.
    Prefer publicationVenue.name (structured) over bare venue string.
    """
    pub_venue = data.get("publicationVenue") or {}
    name = pub_venue.get("name", "").strip()
    if name:
        return name
    bare = (data.get("venue") or "").strip()
    return bare if bare else None


async def _fetch_ss_batch(
    http: httpx.AsyncClient,
    arxiv_ids: list[str],
) -> dict[str, dict]:
    """
    Query SS batch API for a list of arXiv IDs.

    Returns a dict mapping arxiv_id → SS data dict.
    Papers not found in SS are absent from the result.
    """
    payload = {"ids": [f"ArXiv:{aid}" for aid in arxiv_ids]}
    resp = await http.post(
        SS_BATCH_URL,
        json=payload,
        params={"fields": SS_FIELDS},
        headers=_build_headers(),
        timeout=30.0,
    )
    resp.raise_for_status()
    results: list[dict | None] = resp.json()

    out: dict[str, dict] = {}
    for item in results:
        if item is None:
            continue
        ext = item.get("externalIds") or {}
        aid = ext.get("ArXiv")
        if aid:
            out[aid] = item
    return out


# ─── Main enrichment loop ─────────────────────────────────────────────────────

async def enrich(batch_size: int = BATCH_SIZE) -> None:
    """
    Fetch all un-enriched papers from PostgreSQL and fill SS fields.

    Works with or without a SEMANTIC_SCHOLAR_API_KEY in .env.
    Without a key the batch interval is 1.1 s instead of 0.11 s,
    which is still fast enough for hundreds of papers.
    """
    conn = await asyncpg.connect(_get_dsn())

    rows = await conn.fetch(_SELECT_UNENRICHED)
    arxiv_ids = [r["arxiv_id"] for r in rows]

    if not arxiv_ids:
        print("[info]  No papers need enrichment – all already have ss_enriched_at set.")
        await conn.close()
        return

    has_key = bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip())
    request_delay = DELAY_WITH_KEY if has_key else DELAY_WITHOUT_KEY

    print(f"[info]  {len(arxiv_ids)} papers to enrich")
    if has_key:
        print("[info]  API key: yes  (10 req/s)")
    else:
        print("[info]  API key: none (1 req/s – anonymous tier)")
        print("[info]  Tip: set SEMANTIC_SCHOLAR_API_KEY in .env for 10x faster enrichment")
        print("[info]  Apply at: https://www.semanticscholar.org/product/api#api-key-form")
    print(f"[info]  Batch size: {batch_size}\n")

    found      = 0
    not_found  = 0   # on SS but returned null (not indexed yet)
    updated    = 0

    async with httpx.AsyncClient() as http:
        for batch_start in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[batch_start: batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(arxiv_ids) + batch_size - 1) // batch_size

            print(f"  [batch {batch_num}/{total_batches}]  querying {len(batch)} papers ...",
                  end="", flush=True)

            try:
                ss_data = await _fetch_ss_batch(http, batch)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                print(f"\n  [error]  HTTP {status}")
                if status == 429:
                    # Exponential backoff: wait 30s, 60s, 120s
                    for wait in (30, 60, 120):
                        print(f"  [429]    rate limited – waiting {wait}s before retry ...")
                        await asyncio.sleep(wait)
                        try:
                            ss_data = await _fetch_ss_batch(http, batch)
                            break
                        except httpx.HTTPStatusError as e2:
                            if e2.response.status_code != 429:
                                raise
                    else:
                    print("  [error]  Still rate-limited after retries, skipping batch.")
                    await asyncio.sleep(request_delay)
                    continue
                else:
                    print(f"  [error]  {exc.response.text[:200]}")
                    await asyncio.sleep(request_delay)
                    continue

            print(f"  found {len(ss_data)}/{len(batch)}")

            # Update each paper in this batch
            for arxiv_id in batch:
                item = ss_data.get(arxiv_id)
                if item:
                    found += 1
                    venue = _parse_venue(item)
                    await conn.execute(
                        _UPDATE_PAPER,
                        arxiv_id,
                        item.get("paperId"),
                        item.get("citationCount"),
                        item.get("influentialCitationCount"),
                        venue,
                    )
                    updated += 1
                else:
                    # Paper not yet indexed by SS – still mark ss_enriched_at
                    # so we don't hammer the API on every run
                    not_found += 1
                    await conn.execute(
                        _UPDATE_PAPER,
                        arxiv_id,
                        None, None, None, None,
                    )

            await asyncio.sleep(request_delay)

    await conn.close()

    print(f"\n[done]  enriched={updated}  not_on_SS={not_found}")
    if not_found > 0:
        print(f"[note]  {not_found} papers not yet indexed by Semantic Scholar.")
        print("        Re-run this script in a few days to back-fill them.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(enrich())
