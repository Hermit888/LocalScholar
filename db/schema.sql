-- =============================================================================
-- LocalScholar – Database Schema
-- =============================================================================
-- Run this file to create all tables and indexes from scratch.
--
-- Option A – pgAdmin 4:
--   1. Open pgAdmin 4 and connect to the "localscholar" database.
--   2. Click Tools → Query Tool.
--   3. Open this file (File → Open) and press F5 (Execute).
--
-- Option B – psql (command line):
--   psql -U postgres -p 5433 -d localscholar -f db/schema.sql
--
-- Option C – Python (automated):
--   python db/init_db.py
--
-- All statements use IF NOT EXISTS, so re-running is always safe.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Table: papers
-- One row per unique arXiv paper (identified by arxiv_id, no version suffix).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS papers (

    -- Primary key and unique identifier
    id                          BIGSERIAL PRIMARY KEY,
    arxiv_id                    TEXT NOT NULL UNIQUE,   -- e.g. "2301.07041" (no "v2" suffix)
    version                     INTEGER NOT NULL DEFAULT 1,

    -- Core metadata from the arXiv API
    title                       TEXT NOT NULL,
    abstract                    TEXT NOT NULL,
    authors                     TEXT[] NOT NULL,
    submitted_date              DATE NOT NULL,          -- original submission date; used for time-range filtering
    updated_date                DATE NOT NULL,          -- date of the latest version
    categories                  TEXT[] NOT NULL,        -- all category tags, e.g. {"cs.LG","cs.AI"}
    primary_category            TEXT NOT NULL,          -- primary category tag
    abs_url                     TEXT NOT NULL,          -- arXiv abstract page URL
    pdf_url                     TEXT NOT NULL,          -- PDF download URL
    doi                         TEXT,                   -- DOI if published
    journal_ref                 TEXT,                   -- journal/conference string from arXiv metadata
    comment                     TEXT,                   -- author comment (page count, code link, etc.)

    -- PDF cache state (lazy-loaded: only downloaded when the QA Layer is triggered)
    -- Allowed values: not_downloaded | downloading | cached | failed
    pdf_status                  TEXT NOT NULL DEFAULT 'not_downloaded',
    pdf_path                    TEXT,                   -- absolute local path once the PDF is cached
    pdf_cached_at               TIMESTAMPTZ,
    pdf_fail_reason             TEXT,                   -- reason string for the last download failure
    pdf_fail_count              INTEGER NOT NULL DEFAULT 0,  -- retry counter

    -- Semantic Scholar enrichment (filled asynchronously after ingestion)
    ss_paper_id                 TEXT,                   -- Semantic Scholar internal paper ID
    citation_count              INTEGER,                -- citation count (NULL until enriched)
    influential_citation_count  INTEGER,                -- highly-influential citation count
    venue                       TEXT,                   -- journal/conference name; NULL for arXiv-only preprints
    ss_enriched_at              TIMESTAMPTZ,            -- timestamp of the last Semantic Scholar fetch

    -- Housekeeping
    ingested_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_checked_at             TIMESTAMPTZ
);

-- Indexes for papers
-- Time-range filtering (most common filter in Discovery Layer)
CREATE INDEX IF NOT EXISTS idx_papers_submitted_date
    ON papers (submitted_date);

-- Exact match on primary_category
CREATE INDEX IF NOT EXISTS idx_papers_primary_category
    ON papers (primary_category);

-- Multi-category GIN index: supports  categories @> ARRAY['cs.LG']
CREATE INDEX IF NOT EXISTS idx_papers_categories
    ON papers USING GIN (categories);

-- Filter papers by PDF download state
CREATE INDEX IF NOT EXISTS idx_papers_pdf_status
    ON papers (pdf_status);

-- Sort by citation count (NULLs last)
CREATE INDEX IF NOT EXISTS idx_papers_citation_count
    ON papers (citation_count DESC NULLS LAST);

-- Find papers not yet enriched by Semantic Scholar
CREATE INDEX IF NOT EXISTS idx_papers_ss_enriched_at
    ON papers (ss_enriched_at);


-- ---------------------------------------------------------------------------
-- Table: ingestion_jobs
-- One row per user-triggered ingestion run.
-- Tracks progress in real time and supports resume after interruption.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_jobs (

    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Job configuration (parameters the user chose)
    categories      TEXT[] NOT NULL,
    date_from       DATE NOT NULL,
    date_to         DATE NOT NULL,
    max_papers      INTEGER NOT NULL,

    -- Job state
    -- Allowed values: pending | running | done | failed | cancelled
    status          TEXT NOT NULL DEFAULT 'pending',

    -- Live progress counters (updated every ~10 papers)
    total_found     INTEGER,                        -- total papers on arXiv matching the query
    processed       INTEGER NOT NULL DEFAULT 0,     -- successfully written to the DB
    skipped         INTEGER NOT NULL DEFAULT 0,     -- already existed (deduplication)
    failed_count    INTEGER NOT NULL DEFAULT 0,     -- failed to process

    -- Resume support: last successfully consumed arXiv page offset
    arxiv_offset    INTEGER NOT NULL DEFAULT 0,

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,

    -- Error details (populated only on failure)
    error_msg       TEXT
);

-- Indexes for ingestion_jobs
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status
    ON ingestion_jobs (status);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_created_at
    ON ingestion_jobs (created_at DESC);
