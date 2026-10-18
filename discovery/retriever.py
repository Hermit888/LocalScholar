"""
retriever.py – First-stage retrieval for the Discovery Layer.

Given a parsed query dict (output of query2json.parse_query), returns up to
`retrieval_k` candidate papers by fusing two complementary signals:

  1. Semantic vector search via Qdrant + SPECTER2 adhoc_query adapter
     – Captures meaning: "neural network" finds "deep learning" papers too.

  2. BM25 keyword search via rank_bm25 (in-memory, rebuilt each call)
     – Captures exact terms: author names, model names, acronyms (ViT, GPT).

The two ranked lists are merged with Reciprocal Rank Fusion (RRF):
    score(paper) = Σ  1 / (60 + rank_i)
where rank_i is the paper's rank in each individual list (1-indexed).
RRF is parameter-free and robust to score-scale differences between sources.

Data flow:
  PostgreSQL  →  filtered papers (category + date WHERE clause)
                    ↓              ↓
              BM25Okapi        SPECTER2 adhoc_query → Qdrant (has_id filter)
                    ↓              ↓
                  top-50        top-50
                    └──── RRF ────┘
                         ↓
                  top-retrieval_k candidates  →  reranker.py

SPECTER2 adapter choice:
  - embedder.py uses  "proximity"    (document side: title + SEP + abstract)
  - retriever.py uses "adhoc_query"  (query side: joined topics only)
  Loading the wrong adapter gives significantly worse retrieval quality.

Config (from .env):
    POSTGRES_*      – standard PostgreSQL connection variables
    QDRANT_HOST / QDRANT_PORT
    SPECTER2_BASE   – HuggingFace model id, default allenai/specter2_base

Models are cached at module level – loaded once per process, not per query.

Smoke test:
    python test/test_retriever_vector_and_bm25.py
"""

import os
import sys
import uuid
from datetime import date as date_type
from pathlib import Path

import asyncpg
import torch
from adapters import AutoAdapterModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, HasIdCondition
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Config ───────────────────────────────────────────────────────────────────

SPECTER2_BASE    = os.getenv("SPECTER2_BASE", "allenai/specter2_base")
SPECTER2_ADAPTER = "allenai/specter2"
COLLECTION_NAME  = "papers_abstract"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Namespace must match embedder.py so point IDs are identical
_UUID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# RRF rank-smoothing constant (standard value; higher = gentler rank penalty)
_RRF_K = 60

# ─── Module-level model cache (load once per process) ─────────────────────────

_tokenizer = None
_model     = None


def _get_specter2():
    """Load SPECTER2 + adhoc_query adapter on first call, return cached afterwards."""
    global _tokenizer, _model
    if _tokenizer is None:
        print("[retriever] Loading SPECTER2 tokenizer ...")
        _tokenizer = AutoTokenizer.from_pretrained(SPECTER2_BASE)

        print("[retriever] Loading SPECTER2 base model ...")
        _model = AutoAdapterModel.from_pretrained(SPECTER2_BASE)

        print("[retriever] Loading adhoc_query adapter ...")
        _model.load_adapter(
            SPECTER2_ADAPTER,
            source="hf",
            load_as="adhoc_query",
            set_active=True,
        )
        _model.eval()
        print("[retriever] SPECTER2 (adhoc_query) ready.\n")

    return _tokenizer, _model


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _arxiv_id_to_point_id(arxiv_id: str) -> str:
    """Deterministic UUID5 – must match embedder.py exactly."""
    return str(uuid.uuid5(_UUID_NS, arxiv_id))


def _get_dsn() -> str:
    user     = os.getenv("POSTGRES_USER", "").strip()
    password = os.getenv("POSTGRES_PASSWORD", "").strip()
    host     = os.getenv("POSTGRES_HOST", "localhost")
    port     = os.getenv("POSTGRES_PORT", "5433")
    db       = os.getenv("POSTGRES_DB", "localscholar")
    if not user or not password:
        raise RuntimeError("POSTGRES_USER or POSTGRES_PASSWORD not set in .env")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


# ─── PostgreSQL fetch ─────────────────────────────────────────────────────────

async def _fetch_papers(
    category:  str | None,
    date_from: str | None,
    date_to:   str | None,
) -> list[dict]:
    """
    Fetch papers from PostgreSQL filtered by category and/or date range.
    Returns fresh data every call (no in-memory stale state).

    The result set is the BM25 search space AND the Qdrant has_id filter,
    so only papers matching the filter can ever appear in the final ranking.
    """
    conditions: list[str] = []
    params:     list      = []

    if category:
        params.append(category)
        conditions.append(f"primary_category = ${len(params)}")

    if date_from:
        # asyncpg requires datetime.date objects, not ISO strings
        params.append(date_type.fromisoformat(date_from))
        conditions.append(f"submitted_date >= ${len(params)}")

    if date_to:
        params.append(date_type.fromisoformat(date_to))
        conditions.append(f"submitted_date <= ${len(params)}")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT
            arxiv_id,
            title,
            abstract,
            authors,
            submitted_date,
            primary_category,
            abs_url,
            doi,
            journal_ref,
            citation_count,
            venue
        FROM papers
        {where}
        ORDER BY submitted_date DESC
    """

    conn = await asyncpg.connect(_get_dsn())
    try:
        rows = await conn.fetch(sql, *params)
    finally:
        await conn.close()

    return [dict(r) for r in rows]


# ─── BM25 search ──────────────────────────────────────────────────────────────

def _bm25_search(papers: list[dict], topics: list[str], top_k: int) -> list[str]:
    """
    Build a BM25Okapi index over `papers` and return the top_k arxiv_ids
    ordered by BM25 score (best first).

    Corpus: title + abstract (lower-cased, whitespace-tokenised).
    Query: joined topics, same pre-processing.
    """
    if not papers:
        return []

    corpus      = [f"{p['title']} {p['abstract'] or ''}".lower().split() for p in papers]
    bm25        = BM25Okapi(corpus)
    query_toks  = " ".join(topics).lower().split()
    scores      = bm25.get_scores(query_toks)

    # Pair each paper with its BM25 score, sort descending
    ranked = sorted(zip(papers, scores), key=lambda x: x[1], reverse=True)
    return [p["arxiv_id"] for p, _ in ranked[:top_k]]


# ─── Vector search ────────────────────────────────────────────────────────────

def _vector_search(
    topics:   list[str],
    point_ids: list[str],
    top_k:    int,
) -> list[str]:
    """
    Encode the query topics with SPECTER2 adhoc_query adapter, then search
    Qdrant restricted to `point_ids` (the papers that passed the PostgreSQL
    date/category filter).

    Returns the top_k arxiv_ids ordered by cosine similarity (best first).
    """
    if not point_ids:
        return []

    tokenizer, model = _get_specter2()

    # Query input for adhoc_query: just the joined topics, no sep token
    query_text = " ".join(topics)
    inputs = tokenizer(
        query_text,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    with torch.no_grad():
        output = model(**inputs)

    query_vector = output.last_hidden_state[:, 0, :].squeeze(0).tolist()

    # Restrict Qdrant search to the filtered point IDs.
    # qdrant-client >= 1.9 uses query_points() instead of the removed search().
    qdrant   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(must=[HasIdCondition(has_id=point_ids)]),
        limit=top_k,
        with_payload=True,
    )

    return [hit.payload["arxiv_id"] for hit in response.points]


# ─── RRF fusion ───────────────────────────────────────────────────────────────

def _rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = _RRF_K,
) -> list[str]:
    """
    Reciprocal Rank Fusion over multiple ranked lists of arxiv_ids.

    score(paper) = Σ  1 / (k + rank_i)   where rank_i is 1-indexed.

    Papers that appear in more lists (or at higher ranks) accumulate more
    score. Papers absent from a list contribute nothing from that list.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, arxiv_id in enumerate(ranked, start=1):
            scores[arxiv_id] = scores.get(arxiv_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores, key=scores.__getitem__, reverse=True)


# ─── Public interface ─────────────────────────────────────────────────────────

async def retrieve(
    query_json: dict,
    *,
    retrieval_k: int = 50,
) -> list[dict]:
    """
    First-stage retrieval: BM25 + SPECTER2 vector search fused with RRF.

    Args:
        query_json:   output of parse_query(), with keys:
                          topics, category, date_from, date_to, num_results
        retrieval_k:  number of candidate papers to return (before reranking).
                      Should be >= num_results to give the reranker room to work.

    Returns:
        List of paper dicts ordered by RRF score (descending), length <= retrieval_k.
        Each dict contains: arxiv_id, title, abstract, submitted_date,
        primary_category, citation_count, venue, rrf_score.
    """
    topics    = query_json.get("topics", [])
    category  = query_json.get("category")
    date_from = query_json.get("date_from")
    date_to   = query_json.get("date_to")

    if not topics:
        raise ValueError("query_json must contain at least one topic.")

    # 1. Fetch filtered papers from PostgreSQL (fresh every call)
    papers = await _fetch_papers(category, date_from, date_to)
    if not papers:
        print("[retriever] No papers found for the given filters.")
        return []

    print(f"[retriever] {len(papers)} papers match filters "
          f"(category={category}, {date_from} – {date_to})")

    # Build a lookup by arxiv_id for fast result assembly
    paper_by_id = {p["arxiv_id"]: p for p in papers}

    # Convert arxiv_ids → Qdrant point UUIDs (for has_id filter)
    point_ids = [_arxiv_id_to_point_id(aid) for aid in paper_by_id]

    # 2. BM25 keyword search (in-memory, on the filtered subset)
    bm25_ranked  = _bm25_search(papers, topics, top_k=retrieval_k)
    print(f"[retriever] BM25: top-{len(bm25_ranked)} retrieved")

    # 3. SPECTER2 vector search (Qdrant, restricted to filtered point IDs)
    vec_ranked   = _vector_search(topics, point_ids, top_k=retrieval_k)
    print(f"[retriever] Vector: top-{len(vec_ranked)} retrieved")

    # 4. RRF fusion
    fused_ids = _rrf_fuse([bm25_ranked, vec_ranked])[:retrieval_k]

    # Compute RRF scores for each result (for transparency / debugging)
    rrf_scores: dict[str, float] = {}
    for rank_list in [bm25_ranked, vec_ranked]:
        for rank, arxiv_id in enumerate(rank_list, start=1):
            rrf_scores[arxiv_id] = rrf_scores.get(arxiv_id, 0.0) + 1.0 / (_RRF_K + rank)

    # 5. Assemble final result list
    results = []
    for arxiv_id in fused_ids:
        if arxiv_id not in paper_by_id:
            continue   # Qdrant returned a point not in our PostgreSQL filter (shouldn't happen)
        paper = dict(paper_by_id[arxiv_id])
        paper["rrf_score"] = round(rrf_scores.get(arxiv_id, 0.0), 6)
        # Normalise submitted_date to ISO string for JSON serialisability
        if paper.get("submitted_date") and not isinstance(paper["submitted_date"], str):
            paper["submitted_date"] = paper["submitted_date"].isoformat()
        results.append(paper)

    print(f"[retriever] RRF fused: top-{len(results)} candidates")
    return results
