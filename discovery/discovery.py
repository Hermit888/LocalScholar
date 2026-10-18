"""
discovery.py – Discovery Layer orchestrator.

Wires together the three components of the Discovery Layer:

    User query (Chinese or English)
        │
        ▼
    query2json.parse_query()          ← Ollama / Qwen3 (non-thinking mode)
        │  structured JSON
        ▼
    retriever.retrieve()              ← SPECTER2 adhoc_query + BM25 + RRF
        │  top-50 candidates
        ▼
    reranker.rerank()                 ← cross-encoder ms-marco-MiniLM
        │  top-N final papers
        ▼
    List[dict]  (returned to caller)

Each returned paper dict contains:
    arxiv_id         str
    title            str
    abstract         str
    submitted_date   str  (ISO 8601, e.g. "2020-06-15")
    primary_category str  (e.g. "cs.AI")
    citation_count   int | None
    venue            str | None
    rrf_score        float  (RRF fusion score, for debugging)
    rerank_score     float  (cross-encoder logit, higher = more relevant)

Usage:
    import asyncio
    from discovery.discovery import discover

    results = asyncio.run(discover("帮我找2020年关于attention机制的论文"))
    for paper in results:
        print(paper["title"], paper["rerank_score"])

Config (all from .env):
    OLLAMA_HOST / OLLAMA_MODEL   – query understanding model
    POSTGRES_*                   – paper database
    QDRANT_HOST / QDRANT_PORT    – vector store
    SPECTER2_BASE                – encoder model ID
    RERANKER_MODEL               – cross-encoder model ID

Smoke test:
    python test/test_discovery_full_pipeline.py
"""

import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from discovery.query2json        import parse_query
from discovery.query2json        import OLLAMA_MODEL
from discovery.reranker          import rerank
from discovery.retriever         import retrieve
from discovery.summary_retrieval import summarize_papers

# How many candidates the retriever returns to the reranker.
# Should be comfortably larger than the max num_results (default 5).
_RETRIEVAL_K = 50


async def discover(
    user_query: str,
    today: date | None = None,
    summarize: bool = True,
) -> list[dict]:
    """
    End-to-end discovery pipeline: natural language → ranked paper list with summaries.

    Pipeline:
        query2json  →  retriever (BM25 + vector + RRF)  →  reranker  →  summary_retrieval

    Args:
        user_query: raw user input in Chinese or English.
                    e.g. "最近两年有哪些关于图神经网络的好论文"
        today:      reference date for resolving relative time expressions.
                    Defaults to the real current date.
        summarize:  if True (default), generate a per-paper summary via Qwen3.5-2B.
                    Set to False for faster results without summaries (e.g. for testing).

    Returns:
        List of paper dicts ordered by relevance (best first).
        When summarize=True, each dict also contains:
            'summary'      – 2–3 sentence plain-English summary
            'display_card' – formatted display string ready to print/render
            'pub_status'   – publication status (venue, journal, or "arXiv preprint only")

    Raises:
        RuntimeError: if Ollama, PostgreSQL, or Qdrant is unreachable.
        ValueError:   if the LLM returns unparseable output.
    """
    # ── Step 1: parse natural language → structured query ─────────────────────
    print(f"\n[discover] Query: {user_query!r}")
    query_json = parse_query(user_query, today=today)
    print(f"[discover] Parsed: {query_json}")

    # ── Step 2: retrieve top-K candidates (BM25 + vector + RRF) ──────────────
    candidates = await retrieve(query_json, retrieval_k=_RETRIEVAL_K)

    if not candidates:
        print("[discover] No candidates found. Returning empty list.")
        return []

    # ── Step 3: rerank candidates with cross-encoder ──────────────────────────
    query_text = " ".join(query_json["topics"])
    top_k      = query_json["num_results"]

    final = rerank(query_text, candidates, top_k=top_k)
    print(f"[discover] Reranked: top-{len(final)} papers selected.")

    # ── Step 4: generate per-paper summaries (Discovery Layer output) ─────────
    if summarize:
        print(f"[discover] Summarising {len(final)} papers via {OLLAMA_MODEL} ...")
        final = summarize_papers(final, verbose=True)

    print(f"[discover] Done.\n")
    return final
