"""
summary_retrieval.py – Single-document lightweight summarization for the Discovery Layer.

Position in the pipeline:
    query2json → retriever → reranker → [HERE] summary_retrieval → return to user

For each reranked paper, calls Qwen3.5-2B (non-thinking mode, fast) via Ollama to
generate a 2–3 sentence plain-English summary of the paper, then formats a display
card containing all metadata the user needs to decide whether to read the paper:

    Display card per paper:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Title                                                            │
    │ Authors (first 3) · Year · Publication status                   │
    │ arXiv: <abs_url>                                                 │
    │                                                                  │
    │ Summary: <2–3 sentence plain-English summary>                    │
    └──────────────────────────────────────────────────────────────────┘

Why Discovery Layer, not QA Layer?
    Discovery Layer: user hasn't selected papers yet – a per-paper summary helps
    them scan the list and decide which are relevant.
    QA Layer: user has selected papers and asks cross-document questions – that
    requires multi-document synthesis, which is a separate task.

Publication status logic:
    venue (Semantic Scholar) → e.g. "NeurIPS", "IEEE TPAMI"
    journal_ref (arXiv)      → e.g. "ICML 2020"
    doi (arXiv)              → "Published (DOI: 10.xxxx/...)"
    none of the above        → "arXiv preprint only"

Model: Qwen3.5-2B via Ollama, non-thinking mode (think=False).
  Same model as query2json – no extra download needed.
  One Ollama call per paper; for 5 papers ≈ 10–25 seconds total on CPU.

Config (from .env):
    OLLAMA_HOST  – Ollama server URL
    OLLAMA_MODEL – model tag (qwen3.5:2b)

Smoke test:
    python test/test_summary_retrieval_single_paper.py
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Config ───────────────────────────────────────────────────────────────────

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")

# ─── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a scientific paper summarizer. Given a paper title and abstract, write a
concise 2–3 sentence summary in plain English that:
- States the core research problem and the paper's main contribution
- Mentions the key method or approach used
- States the main result or finding (quantitative if mentioned)

Rules:
- Write in English only, regardless of input language
- Do NOT start with "This paper", "The paper", or "The authors" – vary the opening
- Do NOT include citations, formulas, or LaTeX
- Output ONLY the summary text – no labels, no bullet points, no extra text
"""


def _summarize_one(title: str, abstract: str, timeout: float = 90.0) -> str:
    """
    Call Ollama to generate a 2–3 sentence summary of a single paper.

    Args:
        title:    paper title
        abstract: paper abstract
        timeout:  httpx request timeout in seconds

    Returns:
        Summary string (2–3 plain-English sentences).

    Raises:
        RuntimeError: if Ollama is unreachable or model is not downloaded.
    """
    user_content = f"Title: {title}\n\nAbstract: {abstract}"

    payload = {
        "model":  OLLAMA_MODEL,
        "stream": False,
        "think":  False,        # non-thinking mode: fast, no reasoning trace
        "options": {
            "temperature": 0.3,  # slightly higher than query2json for more natural prose
        },
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    }

    try:
        resp = httpx.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_HOST}. "
            "Make sure Ollama is running."
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}"
            )
        raise

    return resp.json()["message"]["content"].strip()


# ─── Publication status ───────────────────────────────────────────────────────

def _publication_status(paper: dict) -> str:
    """
    Determine publication status from available metadata fields.

    Priority:
      1. venue (Semantic Scholar) – most reliable, e.g. "NeurIPS", "ICML"
      2. journal_ref (arXiv)      – raw string, e.g. "ICML 2020 Workshop"
      3. doi (arXiv)              – present for journal papers
      4. fallback                 – arXiv preprint only
    """
    if paper.get("venue"):
        return paper["venue"]
    if paper.get("journal_ref"):
        return paper["journal_ref"]
    if paper.get("doi"):
        return f"Published (DOI: {paper['doi']})"
    return "arXiv preprint only"


# ─── Display card builder ─────────────────────────────────────────────────────

def _format_authors(authors) -> str:
    """Format author list: show first 3 authors, add 'et al.' if more."""
    if not authors:
        return "Unknown authors"
    if isinstance(authors, str):
        # Shouldn't happen but guard against it
        return authors
    shown = list(authors[:3])
    if len(authors) > 3:
        shown.append("et al.")
    return ", ".join(shown)


def _build_display_card(paper: dict, summary: str) -> str:
    """
    Build a formatted text display card for a single paper.

    Example output:
        ══ An Image is Worth 16x16 Words ══════════════════════
        Dosovitskiy A., Beyer L., Kolesnikov A., et al.
        2020 · ICLR · arXiv:2010.11929
        https://arxiv.org/abs/2010.11929

        Pure transformer applied directly to image patches achieves strong
        performance on image classification tasks without CNNs, matching or
        exceeding ResNets when pre-trained on large datasets.
        ────────────────────────────────────────────────────────
    """
    year   = str(paper.get("submitted_date", ""))[:4]
    status = _publication_status(paper)
    arxiv_id = paper.get("arxiv_id", "")
    abs_url  = paper.get("abs_url") or f"https://arxiv.org/abs/{arxiv_id}"
    authors  = _format_authors(paper.get("authors"))
    cit      = paper.get("citation_count")
    cit_str  = f" · {cit} citations" if cit else ""

    title_line = paper.get("title", "Untitled")
    sep = "═" * min(len(title_line) + 4, 64)

    return (
        f"══ {title_line}\n"
        f"{sep}\n"
        f"{authors}\n"
        f"{year} · {status}{cit_str}\n"
        f"{abs_url}\n"
        f"\n"
        f"{summary}\n"
        f"{'─' * 64}"
    )


# ─── Public interface ─────────────────────────────────────────────────────────

def summarize_papers(
    papers: list[dict],
    *,
    verbose: bool = False,
) -> list[dict]:
    """
    Generate a summary and display card for each paper.

    Args:
        papers:  list of paper dicts from reranker.rerank().
                 Each dict must have 'title' and 'abstract'.
        verbose: if True, print progress to stdout.

    Returns:
        Same list of dicts, each enriched with:
            'summary'      – 2–3 sentence plain-English summary (str)
            'display_card' – formatted display string (str)
            'pub_status'   – publication status string (str)
    """
    results = []
    total = len(papers)

    for i, paper in enumerate(papers, 1):
        title    = paper.get("title", "")
        abstract = paper.get("abstract", "")

        if verbose:
            short_title = title[:55] + "..." if len(title) > 55 else title
            print(f"  [{i}/{total}] Summarising: {short_title}", flush=True)

        try:
            summary = _summarize_one(title, abstract)
        except Exception as e:
            # Graceful fallback: use the first two sentences of the abstract
            summary = ". ".join((abstract or "No abstract available.").split(". ")[:2]) + "."
            if verbose:
                print(f"          [warn] Ollama error, using abstract fallback: {e}")

        enriched = dict(paper)
        enriched["summary"]      = summary
        enriched["pub_status"]   = _publication_status(paper)
        enriched["display_card"] = _build_display_card(paper, summary)
        results.append(enriched)

    return results
