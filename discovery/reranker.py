"""
reranker.py – Second-stage reranking for the Discovery Layer.

Takes the top-N candidates from retriever.py and re-scores every
(query, document) pair jointly using a cross-encoder.

Why cross-encoder after RRF?
    BM25 and SPECTER2 each score query and document independently, then
    RRF merges the two rankings. A cross-encoder reads query + document
    together in one forward pass, so every attention head can relate any
    word in the query to any word in the document. This catches fine-grained
    relevance that neither individual scorer can see.

    Example: "Visual Transformer for image classification 2020"
      - BM25 may rank "Attention Is All You Need" highly (keyword overlap)
      - SPECTER2 may rank a ViT paper and a BERT paper similarly close
      - Cross-encoder correctly scores ViT >> BERT for this specific query

Model: cross-encoder/ms-marco-MiniLM-L-6-v2  (HuggingFace, ~22 MB)
    - MiniLM-L6 cross-encoder trained on MS-MARCO passage-ranking data
    - 22M parameters – 10x smaller and ~10x faster than BGE reranker-base on CPU
    - Outputs a single relevance logit; higher = more relevant
    - Downloads via HuggingFace Hub (NOT Ollama) on first run, cached afterwards
    - CPU inference on 50 candidates: ~3–8 seconds (acceptable for local use)

Input format fed to the model:
    [CLS] <topics joined by space> [SEP] <title>. <abstract> [SEP]
    Truncated to 512 tokens total.

Config:
    RERANKER_MODEL env var (optional override), default cross-encoder/ms-marco-MiniLM-L-6-v2

Smoke test:
    python test/test_reranker_cross_encoder.py
"""

import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Config ───────────────────────────────────────────────────────────────────

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# ─── Module-level model cache (load once per process) ─────────────────────────

_tokenizer = None
_model     = None


def _get_cross_encoder():
    """Load the cross-encoder on first call, return cached model afterwards."""
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"[reranker] Loading cross-encoder: {RERANKER_MODEL} ...")
        _tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        _model.eval()
        print("[reranker] Cross-encoder ready.\n")
    return _tokenizer, _model


# ─── Scoring ──────────────────────────────────────────────────────────────────

def _score_pairs(
    query: str,
    documents: list[str],
    batch_size: int = 16,
) -> list[float]:
    """
    Score a list of (query, document) pairs with the cross-encoder.

    Args:
        query:     the search query string (joined topics).
        documents: list of document strings (title + abstract).
        batch_size: how many pairs to feed the model at once.
                    Lower = less RAM; 16 is safe for CPU with 512-token inputs.

    Returns:
        List of raw logit scores, one per document (higher = more relevant).
    """
    tokenizer, model = _get_cross_encoder()

    all_scores: list[float] = []

    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start: start + batch_size]
        queries    = [query] * len(batch_docs)

        features = tokenizer(
            queries,
            batch_docs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**features).logits

        # ms-marco model has a single output neuron → squeeze to 1-D scores
        scores = logits.squeeze(-1).tolist()
        if isinstance(scores, float):
            scores = [scores]   # single-item batch edge case
        all_scores.extend(scores)

    return all_scores


# ─── Public interface ─────────────────────────────────────────────────────────

def rerank(
    query_text: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Re-score candidates with a cross-encoder and return the top_k most relevant.

    Args:
        query_text:  joined search topics, e.g. "attention mechanism transformer".
        candidates:  list of paper dicts from retrieve(), each must have
                     'title' and 'abstract' fields.
        top_k:       number of papers to return (typically == num_results from query_json).

    Returns:
        List of paper dicts, sorted by rerank_score descending, length <= top_k.
        Each dict gains a 'rerank_score' field (raw cross-encoder logit).
    """
    if not candidates:
        return []

    # Build document strings: "title. abstract" – the period helps the model
    # parse where the title ends and the abstract begins.
    documents = [
        f"{c['title']}. {c.get('abstract') or ''}"
        for c in candidates
    ]

    scores = _score_pairs(query_text, documents)

    # Attach scores and sort
    scored = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for candidate, score in scored[:top_k]:
        paper = dict(candidate)
        paper["rerank_score"] = round(float(score), 4)
        results.append(paper)

    return results
