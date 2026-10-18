"""
Smoke test: cross-encoder reranker (ms-marco-MiniLM-L-6-v2).

Does NOT require PostgreSQL, Qdrant, or Ollama.
Uses hand-crafted mock candidates so the expected ranking is known in advance.

Expected outcome:
  Paper about Vision Transformer (ViT) should rank highest for the query
  "visual transformer image classification", because its title and abstract
  directly match. The NLP paper should rank lower.

Downloads the cross-encoder model (~80 MB) from HuggingFace on first run.

Run from the project root:
    python test/test_reranker_cross_encoder.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from discovery.reranker import RERANKER_MODEL, rerank

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Mock candidates (no database needed) ─────────────────────────────────────

MOCK_CANDIDATES = [
    {
        "arxiv_id":        "2010.11929",
        "title":           "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "abstract":        (
            "While the Transformer architecture has become the de-facto standard for "
            "natural language processing tasks, its applications to computer vision remain "
            "limited. In vision, attention is either applied in conjunction with "
            "convolutional networks, or used to replace certain components of convolutional "
            "networks while keeping their overall structure in place. We show that this "
            "reliance on CNNs is not necessary and a pure transformer applied directly to "
            "sequences of image patches can perform very well on image classification tasks."
        ),
        "submitted_date":  "2020-10-22",
        "primary_category": "cs.CV",
        "citation_count":  45000,
        "venue":           "ICLR",
        "rrf_score":       0.030,
    },
    {
        "arxiv_id":        "1706.03762",
        "title":           "Attention Is All You Need",
        "abstract":        (
            "The dominant sequence transduction models are based on complex recurrent or "
            "convolutional neural networks that include an encoder and a decoder. The best "
            "performing models also connect the encoder and decoder through an attention "
            "mechanism. We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms, dispensing with recurrence and "
            "convolutions entirely."
        ),
        "submitted_date":  "2017-06-12",
        "primary_category": "cs.CL",
        "citation_count":  90000,
        "venue":           "NeurIPS",
        "rrf_score":       0.025,
    },
    {
        "arxiv_id":        "2005.12872",
        "title":           "End-to-End Object Detection with Transformers",
        "abstract":        (
            "We present a new method that views object detection as a direct set prediction "
            "problem. Our approach streamlines the detection pipeline, effectively removing "
            "the need for many hand-designed components like a non-maximum suppression "
            "procedure or anchor generation that explicitly encode our prior knowledge "
            "about the task."
        ),
        "submitted_date":  "2020-05-26",
        "primary_category": "cs.CV",
        "citation_count":  12000,
        "venue":           "ECCV",
        "rrf_score":       0.022,
    },
    {
        "arxiv_id":        "1810.04805",
        "title":           "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract":        (
            "We introduce a new language representation model called BERT, which stands for "
            "Bidirectional Encoder Representations from Transformers. Unlike recent language "
            "representation models, BERT is designed to pre-train deep bidirectional "
            "representations from unlabeled text by jointly conditioning on both left and "
            "right context in all layers."
        ),
        "submitted_date":  "2018-10-11",
        "primary_category": "cs.CL",
        "citation_count":  80000,
        "venue":           "NAACL",
        "rrf_score":       0.018,
    },
    {
        "arxiv_id":        "2012.09841",
        "title":           "Training data-efficient image transformers and distillation through attention",
        "abstract":        (
            "Recently, neural networks purely based on attention were shown to address image "
            "understanding tasks such as image classification. However, these visual "
            "transformers are pre-trained with hundreds of millions of images using a large "
            "infrastructure, thereby limiting their adoption."
        ),
        "submitted_date":  "2020-12-15",
        "primary_category": "cs.CV",
        "citation_count":  8000,
        "venue":           "ICML",
        "rrf_score":       0.015,
    },
]

QUERY = "visual transformer image classification"


def main() -> None:
    print("=" * 60)
    print("Reranker smoke test")
    print(f"Model:  {RERANKER_MODEL}")
    print(f"Query:  {QUERY!r}")
    print(f"Input:  {len(MOCK_CANDIDATES)} candidates")
    print("=" * 60, "\n")

    results = rerank(QUERY, MOCK_CANDIDATES, top_k=5)

    print(f"{'─'*60}")
    print("Reranked results (best first):")
    print(f"{'─'*60}")

    for rank, paper in enumerate(results, 1):
        title = paper["title"][:65] + "..." if len(paper["title"]) > 65 else paper["title"]
        print(f"[{rank}] rerank={paper['rerank_score']:>7.3f}  rrf={paper['rrf_score']:.4f}")
        print(f"     {title}")
        print(f"     {paper['submitted_date']}  {paper['primary_category']}"
              f"  citations={paper['citation_count']}")
        print()

    # Basic sanity check: one of the two visual-transformer papers should rank #1
    # (ViT 2010.11929 or DeiT 2012.09841 — both are valid top answers)
    top_id = results[0]["arxiv_id"]
    visual_transformer_ids = {"2010.11929", "2012.09841"}
    status = "PASS" if top_id in visual_transformer_ids else f"FAIL (got {top_id})"
    print(f"Sanity check (visual transformer ranks #1): {status}")


if __name__ == "__main__":
    main()
