"""
Smoke test: single-document summarization via summary_retrieval.py.

Uses two hard-coded paper dicts (no database needed) to verify:
  1. Ollama generates a 2–3 sentence summary in English
  2. Publication status is inferred correctly (venue > journal_ref > doi > preprint)
  3. Display card is formatted without errors

Requires Ollama running with qwen3.5:2b (or whichever model is in .env).

Run from the project root:
    python test/test_summary_retrieval_single_paper.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from discovery.summary_retrieval import OLLAMA_MODEL, summarize_papers

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Mock papers (no database needed) ─────────────────────────────────────────

MOCK_PAPERS = [
    {
        "arxiv_id":        "2010.11929",
        "title":           "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "abstract":        (
            "While the Transformer architecture has become the de-facto standard for natural "
            "language processing tasks, its applications to computer vision remain limited. "
            "In vision, attention is either applied in conjunction with convolutional networks, "
            "or used to replace certain components of convolutional networks. We show that this "
            "reliance on CNNs is not necessary and a pure transformer applied directly to "
            "sequences of image patches can perform very well on image classification tasks. "
            "When pre-trained on large amounts of data and transferred to multiple mid-sized "
            "or small image recognition benchmarks, Vision Transformer (ViT) attains excellent "
            "results compared to state-of-the-art CNNs while requiring substantially fewer "
            "computational resources to train."
        ),
        "authors":         ["Dosovitskiy, A.", "Beyer, L.", "Kolesnikov, A.", "Weissenborn, D.", "Zhai, X."],
        "submitted_date":  "2020-10-22",
        "primary_category": "cs.CV",
        "abs_url":         "https://arxiv.org/abs/2010.11929",
        "doi":             None,
        "journal_ref":     None,
        "citation_count":  45000,
        "venue":           "ICLR",           # ← should show "ICLR"
        "rerank_score":    4.611,
        "rrf_score":       0.0300,
    },
    {
        "arxiv_id":        "2001.08361",
        "title":           "Scaling Laws for Neural Language Models",
        "abstract":        (
            "We study empirical scaling laws for language model performance on the cross-entropy "
            "loss. The loss scales as a power-law with model size, dataset size, and the amount "
            "of compute used for training, with some trends spanning more than seven orders of "
            "magnitude. Other architectural details such as network width or depth have minimal "
            "effects within a wide range. Simple equations govern the dependence of overfitting "
            "on model/dataset size and the dependence of training speed on model size. These "
            "relationships allow us to determine the optimal allocation of a fixed compute budget."
        ),
        "authors":         ["Kaplan, J.", "McCandlish, S.", "Henighan, T."],
        "submitted_date":  "2020-01-22",
        "primary_category": "cs.LG",
        "abs_url":         "https://arxiv.org/abs/2001.08361",
        "doi":             None,
        "journal_ref":     None,           # ← no venue or journal → "arXiv preprint only"
        "citation_count":  8500,
        "venue":           None,
        "rerank_score":    3.201,
        "rrf_score":       0.0250,
    },
]


def main() -> None:
    print("=" * 64)
    print("Summary Retrieval smoke test")
    print(f"Model:  {OLLAMA_MODEL}  (non-thinking mode)")
    print(f"Papers: {len(MOCK_PAPERS)}")
    print("=" * 64, "\n")

    results = summarize_papers(MOCK_PAPERS, verbose=True)

    print()
    for paper in results:
        print(paper["display_card"])
        print()

    # Basic checks
    passed = 0
    checks = [
        ("ViT pub_status == ICLR",         results[0]["pub_status"] == "ICLR"),
        ("Scaling Laws pub_status preprint", "preprint" in results[1]["pub_status"].lower()),
        ("ViT summary non-empty",           len(results[0]["summary"]) > 20),
        ("Scaling Laws summary non-empty",  len(results[1]["summary"]) > 20),
    ]
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {label}")

    print(f"\n{passed}/{len(checks)} checks passed.")


if __name__ == "__main__":
    main()
