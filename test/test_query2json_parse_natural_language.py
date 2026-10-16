"""
Smoke test: parse a natural-language query (Chinese or English) into JSON.

Covers explicit years, relative ranges ("最近两年", "last year"), and queries
with no date. Requires Ollama running and the model named in .env.

Run from the project root:
    python test/test_query2json_parse_natural_language.py
"""

import json
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from discovery.query2json import OLLAMA_HOST, OLLAMA_MODEL, parse_query

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    # Fixed reference date so relative expressions produce predictable output.
    ref_date = date(2026, 10, 13)

    test_cases = [
        ("Explicit year (EN)", "find AI papers from 2020 about attention mechanism"),
        ("Relative date (ZH)", "帮我找最近两年关于注意力机制的论文，要10篇"),
        ("Relative date (EN)", "recent graph neural network papers for traffic prediction"),
        ("No date (ZH)",       "推荐几篇关于大语言模型对齐的论文"),
        ("Half-year (ZH)",     "最近半年有哪些关于强化学习的好文章"),
    ]

    print(f"[model]     {OLLAMA_MODEL}  (non-thinking mode)")
    print(f"[host]      {OLLAMA_HOST}")
    print(f"[ref date]  {ref_date}\n")

    for label, query in test_cases:
        print(f"── {label}")
        print(f"   Input : {query}")
        try:
            result = parse_query(query, today=ref_date)
            print(
                f"   Output: {json.dumps(result, ensure_ascii=False, indent=10)}"
                .replace("\n          ", "\n           ")
            )
        except Exception as e:
            print(f"   Error : {e}")
        print()


if __name__ == "__main__":
    main()
