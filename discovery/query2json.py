"""
query2json.py – Convert a user's natural language query (Chinese or English)
into a structured JSON object for downstream retrieval.

Uses Qwen3 (non-thinking mode) via the local Ollama API.
The model always outputs English academic search terms regardless of the
input language, so that SPECTER2 (English-trained) can embed them correctly.

Output schema:
    {
        "topics":    ["English academic term", ...],   // 1-5 specific terms
        "category":  "cs.AI",                          // arXiv category
        "date_from": "YYYY-MM-DD",                     // null if not specified
        "date_to":   "YYYY-MM-DD",                     // null if not specified
        "num_results": 5                               // default 5
    }

Relative date resolution:
    Today's real date is injected into the system prompt so the model can
    convert expressions like "最近两年" or "last 6 months" into actual
    YYYY-MM-DD timestamps rather than leaving them as vague text.

Non-thinking mode:
    Qwen3 supports think=False via Ollama's API (direct answer, no reasoning
    trace). Faster and sufficient for simple extraction tasks like this one.
    Silently ignored by older models (e.g. qwen2.5).

Ollama API reference: https://github.com/ollama/ollama/blob/main/docs/api.md

Config (from .env):
    OLLAMA_HOST   – Ollama server URL, default http://localhost:11434
    OLLAMA_MODEL  – model tag, default qwen3:1.7b

Smoke test:
    python test/test_query2json_parse_natural_language.py
"""

import json
import os
import re
import sys
from datetime import date, timedelta
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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

# arXiv categories the model can choose from
_KNOWN_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.IR", "cs.RO",
    "math.ST", "math.OC", "physics.comp-ph", "quant-ph",
    "stat.ML", "stat.ME",
]

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ─── Prompt builder ───────────────────────────────────────────────────────────

def _build_system_prompt(today: date) -> str:
    """
    Build the system prompt with today's real date embedded so the model
    can resolve relative time expressions into concrete YYYY-MM-DD dates.

    Pre-computed anchors are shown as examples; the model is free to compute
    any offset itself as long as it always returns real date strings.
    """
    today_str      = today.isoformat()                           # 2026-10-13
    one_year_ago   = today.replace(year=today.year - 1).isoformat()  # 2025-10-13
    two_years_ago  = today.replace(year=today.year - 2).isoformat()  # 2024-10-13
    six_months_ago = (today - timedelta(days=183)).isoformat()        # 2026-04-13

    return f"""\
You are a search parameter extractor for an academic paper retrieval system.
TODAY'S DATE: {today_str}

Return ONLY a valid JSON object. No explanation, no markdown, no extra text.

Output schema (all five fields required):
{{
  "topics":      ["<English academic term>", ...],
  "category":    "<arXiv category or null>",
  "date_from":   "<YYYY-MM-DD or null>",
  "date_to":     "<YYYY-MM-DD or null>",
  "num_results": <integer>
}}

── TOPICS ───────────────────────────────────────────────────────────────────────
- 1 to 5 SPECIFIC academic concepts or method names. Translate Chinese to English.
- GOOD: "attention mechanism", "graph neural network", "reinforcement learning"
- BAD (never include these): "papers", "recent papers", "recommendation", "research",
  "articles", "find", "good", "best", "survey", "review"

── CATEGORY ─────────────────────────────────────────────────────────────────────
- ONLY set category if the user EXPLICITLY mentions a specific field or domain.
- If the query is general ML/AI without a specific domain focus → null.
- If you are not confident → null (null is always safe; wrong category kills results).

  User says "robotics" / "robot arm"         → cs.RO
  User says "image" / "video" / "vision"     → cs.CV
  User says "NLP" / "text" / "language"      → cs.CL
  User says "reinforcement learning"          → cs.AI
  User says "attention" / "transformer"       → null  (cross-domain, don't filter)
  User says "graph neural network"            → null  (appears in many categories)
  General query without domain restriction    → null

- DO NOT default to cs.AI just because the topic sounds like AI.
  Most ML topics appear across cs.AI, cs.LG, cs.CV, cs.CL simultaneously.

── DATE_FROM / DATE_TO ──────────────────────────────────────────────────────────
RULE 1: If the query mentions NO time or date at all → BOTH must be null.
  "recommend papers on LLM alignment"  → null / null  ← NO DATE MENTIONED
  "find papers about attention"        → null / null

RULE 2: If a time expression IS mentioned → use {today_str} as reference:
  "2020年" / "year 2020"          → "2020-01-01" / "2020-12-31"
  "2019–2021"                     → "2019-01-01" / "2021-12-31"
  "最近半年" / "last 6 months"    → "{six_months_ago}" / "{today_str}"
  "最近一年" / "last year"        → "{one_year_ago}" / "{today_str}"
  "最近两年" / "last 2 years"     → "{two_years_ago}" / "{today_str}"
  "最近" / "recent" (standalone)  → "{one_year_ago}" / "{today_str}"
ALWAYS output real YYYY-MM-DD strings, never vague phrases.

── NUM_RESULTS ───────────────────────────────────────────────────────────────────
- Extract if explicitly mentioned ("10篇", "find 5 papers", "give me 20").
- Default to 5 if not mentioned.
"""


# ─── Output validation ────────────────────────────────────────────────────────

_REQUIRED_FIELDS = {"topics", "category", "date_from", "date_to", "num_results"}


def _validate(data: dict) -> dict:
    """
    Validate and normalise the parsed JSON dict.
    Raises ValueError if the structure is unrecoverable.
    """
    missing = _REQUIRED_FIELDS - data.keys()
    if missing:
        raise ValueError(f"Missing fields in model output: {missing}")

    # topics: non-empty list of strings
    if not isinstance(data["topics"], list) or not data["topics"]:
        raise ValueError("'topics' must be a non-empty list")
    data["topics"] = [str(t) for t in data["topics"]]

    # category: null means "no filter" – only keep if it's a known valid value
    cat = data.get("category")
    if cat in (None, "null", "", "None"):
        data["category"] = None
    elif cat not in _KNOWN_CATEGORIES:
        # Model returned something unrecognised; safer to drop it than filter wrongly
        data["category"] = None

    # num_results: positive integer
    try:
        data["num_results"] = max(1, int(data["num_results"]))
    except (TypeError, ValueError):
        data["num_results"] = 5

    # date fields: either a valid YYYY-MM-DD string or Python None
    for field in ("date_from", "date_to"):
        val = data[field]
        if val in (None, "null", "", "None"):
            data[field] = None
        else:
            val = str(val).strip()
            if not _DATE_RE.match(val):
                # Model returned something like "two years ago" despite instructions.
                # Reject this field and set to None rather than passing garbage downstream.
                data[field] = None
            else:
                data[field] = val

    return data


# ─── Ollama API call ──────────────────────────────────────────────────────────

def _call_ollama(user_query: str, today: date, timeout: float = 120.0) -> str:
    """
    Send a chat request to Ollama and return the raw assistant content string.

    Args:
        user_query: raw user input (Chinese or English).
        today:      reference date for resolving relative time expressions.
        timeout:    httpx request timeout in seconds.

    Returns:
        Raw text content from the model (should be a JSON string).

    Raises:
        RuntimeError: if Ollama is unreachable or the model is not downloaded.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model":  OLLAMA_MODEL,
        "stream": False,
        "think":  False,        # Qwen3 non-thinking mode; ignored by qwen2.5
        "options": {
            "temperature": 0.1, # low temperature → deterministic, structured output
        },
        "messages": [
            {"role": "system", "content": _build_system_prompt(today)},
            {"role": "user",   "content": user_query},
        ],
    }

    try:
        resp = httpx.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_HOST}. "
            "Make sure Ollama is running (open the Ollama app or run 'ollama serve')."
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"Model '{OLLAMA_MODEL}' not found. "
                f"Download it first:  ollama pull {OLLAMA_MODEL}"
            )
        raise

    return resp.json()["message"]["content"]


# ─── Public interface ─────────────────────────────────────────────────────────

def parse_query(user_query: str, today: date | None = None) -> dict:
    """
    Convert a natural-language query (Chinese or English) into a structured
    search-parameter dict.

    Args:
        user_query: raw user input,
                    e.g. "帮我找最近两年关于attention机制的AI论文"
        today:      override the reference date (useful for testing).
                    Defaults to the actual current date.

    Returns:
        {
            "topics":      ["attention mechanism", "self-attention"],
            "category":    "cs.AI",
            "date_from":   "2024-10-13",   ← real dates, never vague text
            "date_to":     "2026-10-13",
            "num_results": 5
        }

    Raises:
        RuntimeError: Ollama unreachable or model missing.
        ValueError:   Model returned unparseable output.
    """
    if today is None:
        today = date.today()

    raw = _call_ollama(user_query, today)

    # Strip accidental markdown fences the model might still produce
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    # Parse JSON; fall back to extracting the first {...} block
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"Model returned non-JSON output:\n{raw[:400]}")
        data = json.loads(cleaned[start:end])

    return _validate(data)
