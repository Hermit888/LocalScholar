"""
arXiv API client: fetches paper metadata by subject category and date range.

arXiv API docs: https://info.arxiv.org/help/api/index.html
Rate limit: arXiv recommends >= 3 seconds between requests; this client respects that by default.
Response format: Atom XML, parsed with the stdlib xml.etree.ElementTree (no extra dependencies).
"""

import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from typing import Callable

import httpx

# ─── Constants ────────────────────────────────────────────────────────────────

ARXIV_API_BASE = "https://export.arxiv.org/api/query"

# XML namespaces used in the arXiv Atom feed
_NS = {
    "atom":       "http://www.w3.org/2005/Atom",
    "arxiv":      "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# Supported arXiv categories (used by the frontend selector)
AVAILABLE_CATEGORIES = {
    # Computer Science
    "cs.AI":  "CS - Artificial Intelligence",
    "cs.LG":  "CS - Machine Learning",
    "cs.CV":  "CS - Computer Vision",
    "cs.CL":  "CS - Computation and Language (NLP)",
    "cs.IR":  "CS - Information Retrieval",
    "cs.RO":  "CS - Robotics",
    # Mathematics
    "math.ST": "Math - Statistics Theory",
    "math.OC": "Math - Optimization and Control",
    # Physics
    "physics.comp-ph": "Physics - Computational Physics",
    "quant-ph":        "Quantum Physics",
    # Statistics
    "stat.ML": "Statistics - Machine Learning",
    "stat.ME": "Statistics - Methodology",
}


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ArxivPaper:
    """Paper metadata returned by the arXiv API for a single entry."""

    arxiv_id:         str         # no version suffix, e.g. "2301.07041"
    version:          int         # version number, e.g. 2
    title:            str
    abstract:         str
    authors:          list[str]
    submitted_date:   date        # original submission date
    updated_date:     date        # date of the latest version
    categories:       list[str]  # all category tags
    primary_category: str         # primary category
    abs_url:          str         # arXiv abstract page URL
    pdf_url:          str         # PDF download URL
    doi:              str | None = None
    journal_ref:      str | None = None
    comment:          str | None = None


@dataclass
class FetchResult:
    """Summary returned after fetch_papers() completes."""

    total_found:  int   # total papers on arXiv matching the query (may be >> max_papers)
    fetched:      int   # number of papers actually retrieved this run
    truncated:    bool  # True if the result was cut short by max_papers


# ─── Client ───────────────────────────────────────────────────────────────────

class ArxivClient:
    """
    Async arXiv API client.

    Usage:
        client = ArxivClient()
        papers, result = await client.fetch_papers(
            categories=["cs.AI", "cs.LG"],
            date_from=date(2023, 1, 1),
            date_to=date(2024, 12, 31),
            max_papers=500,
            on_progress=lambda fetched, total: print(f"{fetched}/{total}"),
        )
    """

    def __init__(self, request_delay: float = 3.0):
        """
        Args:
            request_delay: Minimum seconds between API requests.
                           arXiv recommends >= 3 s; lower values risk rate-limiting.
        """
        self.request_delay = request_delay
        self._last_request_time: float = 0.0

    # ── Public interface ──────────────────────────────────────────────────────

    async def fetch_papers(
        self,
        categories: list[str],
        date_from: date,
        date_to: date,
        max_papers: int = 500,
        page_size: int = 100,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[ArxivPaper], FetchResult]:
        """
        Fetch paper metadata matching the given filters.

        Args:
            categories:  arXiv category list, e.g. ["cs.AI", "cs.LG"]
            date_from:   Start date (inclusive).
            date_to:     End date (inclusive).
            max_papers:  Hard cap on the number of papers returned (default 500).
            page_size:   Papers per API request (recommended 100, max 2000).
            on_progress: Optional callback(fetched, expected_total) called after each page.

        Returns:
            (papers, result) tuple.
        """
        if not categories:
            raise ValueError("categories must not be empty")
        if date_from > date_to:
            raise ValueError("date_from must not be later than date_to")
        if max_papers <= 0:
            raise ValueError("max_papers must be a positive integer")

        search_query = self._build_search_query(categories, date_from, date_to)
        papers: list[ArxivPaper] = []
        total_found: int = 0
        offset: int = 0

        async with httpx.AsyncClient(follow_redirects=True) as http:
            while len(papers) < max_papers:
                # Clamp batch size so we never exceed the user's cap
                batch_size = min(page_size, max_papers - len(papers))

                batch_total, batch_papers = await self._fetch_page(
                    http, search_query, start=offset, max_results=batch_size
                )

                # Record the arXiv total on the first request
                if offset == 0:
                    total_found = batch_total
                    effective_total = min(total_found, max_papers)
                    if on_progress:
                        on_progress(0, effective_total)

                if not batch_papers:
                    break  # arXiv returned an empty page; no more results

                papers.extend(batch_papers)
                offset += len(batch_papers)

                if on_progress:
                    on_progress(len(papers), min(total_found, max_papers))

                # Stop if arXiv has no more results
                if offset >= total_found:
                    break

        truncated = total_found > max_papers
        result = FetchResult(
            total_found=total_found,
            fetched=len(papers),
            truncated=truncated,
        )
        return papers, result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_search_query(
        self,
        categories: list[str],
        date_from: date,
        date_to: date,
    ) -> str:
        """
        Build the arXiv API search_query string.

        Example output:
            (cat:cs.AI OR cat:cs.LG) AND submittedDate:[202001010000 TO 202512312359]
        """
        # Join multiple categories with OR
        cat_parts = " OR ".join(f"cat:{c}" for c in categories)
        if len(categories) > 1:
            cat_parts = f"({cat_parts})"

        # arXiv submittedDate format: YYYYMMDDhhmm
        date_filter = (
            f"submittedDate:[{date_from.strftime('%Y%m%d')}0000 "
            f"TO {date_to.strftime('%Y%m%d')}2359]"
        )

        return f"{cat_parts} AND {date_filter}"

    async def _fetch_page(
        self,
        http: httpx.AsyncClient,
        search_query: str,
        start: int,
        max_results: int,
    ) -> tuple[int, list[ArxivPaper]]:
        """
        Fetch a single page of results, respecting the rate limit.

        Returns:
            (total_results_on_arxiv, papers_in_this_page)
        """
        await self._rate_limit()

        params = {
            "search_query": search_query,
            "start":        start,
            "max_results":  max_results,
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",   # newest first
        }

        response = await http.get(ARXIV_API_BASE, params=params, timeout=30.0)
        response.raise_for_status()

        return self._parse_feed(response.text)

    async def _rate_limit(self) -> None:
        """Sleep if needed to ensure at least request_delay seconds between requests."""
        loop = asyncio.get_event_loop()
        now = loop.time()
        elapsed = now - self._last_request_time
        if elapsed < self.request_delay:
            await asyncio.sleep(self.request_delay - elapsed)
        self._last_request_time = loop.time()

    def _parse_feed(self, xml_text: str) -> tuple[int, list[ArxivPaper]]:
        """Parse the arXiv Atom XML feed. Returns (total_results, papers)."""
        root = ET.fromstring(xml_text)

        total_el = root.find("opensearch:totalResults", _NS)
        total = int(total_el.text) if total_el is not None else 0

        papers = []
        for entry in root.findall("atom:entry", _NS):
            paper = self._parse_entry(entry)
            if paper is not None:
                papers.append(paper)

        return total, papers

    def _parse_entry(self, entry: ET.Element) -> ArxivPaper | None:
        """
        Parse a single <entry> element into an ArxivPaper.
        Returns None on failure (logs a warning) so one bad entry never kills the batch.
        """
        try:
            # ── ID and version ────────────────────────────────────────────────
            # Raw format: http://arxiv.org/abs/2301.07041v2
            raw_id = entry.find("atom:id", _NS).text.strip().split("/abs/")[-1]
            if "v" in raw_id.rsplit(".", 1)[-1]:    # version suffix is in the last segment
                arxiv_id, version_str = raw_id.rsplit("v", 1)
                version = int(version_str)
            else:
                arxiv_id = raw_id
                version = 1

            # ── Title (normalize whitespace) ──────────────────────────────────
            title = " ".join(
                entry.find("atom:title", _NS).text.strip().split()
            )

            # ── Abstract ──────────────────────────────────────────────────────
            abstract = entry.find("atom:summary", _NS).text.strip()

            # ── Author list ───────────────────────────────────────────────────
            authors = [
                a.find("atom:name", _NS).text.strip()
                for a in entry.findall("atom:author", _NS)
            ]

            # ── Dates ─────────────────────────────────────────────────────────
            submitted_date = date.fromisoformat(
                entry.find("atom:published", _NS).text[:10]
            )
            updated_date = date.fromisoformat(
                entry.find("atom:updated", _NS).text[:10]
            )

            # ── Categories ────────────────────────────────────────────────────
            categories = [
                cat.get("term")
                for cat in entry.findall("atom:category", _NS)
            ]
            primary_cat_el = entry.find("arxiv:primary_category", _NS)
            primary_category = (
                primary_cat_el.get("term")
                if primary_cat_el is not None
                else categories[0]
            )

            # ── Links ─────────────────────────────────────────────────────────
            abs_url = ""
            pdf_url = ""
            for link in entry.findall("atom:link", _NS):
                rel    = link.get("rel", "")
                title_ = link.get("title", "")
                href   = link.get("href", "")
                if rel == "alternate":
                    abs_url = href
                elif title_ == "pdf":
                    pdf_url = href

            # ── Optional fields ───────────────────────────────────────────────
            def _optional_text(tag: str) -> str | None:
                el = entry.find(tag, _NS)
                return el.text.strip() if el is not None and el.text else None

            return ArxivPaper(
                arxiv_id=arxiv_id,
                version=version,
                title=title,
                abstract=abstract,
                authors=authors,
                submitted_date=submitted_date,
                updated_date=updated_date,
                categories=categories,
                primary_category=primary_category,
                abs_url=abs_url,
                pdf_url=pdf_url,
                doi=_optional_text("arxiv:doi"),
                journal_ref=_optional_text("arxiv:journal_ref"),
                comment=_optional_text("arxiv:comment"),
            )

        except Exception as exc:
            # A single parse failure should not abort the whole batch
            arxiv_id_hint = ""
            try:
                arxiv_id_hint = entry.find("atom:id", _NS).text
            except Exception:
                pass
            print(f"  [warn] Failed to parse entry (skipped): {arxiv_id_hint}  reason: {exc}")
            return None


# ─── Quick smoke test ─────────────────────────────────────────────────────────

async def _demo() -> None:
    """Fetch 5 recent cs.AI papers from 2024 and print a summary."""
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    client = ArxivClient(request_delay=3.0)

    print("Fetching cs.AI papers (max 5, year 2024)...\n")
    papers, result = await client.fetch_papers(
        categories=["cs.AI"],
        date_from=date(2024, 1, 1),
        date_to=date(2024, 12, 31),
        max_papers=5,
        on_progress=lambda fetched, total: print(f"  progress: {fetched}/{total}"),
    )

    print(f"\narXiv total: {result.total_found}  |  fetched: {result.fetched}"
          + ("  (truncated)" if result.truncated else ""))
    print()

    for i, p in enumerate(papers, 1):
        print(f"[{i}] {p.title}")
        print(f"     ID: {p.arxiv_id}v{p.version}  |  date: {p.submitted_date}")
        print(f"     categories: {', '.join(p.categories)}")
        print(f"     authors: {', '.join(p.authors[:3])}{'...' if len(p.authors) > 3 else ''}")
        print(f"     abstract: {p.abstract[:120]}...")
        print()


if __name__ == "__main__":
    asyncio.run(_demo())
