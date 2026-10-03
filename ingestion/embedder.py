"""
Embedder: reads papers from PostgreSQL, encodes them with SPECTER2 (proximity adapter),
and upserts the resulting vectors into Qdrant's `papers_abstract` collection.

SPECTER2 adapter choice:
  - proximity     → used HERE for encoding paper title+abstract (document side)
  - adhoc_query   → used later in Discovery Layer for encoding user queries (query side)

Qdrant collection: papers_abstract
  vector dim : 768
  distance   : Cosine
  point_id   : UUID5(arxiv_id)  – deterministic, safe to re-run
  payload    : arxiv_id, title, submitted_date, primary_category,
               citation_count, venue

Run:
    python ingestion/embedder.py

First run downloads SPECTER2 base model (~440 MB) from HuggingFace.
Subsequent runs use the local cache – no internet needed.
"""

import asyncio
import os
import sys
import uuid
from datetime import date
from pathlib import Path

import asyncpg
import torch
from adapters import AutoAdapterModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from transformers import AutoTokenizer

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Config ───────────────────────────────────────────────────────────────────

SPECTER2_BASE    = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"          # proximity adapter
COLLECTION_NAME  = "papers_abstract"
VECTOR_DIM       = 768
BATCH_SIZE       = 32    # papers per embedding batch (lower = less RAM)

# Read from .env (with sensible fallbacks)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Deterministic namespace for UUID5 point IDs
_UUID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID_NS_URL

# ─── DB helpers ───────────────────────────────────────────────────────────────

def _get_dsn() -> str:
    user     = os.getenv("POSTGRES_USER", "").strip()
    password = os.getenv("POSTGRES_PASSWORD", "").strip()
    host     = os.getenv("POSTGRES_HOST", "localhost")
    port     = os.getenv("POSTGRES_PORT", "5433")
    db       = os.getenv("POSTGRES_DB", "localscholar")
    if not user or not password:
        print("ERROR: POSTGRES_USER or POSTGRES_PASSWORD not set in .env")
        sys.exit(1)
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


_SELECT_PAPERS = """
SELECT
    arxiv_id,
    title,
    abstract,
    submitted_date,
    primary_category,
    citation_count,
    venue
FROM papers
ORDER BY submitted_date DESC
"""


def _arxiv_id_to_point_id(arxiv_id: str) -> str:
    """Convert an arxiv_id to a deterministic UUID string for Qdrant."""
    return str(uuid.uuid5(_UUID_NS, arxiv_id))


# ─── SPECTER2 ─────────────────────────────────────────────────────────────────

def load_specter2():
    """Load SPECTER2 base model + proximity adapter. Downloads on first run."""
    print("[model]  Loading SPECTER2 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(SPECTER2_BASE)

    print("[model]  Loading SPECTER2 base model ...")
    model = AutoAdapterModel.from_pretrained(SPECTER2_BASE)

    print("[model]  Loading proximity adapter ...")
    model.load_adapter(
        SPECTER2_ADAPTER,
        source="hf",
        load_as="proximity",
        set_active=True,
    )
    model.eval()
    return tokenizer, model


def encode_batch(
    tokenizer,
    model,
    texts: list[str],
) -> list[list[float]]:
    """
    Encode a list of strings with SPECTER2.
    Returns a list of 768-dim float vectors.
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    with torch.no_grad():
        output = model(**inputs)

    # CLS token embedding (first token of last hidden state)
    embeddings = output.last_hidden_state[:, 0, :]
    return embeddings.tolist()


# ─── Qdrant helpers ───────────────────────────────────────────────────────────

def ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it doesn't exist yet."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        print(f"[qdrant] Collection '{COLLECTION_NAME}' already exists.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
    )
    print(f"[qdrant] Collection '{COLLECTION_NAME}' created (dim={VECTOR_DIM}, cosine).")


# ─── Main pipeline ────────────────────────────────────────────────────────────

async def embed_and_store(batch_size: int = BATCH_SIZE) -> None:
    """
    Full pipeline:
      1. Load all papers from PostgreSQL.
      2. Encode title+abstract in batches with SPECTER2 (proximity).
      3. Upsert vectors into Qdrant.
    """
    # ── Load model ────────────────────────────────────────────────────────────
    tokenizer, model = load_specter2()
    print()

    # ── Fetch papers from PostgreSQL ──────────────────────────────────────────
    conn = await asyncpg.connect(_get_dsn())
    rows = await conn.fetch(_SELECT_PAPERS)
    await conn.close()
    print(f"[pg]     {len(rows)} papers loaded from PostgreSQL\n")

    if not rows:
        print("[info]   No papers found. Run ingestion/ingestor.py first.")
        return

    # ── Connect to Qdrant ─────────────────────────────────────────────────────
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(qdrant)
    print()

    # ── Encode + upsert in batches ────────────────────────────────────────────
    total   = len(rows)
    upserted = 0

    for batch_start in range(0, total, batch_size):
        batch = rows[batch_start: batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        # Build input strings: "title [SEP] abstract"
        texts = [
            r["title"] + tokenizer.sep_token + (r["abstract"] or "")
            for r in batch
        ]

        print(f"  [batch {batch_num:>3}/{total_batches}]  encoding {len(batch)} papers ...",
              end="", flush=True)

        vectors = encode_batch(tokenizer, model, texts)

        # Build Qdrant points
        points = []
        for row, vector in zip(batch, vectors):
            point_id = _arxiv_id_to_point_id(row["arxiv_id"])

            # submitted_date → ISO string for Qdrant payload
            submitted_date = row["submitted_date"]
            date_str = submitted_date.isoformat() if submitted_date else None

            payload = {
                "arxiv_id":        row["arxiv_id"],
                "title":           row["title"],
                "submitted_date":  date_str,
                "primary_category": row["primary_category"],
                "citation_count":  row["citation_count"],   # may be None
                "venue":           row["venue"],             # may be None
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        upserted += len(points)
        print(f"  upserted (total {upserted}/{total})")

    # ── Final stats ───────────────────────────────────────────────────────────
    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"\n[done]   Qdrant collection '{COLLECTION_NAME}': "
          f"{info.points_count} points stored.")
    print(f"[view]   Open http://localhost:6333/dashboard to inspect.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(embed_and_store())
