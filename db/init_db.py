"""
Initialize the LocalScholar database by executing db/schema.sql.

Run once before starting the application:
    python db/init_db.py

Alternatively, run schema.sql directly in pgAdmin 4 or psql:
    psql -U postgres -p 5433 -d localscholar -f db/schema.sql

Dependencies: asyncpg, python-dotenv
"""

import asyncio
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

# Load .env from the project root
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

# schema.sql lives next to this file
_SCHEMA_FILE = Path(__file__).parent / "schema.sql"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def _get_dsn() -> str:
    """Build the PostgreSQL connection string from environment variables."""
    user     = os.getenv("POSTGRES_USER", "").strip()
    password = os.getenv("POSTGRES_PASSWORD", "").strip()
    host     = os.getenv("POSTGRES_HOST", "localhost")
    port     = os.getenv("POSTGRES_PORT", "5433")
    db       = os.getenv("POSTGRES_DB", "localscholar")

    if not user or not password:
        print("ERROR: POSTGRES_USER or POSTGRES_PASSWORD is not set in .env")
        sys.exit(1)

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


async def init_db() -> None:
    if not _SCHEMA_FILE.exists():
        print(f"ERROR: Schema file not found: {_SCHEMA_FILE}")
        sys.exit(1)

    sql = _SCHEMA_FILE.read_text(encoding="utf-8")

    print(f"[schema] Reading {_SCHEMA_FILE.name} ...")
    print(f"[connect] PostgreSQL ({os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5433')}) ...")

    try:
        conn = await asyncpg.connect(_get_dsn())
    except Exception as e:
        print(f"[error] Connection failed: {e}")
        print("  Check the PostgreSQL settings in .env and make sure the server is running.")
        sys.exit(1)

    try:
        await conn.execute(sql)
        print("[done]  All tables and indexes created (or already existed).")
    except Exception as e:
        print(f"[error] Failed to execute schema: {e}")
        sys.exit(1)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(init_db())
