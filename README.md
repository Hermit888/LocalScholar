# Overview
LocalScholar is a fully locally deployed academic paper retrieval and question-answering system; it handles the entire process from retrieval to generation without relying on any external large model APIs.

# Implementation
1. Install [PostgreSQL](https://www.postgresql.org/download/) and [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Copy and paste ./db/schema.sql into PostgreSQL or run ./db/init_db.py to create table.
3. Run ./ingestion/ingestor.py to fetch and store paper information from arXiv. Option: run ./ingestion/ss_enricher.py to crawl and store each paper's citation count.
4. Store paper vectors in Qdrant. See [Vector database (Qdrant)](#vector-database-qdrant) below.

# Vector database (Qdrant)

Qdrant runs locally in Docker. No Qdrant Cloud account is required. Paper vectors are written by `ingestion/embedder.py`, which reads title and abstract from PostgreSQL, encodes them with SPECTER2 (`proximity` adapter), and upserts them into the `papers_abstract` collection.

- Vector size: 768 (fixed by SPECTER2)
- Distance: cosine
- Point id: deterministic UUID from `arxiv_id` (safe to re-run)
- Payload: `arxiv_id`, `title`, `submitted_date`, `primary_category`, `citation_count`, `venue`

Connection settings live in `.env`:

```
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## 1. Start Docker Desktop

Open Docker Desktop from the Start menu and wait until it is fully running (the whale icon in the taskbar stops animating). The Docker CLI cannot create containers until the Desktop daemon is up.

## 2. Create and start the Qdrant container

From the project root in PowerShell:

```powershell
docker run -d --name qdrant-localscholar -p 6333:6333 -p 6334:6334 -v "${PWD}/data/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

This pulls the `qdrant/qdrant` image on first use, listens on port 6333, and stores data in `data/qdrant_storage` so vectors survive a container restart.

If the container already exists, start it instead of creating it again:

```powershell
docker start qdrant-localscholar
```

## 3. Install embedding dependencies

```powershell
pip install -r requirements.txt
```

`embedder.py` needs `qdrant-client`, `torch`, `transformers`, and `adapters`.

## 4. Encode papers and write them to Qdrant

PostgreSQL must already contain papers (`ingestion/ingestor.py`, and optionally `ingestion/ss_enricher.py`). Then:

```powershell
python ingestion/embedder.py
```

The first run downloads SPECTER2 (`allenai/specter2_base` plus the proximity adapter, about 440 MB) into the Hugging Face cache at `C:\Users\<you>\.cache\huggingface\hub\`. Later runs use that cache.

## 5. Inspect the vectors

Open [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

- Collection `papers_abstract` is the vector table.
- The Points tab lists each paper's id and payload (title, date, category, citation count, venue).
- Graph shows the selected paper (yellow) and its nearest neighbors (green). `limit` is how many neighbors to return.
- Find similar runs a cosine search from that paper's vector and returns the closest papers. 
