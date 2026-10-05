# Overview
LocalScholar is a fully locally deployed academic paper retrieval and question-answering system; it handles the entire process from retrieval to generation without relying on any external large model APIs.

# Implementation
1. Install [PostgreSQL](https://www.postgresql.org/download/) and [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Copy and paste ./db/schema.sql into PostgreSQL or run ./db/init_db.py to create table.
3. Run ./ingestion/ingestor.py to fetch and store paper information from arXiv. Option: run ./ingestion/ss_enricher.py to crawl and store each paper's citation count.
4. 
