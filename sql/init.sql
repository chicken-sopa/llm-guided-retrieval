-- Enable pgvector. Per-model tables are created at build time (build_index.py)
-- with the dimension probed from the chosen embedding model.
CREATE EXTENSION IF NOT EXISTS vector;
