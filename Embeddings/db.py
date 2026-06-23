"""Postgres + pgvector access: connect, create table, upsert, exact top-k search.

The Convention is ~60 articles, so we use EXACT brute-force cosine search (no
ANN index). That is instant at this scale, avoids pgvector's ~2000-dim index
cap (so any embedding dim works), and removes index tuning as a variable in the
comparison.
"""
from __future__ import annotations

from typing import Any, Sequence

import psycopg
from pgvector.psycopg import register_vector


def connect(dsn: str) -> psycopg.Connection:
    conn = psycopg.connect(dsn)
    ensure_extension(conn)
    register_vector(conn)
    return conn


def ensure_extension(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()


def ensure_table(conn: psycopg.Connection, table: str, dim: int) -> None:
    # `table` comes from the trusted MODELS registry, not user input.
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            article_id text PRIMARY KEY,
            part       text,
            title      text,
            content    text,
            embedding  vector({dim})
        )
        """
    )
    conn.commit()


def upsert_documents(conn: psycopg.Connection, table: str, rows: Sequence[dict[str, Any]]) -> None:
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                f"""
                INSERT INTO {table} (article_id, part, title, content, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (article_id) DO UPDATE SET
                    part      = EXCLUDED.part,
                    title     = EXCLUDED.title,
                    content   = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                """,
                (
                    row["article_id"],
                    row.get("part"),
                    row.get("title"),
                    row["content"],
                    row["embedding"],
                ),
            )
    conn.commit()


def search(conn: psycopg.Connection, table: str, query_vec: list[float], k: int) -> list[dict[str, Any]]:
    """Exact cosine top-k. Returns rows with a `similarity` in [0, 1]."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT article_id, part, title, content,
                   1 - (embedding <=> %s) AS similarity
            FROM {table}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_vec, query_vec, k),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, record)) for record in cur.fetchall()]


def count_rows(conn: psycopg.Connection, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        return int(cur.fetchone()[0])
