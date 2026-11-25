"""
scripts/migrate_chroma_to_pgvector.py
Copy all vectors, documents, and metadata from local ChromaDB into Supabase pgvector.
Preserves exact 1024-dim BGE-M3 embeddings without re-embedding.
"""

import json
import os
import time
import chromadb
from chromadb.config import Settings
import psycopg2
from psycopg2.extras import execute_values

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

# psycopg2 expects postgresql:// (not postgresql+psycopg2://)
PG_DSN = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")

COLLECTIONS = ["financial_data", "user_data", "semantic_cache"]
BATCH_SIZE = 250


def main():
    t0 = time.time()
    chroma = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    print("Ensuring vector_documents table and pgvector extension...")
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS vector_documents (
            id TEXT NOT NULL,
            collection TEXT NOT NULL,
            document TEXT NOT NULL,
            metadata JSONB DEFAULT '{}'::jsonb,
            embedding vector(1024),
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (collection, id)
        );
        CREATE INDEX IF NOT EXISTS idx_vector_docs_collection
            ON vector_documents (collection);
    """)
    conn.commit()

    total_migrated = 0
    for coll_name in COLLECTIONS:
        coll = chroma.get_or_create_collection(coll_name)
        count = coll.count()
        print(f"\nMigrating collection '{coll_name}' ({count} docs)...")
        if count == 0:
            continue

        offset = 0
        while offset < count:
            res = coll.get(
                limit=BATCH_SIZE,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            ids = res["ids"]
            if not ids:
                break

            rows = []
            for doc_id, doc_text, meta, emb in zip(
                ids, res["documents"], res["metadatas"], res["embeddings"]
            ):
                vec_str = "[" + ",".join(f"{float(x):.7g}" for x in emb) + "]"
                rows.append((
                    str(doc_id),
                    coll_name,
                    doc_text or "",
                    json.dumps(meta or {}),
                    vec_str,
                ))

            execute_values(
                cur,
                """
                INSERT INTO vector_documents (id, collection, document, metadata, embedding)
                VALUES %s
                ON CONFLICT (collection, id) DO UPDATE SET
                    document = EXCLUDED.document,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                rows,
                template="(%s, %s, %s, %s::jsonb, %s::vector)",
                page_size=BATCH_SIZE,
            )
            conn.commit()
            offset += len(ids)
            total_migrated += len(ids)
            print(f"  {coll_name}: {offset}/{count} ({int(offset/count*100)}%)")

    print("\nCreating HNSW vector index & GIN full-text index (may take ~15-30s)...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_vector_docs_embedding_hnsw
            ON vector_documents USING hnsw (embedding vector_cosine_ops);
        CREATE INDEX IF NOT EXISTS idx_vector_docs_fts
            ON vector_documents USING gin (to_tsvector('simple', document));
    """)
    conn.commit()

    cur.execute("SELECT collection, COUNT(*) FROM vector_documents GROUP BY collection ORDER BY collection;")
    print("\nSupabase pgvector summary:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")

    cur.close()
    conn.close()
    print(f"\nDone! Migrated {total_migrated} vectors in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
