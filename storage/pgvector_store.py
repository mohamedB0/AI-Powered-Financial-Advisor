"""
storage/pgvector_store.py
Supabase / PostgreSQL pgvector backend with a ChromaDB-compatible collection shim.

Allows VectorSearch, SemanticCache, BM25Search (via Postgres FTS), health checks,
and ingestion tasks to run against Supabase pgvector with 0 MB local vector index RAM.
"""

import json
import logging
import os
import uuid
import hashlib
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from storage.relational_db import RelationalDB

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
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
"""


def _vec_literal(vec: List[float]) -> str:
    """Format a Python float list as a pgvector literal '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


class PgVectorCollection:
    """ChromaDB-compatible Collection wrapper backed by Supabase pgvector."""

    def __init__(self, db: RelationalDB, name: str):
        self.db = db
        self.name = name

    def count(self) -> int:
        with self.db.session as s:
            return int(
                s.execute(
                    text("SELECT COUNT(*) FROM vector_documents WHERE collection = :c"),
                    {"c": self.name},
                ).scalar()
                or 0
            )

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not ids:
            return
        metadatas = metadatas or [{} for _ in ids]
        sql = text("""
            INSERT INTO vector_documents (id, collection, document, metadata, embedding)
            VALUES (:id, :collection, :document, CAST(:metadata AS jsonb), CAST(:embedding AS vector))
            ON CONFLICT (collection, id) DO UPDATE SET
                document = EXCLUDED.document,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
        """)
        rows = [
            {
                "id": str(doc_id),
                "collection": self.name,
                "document": doc_text or "",
                "metadata": json.dumps(meta or {}),
                "embedding": _vec_literal(emb),
            }
            for doc_id, emb, doc_text, meta in zip(ids, embeddings, documents, metadatas)
        ]
        with self.db.session as s:
            s.execute(sql, rows)
            s.commit()

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not query_embeddings:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q_vec = _vec_literal(query_embeddings[0])
        where_clauses = ["collection = :collection", "embedding IS NOT NULL"]
        params: Dict[str, Any] = {
            "collection": self.name,
            "qvec": q_vec,
            "limit": int(n_results),
        }

        # Translate simple ChromaDB metadata filters ({k: {"$eq": v}} or {"$and": [...]})
        if where:
            flat_filters: Dict[str, Any] = {}
            if "$and" in where and isinstance(where["$and"], list):
                for cond in where["$and"]:
                    for k, v in cond.items():
                        flat_filters[k] = v["$eq"] if isinstance(v, dict) and "$eq" in v else v
            else:
                for k, v in where.items():
                    flat_filters[k] = v["$eq"] if isinstance(v, dict) and "$eq" in v else v

            for idx, (fk, fv) in enumerate(flat_filters.items()):
                pkey = f"mk_{idx}"
                pval = f"mv_{idx}"
                where_clauses.append(f"metadata->>:{pkey} = :{pval}")
                params[pkey] = str(fk)
                params[pval] = str(fv)

        where_sql = " AND ".join(where_clauses)
        sql = text(f"""
            SELECT id, document, metadata, (embedding <=> CAST(:qvec AS vector)) AS distance
            FROM vector_documents
            WHERE {where_sql}
            ORDER BY embedding <=> CAST(:qvec AS vector)
            LIMIT :limit
        """)

        with self.db.session as s:
            rows = s.execute(sql, params).fetchall()

        ids, docs, metas, dists = [], [], [], []
        for r in rows:
            ids.append(r[0])
            docs.append(r[1])
            metas.append(r[2] if isinstance(r[2], dict) else (json.loads(r[2]) if r[2] else {}))
            dists.append(float(r[3]) if r[3] is not None else 1.0)

        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }

    def get(
        self,
        ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"collection": self.name}
        where_sql = "collection = :collection"
        if ids:
            where_sql += " AND id = ANY(:ids)"
            params["ids"] = [str(i) for i in ids]

        limit_sql = f" LIMIT {int(limit)} OFFSET {int(offset)}" if limit is not None else ""
        sql = text(
            f"SELECT id, document, metadata FROM vector_documents WHERE {where_sql} ORDER BY id{limit_sql}"
        )
        with self.db.session as s:
            rows = s.execute(sql, params).fetchall()

        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [
                r[2] if isinstance(r[2], dict) else (json.loads(r[2]) if r[2] else {})
                for r in rows
            ],
        }

    def delete(self, ids: Optional[List[str]] = None) -> None:
        if not ids:
            return
        with self.db.session as s:
            s.execute(
                text("DELETE FROM vector_documents WHERE collection = :c AND id = ANY(:ids)"),
                {"c": self.name, "ids": [str(i) for i in ids]},
            )
            s.commit()

    def search_fts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Postgres native full-text search (used by BM25Search when on pgvector)."""
        sql = text("""
            SELECT id, document, metadata,
                   ts_rank_cd(to_tsvector('simple', document), websearch_to_tsquery('simple', :q)) AS score
            FROM vector_documents
            WHERE collection = :c
              AND to_tsvector('simple', document) @@ websearch_to_tsquery('simple', :q)
            ORDER BY score DESC
            LIMIT :k
        """)
        with self.db.session as s:
            rows = s.execute(sql, {"c": self.name, "q": query, "k": int(top_k)}).fetchall()
        results = []
        for r in rows:
            meta = r[2] if isinstance(r[2], dict) else (json.loads(r[2]) if r[2] else {})
            results.append({
                "text": r[1],
                "score": float(r[3]),
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 0),
                "collection": self.name,
                "metadata": meta,
            })
        return results


class PgVectorClient:
    """ChromaDB-compatible PersistentClient replacement backed by Supabase pgvector."""

    _schema_ready = False

    def __init__(self, database_url: Optional[str] = None):
        self.db = RelationalDB(database_url=database_url)
        if not PgVectorClient._schema_ready:
            with self.db.session as s:
                s.execute(text(_SCHEMA_SQL))
                s.commit()
            PgVectorClient._schema_ready = True

    def get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> PgVectorCollection:
        return PgVectorCollection(self.db, name)


class PgVectorStore:
    """Drop-in replacement for ChromaStore backed by Supabase pgvector."""

    def __init__(self, collection_name: str = "financial_data", persist_dir: str = ""):
        self.collection_name = collection_name
        self.client = PgVectorClient()
        self.collection = self.client.get_or_create_collection(collection_name)

    def upsert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 128,
    ) -> int:
        if not texts:
            return 0
        if ids is None:
            ids = [
                hashlib.sha256(f"{m.get('source','')}_{m.get('chunk_index', i)}_{t[:100]}".encode()).hexdigest()[:32]
                if m else str(uuid.uuid4())
                for i, (t, m) in enumerate(zip(texts, metadatas))
            ]
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
        return len(texts)

    def count(self) -> int:
        return self.collection.count()
