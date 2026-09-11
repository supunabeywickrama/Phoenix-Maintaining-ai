"""
One-time migration: copy every ManualChunk / InteractionMemory row out of
Postgres/pgvector into Qdrant.

Reuses the embeddings already stored in pgvector rather than re-calling the
embedding model — for a manual with hundreds of chunks that is the difference
between a few seconds and re-paying the full ingestion embedding cost. Reuses
each row's original Postgres integer id as the Qdrant point id too, so running
this script twice overwrites the same points rather than duplicating them.

The source Postgres tables are left untouched (this only reads them) — see the
module docstring in unified_rag/db/models.py for why they're kept as a
rollback path until Qdrant is confirmed working.

Usage (from backend/):
    python scripts/migrate_to_qdrant.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONUTF8", "1")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from unified_rag.db.database import SessionLocal
from unified_rag.db.models import ManualChunk, InteractionMemory
from unified_rag.db import qdrant_store as qs

BATCH_SIZE = 200


def _to_vector(raw) -> list:
    """pgvector.sqlalchemy's Vector column reads back as a numpy array in
    recent pgvector-python versions, a plain list in older ones — Qdrant's
    client wants a plain list of floats either way."""
    if hasattr(raw, "tolist"):
        return raw.tolist()
    return list(raw)


def migrate_manual_chunks() -> int:
    db = SessionLocal()
    migrated = 0
    try:
        total = db.query(ManualChunk).count()
        print(f"[migrate] manual_chunks: {total} rows in Postgres")
        offset = 0
        while offset < total:
            rows = (
                db.query(ManualChunk)
                .order_by(ManualChunk.id)
                .offset(offset)
                .limit(BATCH_SIZE)
                .all()
            )
            if not rows:
                break
            for row in rows:
                if row.embedding is None:
                    print(f"  [skip] chunk {row.id} has no embedding")
                    continue
                qs.upsert_manual_chunk(
                    _to_vector(row.embedding),
                    point_id=row.id,
                    manual_id=row.manual_id,
                    type=row.type,
                    content=row.content,
                    path=row.path,
                    page=row.page,
                    kind=row.kind,
                    render_markdown=row.render_markdown,
                    figure_role=row.figure_role,
                    parent_path=row.parent_path,
                    width=row.width,
                    height=row.height,
                )
                migrated += 1
            offset += BATCH_SIZE
            print(f"  ∟ {min(offset, total)}/{total}")
    finally:
        db.close()
    return migrated


def migrate_interaction_memory() -> int:
    db = SessionLocal()
    migrated = 0
    try:
        rows = db.query(InteractionMemory).order_by(InteractionMemory.id).all()
        print(f"[migrate] interaction_memory: {len(rows)} rows in Postgres")
        for row in rows:
            if row.embedding is None:
                print(f"  [skip] memory {row.id} has no embedding")
                continue
            qs.upsert_interaction_memory(
                _to_vector(row.embedding),
                point_id=row.id,
                machine_id=row.machine_id,
                manual_id=row.manual_id,
                summary=row.summary,
                operator_fix=row.operator_fix,
                timestamp=row.timestamp,
            )
            migrated += 1
    finally:
        db.close()
    return migrated


if __name__ == "__main__":
    print("Ensuring Qdrant collections exist...")
    qs.ensure_collections()

    n1 = migrate_manual_chunks()
    n2 = migrate_interaction_memory()

    print(f"\nDone. Migrated {n1} manual_chunks + {n2} interaction_memory points.")
    print("Postgres source tables were not modified — safe to re-run this script.")
