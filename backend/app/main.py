"""
Phoenix Industries — Manual Q&A and Guided Repair API.

Knowledge-side only: ingest manuals, ask questions grounded in them, and be walked
through a repair. No telemetry, WebSockets, or model training, so the app boots in
seconds and has no local state.
"""
import logging
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from unified_rag.config import settings
from unified_rag.db.database import engine, Base
from unified_rag.db import models  # noqa: F401  — registers tables on Base
from unified_rag.db import qdrant_store
from services.cloudinary_service import LOCAL_DATA_DIR

from unified_rag.api.endpoints import router as manuals_router
from app.assistant_api import router as assistant_router
from app.machines_api import router as machines_router

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)


def init_qdrant() -> None:
    """Ensure the two vector collections exist.

    Same resilience pattern as init_db() below: never crash the app over this.
    Unlike init_db(), a missing/unreachable Qdrant is not necessarily fatal —
    ordinary chat (sessions, machines) still works via Postgres; only
    manual-grounded retrieval degrades. Every query.retrieve() call already
    wraps its Qdrant calls in try/except and fails to an empty result set for
    exactly this reason.
    """
    try:
        qdrant_store.ensure_collections()
        logger.info("✅ Qdrant collections verified at %s.", settings.qdrant_url)
    except Exception as e:
        logger.error(
            "Could not reach Qdrant at %s: %s — vector search (manual Q&A) will "
            "return no results until it's reachable. Sessions/machines still "
            "work via Postgres.", settings.qdrant_url, e,
        )


def init_db() -> None:
    """Ensure the relational tables are present.

    The two vector tables (manual_chunks, interaction_memory) are also created
    here if missing — their SQLAlchemy models are kept registered on Base as a
    legacy read path for scripts/migrate_to_qdrant.py — but the app no longer
    writes to them; see unified_rag/db/models.py.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except Exception as e:
        logger.warning("Could not ensure pgvector extension: %s", e)

    # create_all() only creates missing TABLES, never missing columns, so columns
    # added after a database was first initialised need an explicit nudge.
    def _add_missing_columns() -> None:
        statements = [
            "ALTER TABLE assistant_sessions ADD COLUMN IF NOT EXISTS intent VARCHAR",
            "ALTER TABLE assistant_sessions ADD COLUMN IF NOT EXISTS resolution TEXT",
            "ALTER TABLE manual_chunks ADD COLUMN IF NOT EXISTS figure_role VARCHAR",
            "ALTER TABLE manual_chunks ADD COLUMN IF NOT EXISTS parent_path VARCHAR",
            "ALTER TABLE manual_chunks ADD COLUMN IF NOT EXISTS width INTEGER",
            "ALTER TABLE manual_chunks ADD COLUMN IF NOT EXISTS height INTEGER",
            "ALTER TABLE manual_chunks ADD COLUMN IF NOT EXISTS kind VARCHAR",
            "ALTER TABLE manual_chunks ADD COLUMN IF NOT EXISTS render_markdown TEXT",
            "ALTER TABLE assistant_messages ADD COLUMN IF NOT EXISTS attachments TEXT",
        ]
        try:
            with engine.connect() as conn:
                for stmt in statements:
                    conn.execute(text(stmt))
                conn.commit()
        except Exception as e:
            logger.warning("Column migration skipped: %s", e)

    for attempt in range(3):
        try:
            Base.metadata.create_all(bind=engine)
            _add_missing_columns()
            logger.info("✅ Database tables verified.")
            return
        except Exception as e:
            logger.error("Database init failed (attempt %d/3): %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2)
    logger.critical("❌ Could not reach the database. Check DATABASE_URL in .env.")


init_db()
init_qdrant()

app = FastAPI(
    title="Phoenix Industries — Maintenance Copilot",
    description="Ask questions about machine problems and get manual-grounded, step-by-step fixes.",
    version="1.0.0",
)

# Single trusted origin rather than "*": credentials are allowed, and the two
# cannot be combined safely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url.rstrip("/")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(manuals_router, tags=["Manuals"])
app.include_router(machines_router)
app.include_router(assistant_router)

# Serves whatever CloudinaryService._upload_local() wrote when CLOUDINARY_* is
# unset — figures/manuals fall back to this instead of real Cloudinary.
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(LOCAL_DATA_DIR)), name="static")


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "service": "phoenix-copilot", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
