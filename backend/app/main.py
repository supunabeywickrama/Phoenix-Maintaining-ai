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
from sqlalchemy import text

from unified_rag.config import settings
from unified_rag.db.database import engine, Base
from unified_rag.db import models  # noqa: F401  — registers tables on Base

from unified_rag.api.endpoints import router as manuals_router
from app.assistant_api import router as assistant_router
from app.machines_api import router as machines_router

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)


def init_db() -> None:
    """Ensure pgvector exists and the six tables are present."""
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except Exception as e:
        logger.warning("Could not ensure pgvector extension: %s", e)

    for attempt in range(3):
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("✅ Database tables verified.")
            return
        except Exception as e:
            logger.error("Database init failed (attempt %d/3): %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2)
    logger.critical("❌ Could not reach the database. Check DATABASE_URL in .env.")


init_db()

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


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "service": "phoenix-copilot", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
