"""
Manual ingestion endpoints.

Ingestion runs as a background task rather than inside the request: a large
illustrated manual makes one vision call per figure and can run for many minutes,
long enough that a browser abandons the request while the server is still working.
The client gets an immediate acknowledgement and polls
GET /ingest-manual/status/{manual_id} for progress.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
import os
import tempfile
import time
from datetime import datetime
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from unified_rag.ingestion.pipeline import process_manual_async
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import Manual, Machine
from unified_rag.db import qdrant_store as qs

router = APIRouter()

# All Phoenix blobs live under their own Cloudinary prefix so client media is
# never interleaved with another deployment's.
CLOUDINARY_MANUAL_FOLDER = "phoenix/manuals"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ManualSummary(BaseModel):
    manual_id: str
    filename: Optional[str] = None
    url: Optional[str] = None
    created_at: Optional[str] = None
    chunks: int = 0


@router.post("/ingest-manual")
async def ingest_manual(
    background_tasks: BackgroundTasks,
    manual_id: str = Form(...),
    file: UploadFile = File(...),
    replace: bool = Form(True),
):
    print(f"\n🚀 [API] Received ingestion request for Manual ID: {manual_id}")
    print(f"📄 [API] File: {file.filename}")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    manual_id = manual_id.strip()
    if not manual_id:
        raise HTTPException(status_code=400, detail="manual_id is required.")

    pdf_bytes = await file.read()

    # Store the source PDF so the original stays retrievable after ingestion.
    try:
        from services.cloudinary_service import CloudinaryService

        cloud = CloudinaryService()
        if cloud.enabled:
            print(f"☁️ [API] Uploading source PDF for {manual_id}...")
            url = cloud.upload_file(
                pdf_bytes,
                public_id=f"manual_{manual_id}",
                folder=CLOUDINARY_MANUAL_FOLDER,
                resource_type="raw",
            )
            if url:
                db = SessionLocal()
                try:
                    record = db.query(Manual).filter(Manual.manual_id == manual_id).first()
                    if not record:
                        record = Manual(manual_id=manual_id)
                        db.add(record)
                    record.filename = file.filename
                    record.url = url
                    record.created_at = datetime.now().isoformat()
                    db.commit()
                    print(f"✅ [API] Manual {manual_id} registered: {url}")
                finally:
                    db.close()
    except Exception as e:
        # Losing the source-PDF copy must not stop us vectorising the content.
        print(f"⚠️ [API] Source PDF cloud sync failed: {e}")

    # Re-ingesting the same manual_id used to append a second full set of chunks,
    # leaving the old ones to compete in retrieval. Updating a manual means
    # replacing its knowledge, so clear it first.
    if replace:
        try:
            removed = qs.delete_manual_chunks(manual_id)
            if removed:
                print(f"♻️ [API] Replacing {manual_id}: cleared {removed} existing chunks.")
        except Exception as e:
            print(f"⚠️ [API] Could not clear existing chunks for {manual_id}: {e}")

    # Parsing needs a real filesystem path (Camelot has no in-memory API), so use
    # a momentary OS-tempdir file that the background task deletes when finished.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    _set_job(manual_id, status="processing", filename=file.filename, chunks=None, error=None)
    background_tasks.add_task(_run_ingestion, tmp_path, manual_id)
    print(f"✅ [API] Ingestion for {manual_id} queued in background.")

    return {
        "message": "Ingestion started",
        "manual_id": manual_id,
        "status": "processing",
        "poll": f"/ingest-manual/status/{manual_id}",
    }


# ── Background ingestion job tracking ────────────────────────────────────────
# In-process registry: advisory progress reporting only. The chunks themselves
# are the source of truth and land in Postgres.
_ingestion_jobs: dict[str, dict] = {}


def _set_job(manual_id: str, **fields) -> None:
    job = _ingestion_jobs.setdefault(manual_id, {"manual_id": manual_id})
    job.update(fields)
    job["updated_at"] = datetime.now().isoformat()


async def _run_ingestion(tmp_path: str, manual_id: str) -> None:
    """Execute the ingestion pipeline outside the request/response cycle."""
    try:
        result = await process_manual_async(tmp_path, manual_id)
        # "partial" is reported distinctly from "success": chunks that failed to
        # embed or store are missing from the index, and an operator asking a
        # question about the very page that dropped out deserves to know the
        # manual went in incomplete rather than silently getting no answer.
        _set_job(
            manual_id,
            status=result.get("status", "success"),
            chunks=result.get("chunks"),
            failed=result.get("failed", 0),
            failure_samples=result.get("failure_samples") or [],
            error=None,
        )
        print(f"🏁 [API] Ingestion finished for {manual_id} "
              f"({result.get('chunks')} chunks, {result.get('failed', 0)} failed)!")
    except Exception as e:
        print(f"🔥 [API] CRITICAL ERROR during ingestion: {e}")
        import traceback
        traceback.print_exc()
        _set_job(manual_id, status="failed", error=str(e))
    finally:
        _cleanup_temp_pdf(tmp_path)


@router.get("/ingest-manual/status/{manual_id}")
async def ingestion_status(manual_id: str):
    """Progress for a queued ingestion. 'unknown' once the server has restarted."""
    return _ingestion_jobs.get(manual_id, {"manual_id": manual_id, "status": "unknown"})


def _cleanup_temp_pdf(tmp_path: Optional[str]) -> None:
    """
    Delete the scratch PDF, retrying briefly while a parser releases its handle.

    Best-effort by design: on Windows the PDF backends can still hold the file
    here, and letting that raise would turn a completed ingestion into a failure.
    """
    if not tmp_path or not os.path.exists(tmp_path):
        return
    for attempt in range(5):
        try:
            os.remove(tmp_path)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
        except OSError as e:
            print(f"⚠️ [API] Could not remove temp file {tmp_path}: {e}")
            return
    print(f"⚠️ [API] Temp file still locked, leaving for OS cleanup: {tmp_path}")


@router.get("/api/manuals", response_model=list[ManualSummary])
async def list_manuals(db: Session = Depends(get_db)):
    """
    Every manual that has content in the knowledge base.

    Includes manual_ids present only as chunks (ingested before the Manual row
    existed, or whose PDF upload failed) so the UI never hides usable knowledge.
    """
    counts = qs.count_all_manual_chunks()

    manuals = {
        m.manual_id: ManualSummary(
            manual_id=m.manual_id,
            filename=m.filename,
            url=m.url,
            created_at=m.created_at,
            chunks=counts.get(m.manual_id, 0),
        )
        for m in db.query(Manual).all()
    }
    for manual_id, count in counts.items():
        if manual_id not in manuals:
            manuals[manual_id] = ManualSummary(manual_id=manual_id, chunks=count)

    return sorted(manuals.values(), key=lambda m: m.manual_id.lower())


class RenameRequest(BaseModel):
    new_manual_id: str


def _delete_local_figures(manual_id: str) -> int:
    """Remove figures written by CloudinaryService's local fallback. Cloudinary
    assets are left alone — they're shared storage and may be referenced elsewhere."""
    from services.cloudinary_service import LOCAL_DATA_DIR

    removed = 0
    manual_dir = LOCAL_DATA_DIR / "phoenix" / "manuals"
    if not manual_dir.exists():
        return 0
    for f in manual_dir.glob(f"{manual_id}_*"):
        try:
            f.unlink()
            removed += 1
        except OSError as e:
            print(f"⚠️ [API] Could not delete {f}: {e}")
    return removed


@router.delete("/api/manuals/{manual_id}")
async def delete_manual(manual_id: str, db: Session = Depends(get_db)):
    """Remove a manual and everything derived from it.

    Machines pointing at it are left in place but flagged in the response — the
    caller decides whether to relink or delete them, since a machine outliving
    its manual is a real situation on the floor.
    """
    chunks = qs.delete_manual_chunks(manual_id)
    record = db.query(Manual).filter(Manual.manual_id == manual_id).first()
    if record:
        db.delete(record)

    if not chunks and not record:
        raise HTTPException(status_code=404, detail=f"Manual '{manual_id}' not found.")

    orphaned = [m.machine_id for m in db.query(Machine).filter(Machine.manual_id == manual_id).all()]
    db.commit()

    files = _delete_local_figures(manual_id)
    _ingestion_jobs.pop(manual_id, None)
    print(f"🗑️ [API] Deleted manual {manual_id}: {chunks} chunks, {files} figures.")

    return {
        "status": "deleted",
        "manual_id": manual_id,
        "chunks_removed": chunks,
        "figures_removed": files,
        "orphaned_machines": orphaned,
    }


@router.patch("/api/manuals/{manual_id}")
async def rename_manual(manual_id: str, req: RenameRequest, db: Session = Depends(get_db)):
    """Rename a manual_id, cascading to its chunks and any machines bound to it."""
    new_id = req.new_manual_id.strip()
    if not new_id:
        raise HTTPException(status_code=400, detail="new_manual_id is required.")
    if new_id == manual_id:
        return {"status": "unchanged", "manual_id": manual_id}

    if db.query(Manual).filter(Manual.manual_id == new_id).first() or qs.count_manual_chunks(new_id):
        raise HTTPException(status_code=409, detail=f"Manual '{new_id}' already exists.")

    chunks = qs.rename_manual_chunks(manual_id, new_id)
    machines = db.query(Machine).filter(Machine.manual_id == manual_id).update(
        {Machine.manual_id: new_id}
    )
    record = db.query(Manual).filter(Manual.manual_id == manual_id).first()
    if record:
        record.manual_id = new_id
    elif not chunks:
        raise HTTPException(status_code=404, detail=f"Manual '{manual_id}' not found.")

    db.commit()
    print(f"✏️ [API] Renamed {manual_id} -> {new_id} ({chunks} chunks, {machines} machines).")
    return {
        "status": "renamed",
        "manual_id": new_id,
        "previous_manual_id": manual_id,
        "chunks_updated": chunks,
        "machines_updated": machines,
    }
