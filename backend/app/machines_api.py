"""
Machine registry.

Deliberately thin compared with the Zynaptrix product: Phoenix has no sensors, so
there are no datasheets to parse and no model to train. A machine here exists only
to scope a question to the right manual and to give past fixes something to attach to.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from unified_rag.db.database import SessionLocal
from unified_rag.db.models import Machine, ManualChunk

router = APIRouter(prefix="/api/machines", tags=["Machine Registry"])
logger = logging.getLogger(__name__)

_PLACEHOLDER_IDS = {"", "null", "undefined", "none", "nan"}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _clean_id(value: str, field: str) -> str:
    """
    Reject placeholder identifiers.

    A frontend that interpolates an unset selection into a URL sends the literal
    text "null"; without this it would be stored and looked up as a real id.
    """
    cleaned = (value or "").strip()
    if cleaned.lower() in _PLACEHOLDER_IDS:
        raise HTTPException(status_code=400, detail=f"Invalid {field}: {value!r}")
    return cleaned


class MachineIn(BaseModel):
    machine_id: str = Field(..., description="Unique identifier, e.g. PRESS-04")
    name: str
    location: Optional[str] = ""
    manual_id: str = Field(..., description="Manual that documents this machine")


class MachineOut(BaseModel):
    machine_id: str
    name: str
    location: Optional[str] = ""
    manual_id: str
    manual_chunks: int = 0   # 0 means the manual has no ingested content yet

    class Config:
        from_attributes = True


def _chunk_counts(db: Session) -> dict:
    return dict(
        db.query(ManualChunk.manual_id, func.count(ManualChunk.id))
        .group_by(ManualChunk.manual_id)
        .all()
    )


@router.get("", response_model=List[MachineOut])
async def list_machines(db: Session = Depends(get_db)):
    counts = _chunk_counts(db)
    return [
        MachineOut(
            machine_id=m.machine_id,
            name=m.name,
            location=m.location or "",
            manual_id=m.manual_id,
            manual_chunks=counts.get(m.manual_id, 0),
        )
        for m in db.query(Machine).order_by(Machine.machine_id).all()
    ]


@router.post("", response_model=MachineOut)
async def upsert_machine(payload: MachineIn, db: Session = Depends(get_db)):
    """Create a machine, or update it if the id already exists."""
    machine_id = _clean_id(payload.machine_id, "machine_id")
    manual_id = _clean_id(payload.manual_id, "manual_id")

    existing = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if existing:
        existing.name = payload.name
        existing.location = payload.location or ""
        existing.manual_id = manual_id
        machine = existing
    else:
        machine = Machine(
            machine_id=machine_id,
            name=payload.name,
            location=payload.location or "",
            manual_id=manual_id,
        )
        db.add(machine)
    db.commit()
    db.refresh(machine)

    chunks = _chunk_counts(db).get(manual_id, 0)
    if chunks == 0:
        # Not an error: operators often register equipment before uploading its
        # manual. Answers will carry the missing-documentation disclaimer until then.
        logger.warning(
            "Machine %s points at manual %s, which has no ingested content yet.",
            machine_id, manual_id,
        )

    return MachineOut(
        machine_id=machine.machine_id,
        name=machine.name,
        location=machine.location or "",
        manual_id=machine.manual_id,
        manual_chunks=chunks,
    )


@router.post("/delete/{machine_id}")
async def delete_machine(machine_id: str, db: Session = Depends(get_db)):
    machine = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    db.delete(machine)
    db.commit()
    logger.info("Removed machine %s", machine_id)
    return {"status": "deleted", "machine_id": machine_id}
