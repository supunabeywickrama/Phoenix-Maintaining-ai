"""
Phoenix assistant: ask about a machine problem, get a manual-grounded answer, and
optionally be walked through the repair step by step.

Differences from the Zynaptrix assistant this is adapted from:

- No GUIDE/ONBOARDING/SEARCH intents. Those explain the Zynaptrix product itself
  and mean nothing to a Phoenix technician.
- The wizard is requested explicitly via `mode`, not inferred from the word
  "step" appearing in the query, which fired on "what are the steps of the
  cycle?" and missed "walk me through fixing it".
- No fallback to a hardcoded manual. If the selected equipment has no ingested
  documentation the answer carries a disclaimer, because silently answering from
  a different customer's manual is worse than admitting the gap.
"""
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from unified_rag.ai_client import get_client, MODEL_CHAT, MODEL_CHAT_LIGHT
from unified_rag.config import settings
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import (
    AssistantMessage,
    AssistantSession,
    Machine,
    ManualChunk,
)
from unified_rag.retrieval.rag import RAGGenerator, RAGMode
from services.incident_summarizer import summarize_and_archive

router = APIRouter(prefix="/api/assistant", tags=["Assistant"])
logger = logging.getLogger(__name__)

# Built once: RAGGenerator holds a RetrievalEngine and is stateless per call.
_rag = RAGGenerator()

# Flag understood by RAGGenerator.generate_response — it strips the marker before
# embedding the query and prepends a "no manual on file" disclaimer to the answer.
MISSING_MANUAL_FLAG = "[DISCLAIMER_REQUIRED: MISSING_MANUAL]"

HISTORY_TURNS = 10


class AskMode(str, Enum):
    ANSWER = "answer"   # concise diagnostic summary
    WIZARD = "wizard"   # conversational step-by-step repair


class AssistantQuery(BaseModel):
    query: str
    session_id: Optional[int] = None
    machine_id: Optional[str] = None
    manual_id: Optional[str] = None   # ask a manual directly, without a machine
    mode: AskMode = AskMode.ANSWER


class SessionOut(BaseModel):
    id: int
    machine_id: Optional[str]
    title: str
    timestamp: str
    resolved: bool = False


class MessageOut(BaseModel):
    role: str
    content: str
    type: str
    step_data: Optional[Dict[str, Any]] = None
    images: List[str] = []
    timestamp: str


class ResolveRequest(BaseModel):
    operator_fix: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _generate_title(query: str) -> str:
    try:
        res = get_client().chat.completions.create(
            model=MODEL_CHAT_LIGHT,
            messages=[{
                "role": "user",
                "content": f"Give a 3-5 word title for a maintenance chat that starts with: '{query}'. Reply with the title only.",
            }],
            max_tokens=24,
        )
        return res.choices[0].message.content.strip().strip('"')[:80]
    except Exception as e:
        logger.warning("Title generation failed: %s", e)
        return query[:60] or "New enquiry"


def _absolute_image_urls(paths: List[str]) -> List[str]:
    """
    Figures are uploaded to Cloudinary during ingestion, so `path` is normally
    already absolute. Legacy relative paths are mapped onto the API host.
    """
    api_url = settings.api_url.rstrip("/")
    urls = []
    for path in paths:
        if path.startswith("http"):
            urls.append(path)
            continue
        web_path = path.replace("\\", "/").replace("data/", "/static/")
        if not web_path.startswith("/"):
            web_path = "/" + web_path
        urls.append(f"{api_url}{web_path}")
    return urls


def _resolve_manual(db: Session, machine_id: Optional[str], manual_id: Optional[str]) -> Optional[str]:
    """Explicit manual wins; otherwise fall back to the machine's manual."""
    if manual_id and manual_id.strip():
        return manual_id.strip()
    if machine_id and machine_id.strip():
        machine = db.query(Machine).filter(Machine.machine_id == machine_id.strip()).first()
        if machine:
            return machine.manual_id
    return None


def _has_content(db: Session, manual_id: str) -> int:
    return db.query(ManualChunk).filter(ManualChunk.manual_id == manual_id).count()


# ── Sessions ─────────────────────────────────────────────────────────────────

@router.get("/sessions", response_model=List[SessionOut])
async def list_sessions(db: Session = Depends(get_db)):
    sessions = db.query(AssistantSession).order_by(desc(AssistantSession.updated_at)).all()
    return [
        SessionOut(
            id=s.id,
            machine_id=s.machine_id,
            title=s.title,
            timestamp=s.updated_at,
            resolved=bool(s.resolved_at),
        )
        for s in sessions
    ]


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: int, db: Session = Depends(get_db)):
    db.query(AssistantMessage).filter(AssistantMessage.session_id == session_id).delete()
    db.query(AssistantSession).filter(AssistantSession.id == session_id).delete()
    db.commit()
    return {"status": "deleted", "session_id": session_id}


@router.get("/sessions/{session_id}/history", response_model=List[MessageOut])
async def session_history(session_id: int, db: Session = Depends(get_db)):
    # Ordered by id, not timestamp: timestamps have second granularity, so two
    # messages in the same second could otherwise come back reversed.
    messages = (
        db.query(AssistantMessage)
        .filter(AssistantMessage.session_id == session_id)
        .order_by(AssistantMessage.id)
        .all()
    )
    return [
        MessageOut(
            role=m.role,
            content=m.content,
            type=m.type or "text",
            step_data=json.loads(m.step_data) if m.step_data else None,
            images=json.loads(m.images) if m.images else [],
            timestamp=m.timestamp,
        )
        for m in messages
    ]


# ── Ask ──────────────────────────────────────────────────────────────────────

@router.post("")
async def ask(req: AssistantQuery, db: Session = Depends(get_db)):
    """Answer a maintenance question, grounded in the selected machine's manual."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    session = None
    if req.session_id:
        session = db.query(AssistantSession).filter(AssistantSession.id == req.session_id).first()
    if session is None:
        session = AssistantSession(
            machine_id=req.machine_id,
            title=_generate_title(req.query),
            created_at=_now(),
            updated_at=_now(),
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    elif req.machine_id:
        session.machine_id = req.machine_id

    machine_id = req.machine_id or session.machine_id

    db.add(AssistantMessage(
        session_id=session.id, role="user", content=req.query,
        type="text", timestamp=_now(),
    ))
    db.commit()

    # Prior turns, oldest first, excluding the message just stored.
    prior = (
        db.query(AssistantMessage)
        .filter(AssistantMessage.session_id == session.id)
        .order_by(desc(AssistantMessage.id))
        .offset(1)
        .limit(HISTORY_TURNS)
        .all()
    )
    history = list(reversed(prior))

    manual_id = _resolve_manual(db, machine_id, req.manual_id)
    images: List[str] = []

    if manual_id:
        chunk_count = _has_content(db, manual_id)
        if chunk_count == 0:
            logger.warning("Manual %s has no ingested content — answering with disclaimer.", manual_id)
        else:
            logger.info("[Provenance] %d manual chunks found for %s.", chunk_count, manual_id)

        mode = RAGMode.CONVERSATIONAL_WIZARD if req.mode == AskMode.WIZARD else RAGMode.SUMMARY
        query = req.query if chunk_count else f"{MISSING_MANUAL_FLAG} {req.query}"
        history_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in history)

        try:
            result = _rag.generate_response(
                query, manual_id, machine_id or manual_id,
                mode=mode, chat_history=history_text,
            )
            # Returned verbatim: a second summarising pass strips the [IMAGE_n]
            # markers the UI needs to interleave diagrams.
            answer = result.get("answer") or "No response generated."
            images = _absolute_image_urls(result.get("images", []))
            source = f"Manual: {manual_id}" + ("" if chunk_count else " (no content on file)")
        except Exception as e:
            logger.error("RAG failed for manual %s: %s", manual_id, e)
            raise HTTPException(status_code=502, detail=f"Retrieval failed: {e}")
    else:
        # Nothing selected — answer conversationally and point them at the selector
        # rather than guessing which equipment they mean.
        chat = [{"role": ("assistant" if m.role == "agent" else "user"), "content": m.content} for m in history]
        res = get_client().chat.completions.create(
            model=MODEL_CHAT_LIGHT,
            messages=[{
                "role": "system",
                "content": (
                    "You are a maintenance assistant for Phoenix Industries. No machine is "
                    "selected, so you have no manual to draw on. Answer briefly from general "
                    "engineering knowledge and tell the user to select a machine for guidance "
                    "specific to their equipment."
                ),
            }] + chat + [{"role": "user", "content": req.query}],
            max_tokens=600,
        )
        answer = res.choices[0].message.content
        source = "General knowledge (no machine selected)"

    db.add(AssistantMessage(
        session_id=session.id, role="agent", content=answer,
        type="wizard_step" if req.mode == AskMode.WIZARD else "text",
        images=json.dumps(images) if images else None,
        timestamp=_now(),
    ))
    session.updated_at = _now()
    db.commit()

    return {
        "role": "agent",
        "content": answer,
        "session_id": session.id,
        "machine_id": machine_id,
        "manual_id": manual_id,
        "images": images,
        "mode": req.mode.value,
        "context_source": source,
        "timestamp": _now(),
    }


# ── Resolve: turn a successful repair into retrievable memory ────────────────

@router.post("/sessions/{session_id}/resolve")
async def resolve_session(session_id: int, req: ResolveRequest, db: Session = Depends(get_db)):
    """
    Record what actually fixed the machine.

    The summary is embedded into InteractionMemory, which RetrievalEngine queries
    alongside the manual — so the next person asking about this machine sees the
    fix that worked, not just the documentation.
    """
    session = db.query(AssistantSession).filter(AssistantSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.machine_id:
        raise HTTPException(
            status_code=400,
            detail="This session is not tied to a machine, so the fix cannot be filed against one.",
        )
    if not req.operator_fix.strip():
        raise HTTPException(status_code=400, detail="operator_fix is required")

    messages = (
        db.query(AssistantMessage)
        .filter(AssistantMessage.session_id == session_id)
        .order_by(AssistantMessage.id)
        .all()
    )
    history_text = "\n".join(f"{m.role}: {m.content}" for m in messages)

    try:
        summary = summarize_and_archive(history_text, req.operator_fix, session.machine_id, db)
        session.resolved_at = _now()
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("Failed to archive fix for session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "resolved", "session_id": session_id, "summary": summary}


# ── Report ───────────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/report")
async def session_report(session_id: int, db: Session = Depends(get_db)):
    """Structured maintenance report for a session, for PDF export."""
    session = db.query(AssistantSession).filter(AssistantSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = (
        db.query(AssistantMessage)
        .filter(AssistantMessage.session_id == session_id)
        .order_by(AssistantMessage.id)
        .all()
    )
    if not messages:
        raise HTTPException(status_code=400, detail="Session has no messages")

    conversation, all_images = [], []
    for m in messages:
        conversation.append(f"{m.role.upper()}: {m.content}")
        if m.images:
            try:
                all_images.extend(json.loads(m.images))
            except (ValueError, TypeError):
                pass

    extracted = {"problem": "N/A", "diagnosis": "N/A", "solution_steps": []}
    try:
        res = get_client().chat.completions.create(
            model=MODEL_CHAT_LIGHT,
            messages=[
                {"role": "system", "content": (
                    "You are a technical documentation expert. Turn this maintenance "
                    "conversation into a report. Reply with JSON containing: problem, "
                    "diagnosis, solution_steps (list of strings)."
                )},
                {"role": "user", "content": "\n\n".join(conversation)},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        extracted = json.loads(res.choices[0].message.content) or extracted
    except Exception as e:
        logger.error("Report generation failed: %s", e)

    seen, ordered_images = set(), []
    for url in all_images:
        if url not in seen:
            seen.add(url)
            ordered_images.append(url)

    return {
        "sessionId": session.id,
        "machineId": session.machine_id,
        "title": session.title,
        "problemDescription": extracted.get("problem", "N/A"),
        "diagnosis": extracted.get("diagnosis", "N/A"),
        "solutionSteps": extracted.get("solution_steps", []),
        "images": [{"url": u, "caption": f"Figure {i + 1}"} for i, u in enumerate(ordered_images)],
        "resolvedAt": session.resolved_at,
        "timestamp": session.created_at,
    }
