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
import base64
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from unified_rag.ai_client import (
    get_client, chat_text, chat_json, MODEL_CHAT, MODEL_CHAT_LIGHT, MODEL_VISION,
)
from unified_rag.embeddings.embedder import embedder
from services.llm_json import loads_tolerant
from unified_rag.config import settings
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import (
    AssistantMessage,
    AssistantSession,
    Machine,
)
from unified_rag.db import qdrant_store
from unified_rag.retrieval.rag import RAGGenerator, RAGMode
from services.cloudinary_service import CloudinaryService
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


class ChatIntent(str, Enum):
    """Why the technician opened this chat.

    A fault on the floor and a training question need different answers from the
    same manual: one wants a diagnosis and a safe repair path, the other wants an
    explanation. Asking once at the start beats inferring it per message, which
    got it wrong on phrasings like "why does the pump cavitate?".
    """
    TROUBLESHOOT = "troubleshoot"
    LEARN = "learn"


class AssistantQuery(BaseModel):
    query: str
    session_id: Optional[int] = None
    machine_id: Optional[str] = None
    manual_id: Optional[str] = None   # ask a manual directly, without a machine
    mode: AskMode = AskMode.ANSWER
    intent: Optional[ChatIntent] = None   # set on the first message of a session


class SessionOut(BaseModel):
    id: int
    machine_id: Optional[str]
    intent: Optional[str] = None
    title: str
    timestamp: str
    resolved: bool = False


class MessageOut(BaseModel):
    role: str
    content: str
    attachments: List[Dict[str, Any]] = []
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
        title = chat_text(
            MODEL_CHAT_LIGHT,
            f"Give a 3-5 word title for a maintenance chat that starts with: '{query}'. "
            "Reply with the title only.",
            max_tokens=200, temperature=0.2,
        )
        title = (title or "").strip().strip('"').splitlines()[0] if title else ""
        return title[:80] or (query[:60] or "New enquiry")
    except Exception as e:
        logger.warning("Title generation failed: %s", e)
        return query[:60] or "New enquiry"


# An uploaded diagram is matched against the manual's own indexed figures before
# anything is said about it. Above STRONG we name the manual figure as the same
# drawing; between POSSIBLE and STRONG it is offered as the closest thing found
# and flagged as unconfirmed; below POSSIBLE nothing is claimed at all.
FIGURE_MATCH_STRONG = 0.55
FIGURE_MATCH_POSSIBLE = 0.38


def _describe_uploaded_image(image_b64: str) -> dict:
    """Structured read of the user's image: what it is, and which callout codes
    are printed on it. Deliberately does NOT ask what the numbers mean — that
    is what the manual's legend is for, and guessing it is the single biggest
    source of confident, wrong answers about a numbered diagram."""
    prompt = (
        "Look at this image from a technical manual and return JSON with exactly these keys:\n"
        '  "kind": one of "diagram", "schematic", "chart", "flowchart", "exploded_view", '
        '"photo", "table".\n'
        '  "component": short name of the main thing shown (3-6 words), based ONLY on shapes '
        "and any printed text you can actually read.\n"
        '  "codes": array of every numbered/coded callout printed on the image, as strings, '
        'in order (e.g. ["1","2","3"]). Scan the whole image; do not stop early.\n'
        '  "visible_text": array of any words actually printed on the image (labels, drawing '
        'numbers like "CA-0601", dimensions).\n'
        '  "description": 60-120 words describing the LAYOUT and what is drawn - shapes, how '
        "they connect, left to right. Describe the geometry, not the identity of numbered "
        "parts.\n\n"
        "CRITICAL: a bare number on a leader line does not tell you what that part is. Never "
        "state that a number 'is' a particular component. Report the numbers in \"codes\" and "
        "leave their meaning to the manual."
    )
    try:
        raw = chat_json(MODEL_VISION, prompt, image_b64=image_b64, max_tokens=1200, temperature=0.0)
        data = loads_tolerant(raw) if raw else None
    except Exception as e:
        logger.warning("Uploaded-image description failed: %s", e)
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        "kind": str(data.get("kind") or "diagram"),
        "component": str(data.get("component") or "").strip(),
        "codes": [str(c).strip() for c in (data.get("codes") or []) if str(c).strip()],
        "visible_text": [str(t).strip() for t in (data.get("visible_text") or []) if str(t).strip()],
        "description": str(data.get("description") or "").strip(),
    }


def _match_uploaded_figure(manual_id: str, desc: dict) -> dict:
    """Find the manual's own figure that the uploaded image most resembles, and
    pull in that figure's legend table.

    This is the difference between "an LLM looks at a picture and speculates"
    and "we located this exact drawing in your manual and can read its key":
    the matched figure's stored caption and legend are real manual content, so
    the numbers get their true meanings instead of invented ones.
    """
    if not manual_id:
        return {}
    probe = " ".join(filter(None, [
        desc.get("component", ""),
        " ".join(desc.get("visible_text") or []),
        desc.get("description", ""),
    ])).strip()
    if not probe:
        return {}

    try:
        emb = embedder.embed_text(probe)
        hits = qdrant_store.search_manual_chunks(
            emb, manual_id=manual_id, types=["image"], limit=5
        )
    except Exception as e:
        logger.warning("Figure match lookup failed: %s", e)
        return {}

    hits = [h for h in hits if h.path]
    if not hits:
        return {}
    best = hits[0]
    score = best.relevance or 0.0
    if score < FIGURE_MATCH_POSSIBLE:
        return {}

    # Look the legend up with the codes read off the UPLOADED image, not the
    # ones resolved for the stored figure at ingestion time. The upload is the
    # thing being asked about, its numbers are right there in the picture, and
    # this path then works even for manuals ingested before callout resolution
    # was fixed.
    legend = None
    try:
        codes = desc.get("codes") or []
        legend = _rag.retriever.find_legend_table(manual_id, best.page, codes)
        legend_near = legend is not None
        if legend is None:
            # Nothing near the figure — sweep the whole manual, under a much
            # stricter bar (see find_legend_table_anywhere). A table found this
            # way shares the figure's item numbers but NOT its page, so it may
            # belong to a different drawing; the caller must present it as
            # unconfirmed rather than as this figure's key.
            legend = _rag.retriever.find_legend_table_anywhere(manual_id, codes)
    except Exception as e:
        logger.warning("Legend lookup for matched figure failed: %s", e)

    return {
        "figure": best,
        "score": score,
        "confident": score >= FIGURE_MATCH_STRONG,
        "legend": legend,
        # False when the key was found by sweeping the whole manual rather than
        # on a page beside the figure — same numbers, unproven connection.
        "legend_near": legend_near if legend is not None else False,
    }


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
    return qdrant_store.count_manual_chunks(manual_id)


def _original_topic(db: Session, session_id: int) -> Optional[str]:
    """The session's first user message - the actual topic being discussed.

    Wizard continuations ("next step", "I'm stuck", "teach me step by step")
    carry no topical content of their own. Embedding them alone for retrieval
    drifts onto whatever generic content happens to sit nearest in the vector
    index, rather than staying on what was actually asked about.
    """
    first = (
        db.query(AssistantMessage)
        .filter(AssistantMessage.session_id == session_id, AssistantMessage.role == "user")
        .order_by(AssistantMessage.id.asc())
        .first()
    )
    return first.content if first else None


# ── Sessions ─────────────────────────────────────────────────────────────────

@router.get("/sessions", response_model=List[SessionOut])
async def list_sessions(db: Session = Depends(get_db)):
    sessions = db.query(AssistantSession).order_by(desc(AssistantSession.updated_at)).all()
    return [
        SessionOut(
            id=s.id,
            machine_id=s.machine_id,
            intent=s.intent,
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
            attachments=json.loads(m.attachments) if m.attachments else [],
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
            intent=(req.intent.value if req.intent else ChatIntent.TROUBLESHOOT.value),
            title=_generate_title(req.query),
            created_at=_now(),
            updated_at=_now(),
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    elif req.machine_id:
        session.machine_id = req.machine_id
    if req.intent:
        session.intent = req.intent.value

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
    attachments: List[Dict[str, Any]] = []
    cross_manual: List[Dict[str, Any]] = []

    if manual_id:
        chunk_count = _has_content(db, manual_id)
        if chunk_count == 0:
            logger.warning("Manual %s has no ingested content — answering with disclaimer.", manual_id)
        else:
            logger.info("[Provenance] %d manual chunks found for %s.", chunk_count, manual_id)

        intent = req.intent.value if req.intent else (session.intent or ChatIntent.TROUBLESHOOT.value)
        if req.mode == AskMode.WIZARD:
            # A learning walkthrough is not a repair: the repair wizard opens with
            # lockout/tagout, which is the wrong first move when nothing is broken.
            mode = (
                RAGMode.LEARNING_WALKTHROUGH
                if intent == ChatIntent.LEARN.value
                else RAGMode.CONVERSATIONAL_WIZARD
            )
        elif intent == ChatIntent.LEARN.value:
            mode = RAGMode.LEARNING
        else:
            mode = RAGMode.SUMMARY
        query = req.query if chunk_count else f"{MISSING_MANUAL_FLAG} {req.query}"
        history_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in history)

        # Wizard turns are continuations of one topic, not new questions, so
        # retrieval is anchored on what the session actually opened with —
        # see _original_topic(). Ordinary Q&A keeps using the live query: each
        # new question there is legitimately free to change topic.
        retrieval_query = None
        if req.mode == AskMode.WIZARD:
            topic = _original_topic(db, session.id)
            if topic and topic.strip() and topic.strip() != req.query.strip():
                retrieval_query = f"{topic}. {req.query}"

        try:
            result = _rag.generate_response(
                query, manual_id, machine_id or manual_id,
                mode=mode, chat_history=history_text, retrieval_query=retrieval_query,
            )
            # Returned verbatim: a second summarising pass strips the [IMAGE_n]
            # markers the UI needs to interleave diagrams.
            answer = result.get("answer") or "No response generated."
            images = _absolute_image_urls(result.get("images", []))
            cross_manual = result.get("cross_manual", []) or []
            # Figure URLs are stored repo-relative when Cloudinary is off, so the
            # same absolutising the image list gets has to reach inside attachments.
            attachments = []
            for asset in result.get("attachments", []) or []:
                asset = dict(asset)
                if asset.get("url"):
                    asset["url"] = _absolute_image_urls([asset["url"]])[0]
                attachments.append(asset)
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
                    "selected, so you have no manual to draw on. "
                    + (
                        "The technician wants to understand how something works: explain it "
                        "clearly from general engineering knowledge. "
                        if (req.intent or session.intent) == ChatIntent.LEARN.value
                        else "Answer briefly from general engineering knowledge. "
                    )
                    + "Tell the user to select a machine for guidance specific to their equipment."
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
        attachments=json.dumps(attachments) if attachments else None,
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
        # Everything shown with this reply, in presentation order: full view first,
        # then its components, then the tables backing the numbers.
        "attachments": attachments,
        # Where else this topic is documented. Attributed, never merged into the
        # answer as though it described this machine.
        "cross_manual": cross_manual,
        "mode": req.mode.value,
        "intent": session.intent,
        "context_source": source,
        "timestamp": _now(),
    }


@router.post("/ask-with-image")
async def ask_with_image(
    query: str = Form(...),
    image: UploadFile = File(...),
    session_id: Optional[int] = Form(None),
    machine_id: Optional[str] = Form(None),
    manual_id: Optional[str] = Form(None),
    intent: Optional[ChatIntent] = Form(None),
    db: Session = Depends(get_db),
):
    """Ask about a photo or diagram attached directly in chat.

    Separate from POST / (JSON body) rather than an extra field on it, because
    FastAPI cannot mix a Pydantic JSON body with a multipart file upload on one
    endpoint. This is for an ad-hoc image the technician has in hand right now —
    a phone photo of a leak, a page they photographed — not the manual's own
    figures, which already flow through ingestion and ordinary retrieval.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    MAX_UPLOAD_BYTES = 10 * 1024 * 1024
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Image too large (10 MB limit).")

    session = None
    if session_id:
        session = db.query(AssistantSession).filter(AssistantSession.id == session_id).first()
    if session is None:
        session = AssistantSession(
            machine_id=machine_id,
            intent=(intent.value if intent else ChatIntent.TROUBLESHOOT.value),
            title=_generate_title(query),
            created_at=_now(),
            updated_at=_now(),
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    elif machine_id:
        session.machine_id = machine_id
    if intent:
        session.intent = intent.value

    resolved_machine_id = machine_id or session.machine_id
    resolved_manual_id = _resolve_manual(db, resolved_machine_id, manual_id)

    # Stored under the same "phoenix/manuals" convention as ingested figures, so
    # it goes through the same Cloudinary-or-local path (see cloudinary_service.py).
    cloud = CloudinaryService()
    public_id = f"chat_upload_{session.id}_{int(datetime.now().timestamp())}"
    uploaded_url = cloud.upload_image(image_bytes, public_id, folder="phoenix/chat_uploads")
    user_images = _absolute_image_urls([uploaded_url]) if uploaded_url else []

    db.add(AssistantMessage(
        session_id=session.id, role="user", content=query,
        type="text", images=json.dumps(user_images) if user_images else None,
        timestamp=_now(),
    ))
    db.commit()

    # Prior turns for conversational continuity, same window as the text-only path.
    prior = (
        db.query(AssistantMessage)
        .filter(AssistantMessage.session_id == session.id)
        .order_by(desc(AssistantMessage.id))
        .offset(1)
        .limit(HISTORY_TURNS)
        .all()
    )
    history_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in reversed(prior))

    # Manual text is pulled in as background, same retriever the text-only path
    # uses, but there is no figure/table relevance gate here: the user's own
    # photo is the primary subject, manual text is supporting context for it,
    # not a set of candidate attachments competing for slots.
    manual_context = ""
    chunk_count = 0
    if resolved_manual_id:
        chunk_count = _has_content(db, resolved_manual_id)
        if chunk_count:
            try:
                retrieved = _rag.retriever.retrieve(
                    db, query, resolved_manual_id, resolved_machine_id, include_other_manuals=False
                )
                for c in retrieved.get("text_chunks", [])[:3]:
                    manual_context += f"--- Manual, page {c.page} ---\n{c.content}\n\n"
            except Exception as e:
                logger.warning("Manual context lookup failed for image question: %s", e)

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Read the upload, then find it in the manual. Doing this BEFORE the answer
    # is what stops the model speculating about numbered callouts: if the same
    # drawing is indexed, its real caption and legend are available as fact.
    desc = _describe_uploaded_image(b64)
    match = _match_uploaded_figure(resolved_manual_id, desc) if desc else {}

    upload_block = ""
    if desc:
        codes = ", ".join(desc.get("codes") or []) or "none detected"
        seen = ", ".join(desc.get("visible_text") or []) or "none"
        upload_block = (
            "WHAT THE ATTACHED IMAGE CONTAINS (read from the image itself):\n"
            f"- Appears to be: {desc.get('component') or 'unidentified'} ({desc.get('kind')})\n"
            f"- Callout numbers printed on it: {codes}\n"
            f"- Text printed on it: {seen}\n"
            f"- Layout: {desc.get('description')}\n\n"
        )

    match_block = ""
    attachments: List[Dict[str, Any]] = []
    if match:
        fig = match["figure"]
        confident = match["confident"]
        legend = match.get("legend")
        match_block = (
            ("MATCHING FIGURE IN THIS MANUAL — this is the same drawing, and the "
             "description below is the manual's own, so prefer it over your own reading "
             "of the image:\n"
             if confident else
             "CLOSEST FIGURE FOUND IN THIS MANUAL — this is a probable, NOT confirmed, "
             "match. Say so, and only use it if it genuinely corresponds to what is in "
             "the attached image:\n")
            + f"(page {fig.page}, similarity {match['score']:.2f})\n{fig.content}\n\n"
        )
        attachments.append({
            "tag": "IMAGE_0", "type": "image",
            "kind": getattr(fig, "kind", None) or "diagram",
            "role": getattr(fig, "figure_role", None) or "full",
            "title": f"Manual figure, page {fig.page}",
            "page": fig.page,
            "url": _absolute_image_urls([fig.path])[0] if fig.path else None,
        })
        if legend is not None and getattr(legend, "render_markdown", None):
            if match.get("legend_near"):
                match_block += (
                    "LEGEND / KEY FOR THAT FIGURE (printed beside the figure) — this is the "
                    "ONLY authoritative source for what each callout number means:\n"
                    f"{legend.render_markdown}\n\n"
                )
            else:
                # Found by sweeping the manual: same item numbers, but not
                # printed with this figure, so it may key a different drawing.
                match_block += (
                    f"POSSIBLE KEY, from page {legend.page} — it uses the same item numbers as "
                    "the attached image, but it is NOT printed beside this figure and may "
                    "belong to a different drawing. You may offer it as a lead, but you MUST "
                    "say it is unconfirmed and must NOT state its names as this figure's "
                    f"parts:\n{legend.render_markdown}\n\n"
                )
            attachments.append({
                "tag": "TABLE_0", "type": "table", "kind": "table", "role": None,
                "title": (
                    (((legend.content or "").splitlines() or [""])[0][:70])
                    or f"Key, page {legend.page}"
                ) + ("" if match.get("legend_near") else " (unconfirmed)"),
                "page": legend.page,
                "markdown": legend.render_markdown,
            })

    system_prompt = (
        "You are a maintenance engineer helping a technician who has just attached a photo or "
        "diagram in chat.\n"
        f"CONVERSATION SO FAR:\n{history_text}\n\n"
        + upload_block
        + match_block
        + (
            f"RELEVANT MANUAL TEXT (page-cited background — use it if it applies to what is in "
            f"the photo, ignore it if it does not):\n{manual_context}\n\n"
            if manual_context else
            "No manual is ingested for this machine, or nothing in it looked relevant — answer "
            "from the image and general engineering knowledge, and say so rather than guessing "
            "at manual-specific part numbers or torque values you cannot see.\n\n"
        )
        + "TASK: Look at the attached image and answer the technician's question about it.\n"
        "1. Describe what the image actually shows before diagnosing anything.\n"
        "2. If it shows a fault (damage, leak, wear, wrong assembly), say what is wrong and why.\n"
        "3. Ground any part names or specs in the manual text above where it genuinely matches; "
        "do not invent a part number or value that is not there.\n"
        "4. If the image is unclear, blurry, or does not show enough to answer, say exactly what "
        "additional photo or angle would help, rather than guessing.\n"
        "5. Cite the manual page as (page N) when you use it.\n"
        "6. NUMBERED CALLOUTS — THE RULE THAT MATTERS MOST HERE: a number on a leader line does "
        "not tell you what the part is. If a LEGEND is given above, use it and only it: say "
        "'(3) clutch housing' because the key says so. If there is NO legend above, you MUST "
        "NOT assign meanings — list the numbers you can see and say plainly that the key for "
        "this figure is not in the indexed manual, and ask which page the key is on. Writing "
        "'11 is the blade cap' from the shape of a drawing is a fabrication, even when it "
        "sounds reasonable.\n"
        "7. Do not narrate your reasoning. Give the answer directly, with no preamble about "
        "what you are about to do."
    )

    try:
        answer = chat_text(
            MODEL_VISION, query, image_b64=b64, system=system_prompt,
            max_tokens=1200, temperature=0.2,
        ) or "Could not analyze that image."
    except Exception as e:
        logger.error("Vision Q&A failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Image analysis failed: {e}")

    db.add(AssistantMessage(
        session_id=session.id, role="agent", content=answer,
        type="text", timestamp=_now(),
        attachments=json.dumps(attachments) if attachments else None,
    ))
    session.updated_at = _now()
    db.commit()

    return {
        "role": "agent",
        "content": answer,
        "session_id": session.id,
        "machine_id": resolved_machine_id,
        "manual_id": resolved_manual_id,
        "user_image": user_images[0] if user_images else None,
        "images": [a["url"] for a in attachments if a.get("type") == "image" and a.get("url")],
        "attachments": attachments,
        "mode": "answer",
        "intent": session.intent,
        "context_source": (
            # Naming the matched figure is the point of the feature: the
            # technician should see WHICH manual drawing this was read against,
            # not just that "a manual" was consulted.
            f"Manual: {resolved_manual_id} · matched figure on page {match['figure'].page}"
            f"{'' if match['confident'] else ' (probable match)'}"
            if match else
            f"Manual: {resolved_manual_id}" if manual_context
            else "Photo analysis (no manual context)"
        ),
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
