"""
incident_summarizer.py — summarize -> embed -> archive a fix an engineer has
confirmed, so the next time the same fault is reported on the same machine the
chat can say what happened last time and what fixed it.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from unified_rag.ai_client import chat_text, MODEL_CHAT_LIGHT
from unified_rag.embeddings.embedder import embedder
from unified_rag.db import qdrant_store as qs

METHOD_LABELS = {
    "hands_on": "by hand, from the engineer's own experience",
    "system_guided": "by following the system's step-by-step instructions",
    "both": "partly by hand and partly by following the system's instructions",
}


def _fallback_summary(symptom: str, fix: dict) -> str:
    """Built from the fields alone. Used when the LLM returns nothing, so an
    empty string is never embedded (an empty vector-matches nothing useful and
    the record would be unrecallable)."""
    parts = []
    if symptom:
        parts.append(f"Reported: {symptom.strip()}")
    if fix.get("root_cause"):
        parts.append(f"Cause: {fix['root_cause'].strip()}")
    if fix.get("actions"):
        parts.append(f"Fixed by: {fix['actions'].strip()}")
    if fix.get("parts_replaced"):
        parts.append(f"Parts replaced: {fix['parts_replaced'].strip()}")
    return ". ".join(parts)


def summarize_and_archive(
    history_text: str,
    fix: dict,
    machine_id: str,
    manual_id: Optional[str],
    session_id: int,
    symptom: Optional[str],
    db: Session,
) -> str:
    """Summarize a resolved incident and write it to Qdrant's
    interaction_memory. Returns the summary text.

    `fix` keys: engineer, root_cause, actions, method, parts_replaced.

    `db` is accepted but unused — the relational copy of the record is written
    by the caller onto the session row.
    """
    symptom = (symptom or "").strip()
    method = fix.get("method") or ""
    prompt = (
        "Below is a maintenance chat and the fix an engineer confirmed actually worked.\n"
        "Write one concise technical entry (max 120 words) for a future technician who "
        "reports the same fault on this machine: the symptom, the real cause, and what "
        "fixed it. Use ONLY facts from the engineer's record and the chat. Do not add "
        "steps, values or part numbers that are not stated.\n\n"
        f"CHAT:\n{history_text[-6000:]}\n\n"
        f"ENGINEER'S RECORD:\n"
        f"- Reported symptom: {symptom or 'not stated'}\n"
        f"- Root cause: {fix.get('root_cause') or 'not stated'}\n"
        f"- What was done: {fix.get('actions') or 'not stated'}\n"
        f"- Method: {METHOD_LABELS.get(method, 'not stated')}\n"
        f"- Parts replaced: {fix.get('parts_replaced') or 'none stated'}\n"
    )
    try:
        # chat_text, not get_client().chat.completions: the OpenAI-compatible
        # endpoint lets local Qwen3 spend its whole budget on hidden reasoning
        # and return empty content, which is what this used to archive.
        summary = (chat_text(MODEL_CHAT_LIGHT, prompt, max_tokens=400, temperature=0.1) or "").strip()
    except Exception as e:
        print(f"⚠️ [IncidentSummarizer] Summary generation failed, using field summary: {e}")
        summary = ""
    if not summary:
        summary = _fallback_summary(symptom, fix)

    # Embedded text leads with the symptom: a later report is phrased as a
    # symptom ("stopped, overheating"), so that is what it has to land near.
    embed_text = "\n".join(filter(None, [
        f"Symptom: {symptom}" if symptom else "",
        f"Root cause: {fix.get('root_cause', '')}".strip(),
        f"Fix: {fix.get('actions', '')}".strip(),
        summary,
    ]))

    qs.upsert_interaction_memory(
        embedder.embed_text(embed_text),
        machine_id=machine_id,
        manual_id=manual_id,
        session_id=session_id,
        symptom=symptom,
        root_cause=fix.get("root_cause"),
        actions=fix.get("actions"),
        method=method or None,
        parts_replaced=fix.get("parts_replaced") or None,
        engineer=fix.get("engineer"),
        summary=summary,
        # Kept populated for anything still reading the old single-field shape.
        operator_fix=fix.get("actions"),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    return summary
