"""
incident_summarizer.py — shared summarize -> embed -> archive flow for resolved
incidents. Both the Diagnostic Copilot (app/copilot_api.py) and Machine
Registry (app/machine_api.py) resolve endpoints used to duplicate this logic
independently; consolidated here so the prompt/model/embedding choice only
needs to be maintained in one place (and so both routes always write
same-dimension vectors into InteractionMemory).
"""
from datetime import datetime
from sqlalchemy.orm import Session

from unified_rag.ai_client import get_client, MODEL_CHAT
from unified_rag.embeddings.embedder import embedder
from unified_rag.db import qdrant_store as qs


def summarize_and_archive(
    history_text: str,
    operator_fix: str,
    machine_id: str,
    db: Session,
) -> str:
    """
    Summarize a resolved incident's chat history + operator fix, embed the
    summary, and write it to Qdrant's interaction_memory collection.
    Returns the generated summary text.

    `db` (a Postgres session) is accepted but unused — kept so callers don't
    need to change; interaction memory moved to Qdrant with manual_chunks,
    Postgres now only holds the relational session/message rows around this.
    """
    summary_prompt = (
        "You are a Technical Scribe. Below is a diagnostic chat history and the operator's actual fix.\n"
        "Summarize them into a concise, action-oriented technical entry (max 150 words) for future retrieval.\n\n"
        f"History:\n{history_text}\n\nOperator Fix: {operator_fix}"
    )
    res = get_client().chat.completions.create(
        model=MODEL_CHAT,
        messages=[{"role": "user", "content": summary_prompt}],
    )
    summary = res.choices[0].message.content

    qs.upsert_interaction_memory(
        embedder.embed_text(summary),
        machine_id=machine_id,
        manual_id="Historical_Knowledge",
        summary=summary,
        operator_fix=operator_fix,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    return summary
