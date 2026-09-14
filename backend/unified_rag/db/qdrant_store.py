"""
Qdrant vector store — replaces pgvector for the two tables that carried an
`embedding` column: ManualChunk and InteractionMemory.

Everything else (Manual, Machine, AssistantSession, AssistantMessage) stays in
Postgres, in unified_rag/db/models.py — those are relational data with no
vector search need, and Qdrant has no joins/foreign-key story for them.

Result objects below duck-type the old SQLAlchemy models' attribute surface
(`.content`, `.page`, `.path`, `.figure_role`, ...) so retriever.py's callers
(rag.py, image_relevance.py) don't need to change how they READ a result —
only how results get fetched changes, contained entirely in retriever.py.
"""
from functools import lru_cache
from typing import Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from unified_rag.config import settings

MANUAL_CHUNKS = "manual_chunks"
INTERACTION_MEMORY = "interaction_memory"

# Seconds. Generous because ingestion pushes multi-megabyte batches; searches
# return in milliseconds and never come near it.
QDRANT_TIMEOUT_S = 120

# Points per upsert request. Tuned for this embedding size rather than picked
# round: at 2560 dims a point is ~50 KB of JSON, so 32 is ~1.6 MB per request —
# large enough to cut round-trips by an order of magnitude, small enough to stay
# well inside the timeout on a slow link.
UPSERT_BATCH = 32


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    # settings.qdrant_url always has a value (defaults to the local Docker
    # port) — actual connection failures (nothing running there, wrong Cloud
    # URL, etc.) surface from the calls below instead, which every caller in
    # this module and retriever.py already wraps in try/except.
    #
    # The explicit timeout matters at this embedding size: one 2560-dim vector
    # is ~50 KB of JSON, so even a modest batch is multiple megabytes on the
    # wire. The client's 5 s default silently turns a working upsert into a
    # ReadTimeout against a remote instance (reproduced against the configured
    # server — a 200-point batch timed out; the same batch succeeds with this).
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=QDRANT_TIMEOUT_S,
    )


def ensure_collections() -> None:
    """Create both collections if missing, with payload indexes on the fields
    every query filters by. Safe to call on every boot — no-ops once present."""
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if MANUAL_CHUNKS not in existing:
        client.create_collection(
            collection_name=MANUAL_CHUNKS,
            vectors_config=qm.VectorParams(size=settings.embedding_dim, distance=qm.Distance.COSINE),
        )
        for field, schema in [
            ("manual_id", qm.PayloadSchemaType.KEYWORD),
            ("type", qm.PayloadSchemaType.KEYWORD),
            ("page", qm.PayloadSchemaType.INTEGER),
            ("path", qm.PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(MANUAL_CHUNKS, field_name=field, field_schema=schema)

    if INTERACTION_MEMORY not in existing:
        client.create_collection(
            collection_name=INTERACTION_MEMORY,
            vectors_config=qm.VectorParams(size=settings.embedding_dim, distance=qm.Distance.COSINE),
        )
        client.create_payload_index(
            INTERACTION_MEMORY, field_name="machine_id", field_schema=qm.PayloadSchemaType.KEYWORD
        )

    # Added after the collection first shipped, so applied to existing
    # collections too. Creating an index that already exists is a no-op
    # server-side; the except only covers servers that reject the repeat.
    try:
        client.create_payload_index(
            INTERACTION_MEMORY, field_name="session_id", field_schema=qm.PayloadSchemaType.INTEGER
        )
    except Exception:
        pass


class ChunkResult:
    """One manual_chunks point. `relevance` is set by callers after search
    (cosine similarity, 0-1, higher is better) — mirrors the old
    `chunk.relevance = 1.0 - distance` pattern retriever.py already used."""
    __slots__ = ("id", "manual_id", "type", "content", "page", "path", "figure_role",
                 "parent_path", "width", "height", "kind", "render_markdown", "relevance")

    def __init__(self, id, payload: dict, score: Optional[float] = None):
        p = payload or {}
        self.id = id
        self.manual_id = p.get("manual_id")
        self.type = p.get("type")
        self.content = p.get("content")
        self.page = p.get("page")
        self.path = p.get("path")
        self.figure_role = p.get("figure_role")
        self.parent_path = p.get("parent_path")
        self.width = p.get("width")
        self.height = p.get("height")
        self.kind = p.get("kind")
        self.render_markdown = p.get("render_markdown")
        self.relevance = score


MEMORY_FIELDS = (
    "machine_id", "manual_id", "session_id", "symptom", "root_cause", "actions",
    "method", "parts_replaced", "engineer", "summary", "operator_fix", "timestamp",
)


class MemoryResult:
    """One interaction_memory point: a fix an engineer confirmed on a machine.

    Points written before structured fix records existed only carry
    `summary`/`operator_fix`; the newer fields simply come back as None.
    """
    __slots__ = ("id", "relevance") + MEMORY_FIELDS

    def __init__(self, id, payload: dict, score: Optional[float] = None):
        p = payload or {}
        self.id = id
        self.relevance = score
        for field in MEMORY_FIELDS:
            setattr(self, field, p.get(field))

    def as_dict(self) -> dict:
        """The shape the API returns and the chat's past-fix card renders."""
        return {
            "date": self.timestamp,
            "engineer": self.engineer,
            "symptom": self.symptom,
            "root_cause": self.root_cause,
            # Old points have no structured actions; their free-text fix is the
            # closest thing to "what was done".
            "actions": self.actions or self.operator_fix,
            "method": self.method,
            "parts_replaced": self.parts_replaced,
            "summary": self.summary,
            "session_id": self.session_id,
            "similarity": round(self.relevance, 3) if self.relevance is not None else None,
        }


def _chunk_payload(**fields) -> dict:
    # Qdrant payloads skip None values fine, but keeping them explicit (vs
    # **kwargs passthrough) makes the schema visible in one place.
    return {
        "manual_id": fields.get("manual_id"),
        "type": fields.get("type"),
        "content": fields.get("content"),
        "page": fields.get("page"),
        "path": fields.get("path"),
        "figure_role": fields.get("figure_role"),
        "parent_path": fields.get("parent_path"),
        "width": fields.get("width"),
        "height": fields.get("height"),
        "kind": fields.get("kind"),
        "render_markdown": fields.get("render_markdown"),
    }


def upsert_manual_chunk(vector: list, point_id=None, **fields) -> str:
    """Insert or replace one manual_chunks point. Returns the point id used —
    callers that don't care can ignore it; the migration script uses it to
    reuse the original Postgres row id, so re-running migration overwrites
    rather than duplicates."""
    pid = point_id if point_id is not None else str(uuid.uuid4())
    get_client().upsert(
        collection_name=MANUAL_CHUNKS,
        points=[qm.PointStruct(id=pid, vector=vector, payload=_chunk_payload(**fields))],
    )
    return pid


def upsert_manual_chunks(items: list) -> int:
    """Upsert many manual_chunks points in ONE request.

    Each item is {"vector": [...], "point_id": optional, **payload fields}.
    Ingesting a manual one point per call meant one network round-trip per
    chunk — a few hundred of them, serialised — which is slow against a local
    Qdrant and much worse against a remote one. Returns how many were sent.
    """
    if not items:
        return 0
    points = []
    for item in items:
        fields = {k: v for k, v in item.items() if k not in ("vector", "point_id")}
        points.append(qm.PointStruct(
            id=item.get("point_id") or str(uuid.uuid4()),
            vector=item["vector"],
            payload=_chunk_payload(**fields),
        ))
    get_client().upsert(collection_name=MANUAL_CHUNKS, points=points)
    return len(points)


def upsert_interaction_memory(vector: list, point_id=None, **fields) -> str:
    pid = point_id if point_id is not None else str(uuid.uuid4())
    payload = {field: fields.get(field) for field in MEMORY_FIELDS}
    get_client().upsert(
        collection_name=INTERACTION_MEMORY,
        points=[qm.PointStruct(id=pid, vector=vector, payload=payload)],
    )
    return pid


def _filter(must: list = None, must_not: list = None) -> Optional[qm.Filter]:
    if not must and not must_not:
        return None
    return qm.Filter(must=must or None, must_not=must_not or None)


def search_manual_chunks(vector: list, manual_id: str = None, exclude_manual_id: str = None,
                          types: list = None, limit: int = 10) -> list:
    """Cosine-similarity search over manual_chunks. `.relevance` on each result
    is Qdrant's native cosine similarity (higher = better) — equivalent to the
    old `1 - cosine_distance` the pgvector code computed by hand."""
    must, must_not = [], []
    if manual_id:
        must.append(qm.FieldCondition(key="manual_id", match=qm.MatchValue(value=manual_id)))
    if exclude_manual_id:
        must_not.append(qm.FieldCondition(key="manual_id", match=qm.MatchValue(value=exclude_manual_id)))
    if types:
        must.append(qm.FieldCondition(key="type", match=qm.MatchAny(any=types)))

    # .search() was removed in qdrant-client 1.19; .query_points() is the
    # current API and returns a QueryResponse wrapping the hit list in .points.
    resp = get_client().query_points(
        collection_name=MANUAL_CHUNKS,
        query=vector,
        query_filter=_filter(must, must_not),
        limit=limit,
        with_payload=True,
    )
    return [ChunkResult(h.id, h.payload, h.score) for h in resp.points]


def get_manual_chunks_by_path(manual_id: str, paths: list) -> list:
    """Exact lookup (not similarity search) — used to pull in a component's
    parent full-figure by its stored path."""
    if not paths:
        return []
    hits, _ = get_client().scroll(
        collection_name=MANUAL_CHUNKS,
        scroll_filter=qm.Filter(must=[
            qm.FieldCondition(key="manual_id", match=qm.MatchValue(value=manual_id)),
            qm.FieldCondition(key="path", match=qm.MatchAny(any=paths)),
        ]),
        limit=len(paths),
        with_payload=True,
    )
    return [ChunkResult(h.id, h.payload) for h in hits]


def get_manual_tables_near(manual_id: str, pages: list) -> list:
    """Table chunks for a manual on any of the given pages, ignoring the
    query entirely - used to pair a figure's numbered callouts with its
    legend/key table even when the user's wording doesn't happen to make
    that table rank on its own merits in a similarity search."""
    if not pages:
        return []
    hits, _ = get_client().scroll(
        collection_name=MANUAL_CHUNKS,
        scroll_filter=qm.Filter(must=[
            qm.FieldCondition(key="manual_id", match=qm.MatchValue(value=manual_id)),
            qm.FieldCondition(key="type", match=qm.MatchValue(value="table")),
            qm.FieldCondition(key="page", match=qm.MatchAny(any=pages)),
        ]),
        limit=50,
        with_payload=True,
    )
    return [ChunkResult(h.id, h.payload) for h in hits]


def get_manual_tables(manual_id: str, limit: int = 200) -> list:
    """Every table chunk in a manual. Used as a last resort when a figure's key
    is not on a page adjacent to the figure itself — some manuals collect all
    their keys in one place."""
    hits, _ = get_client().scroll(
        collection_name=MANUAL_CHUNKS,
        scroll_filter=qm.Filter(must=[
            qm.FieldCondition(key="manual_id", match=qm.MatchValue(value=manual_id)),
            qm.FieldCondition(key="type", match=qm.MatchValue(value="table")),
        ]),
        limit=limit,
        with_payload=True,
    )
    return [ChunkResult(h.id, h.payload) for h in hits]


def search_interaction_memory(vector: list, machine_id: str, limit: int = 2) -> list:
    resp = get_client().query_points(
        collection_name=INTERACTION_MEMORY,
        query=vector,
        query_filter=qm.Filter(must=[
            qm.FieldCondition(key="machine_id", match=qm.MatchValue(value=machine_id))
        ]),
        limit=limit,
        with_payload=True,
    )
    # The score is kept: callers need it to tell a genuine repeat of the same
    # fault from the least-unrelated fix that happens to be on file.
    return [MemoryResult(h.id, h.payload, h.score) for h in resp.points]


def delete_interaction_memory_for_session(session_id: int) -> None:
    """Remove the fix(es) filed from one chat session. Deleting a session whose
    recorded fix was wrong is how that fix stops being recalled."""
    get_client().delete(
        collection_name=INTERACTION_MEMORY,
        points_selector=qm.FilterSelector(filter=qm.Filter(must=[
            qm.FieldCondition(key="session_id", match=qm.MatchValue(value=session_id))
        ])),
    )


def count_interaction_memory() -> int:
    try:
        return get_client().count(collection_name=INTERACTION_MEMORY, exact=True).count
    except Exception as e:
        print(f"[qdrant_store] count_interaction_memory unavailable: {e}")
        return 0


def count_manual_chunks(manual_id: str) -> int:
    """0 if Qdrant is unreachable or unconfigured, same as "no chunks found" —
    used for display/listing, where the caller has no other graceful fallback
    (unlike search, which retriever.py already wraps per-call)."""
    try:
        result = get_client().count(
            collection_name=MANUAL_CHUNKS,
            count_filter=qm.Filter(must=[
                qm.FieldCondition(key="manual_id", match=qm.MatchValue(value=manual_id))
            ]),
            exact=True,
        )
        return result.count
    except Exception as e:
        print(f"[qdrant_store] count_manual_chunks unavailable: {e}")
        return 0


def count_all_manual_chunks() -> dict:
    """manual_id -> chunk count, for every manual that has any chunks.
    Qdrant has no GROUP BY, so this scrolls all points once and tallies
    client-side — fine at the size a maintenance-manual index runs at, and it
    is the same shape as the old `func.count(...).group_by(...)` query.

    Empty dict if Qdrant is unreachable or unconfigured — the manuals/machines
    listing endpoints have no other graceful fallback for this (unlike search,
    which retriever.py already wraps per-call), so a manual just shows 0
    chunks rather than 500ing the whole listing.
    """
    try:
        client = get_client()
    except Exception as e:
        print(f"[qdrant_store] count_all_manual_chunks unavailable: {e}")
        return {}

    counts: dict = {}
    offset = None
    while True:
        try:
            points, offset = client.scroll(
                collection_name=MANUAL_CHUNKS,
                with_payload=["manual_id"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
        except Exception as e:
            print(f"[qdrant_store] count_all_manual_chunks scroll failed: {e}")
            break
        for p in points:
            mid = (p.payload or {}).get("manual_id")
            if mid:
                counts[mid] = counts.get(mid, 0) + 1
        if offset is None:
            break
    return counts


def delete_manual_chunks(manual_id: str) -> int:
    """Delete every point for a manual. Returns how many existed (counted
    first since Qdrant's delete-by-filter does not report a row count)."""
    n = count_manual_chunks(manual_id)
    if n:
        get_client().delete(
            collection_name=MANUAL_CHUNKS,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                qm.FieldCondition(key="manual_id", match=qm.MatchValue(value=manual_id))
            ])),
        )
    return n


def rename_manual_chunks(old_manual_id: str, new_manual_id: str) -> int:
    """Point every chunk's manual_id payload field at the new id, in place —
    no re-embedding needed since the vectors themselves don't change."""
    n = count_manual_chunks(old_manual_id)
    if n:
        get_client().set_payload(
            collection_name=MANUAL_CHUNKS,
            payload={"manual_id": new_manual_id},
            points=qm.FilterSelector(filter=qm.Filter(must=[
                qm.FieldCondition(key="manual_id", match=qm.MatchValue(value=old_manual_id))
            ])),
        )
    return n
