from sqlalchemy.orm import Session
from unified_rag.db.models import ManualChunk, InteractionMemory
from unified_rag.embeddings.embedder import embedder

# Cosine distance beyond which a figure is not really "about" the query. Vector
# search always returns its top-k, so without this a manual whose only diagrams
# are of the hydraulic pump will happily attach them to a question about the
# control panel. A wrong diagram is worse than no diagram.
MAX_IMAGE_DISTANCE = 0.62

# Distance gate for content pulled from OTHER manuals. Deliberately tighter than
# the in-manual gate: a cross-manual hit is only worth surfacing when it is a
# strong match, because it is by definition about different equipment.
MAX_CROSS_MANUAL_DISTANCE = 0.45

# A component crop this small is unreadable on a phone in a plant; SAM sometimes
# isolates a bolt head or a fragment of a label. It stays in the index (the
# caption is still searchable) but is not shown in chat.
MIN_PART_SIDE = 80
MIN_PART_AREA = 12000


def _is_part(chunk) -> bool:
    return (chunk.figure_role or "full") == "part"


def _too_small_to_show(chunk) -> bool:
    """Only applied to parts; a full figure is shown whatever its size."""
    if not _is_part(chunk):
        return False
    w, h = chunk.width, chunk.height
    if not w or not h:
        return False  # pre-dates the size columns, so do not guess
    return w < MIN_PART_SIDE or h < MIN_PART_SIDE or (w * h) < MIN_PART_AREA


class RetrievalEngine:
    def __init__(self, top_k_text=3, top_k_image=3, top_k_memory=2, top_k_cross=3,
                 top_k_table=2):
        self.top_k_text = top_k_text
        self.top_k_image = top_k_image
        self.top_k_memory = top_k_memory
        self.top_k_cross = top_k_cross
        self.top_k_table = top_k_table

    def retrieve(self, db: Session, query: str, manual_id: str, machine_id: str = None,
                 include_other_manuals: bool = True):
        """
        Dual-Source Vector Search:
        1. Manual Documentation (Theoretical Knowledge)
        2. Interaction Memory (Historical Field Fixes)
        Plus an explicitly attributed sweep of the other manuals on file.
        """
        query_emb = embedder.embed_text(query)

        # 1. Search this manual (theory)
        try:
            text_results = db.query(ManualChunk).filter(
                ManualChunk.manual_id == manual_id,
                ManualChunk.type.in_(["text", "table"])
            ).order_by(
                ManualChunk.embedding.cosine_distance(query_emb)
            ).limit(self.top_k_text).all()
        except Exception:
            text_results = []

        # 2. Images: distance-gated, size-filtered, ordered full-before-parts
        image_results = self._retrieve_images(db, query_emb, manual_id)

        # 2b. Tables get their own slots. Sharing the text budget meant a torque
        #     schedule lost to three paragraphs of prose and was never shown, even
        #     when the question was explicitly about torque figures.
        table_results = self._retrieve_tables(db, query_emb, manual_id)

        # 3. Interaction memory (history), scoped to the machine
        historical_fixes = []
        try:
            if machine_id:
                historical_fixes = db.query(InteractionMemory).filter(
                    InteractionMemory.machine_id == machine_id
                ).order_by(
                    InteractionMemory.embedding.cosine_distance(query_emb)
                ).limit(self.top_k_memory).all()
        except Exception as e:
            print(f"Error retrieving historical fixes: {e}")

        # 4. Other manuals: surfaced as clearly attributed pointers, never mixed
        #    silently into this machine's answer.
        cross_refs = []
        if include_other_manuals:
            cross_refs = self._retrieve_cross_manual(db, query_emb, manual_id)

        return {
            "text_chunks": text_results,
            "tables": table_results,
            "images": image_results,
            "historical_fixes": historical_fixes,
            "cross_manual": cross_refs,
        }

    def _retrieve_images(self, db: Session, query_emb, manual_id: str):
        """Figures for the answer, ordered so the full drawing comes before the
        components cropped out of it."""
        try:
            distance = ManualChunk.embedding.cosine_distance(query_emb).label("distance")
            rows = (
                db.query(ManualChunk, distance)
                .filter(
                    ManualChunk.manual_id == manual_id,
                    ManualChunk.type == "image",
                )
                .order_by(distance)
                .limit(self.top_k_image * 4)
                .all()
            )
        except Exception as e:
            print(f"Error retrieving images: {e}")
            return []

        picked, seen_paths = [], set()
        for chunk, dist in rows:
            if dist is not None and dist > MAX_IMAGE_DISTANCE:
                break
            if chunk.path in seen_paths or _too_small_to_show(chunk):
                continue
            seen_paths.add(chunk.path)
            chunk.relevance = 1.0 - float(dist or 0.0)
            picked.append(chunk)
            if len(picked) >= self.top_k_image:
                break

        if not picked:
            return []

        # Pull in the full figure behind any matched component, so the answer can
        # orient the technician on the whole drawing before zooming in.
        parent_paths = {c.parent_path for c in picked if _is_part(c) and c.parent_path}
        have = {c.path for c in picked}
        missing = parent_paths - have
        if missing:
            try:
                parents = db.query(ManualChunk).filter(
                    ManualChunk.manual_id == manual_id,
                    ManualChunk.type == "image",
                    ManualChunk.path.in_(missing),
                ).all()
                for parent in parents:
                    parent.relevance = max(
                        (c.relevance for c in picked if c.parent_path == parent.path),
                        default=0.0,
                    )
                    picked.append(parent)
            except Exception as e:
                print(f"Error fetching parent figures: {e}")

        return self._order_full_first(picked)

    @staticmethod
    def _order_full_first(chunks):
        """Group each full figure with its own parts, strongest group first.

        The order matters: it becomes [IMAGE_0], [IMAGE_1] ... in the prompt, and
        the answer is instructed to introduce the full diagram before its parts.
        """
        by_path = {c.path: c for c in chunks}
        groups = {}
        for c in chunks:
            key = c.parent_path if (_is_part(c) and c.parent_path in by_path) else c.path
            groups.setdefault(key, []).append(c)

        def group_score(members):
            return max((getattr(m, "relevance", 0.0) or 0.0) for m in members)

        ordered = []
        for _key, members in sorted(groups.items(), key=lambda kv: group_score(kv[1]), reverse=True):
            full = [m for m in members if not _is_part(m)]
            parts = sorted(
                [m for m in members if _is_part(m)],
                key=lambda m: (getattr(m, "relevance", 0.0) or 0.0),
                reverse=True,
            )
            ordered.extend(full + parts)
        return ordered

    def _retrieve_tables(self, db: Session, query_emb, manual_id: str):
        """Tables that can actually be rendered, searched independently of prose."""
        try:
            distance = ManualChunk.embedding.cosine_distance(query_emb).label("distance")
            rows = (
                db.query(ManualChunk, distance)
                .filter(
                    ManualChunk.manual_id == manual_id,
                    ManualChunk.type == "table",
                    ManualChunk.render_markdown.isnot(None),
                )
                .order_by(distance)
                .limit(self.top_k_table * 2)
                .all()
            )
        except Exception as e:
            print(f"Error retrieving tables: {e}")
            return []

        out = []
        for chunk, dist in rows:
            if dist is not None and dist > MAX_IMAGE_DISTANCE:
                break
            chunk.relevance = 1.0 - float(dist or 0.0)
            out.append(chunk)
            if len(out) >= self.top_k_table:
                break
        return out

    def _retrieve_cross_manual(self, db: Session, query_emb, manual_id: str):
        """Strong matches living in OTHER manuals.

        Returned separately and always attributed to their manual and page: the
        point is to tell the technician where else this is documented, not to
        answer from another machine's paperwork as if it were this one's.
        """
        try:
            distance = ManualChunk.embedding.cosine_distance(query_emb).label("distance")
            rows = (
                db.query(ManualChunk, distance)
                .filter(
                    ManualChunk.manual_id != manual_id,
                    ManualChunk.type.in_(["text", "table", "image"]),
                )
                .order_by(distance)
                .limit(self.top_k_cross * 3)
                .all()
            )
        except Exception as e:
            print(f"Error retrieving cross-manual matches: {e}")
            return []

        out, seen = [], set()
        for chunk, dist in rows:
            if dist is not None and dist > MAX_CROSS_MANUAL_DISTANCE:
                break
            key = (chunk.manual_id, chunk.page)
            if key in seen:
                continue
            seen.add(key)
            chunk.relevance = 1.0 - float(dist or 0.0)
            out.append(chunk)
            if len(out) >= self.top_k_cross:
                break
        return out
