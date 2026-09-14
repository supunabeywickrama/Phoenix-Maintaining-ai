import re

from unified_rag.db import qdrant_store as qs
from unified_rag.embeddings.embedder import embedder

# Cosine SIMILARITY below which a figure is not really "about" the query.
# Qdrant returns similarity directly (higher = better); the old pgvector code
# computed `1 - distance` by hand to get this same number, so the thresholds
# here are unchanged from before — just compared the other way round.
MIN_IMAGE_RELEVANCE = 1 - 0.62

# Similarity floor for content pulled from OTHER manuals. Deliberately higher
# than the in-manual floor: a cross-manual hit is only worth surfacing when it
# is a strong match, because it is by definition about different equipment.
MIN_CROSS_MANUAL_RELEVANCE = 1 - 0.45

# Similarity a past fix on the same machine needs before it is shown as "this
# happened before". Higher than the manual floors on purpose: a manual passage
# that is loosely related is still useful background, but telling a technician
# "last time this was the clutch" for an unrelated fault sends them the wrong
# way. No match shows nothing.
#
# Calibrated against a stored "stopped / overheating / injection module" fix
# (qwen3-embedding:4b): six rewordings of that same fault scored 0.536-0.707,
# six unrelated questions on the same machine scored 0.232-0.403. 0.47 sits in
# the gap; 0.55 was tried first and missed "the heater is too hot and the
# machine tripped" (0.536), a genuine repeat.
MIN_MEMORY_RELEVANCE = 0.47

# Part numbers as printed in parts lists: "6710415", "17C-612", "83FN-3".
# Pulled out of a question for an exact lookup, since an embedding of a bare
# code carries little meaning to match on.
_PART_NUMBER = re.compile(r"\b(?:\d{6,8}|[0-9]{1,3}[A-Z]{1,3}-\d{1,4}[A-Z]?)\b", re.IGNORECASE)


def _is_part(chunk) -> bool:
    return (chunk.figure_role or "full") == "part"


# services/callout_linker.py appends "Parts shown in this diagram: 1 = Cutter
# bar assembly; 2 = Gearbox..." to a figure's caption at ingestion. Pulled back
# out here to find that same figure's legend/key table at answer time.
_RESOLVED_PARTS_RE = re.compile(r"Parts shown in this diagram:\s*(.+)")


def _resolved_codes(content: str) -> list:
    m = _RESOLVED_PARTS_RE.search(content or "")
    if not m:
        return []
    codes = []
    for part in m.group(1).split(";"):
        code = part.split("=", 1)[0].strip().rstrip(".")
        if code:
            codes.append(code)
    return codes


class RetrievalEngine:
    def __init__(self, top_k_text=3, top_k_image=3, top_k_memory=3, top_k_cross=3,
                 top_k_table=2):
        self.top_k_text = top_k_text
        self.top_k_image = top_k_image
        self.top_k_memory = top_k_memory
        self.top_k_cross = top_k_cross
        self.top_k_table = top_k_table

    def retrieve(self, db, query: str, manual_id: str, machine_id: str = None,
                 include_other_manuals: bool = True):
        """
        Dual-Source Vector Search over Qdrant:
        1. Manual Documentation (Theoretical Knowledge)
        2. Interaction Memory (Historical Field Fixes)
        Plus an explicitly attributed sweep of the other manuals on file.

        `db` (a Postgres session) is accepted but unused here — kept so call
        sites that still pass one for the relational tables don't need to
        change; vector search itself no longer touches Postgres at all.
        """
        query_emb = embedder.embed_text(query)

        # 1. Search this manual (theory). "part" = one parts-list entry.
        try:
            text_results = qs.search_manual_chunks(
                query_emb, manual_id=manual_id, types=["text", "table", "part"], limit=self.top_k_text
            )
        except Exception as e:
            print(f"Error retrieving text: {e}")
            text_results = []

        # 1b. A part number named in the question is looked up exactly and put
        #     first - "what is 6710415?" should never depend on vector luck.
        exact = self._exact_part_matches(query, manual_id)
        exact_ids = {e.id for e in exact}
        text_results = exact + [c for c in text_results if c.id not in exact_ids]

        # 2. Images: relevance-gated, ordered full-before-parts
        image_results = self._retrieve_images(query_emb, manual_id)

        # 2a. A matched parts-list entry brings the drawing it is a callout on,
        #     so an answer about "the caliper hose" can show where ref 30 is.
        image_results = self._with_part_figures(text_results, image_results, manual_id)

        # 2b. Tables get their own slots. Sharing the text budget meant a torque
        #     schedule lost to three paragraphs of prose and was never shown, even
        #     when the question was explicitly about torque figures.
        table_results = self._retrieve_tables(query_emb, manual_id)

        # 3. Interaction memory (history): same machine only, and only fixes
        #    that genuinely resemble this report.
        historical_fixes = self._retrieve_memory(query_emb, machine_id)

        # 4. Other manuals: surfaced as clearly attributed pointers, never mixed
        #    silently into this machine's answer.
        cross_refs = []
        if include_other_manuals:
            cross_refs = self._retrieve_cross_manual(query_emb, manual_id)

        return {
            "text_chunks": text_results,
            "tables": table_results,
            "images": image_results,
            "historical_fixes": historical_fixes,
            "cross_manual": cross_refs,
        }

    def _exact_part_matches(self, query: str, manual_id: str) -> list:
        numbers = sorted({m.group(0).upper() for m in _PART_NUMBER.finditer(query or "")})
        if not numbers or not manual_id:
            return []
        try:
            return qs.get_parts_by_number(manual_id, numbers)
        except Exception as e:
            print(f"Error in exact part-number lookup: {e}")
            return []

    def _with_part_figures(self, text_results: list, image_results: list, manual_id: str) -> list:
        wanted = {}
        for c in text_results:
            if c.type == "part" and c.parent_path:
                wanted[c.parent_path] = max(wanted.get(c.parent_path, 0.0), c.relevance or 0.0)
        missing = [p for p in wanted if p not in {i.path for i in image_results}]
        if not missing:
            return image_results
        try:
            figures = qs.get_manual_chunks_by_path(manual_id, missing)
        except Exception as e:
            print(f"Error fetching figures for matched parts: {e}")
            return image_results
        for f in figures:
            f.relevance = wanted.get(f.path, 0.0)
        return self._order_full_first(image_results + figures)

    def _retrieve_memory(self, query_emb, machine_id: str):
        if not machine_id:
            return []
        try:
            hits = qs.search_interaction_memory(
                query_emb, machine_id=machine_id, limit=self.top_k_memory
            )
        except Exception as e:
            print(f"Error retrieving historical fixes: {e}")
            return []
        return [h for h in hits if (h.relevance or 0.0) >= MIN_MEMORY_RELEVANCE]

    def retrieve_many(self, db, queries: list, manual_id: str, machine_id: str = None,
                      max_text: int = 8, max_tables: int = 4):
        """Retrieval for a fault report that names several symptoms at once.

        "Stopped, X part overheating, injection module not working" is three
        faults to look up, and one embedding of the whole sentence lands near
        whichever symptom dominates it. Each query is searched separately and
        the text/table hits are merged, keeping each chunk's best score.

        Images, cross-manual pointers and memory come from the FIRST query (the
        technician's own words) only, so the relevance gate and the past-fix
        floor judge against what was actually reported.
        """
        queries = [q.strip() for q in queries if q and q.strip()]
        if not queries:
            return {"text_chunks": [], "tables": [], "images": [],
                    "historical_fixes": [], "cross_manual": []}

        base = self.retrieve(db, queries[0], manual_id, machine_id)
        text_by_id = {c.id: c for c in base["text_chunks"]}
        table_by_id = {c.id: c for c in base["tables"]}

        def keep_best(bucket: dict, chunk):
            prev = bucket.get(chunk.id)
            if prev is None or (chunk.relevance or 0) > (prev.relevance or 0):
                bucket[chunk.id] = chunk

        for q in queries[1:]:
            try:
                emb = embedder.embed_text(q)
                for c in qs.search_manual_chunks(
                    emb, manual_id=manual_id, types=["text", "table", "part"], limit=self.top_k_text
                ):
                    keep_best(text_by_id, c)
                for c in self._retrieve_tables(emb, manual_id):
                    keep_best(table_by_id, c)
            except Exception as e:
                print(f"Error retrieving for sub-query '{q[:60]}': {e}")

        by_score = lambda c: -(c.relevance or 0.0)
        base["text_chunks"] = sorted(text_by_id.values(), key=by_score)[:max_text]
        base["tables"] = sorted(table_by_id.values(), key=by_score)[:max_tables]
        return base

    def _retrieve_images(self, query_emb, manual_id: str):
        """Figures for the answer, ordered so the full drawing comes before the
        components cropped out of it."""
        try:
            candidates = qs.search_manual_chunks(
                query_emb, manual_id=manual_id, types=["image"], limit=self.top_k_image * 4
            )
        except Exception as e:
            print(f"Error retrieving images: {e}")
            return []

        picked, seen_paths = [], set()
        for chunk in candidates:  # already sorted best-first by Qdrant
            if chunk.relevance is not None and chunk.relevance < MIN_IMAGE_RELEVANCE:
                break
            # No size-based hiding: whether a figure is worth showing is decided
            # by vision at ingestion and the relevance check at answer time.
            if chunk.path in seen_paths:
                continue
            seen_paths.add(chunk.path)
            picked.append(chunk)
            if len(picked) >= self.top_k_image:
                break

        if not picked:
            return []

        # Pull in the full figure behind any matched component, so the answer can
        # orient the technician on the whole drawing before zooming in.
        parent_paths = {c.parent_path for c in picked if _is_part(c) and c.parent_path}
        have = {c.path for c in picked}
        missing = list(parent_paths - have)
        if missing:
            try:
                parents = qs.get_manual_chunks_by_path(manual_id, missing)
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

    def _retrieve_tables(self, query_emb, manual_id: str):
        """Tables that can actually be rendered, searched independently of prose."""
        try:
            candidates = qs.search_manual_chunks(
                query_emb, manual_id=manual_id, types=["table"], limit=self.top_k_table * 2
            )
        except Exception as e:
            print(f"Error retrieving tables: {e}")
            return []

        out = []
        for chunk in candidates:
            if not chunk.render_markdown:
                continue
            if chunk.relevance is not None and chunk.relevance < MIN_IMAGE_RELEVANCE:
                break
            out.append(chunk)
            if len(out) >= self.top_k_table:
                break
        return out

    @staticmethod
    def _legend_score(cand, codes: list) -> tuple:
        """(matched code count, has legend-style caption) for one table.

        A code counts only when it is a WHOLE CELL of the rendered grid, which
        is what an item-number column actually looks like. Substring matching
        on the flat text is far too loose: an engine-spec table containing
        "1.05 kW at 7 500 rpm" and "2 800 ± 200 rpm" scores hits for codes 1, 5,
        7, 2 and 8 without holding a single item number — that false match
        really did beat the true legend before this was tightened.
        """
        cells = set()
        for line in (cand.render_markdown or "").splitlines():
            for cell in line.split("|"):
                cell = cell.strip()
                if cell and cell != "---":
                    cells.add(cell)
        hits = sum(1 for c in codes if c in cells)
        # Only the title line is considered for the caption signal, and only
        # from the markdown/caption - not the whole body, where the phrase can
        # appear incidentally (or as a stale section header on older chunks).
        first_line = ((cand.content or "").splitlines() or [""])[0]
        titled = bool(re.search(r"key to fig|legend|parts? list", first_line, re.IGNORECASE))
        return hits, titled

    def find_legend_table_anywhere(self, manual_id: str, codes: list, exclude_ids=()):
        """Last-resort sweep of every table in the manual.

        Held to a much higher bar than the nearby-page search, because without
        page proximity the only evidence is the numbers themselves, and figure
        keys across a manual all use 1, 2, 3... — so a majority of the codes
        must be present AND the table must actually read like a key.
        """
        if not manual_id or len(codes) < 4:
            return None
        try:
            candidates = qs.get_manual_tables(manual_id)
        except Exception as e:
            print(f"Error sweeping tables for a legend: {e}")
            return None

        best, best_hits = None, 0
        for cand in candidates:
            if cand.id in exclude_ids or not cand.render_markdown:
                continue
            hits, _titled = self._legend_score(cand, codes)
            # Whole-cell matching is precise enough to stand on its own here;
            # the caption is not required, because a manual that prints its key
            # without the words "key"/"legend" is common and the old title data
            # cannot be trusted anyway.
            if hits >= max(4, int(len(codes) * 0.6)) and hits > best_hits:
                best, best_hits = cand, hits
        return best

    def find_legend_table(self, manual_id: str, page, codes: list, exclude_ids=()):
        """The legend/key table for a figure on `page` that carries `codes`.

        Takes the codes explicitly rather than reading them off a stored
        caption, so it also works for an image the user has just uploaded —
        there the callout numbers are read straight from the picture, and no
        ingestion-time resolution needs to have happened first.
        """
        if not manual_id or not codes or not page:
            return None
        # +-2 pages: a key usually faces its figure, but a full-page drawing can
        # push it one spread further. The scoring gate below is what keeps this
        # honest, not the narrowness of the window.
        near_pages = [p for p in range(page - 2, page + 3) if p > 0]
        try:
            candidates = qs.get_manual_tables_near(manual_id, near_pages)
        except Exception as e:
            print(f"Error searching for legend table: {e}")
            return None

        best, best_score = None, 0
        for cand in candidates:
            if cand.id in exclude_ids or not cand.render_markdown:
                continue
            hits, titled = self._legend_score(cand, codes)
            score = hits + (2 if titled else 0)
            if score > best_score:
                best, best_score = cand, score

        # Require at least 2 matching item numbers (or the caption bonus plus
        # one) so an unrelated torque table on the same page isn't pulled in
        # just for sharing a page number.
        return best if best and best_score >= 2 else None

    def attach_legend_tables(self, manual_id: str, images: list, tables: list):
        """Guarantee a numbered figure ships with the table that decodes its
        callouts, e.g. a "Key to Figure 1.1" parts list for a diagram whose
        circles just say "1", "2", "3...". Vector search finds that table only
        when the user's own wording happens to resemble it; the pairing here
        is instead by hard evidence - the same item numbers appearing in both
        - so it works regardless of how the question was phrased.

        Called AFTER the relevance gate: a legend is only worth attaching once
        its figure has already been judged relevant to answer with.
        """
        have_ids = {t.id for t in tables}
        extra = []
        for img in images:
            if (getattr(img, "figure_role", None) or "full") != "full":
                continue
            codes = _resolved_codes(img.content)
            if not codes:
                continue
            best = self.find_legend_table(manual_id, img.page, codes, exclude_ids=have_ids)
            if best:
                extra.append(best)
                have_ids.add(best.id)

        return tables + extra

    def _retrieve_cross_manual(self, query_emb, manual_id: str):
        """Strong matches living in OTHER manuals.

        Returned separately and always attributed to their manual and page: the
        point is to tell the technician where else this is documented, not to
        answer from another machine's paperwork as if it were this one's.
        """
        try:
            candidates = qs.search_manual_chunks(
                query_emb, exclude_manual_id=manual_id, types=["text", "table", "image", "part"],
                limit=self.top_k_cross * 3,
            )
        except Exception as e:
            print(f"Error retrieving cross-manual matches: {e}")
            return []

        out, seen = [], set()
        for chunk in candidates:
            if chunk.relevance is not None and chunk.relevance < MIN_CROSS_MANUAL_RELEVANCE:
                break
            key = (chunk.manual_id, chunk.page)
            if key in seen:
                continue
            seen.add(key)
            out.append(chunk)
            if len(out) >= self.top_k_cross:
                break
        return out
