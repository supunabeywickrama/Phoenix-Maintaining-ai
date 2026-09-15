"""
Callout resolution — turn the codes printed on a diagram into part names.

Engineering drawings label parts with a bare code on a leader line ("P-100",
"7", "3a") and put the actual name in a parts table or a paragraph elsewhere in
the manual. Retrieval that only sees the figure therefore can't answer "where is
the pressure relief valve" — the diagram never says those words.

This runs during ingestion, while every text and table chunk for the manual is
still in memory, and appends the resolved names to each figure's caption before
it gets embedded. No second embedding pass, no extra vision calls.
"""
import re
from collections import defaultdict

from unified_rag.ai_client import chat_json, MODEL_CHAT_LIGHT
from services.llm_json import loads_tolerant

MAX_SNIPPETS_PER_CODE = 4
SNIPPET_WINDOW = 160


def _code_pattern(code: str):
    """Match the code as a standalone token. Bare numbers need tighter anchoring
    ('7' otherwise matches inside '1700 rpm' or a torque spec)."""
    esc = re.escape(code)
    if code.isdigit():
        return re.compile(
            # (7)  7.  7)  7 -
            rf"(?<![\w.-])[\(\[]?{esc}[\)\].:\-–]"
            # item 7 / Item: 7 / item-7. The separator was a bare \s+ before,
            # which matched none of the manual's legend rows: they render as
            # "Item: 7 | Component: ...", so every numeric callout in a figure
            # key silently failed to resolve and the diagram kept only its
            # meaningless bare numbers.
            rf"|\bitems?\s*[:.\-]?\s*{esc}(?!\w)",
            re.IGNORECASE,
        )
    return re.compile(rf"(?<![\w-]){esc}(?![\w-])", re.IGNORECASE)


def _gather_snippets(codes, text_chunks) -> dict:
    """code -> list of surrounding text snippets from the manual."""
    found = defaultdict(list)
    for code in codes:
        pattern = _code_pattern(code)
        for chunk in text_chunks:
            content = chunk.get("content") or ""
            if not content:
                continue
            for m in pattern.finditer(content):
                start = max(0, m.start() - SNIPPET_WINDOW)
                end = min(len(content), m.end() + SNIPPET_WINDOW)
                snippet = " ".join(content[start:end].split())
                page = chunk.get("page")
                found[code].append(f"(page {page}) …{snippet}…")
                if len(found[code]) >= MAX_SNIPPETS_PER_CODE:
                    break
            if len(found[code]) >= MAX_SNIPPETS_PER_CODE:
                break
    return found


def resolve_callouts(codes, text_chunks) -> dict:
    """code -> short part name. Only codes with real textual evidence are
    returned; anything unresolved is left out rather than guessed at."""
    codes = sorted({c.strip() for c in codes if c and c.strip()})
    if not codes or not text_chunks:
        return {}

    snippets = _gather_snippets(codes, text_chunks)
    if not snippets:
        print("      🔗 [Callouts] No textual references found for any code.")
        return {}

    evidence = "\n\n".join(
        f"CODE {code}:\n" + "\n".join(f"  - {s}" for s in snips)
        for code, snips in snippets.items()
    )
    prompt = (
        "You are reading a machine maintenance manual. Each CODE below is a callout "
        "printed on a diagram. The snippets are places that code appears in the manual text "
        "and tables.\n\n"
        f"{evidence}\n\n"
        "Return a JSON object with one key \"parts\": an array of "
        '{"code": "...", "name": "short part name (2-6 words)"}.\n'
        "Rules:\n"
        "- Only include a code if the snippets genuinely identify what it is.\n"
        "- Omit codes where the evidence is ambiguous or is just a measurement/quantity.\n"
        "- Use the manual's own terminology. Do not invent names."
    )

    try:
        raw = chat_json(MODEL_CHAT_LIGHT, prompt, max_tokens=900, temperature=0.0)
        data = loads_tolerant(raw) if raw else None
    except Exception as e:
        print(f"      ⚠️ [Callouts] Resolution call failed: {e}")
        return {}

    resolved = {}
    if isinstance(data, dict):
        for entry in data.get("parts") or []:
            if not isinstance(entry, dict):
                continue
            code, name = str(entry.get("code", "")).strip(), str(entry.get("name", "")).strip()
            if code and name and code in snippets:
                resolved[code] = name

    if resolved:
        print(f"      🔗 [Callouts] Resolved {len(resolved)}/{len(codes)}: "
              + ", ".join(f"{c}={n}" for c, n in resolved.items()))
    return resolved


def link_image_chunks(chunks) -> int:
    """Append resolved part names to each figure caption, in place.

    Those names are what makes the figure findable by part name instead of only
    by the code stamped on it. Returns how many figures were enriched.
    """
    # Figures already named from their own parts list are exact; guessing names
    # from text snippets could only add a second, worse listing.
    # Scanned figures are skipped too: their numbers were read from circles on
    # the drawing, and guessing names from nearby text invented "parts" for a
    # photographed controller screen from its setting values.
    image_chunks = [c for c in chunks if c.get("type") == "image"
                    and not (c.get("metadata") or {}).get("parts_resolved")
                    and not (c.get("metadata") or {}).get("scanned")]
    text_chunks = [c for c in chunks if c.get("type") in ("text", "table")]
    if not image_chunks or not text_chunks:
        return 0

    all_codes = {code for c in image_chunks for code in (c.get("metadata") or {}).get("codes", [])}
    if not all_codes:
        return 0

    resolved = resolve_callouts(all_codes, text_chunks)
    if not resolved:
        return 0

    enriched = 0
    for chunk in image_chunks:
        codes = (chunk.get("metadata") or {}).get("codes", [])
        named = [(c, resolved[c]) for c in codes if c in resolved]
        if not named:
            continue
        listing = "; ".join(f"{code} = {name}" for code, name in named)
        chunk["content"] = (chunk.get("content") or "") + f"\nParts shown in this diagram: {listing}."
        chunk.setdefault("metadata", {})["resolved_parts"] = dict(named)
        enriched += 1

    return enriched
