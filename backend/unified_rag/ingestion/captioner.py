"""
Vision captioning for extracted figures.

Produces a retrieval-optimised description AND the raw callout codes printed on
the diagram (e.g. "P-100", "7", "V-220"). Industrial drawings routinely label
parts with a bare code and a leader line, with the actual part name living in a
table or paragraph elsewhere in the manual — services/callout_linker.py resolves
those codes afterwards so a technician searching the part *name* still finds the
diagram.
"""
import base64
import json
import os
import re
from pathlib import Path
from typing import Optional

from unified_rag.ai_client import chat_json, chat_text, MODEL_VISION
from services.llm_json import loads_tolerant

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PAGE_TEXT_PREVIEW = 900


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _resolve_local(path: str) -> Path:
    """Local figures are stored as a repo-relative 'data/...' path (see
    CloudinaryService._upload_local). Resolve against the backend dir so this
    works regardless of the process's working directory."""
    p = Path(path)
    return p if p.is_absolute() else (BACKEND_DIR / p)


# Longest side sent to the vision model. A full exploded-view page stored at
# 300 DPI is ~2000x2900 px; the stored figure keeps that, but sending it whole
# costs a large image-token budget for no gain in reading the drawing.
VISION_MAX_SIDE = 1800


def _image_b64(image_path: str) -> Optional[str]:
    """Base64 for the vision call. Fetches URLs, reads local paths."""
    try:
        if _is_url(image_path):
            import requests
            r = requests.get(image_path, timeout=30)
            r.raise_for_status()
            data = r.content
        else:
            with open(_resolve_local(image_path), "rb") as f:
                data = f.read()
    except Exception as e:
        print(f"      ❌ [Vision] Could not load image {image_path}: {e}")
        return None
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None and max(img.shape[:2]) > VISION_MAX_SIDE:
            scale = VISION_MAX_SIDE / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".png", img)
            if ok:
                data = buf.tobytes()
    except Exception:
        pass  # send the original bytes
    return base64.b64encode(data).decode("utf-8")


# Parts-list abbreviations and the words a description spells them out as.
_ABBREVIATIONS = {
    "assy": "assembly", "asm": "assembly", "cyl": "cylinder", "brkt": "bracket",
    "hdl": "handle", "wshr": "washer", "hsg": "housing", "mtg": "mounting", "adj": "adjuster",
}


def _name_words(text: str) -> set:
    """Comparable words: lower-case, abbreviations spelled out, plural 's' off."""
    out = set()
    for w in re.findall(r"[a-z]{3,}", (text or "").lower()):
        w = _ABBREVIATIONS.get(w, w)
        # nuts -> nut, hoses -> hose; but not glass -> glas.
        out.add(w[:-1] if len(w) >= 4 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def _citation_problems(caption: str, groups: list) -> list:
    """Every "(N)" / "[N]" part reference in a caption that disagrees with the
    parts list, as plain sentences the model can be told to fix.

    Zero tolerance. An earlier version let a quarter of references be wrong
    so that loose phrasing would not be rejected, and exactly that let "the
    caliper mount (5)" through when the list says part 5 is the STATOR ASSY.,
    and "the main unit (19)" when it is INSULATION. A single wrong name tells
    a technician the wrong part.

    The name checked is the few words immediately before the bracket (the
    noun the number is attached to), not the whole sentence, so "the piston
    holds the washer (7)" is caught when part 7 is the piston.
    """
    if not caption:
        return []
    names, labels = {}, {}
    for g in groups:
        if g.get("ref"):
            descs = [r.get("description") or "" for r in g["rows"] if r.get("description")]
            names[g["ref"]] = set().union(*(_name_words(d) for d in descs)) if descs else set()
            labels[g["ref"]] = (" / ".join(dict.fromkeys(descs)) or "an unnamed part").rstrip(".")

    problems = []
    for m in re.finditer(r"([A-Za-z][A-Za-z .\-/']{0,80})[(\[](\d{1,3}(?:\s*,\s*\d{1,3})*)[)\]]", caption):
        # A sentence break ends the phrase - but not an abbreviation's full stop
        # ("BRAKE ASSY. (26)"), hence the case-sensitive capital check.
        phrase = re.split(r"[,;:]|(?-i:(?<![A-Z]))\.\s|\band\b|\bwith\b|\bvia\b|\bto\b|\bon\b|\bby\b",
                          m.group(1).strip(), flags=re.IGNORECASE)[-1].strip()
        near = _name_words(" ".join(phrase.split()[-3:]))
        for ref in re.findall(r"\d{1,3}", m.group(2)):
            if ref not in names:
                problems.append(f"({ref}) is not a part in the list - remove it.")
            elif names[ref] and not (names[ref] & near):
                problems.append(f"\"{phrase} ({ref})\" is wrong - the list calls part {ref} "
                                f"{labels[ref]}.")
    return list(dict.fromkeys(problems))


MIN_KEPT_SENTENCES = 2


def _keep_correct_sentences(caption: str, groups: list) -> str:
    """Drop only the sentences that name a part wrongly; keep the rest.

    Used when a rewrite still gets a name wrong. Seen for real: the Brake
    view twice called part 5 a "mounting bracket" (the list says STATOR ASSY.)
    in one sentence while its other sentences were right - discarding the
    whole description threw the correct ones away too. Returns "" when too
    little correct text is left to be worth storing.
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", (caption or "").strip())
    kept = [x for x in sentences if x and not _citation_problems(x, groups)]
    if len(kept) < MIN_KEPT_SENTENCES:
        return ""
    return " ".join(kept)


class ImageCaptioner:
    def describe(self, image_path: str, metadata: dict = None) -> dict:
        """Returns {'caption': str, 'codes': [str], 'component': str, 'kind': str}."""
        metadata = metadata or {}
        page = metadata.get("page", "Unknown")
        # Section headers can span lines ("EUROPEAN ACCESSORIES" / "(Sound Option)").
        section = " ".join(str(metadata.get("section") or "Unknown Section").split())
        label = metadata.get("label", "Diagram")
        parent_ctx = metadata.get("parent_context", "")
        page_text = (metadata.get("page_text") or "").strip()

        context_str = f"This is a technical illustration labelled '{label}' on page {page} of the manual."
        if section != "Unknown Section":
            context_str += f" It sits in the section: '{section}'."
        if parent_ctx:
            context_str += f" Context: {parent_ctx}."

        print(f"      📸 [Vision] Captioning {label} (page {page}) via {MODEL_VISION}")

        b64 = _image_b64(image_path)
        if not b64:
            return {"caption": "", "codes": [], "component": label, "kind": "diagram"}

        # Grounding text from the SAME page. Caught this for real: without it, a
        # diagram of a petrol two-stroke tea-pruning machine got captioned as a
        # "pneumatic reciprocating saw" — the vision model pattern-matched the
        # blade shape to a common stock category instead of using what the
        # manual itself says is on the page. A crop with no manual text nearby
        # still gets a caption; it just has less to check itself against.
        grounding = ""
        if page_text:
            snippet = " ".join(page_text.split())[:PAGE_TEXT_PREVIEW]
            grounding = (
                f"\n\nTEXT FROM THIS SAME PAGE OF THE MANUAL (ground your description in this - "
                f"do not describe a different machine, mechanism or power source than what it "
                f"says):\n\"{snippet}\"\n"
            )

        prompt = (
            f"You are a senior industrial maintenance engineer. {context_str}{grounding}\n\n"
            "Study the image and return a JSON object with exactly these keys:\n"
            '  "kind": what this image IS - one of "diagram", "schematic", "chart", '
            '"flowchart", "exploded_view", "photo", "table".\n'
            '  "component": short name of the main component shown (3-6 words).\n'
            '  "codes": array of EVERY numbered or coded callout printed on the image, including '
            'plain numbers in circles or with leader lines (e.g. "1", "2", ... up to whatever the '
            'highest number visible is), plus any alphanumeric codes (e.g. "P-100", "V-220"). Scan '
            'the whole image systematically left-to-right, top-to-bottom - do not stop after the '
            'first few. Codes only, no descriptions.\n'
            '  "leader_labels": array of any TEXT labels connected to the diagram by a leader line '
            'or pointer (e.g. "Harness attachment point", "Drive shaft tube") - the exact wording '
            'shown, not numbered callouts (those go in "codes").\n'
            '  "caption": a dense 100-180 word description written for search retrieval. Cover: '
            'what the component is, its function, and how its parts connect. Use the page text '
            'above to get the machine, power source and terminology right. Use standard engineering '
            'terminology a technician would search for.\n\n'
            "Describe only what is actually visible. Do not invent part numbers, and do not guess "
            "a different type of machine or power source than the page text states.\n\n"
            "NUMBERED CALLOUTS - READ CAREFULLY: a bare number on a leader line does NOT tell you "
            "what the part is. The manual says what it is, in a separate key/legend table that you "
            "cannot see. So NEVER write a sentence of the form 'N is the <part>' for a numbered "
            "callout unless the page text above literally states it. Put the numbers in \"codes\" "
            "and describe the drawing around them instead. Guessing here is worse than silence: it "
            "produces confident, wrong part lists that contradict the manual's own legend."
        )

        # A figure whose parts list was found takes its own path (below): the
        # generic prompt is built for drawings whose numbers are unknown.
        parts = metadata.get("parts_list") or {}
        groups = parts.get("groups") or []
        if groups:
            return self._describe_parts_view(b64, metadata, parts, groups, page, label)

        raw = None
        try:
            # 1200 was tight once "leader_labels" and "scan the whole image
            # systematically" were added - a dense composite with a dozen
            # callouts needs room to actually enumerate them all rather than
            # stopping at the first few.
            raw = chat_json(MODEL_VISION, prompt, image_b64=b64, max_tokens=1800, temperature=0.1)
            data = loads_tolerant(raw) if raw else None
        except Exception as e:
            print(f"      ⚠️ [Vision] Structured caption failed for {label}: {e}")
            data = None

        VALID_KINDS = {"diagram", "schematic", "chart", "flowchart",
                       "exploded_view", "photo", "table"}
        if isinstance(data, dict) and data.get("caption"):
            component = str(data.get("component") or label).strip()
            codes = [str(c).strip() for c in (data.get("codes") or []) if str(c).strip()]
            leader_labels = [str(c).strip() for c in (data.get("leader_labels") or []) if str(c).strip()]
            caption = str(data["caption"]).strip()
            kind = str(data.get("kind", "")).strip().lower()
            kind = kind if kind in VALID_KINDS else "diagram"
        else:
            # Structured mode failed — fall back to a plain prose caption rather
            # than storing an empty chunk that can never be retrieved.
            print(f"      ↩️ [Vision] Falling back to prose caption for {label}")
            try:
                caption = (chat_text(
                    MODEL_VISION,
                    f"{context_str}{grounding}\n\nDescribe this technical component in 80-150 words for a "
                    "maintenance technician: what it is, its function, visible labels and part codes, "
                    "and the faults this diagram helps diagnose.",
                    image_b64=b64, max_tokens=900, temperature=0.2,
                ) or "").strip()
            except Exception as e:
                # One figure's failed call must not abort the whole manual:
                # captioning runs inside asyncio.gather, where an unhandled
                # error here failed every other figure's ingestion with it.
                print(f"      ⚠️ [Vision] Prose caption failed for {label} (page {page}): {e}")
                caption = ""
            component, codes, leader_labels, kind = label, [], [], "diagram"

        if not caption and not groups:
            return {"caption": "", "codes": [], "component": label, "kind": "diagram"}
        if not caption:
            # The vision caption failed, but the parts list is verified text from
            # the manual - enough on its own to make the figure findable, so the
            # figure is not dropped for want of a description.
            caption = "Exploded parts view: " + (" ".join(filter(None, [parts.get("title"), parts.get("subtitle")])) or label) + "."
            kind = "exploded_view"

        # Numbers printed in the PDF's text layer inside the figure are exact;
        # the vision model's reading of small callout digits is not.
        if metadata.get("callouts"):
            codes = list(metadata["callouts"])

        parts_line = ""
        if groups:
            # The manual's own title beats a component name the model made up.
            heading = " ".join(filter(None, [parts.get("title"), parts.get("subtitle")]))
            if heading:
                component = heading
            # Written here from the parts list, never by the model, so it cannot
            # drift from the printed table. This exact "Parts shown in this
            # diagram: N = ...; " shape is what retrieval reads back to pair
            # the figure with its parts-list table (retriever._resolved_codes).
            entries = []
            for g in groups:
                if not g.get("ref"):
                    continue
                names = [r["description"] for r in g["rows"] if r.get("description")]
                numbers = [r["part_number"] for r in g["rows"] if r.get("part_number")]
                label_text = " / ".join(dict.fromkeys(names)) or "part"
                entries.append(f"{g['ref']} = {label_text} ({', '.join(numbers)})".replace(";", ","))
            parts_line = "\nParts shown in this diagram: " + "; ".join(entries) + "."
            if not parts.get("verified", True):
                parts_line += " (Parts list read by vision from a scanned page - unverified.)"
            missing = parts.get("missing_callouts") or []
            if missing:
                parts_line += f"\nCallouts with no entry in the parts list: {', '.join(missing)}."

        drawing = metadata.get("drawing_code")
        pretty_kind = kind.replace("_", " ").title()
        header = f"### {component} ({pretty_kind}) — page {page}"
        if drawing:
            header += f" — drawing {drawing}"
        if section != "Unknown Section":
            header += f" (section: {section})"
        code_line = f"\nCallout codes shown: {', '.join(codes)}" if codes and not groups else ""
        label_line = f"\nLeader-line labels: {', '.join(leader_labels)}" if leader_labels else ""
        return {
            "parts_resolved": bool(groups),
            "caption": f"{header}\n\n{caption}{parts_line}{code_line}{label_line}",
            # Both feed services/callout_linker.py: numeric/alphanumeric codes
            # resolve against "(1)"/"item 1"-style manual text, free-text leader
            # labels resolve as literal phrase matches against the same text.
            "codes": codes + leader_labels,
            "component": component,
            "kind": kind,
        }

    def _describe_parts_view(self, b64: str, metadata: dict, parts: dict, groups: list,
                             page, label: str) -> dict:
        """Caption an exploded parts view whose parts list is known.

        Kept separate from the generic prompt because that one failed on these
        figures in testing:
        - asking the model to enumerate 30+ callouts in JSON came back truncated,
          so the Cab Windows view (35 parts) got no description at all;
        - its plain-text fallback had no parts list and used the section name,
          which the layout order leaves stale, so the Sound Option view was
          described as "the Debris Ejector Kit... removes debris via airflow".
        Here the manual's own title and list are the anchor, the answer is
        plain text, and the section name is never used.
        """
        drawing = metadata.get("drawing_code") or ""
        # Where the sheet title could not be read, the drawing number names it.
        # Falling back to a generic label produced captions that called the
        # drawing itself "This Parts list ...".
        heading = (" ".join(filter(None, [parts.get("title"), parts.get("subtitle")]))
                   or (f"drawing {drawing}" if drawing else "this exploded view"))
        page_text = " ".join((metadata.get("page_text") or "").split())[:PAGE_TEXT_PREVIEW]
        refs = [g for g in groups if g.get("ref")]
        listing = "\n".join(
            f"  {g['ref']}: " + " / ".join(dict.fromkeys(
                r["description"] for r in g["rows"] if r.get("description")
            ))
            for g in refs[:80]
        )
        prompt = (
            f"This image is the exploded parts view \"{heading}\""
            f"{f' (drawing {drawing})' if drawing else ''} from a machine parts manual.\n"
            f"Text printed on the same page: \"{page_text}\"\n\n"
            "ITS PARTS LIST - authoritative, read from the manual's own table "
            f"(ref number: description):\n{listing}\n\n"
            "Return JSON {\"caption\": str}. The caption is 100-180 words for a maintenance "
            "technician: first what this assembly is and what it is for, then how the listed parts "
            "fit together as the drawing shows, naming each part by the list's description and its "
            "ref number in round brackets, e.g. 'the brake assembly (26) mounts on the caliper "
            "mount (27)'.\n"
            "RULES:\n"
            f"- The assembly is \"{heading}\". Never call it a different kit, system or machine.\n"
            "- For a numbered part use ONLY its description from the list, with its own number.\n"
            "- If the drawing does not show how a part connects, just name it as part of the kit. "
            "Do not invent a function, a connection, a wire or a label that is not visible.\n"
            "- No part numbers, no lists, no headings inside the caption."
        )
        print(f"      📸 [Vision] Captioning parts view '{heading}' (page {page}, {len(refs)} parts)")

        def ask(text: str) -> str:
            try:
                # JSON mode, not plain text: in plain-text mode the local qwen3-vl
                # can ignore "no thinking", spend the whole budget on hidden
                # reasoning and return nothing (done_reason=length, empty
                # content) - that left 7 of 8 parts views with no description in
                # testing. JSON mode makes it answer directly (~5 s per figure).
                data = loads_tolerant(chat_json(MODEL_VISION, text, image_b64=b64,
                                                max_tokens=900, temperature=0.1) or "")
                return str(data.get("caption") or "").strip() if isinstance(data, dict) else ""
            except Exception as e:
                print(f"      ⚠️ [Vision] Parts-view caption failed for page {page}: {e}")
                return ""

        # Every "(N)" the model writes is checked against the list. On any
        # mismatch it gets one rewrite, told exactly which names were wrong;
        # if the rewrite is still wrong the description is not stored at all.
        caption = ask(prompt)
        problems = _citation_problems(caption, groups)
        if caption and problems:
            print(f"      🔁 [Vision] Page {page}: {len(problems)} wrong part name(s), asking for a "
                  f"rewrite: {problems[:3]}")
            caption = ask(
                prompt
                + f"\n\nYOUR PREVIOUS CAPTION:\n{caption}\n\nIT NAMED THESE PARTS WRONG:\n- "
                + "\n- ".join(problems)
                + "\n\nRewrite the caption. Every part number in brackets must follow that part's "
                  "description exactly as the list gives it. If you are not sure which part a "
                  "number is, leave the number out."
            )
            problems = _citation_problems(caption, groups)
            if caption and problems:
                pruned = _keep_correct_sentences(caption, groups)
                if pruned:
                    print(f"      ✂️ [Vision] Page {page}: rewrite still named {len(problems)} part(s) "
                          f"wrong - removed those sentences, kept the correct ones.")
                else:
                    print(f"      🚫 [Vision] Page {page}: rewrite still names parts wrong "
                          f"({problems[:2]}) - description not stored.")
                caption = pruned
        if not caption:
            names = ", ".join(
                f"{' / '.join(dict.fromkeys(r['description'] for r in g['rows'] if r.get('description')))} ({g['ref']})"
                for g in refs
            )
            caption = f"Exploded parts view of {heading}. The drawing shows: {names}."

        entries = []
        for g in refs:
            descs = [r["description"] for r in g["rows"] if r.get("description")]
            numbers = [r["part_number"] for r in g["rows"] if r.get("part_number")]
            entries.append(f"{g['ref']} = {' / '.join(dict.fromkeys(descs)) or 'part'} "
                           f"({', '.join(numbers)})".replace(";", ","))
        # Written from the parts list, never by the model. This exact
        # "Parts shown in this diagram: N = ...; " shape is what retrieval reads
        # back to pair the figure with its parts-list table.
        parts_line = "\nParts shown in this diagram: " + "; ".join(entries) + "."
        if not parts.get("verified", True):
            parts_line += " (Parts list read by vision from a scanned page - unverified.)"
        if parts.get("missing_callouts"):
            parts_line += f"\nCallouts with no entry in the parts list: {', '.join(parts['missing_callouts'])}."

        header = f"### {heading} (Exploded View) — page {page}"
        if drawing:
            header += f" — drawing {drawing}"
        return {
            "parts_resolved": True,
            "caption": f"{header}\n\n{caption}{parts_line}",
            "codes": list(metadata.get("callouts") or [g["ref"] for g in refs]),
            "component": heading,
            "kind": "exploded_view",
        }

    def generate_caption(self, image_path: str, metadata: dict = None) -> str:
        """Back-compatible string form."""
        return self.describe(image_path, metadata)["caption"]
