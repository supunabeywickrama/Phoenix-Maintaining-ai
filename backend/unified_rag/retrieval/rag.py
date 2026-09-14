import re
from enum import Enum
from unified_rag.retrieval.retriever import RetrievalEngine
from unified_rag.db.database import SessionLocal
from unified_rag.ai_client import get_client, chat_text, chat_json, MODEL_CHAT, MODEL_CHAT_LIGHT
from services.image_relevance import verify_images
from services.llm_json import loads_tolerant
from services.table_validator import markdown_is_usable

class RAGMode(Enum):
    SUMMARY = "summary"
    DIAGNOSIS = "diagnosis"
    PROCEDURE = "procedure"
    CLARIFICATION = "clarification"
    EVALUATION = "evaluation"
    CONVERSATIONAL_WIZARD = "conversational_wizard"
    LEARNING = "learning"
    LEARNING_WALKTHROUGH = "learning_walkthrough"


# A "### Happened before..." markdown section, up to the next heading or the end.
_PAST_SECTION = re.compile(
    r"^#{1,4}\s*Happened before[^\n]*\n.*?(?=^#{1,4}\s|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

# "[IMAGE_0] DIAGRAM, FULL VIEW - title (page N)" / "[TABLE_1] TABLE - title (page N)":
# the exact shape of an AVAILABLE MATERIAL line, echoed into the answer.
_TAG_LISTING_LEAK = re.compile(
    r"(\[(?:IMAGE|TABLE)_\d+\])[ \t]+(?:DIAGRAM|SCHEMATIC|CHART|FLOWCHART|EXPLODED VIEW|PHOTO|TABLE)\b"
    r"[^\n]*\(page[^\n]*"
)

METHOD_LABELS = {
    "hands_on": "by hand (engineer's own experience)",
    "system_guided": "by following the system's instructions",
    "both": "by hand and the system's instructions",
}


def _diagnosis_queries(report: str) -> list:
    """Split a fault report into separate lookups.

    "Stopped, X part overheating, injection module not working" names three
    symptoms; searched as one sentence it only finds whichever dominates. One
    cheap light-model call pulls them apart. On any failure the report alone is
    searched, which is exactly the previous behaviour.
    """
    queries = [report]
    prompt = (
        "A technician reported a machine fault. Extract what to look up in the machine's "
        "manual. Return JSON with keys:\n"
        '  "symptoms": short phrases, one per distinct symptom (max 4)\n'
        '  "components": parts or systems named (max 4)\n'
        '  "fault_codes": any codes or alarm numbers mentioned\n'
        "Use the technician's own words, corrected for spelling. Do not add symptoms "
        f"that were not reported.\n\nREPORT: {report}"
    )
    try:
        data = loads_tolerant(chat_json(MODEL_CHAT_LIGHT, prompt, max_tokens=300, temperature=0.0) or "")
    except Exception as e:
        print(f"Symptom extraction skipped: {e}")
        return queries
    if not isinstance(data, dict):
        return queries

    def items(key, n):
        return [str(x).strip() for x in (data.get(key) or []) if str(x).strip()][:n]

    queries += [f"{s} cause troubleshooting" for s in items("symptoms", 4)]
    queries += items("components", 4)
    queries += [f"fault code {c}" for c in items("fault_codes", 3)]
    seen, out = set(), []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out

def _title_of(chunk, fallback: str) -> str:
    """First line of a caption doubles as its title - the captioner writes
    '### <component> (<Kind>) - page N'."""
    text = (chunk.content or "").strip()
    if text:
        head = text.splitlines()[0].lstrip("#").strip()
        head = head.split("(Context:")[0].strip()
        head = head.replace("**", "").replace("__", "").strip(" :*-")
        if head:
            return head[:70]
    return fallback


def _pretty_kind(kind: str) -> str:
    return (kind or "diagram").replace("_", " ").title()


class RAGGenerator:
    """
    The core Multimodal Retrieval-Augmented Generation (RAG) engine.
    Now supports universal provenance checking and simple-English pointwise clarifications.
    """
    def __init__(self):
        self.retriever = RetrievalEngine()
        
    def generate_response(self, query: str, manual_id: str, machine_id: str, mode: RAGMode = RAGMode.SUMMARY,
                          chat_history: str = "", retrieval_query: str = None) -> dict:
        """
        Executes the Multimodal RAG Pipeline with strict mode enforcement and provenance checks.
        """
        db = None
        retrieved_data = {"text_chunks": [], "images": [], "historical_fixes": []}
        
        # PROVENANCE PRE-CHECK
        manual_missing = "[DISCLAIMER_REQUIRED: MISSING_MANUAL]" in query
        if manual_missing:
            # Strip the flag from the query before search to prevent vector pollution
            query = query.replace("[DISCLAIMER_REQUIRED: MISSING_MANUAL]", "").strip()

        # retrieval_query differs from `query` for wizard continuations: "next
        # step", "I'm stuck" and "teach me step by step" carry no topical
        # content of their own, so embedding them alone drifts retrieval onto
        # whatever is generically nearby instead of what was actually asked
        # about. The caller anchors this on the session's original question;
        # `query` itself is untouched, since it is also what the LLM sees as
        # "the current instruction".
        search_query = (retrieval_query or query).strip() or query

        try:
            # STAGE 1: SEMANTIC RETRIEVAL (Fault Tolerant)
            try:
                db = SessionLocal()
                if mode == RAGMode.DIAGNOSIS:
                    # A diagnosis has to consider every cause the manual gives
                    # for every symptom reported, not just the nearest passage.
                    retrieved_data = self.retriever.retrieve_many(
                        db, _diagnosis_queries(search_query), manual_id, machine_id
                    )
                else:
                    retrieved_data = self.retriever.retrieve(db, search_query, manual_id, machine_id)
            except Exception as e:
                print(f"RAG_DB_ERROR: {e}. Falling back to manual-only context.")
            finally:
                if db:
                    db.close()
            
            # STAGE 1b: RELEVANCE GATE (figures AND tables)
            # Vector search returns its top-k regardless of fit. A wrong diagram
            # or a table that does not actually bear on the question is worse
            # than showing nothing - tables in particular are supporting
            # evidence, not a checklist item every answer must include.
            #
            # The pages the answer's own TEXT is grounded on give the judge a
            # much harder signal than caption wording alone: a figure from a
            # page nobody cited, that just LOOKS structurally similar (e.g. a
            # different numbered test using the same feeler-gauge/dial-
            # indicator style diagram), is very likely a different procedure.
            text_pages = {c.page for c in retrieved_data.get("text_chunks", []) if c.page is not None}
            if retrieved_data.get("images"):
                try:
                    retrieved_data["images"] = verify_images(
                        search_query, retrieved_data["images"], context_pages=text_pages
                    )
                except Exception as e:
                    print(f"Image relevance check skipped: {e}")
            if retrieved_data.get("tables"):
                try:
                    retrieved_data["tables"] = verify_images(
                        search_query, retrieved_data["tables"], context_pages=text_pages
                    )
                except Exception as e:
                    print(f"Table relevance check skipped: {e}")

            # STAGE 1c: LEGEND PAIRING
            # A figure whose callouts are just "1", "2", "3..." on the drawing
            # only answers anything once matched to the table that names them
            # (e.g. "Key to Figure 1.1"). That table doesn't necessarily rank on
            # its own for the user's wording, so it's paired in by hard evidence
            # - shared item numbers - rather than left to vector search luck.
            if retrieved_data.get("images"):
                try:
                    retrieved_data["tables"] = self.retriever.attach_legend_tables(
                        manual_id, retrieved_data["images"], retrieved_data.get("tables", [])
                    )
                except Exception as e:
                    print(f"Legend table pairing skipped: {e}")

            # STAGE 2: CONTEXT BUILDER
            text_context = ""
            pages = set()
            image_references = []
            
            for i, chunk in enumerate(retrieved_data["text_chunks"]):
                text_context += f"--- Manual Context {i+1} (Page {chunk.page}) ---\n{chunk.content}\n\n"
                if chunk.page is not None: pages.add(chunk.page)
            
            # Matches from OTHER manuals, kept in their own block and always
            # attributed. Pointers to where else this is documented, never folded
            # into this machine's answer as if it were the same equipment.
            cross_context = ""
            cross_refs = []
            for ref in retrieved_data.get("cross_manual", []):
                body = " ".join((ref.content or "").split())[:500]
                cross_context += (
                    f"--- MANUAL {ref.manual_id}, page {ref.page} ({ref.type}) ---\n{body}\n\n"
                )
                cross_refs.append({
                    "manual_id": ref.manual_id,
                    "page": ref.page,
                    "type": ref.type,
                })

            # Table summaries belong in the prompt text too, so the model can quote
            # the numbers rather than only gesturing at the rendered grid.
            for t in retrieved_data.get("tables", []):
                text_context += f"--- Table (page {t.page}) ---\n{t.content}\n\n"
                if t.page is not None:
                    pages.add(t.page)

            # Past fixes on this same machine that cleared the retriever's
            # similarity floor. Handed back as structured records too, so the
            # chat can show them from the database rather than as model prose.
            history_context = ""
            past_incidents = []
            for i, fix in enumerate(retrieved_data.get("historical_fixes", [])):
                history_context += (
                    f"--- PREVIOUS INCIDENT {i+1} on this machine ({fix.timestamp}) ---\n"
                    f"Reported: {fix.symptom or 'not recorded'}\n"
                    f"Root cause found: {fix.root_cause or 'not recorded'}\n"
                    f"What fixed it: {fix.actions or fix.operator_fix or 'not recorded'}\n"
                    f"Method: {METHOD_LABELS.get(fix.method or '', 'not recorded')}\n"
                    f"Parts replaced: {fix.parts_replaced or 'none recorded'}\n"
                    f"Engineer: {fix.engineer or 'not recorded'}\n"
                    f"Summary: {fix.summary or ''}\n\n"
                )
                past_incidents.append(fix.as_dict())

            # Attachments are what the reader actually sees, in the order they
            # should meet them: the whole drawing for orientation, then the parts
            # cropped from it, then the data tables that back the numbers up.
            attachments = []

            for i, img in enumerate(retrieved_data["images"]):
                if img.path:
                    image_references.append(img.path)
                    attachments.append({
                        "tag": f"IMAGE_{len(image_references) - 1}",
                        "type": "image",
                        "kind": getattr(img, "kind", None) or "diagram",
                        "role": getattr(img, "figure_role", None) or "full",
                        "title": _title_of(img, f"Figure on page {img.page}"),
                        "page": img.page,
                        "url": img.path,
                    })
                text_context += f"--- Image Description {i+1} (Page {img.page}) ---\n{img.content}\n\n"

            # Tables retrieved as text chunks carry the original grid, so they can be
            # shown as a table instead of being paraphrased into a sentence.
            table_index = 0
            seen_tables = set()
            table_sources = list(retrieved_data.get("tables", [])) + [
                c for c in retrieved_data["text_chunks"] if c.type == "table"
            ]
            for chunk in table_sources:
                markdown = getattr(chunk, "render_markdown", None)
                # Guards data ingested before table_validator.is_valid_table()
                # existed. A garbage grid must never reach the chat regardless
                # of when it was stored.
                if not markdown or chunk.id in seen_tables or not markdown_is_usable(markdown):
                    continue
                seen_tables.add(chunk.id)
                attachments.append({
                    "tag": f"TABLE_{table_index}",
                    "type": "table",
                    "kind": "table",
                    "role": None,
                    "title": _title_of(chunk, f"Table on page {chunk.page}"),
                    "page": chunk.page,
                    "markdown": markdown,
                })
                table_index += 1
                
            # STAGE 3: MODE SELECTION (Prompt Construction)
            if mode == RAGMode.DIAGNOSIS:
                system_prompt = self._build_diagnosis_prompt(
                    manual_id, text_context, history_context, chat_history, manual_missing
                )
            elif mode == RAGMode.PROCEDURE:
                system_prompt = self._build_procedure_prompt(manual_id, text_context, history_context, image_references)
            elif mode == RAGMode.CLARIFICATION:
                system_prompt = self._build_clarification_prompt(manual_id, query, text_context, image_references, manual_missing)
            elif mode == RAGMode.EVALUATION:
                system_prompt = self._build_evaluation_prompt(manual_id, query, text_context, image_references)
            elif mode == RAGMode.LEARNING:
                system_prompt = self._build_learning_prompt(
                    manual_id, text_context, image_references, chat_history, manual_missing
                )
            elif mode == RAGMode.LEARNING_WALKTHROUGH:
                system_prompt = self._build_learning_walkthrough_prompt(
                    manual_id, query, text_context, image_references, chat_history, manual_missing
                )
            elif mode == RAGMode.CONVERSATIONAL_WIZARD:
                system_prompt = self._build_conversational_wizard_prompt(
                    manual_id, 
                    query, 
                    text_context, 
                    history_context,
                    image_references,
                    chat_history,
                    manual_missing
                )
            else:
                system_prompt = self._build_summary_prompt(manual_id, text_context, history_context, manual_missing)
            
            # A bare [IMAGE_n] tag tells the model nothing about what it is
            # showing. Naming each one, and saying which are components of which
            # full drawing, is what makes "show the whole figure first" possible.
            if attachments:
                lines = []
                for a in attachments:
                    if a["type"] == "table":
                        lines.append(f"  - [{a['tag']}] TABLE - {a['title']} (page {a['page']})")
                    else:
                        role = "FULL VIEW" if a["role"] != "part" else "COMPONENT of the full view above"
                        lines.append(
                            f"  - [{a['tag']}] {_pretty_kind(a['kind']).upper()}, {role} - "
                            f"{a['title']} (page {a['page']})"
                        )
                system_prompt += (
                    "\n\nAVAILABLE MATERIAL (tags are listed in the order they should be shown):\n"
                    + "\n".join(lines)
                    + "\n\nHOW TO PRESENT IT:\n"
                    "1. ORIENTATION FIRST: show the FULL VIEW tag before any COMPONENT taken from "
                    "it, so the reader sees the whole assembly before a close-up.\n"
                    "2. THEN DETAIL: take the components in turn, saying what each is and does.\n"
                    "3. THEN EVIDENCE: when you state a figure, limit, torque, fault code or part "
                    "number that appears in a table, place that [TABLE_n] tag right there.\n"
                    "4. Do NOT retype a table's contents as prose - place the tag and comment on "
                    "what it shows.\n"
                    "5. Place every tag on its own line at the point where the reader should look "
                    "at it. Never dump tags at the end. Write ONLY the bare tag, e.g. [IMAGE_0] - "
                    "never copy the description text from the list above next to it.\n"
                    "6. Use each tag EXACTLY ONCE. Referring to the same table three times "
                    "renders it three times; refer back to it in words instead.\n"
                    "7. Name the material in the sentence that introduces it, e.g. 'the wiring "
                    "schematic below' or 'the torque table below', so it is clear what is coming.\n"
                    "8. Never present a component crop as though it were the whole assembly.\n"
                    "9. Use ONLY the tags listed above. If no table is listed, do not write a "
                    "[TABLE_n] tag at all - state the values in your own words instead.\n"
                    "10. NUMBERED CALLOUTS: if a figure's description lists 'Parts shown in this "
                    "diagram: 1 = ..., 2 = ...' and a [TABLE_n] with a title like 'Key to Figure' "
                    "is available for it, that table is the legend for those exact numbers. Show "
                    "the figure, then the [TABLE_n] tag right after it, then walk through the "
                    "callouts by number in your own words, e.g. '(1) is the cutter bar assembly, "
                    "(2) the gearbox and eccentric drive, ...' - do not just say 'see the table "
                    "below' and stop. If the question named a component (e.g. 'what is the "
                    "clutch housing'), find its number in the list and say which callout it is.\n"
                )

            if cross_context:
                system_prompt += (
                    "\n\nRELATED CONTENT IN OTHER MANUALS:\n"
                    + cross_context
                    + "HOW TO USE THIS: this is documentation for DIFFERENT equipment. Do not "
                    "present it as describing this machine. If it is genuinely relevant, add a "
                    "short closing note in the form: 'Also documented in <manual> (page N): "
                    "<what is there>.' so the technician knows where else to look. If it is not "
                    "relevant, ignore it entirely and say nothing about it.\n"
                )

            # STAGE 4: LLM INFERENCE
            try:
                # Differentiate user directive by mode for better steerability
                if mode == RAGMode.CLARIFICATION:
                    user_content = f"Please explain the following task in very simple, pointwise English using bullet points: '{query}'"
                elif mode == RAGMode.EVALUATION:
                    user_content = f"Please evaluate the technician's progress: '{query}'"
                elif mode == RAGMode.LEARNING:
                    user_content = f"Explain this so I understand how it works: '{query}'"
                elif mode == RAGMode.LEARNING_WALKTHROUGH:
                    user_content = f"Teach me this one part at a time: '{query}'"
                elif mode == RAGMode.DIAGNOSIS:
                    user_content = f"Fault report from the technician: '{query}'"
                else:
                    user_content = f"Technician query: '{query}'"
                
                answer = chat_text(
                    MODEL_CHAT, user_content, system=system_prompt,
                    max_tokens=3000, temperature=0.1,
                )
            except Exception as e:
                print(f"LLM API call failed: {e}")
                answer = "Error generating response from LLM."

            # Hard guard, independent of the prompt: with no confirmed fix on
            # record, any "happened before" section is invented history and is
            # removed. The past-fix card the chat shows comes from the database
            # rows, so this section is never the only place a real one appears.
            if not past_incidents and answer:
                answer = _PAST_SECTION.sub("", answer)
            # The model sometimes copies a whole AVAILABLE MATERIAL listing line
            # ("[IMAGE_0] DIAGRAM, FULL VIEW - Service Decal ... (page 23)")
            # instead of just the tag; the UI swaps the tag for the figure and
            # would leave the rest as stray text beside it.
            if answer:
                answer = _TAG_LISTING_LEAK.sub(r"\1", answer)

            return {
                "answer": answer,
                "images": image_references,
                "attachments": attachments,
                "pages": sorted(list(pages)),
                "cross_manual": cross_refs,
                "past_incidents": past_incidents,
            }
        except Exception as e:
            print(f"RAG formulation failed: {e}")
            raise e

    def _get_disclaimer(self) -> str:
        return (
            "CRITICAL: I do not have the specific technical manual for this machine in my database. "
            "YOU MUST START YOUR RESPONSE WITH THIS EXACT DISCLAIMER: "
            "'⚠️ Documentation Alert: I do not have the specific technical manual for this machine in my database. "
            "The following steps are based on general industrial best practices. Please consult local site safety protocols before proceeding.'\n\n"
        )

    def _build_diagnosis_prompt(self, manual_id: str, text_context: str, history_context: str,
                                history: str = "", missing_manual: bool = False) -> str:
        """First reply to a fault report: make it safe, say what is probably
        wrong, and give the checks in the order to do them — then ask for the
        result of the first check so the fix can be steered by evidence.

        Replaces SUMMARY for fault reports: a technician standing at a stopped
        machine needs to know what to check, and SUMMARY forbade saying.
        """
        disclaimer = self._get_disclaimer() if missing_manual else ""
        # The past-incident section is only described to the model when there
        # are incidents. Telling it "leave this out if there are none" was not
        # enough: with nothing on record it still wrote the heading and filled
        # it with a made-up history. What it is never told about, it can't write.
        if history_context:
            past = (
                "PAST INCIDENTS ON THIS SAME MACHINE (confirmed fixes from the maintenance "
                f"database):\n{history_context}"
            )
            past_section = (
                "### Happened before on this machine\n"
                "One short paragraph per past incident listed above - when, what the cause was, "
                "what fixed it, and how (by hand or following the system's instructions). Say "
                "whether it matches this report closely or only partly. Use only what the record "
                "says.\n\n"
            )
        else:
            past, past_section = "", ""
        return (
            f"""You are a senior maintenance engineer diagnosing a fault on: {manual_id}.

{disclaimer}CONVERSATION SO FAR:
{history or '(this is the first message)'}

{past}
MANUAL SOURCE OF TRUTH:
{text_context}

WRITE YOUR REPLY IN EXACTLY THIS STRUCTURE (markdown):

[PHASE: Diagnosis]

### Safety first
- 2-4 bullets on making the machine safe to inspect for THESE symptoms: how to stop and
  isolate it, and the hazards this particular fault creates. Safety bullets are about
  making it safe, not about troubleshooting. Only hazards the manual states for this
  machine, or that the reported symptoms directly create (an overheating engine means hot
  surfaces). Do NOT list a hazard just because machines commonly have it - if the manual
  context says nothing about stored pressure or electrics on this machine, do not mention
  them. Cite (page N) where the manual states it.

{past_section}### What is likely happening
A numbered list of probable causes, most likely first. For each: **the cause** - why it
fits what the technician reported - (page N). Consider every symptom reported, and causes
that connect them (e.g. one failure that explains both the overheating and the stop).
If a past incident on this machine matches, weigh that cause higher and say so.

### Check in this order
A numbered list of 3-5 checks that between them cover the likely causes above - safest
and quickest first, then most likely cause. For each:
**Check:** what part or reading to look at
**How:** exactly how - tool, location, spec or limit from the manual - (page N)
**Result → meaning:** what each result tells you, pointing to a cause number above or to
the next check (e.g. "Below 0.75 MPa → cause 2 · Normal → go to check 3").

Then end with EXACTLY ONE question on its own line, for the result of the FIRST check, in
this format, with 2-4 short answers the technician can tap:
[ASK: <question> | <answer> | <answer> | <answer>]
The answers are what the technician OBSERVES ("Yes, fins blocked", "No, fins clear",
"Below 0.75 MPa"), never instructions ("clean the fins") - you decide what to do next
from their answer.

RULES:
- Use ONLY values, limits, part names and procedures that appear in the manual context or
  the past incidents. If the manual does not give a spec, say "the manual does not give a
  value for this" - never invent one.
- Do not write the repair procedure yet. That comes after the technician reports what the
  checks show.
- Do not emit any [SUGGESTION: ...] tag.
- Do not narrate your reasoning; write the answer directly.
"""
        )

    def _build_summary_prompt(self, manual_id: str, text_context: str, history_context: str, missing_manual: bool = False) -> str:
        disclaimer = self._get_disclaimer() if missing_manual else ""
        return (
            f"You are an Industrial Diagnostic AI for: {manual_id}.\n\n"
            f"{disclaimer}"
            "YOUR TASK: Provide a SHORT diagnostic summary (3-5 sentences maximum).\n\n"
            "ABSOLUTE RULES:\n"
            "1. DO NOT provide ANY maintenance steps, repair instructions, or numbered procedures.\n"
            "2. DO NOT use bullet points or numbered lists to describe what to do.\n"
            "3. DO NOT say 'Step 1', 'Step 2', etc.\n"
            "4. DO NOT provide checklists or action items.\n"
            "5. ONLY describe WHAT the problem likely is and WHY based on sensor data.\n"
            "6. Keep your response under 5 sentences.\n"
            "6b. Cite the manual page for any specific fact, as (page N), using the "
            "page numbers shown in the context below.\n"
            "7. You MUST end your response with EXACTLY this tag on its own line to trigger the repair wizard:\n"
            "   [SUGGESTION: Generate full step-by-step repair procedure]\n\n"
            f"MANUAL CONTEXT:\n{text_context}\n\n"
            f"HISTORICAL REPAIRS:\n{history_context}"
        )
    
    def _build_procedure_prompt(self, manual_id: str, text_context: str, history_context: str, image_references: list) -> str:
        # Which tags exist, their kind, and full-view-before-parts ordering are
        # appended uniformly after mode selection (the "AVAILABLE MATERIAL"
        # block) - no separate list here, so there is only ever one ordering
        # rule in the prompt instead of two that can disagree.
        return (
            f"You are an Industrial Maintenance Procedure Generator for: {manual_id}.\n\n"
            "YOUR TASK: Generate a COMPLETE structured repair procedure in JSON format.\n\n"
            "ABSOLUTE RULES:\n"
            "1. Output ONLY the JSON wrapped between [PROCEDURE_START] and [PROCEDURE_END] tags.\n"
            "2. Do NOT write any text before or after the JSON block. No introductions. No summaries.\n"
            "3. The FIRST phase MUST be type 'safety' with lockout/tagout and PPE steps.\n"
            "4. Mark safety/critical tasks with '\"critical\": true'.\n"
            "5. Include image/table references like [IMAGE_0], [TABLE_0] in task text where "
            "relevant, using the material listed below (see AVAILABLE MATERIAL).\n\n"
            "EXACT OUTPUT FORMAT:\n"
            "[PROCEDURE_START]\n"
            '{"phases": [...]}\n'
            "[PROCEDURE_END]\n\n"
            f"MANUAL CONTEXT:\n{text_context}\n\n"
            f"HISTORICAL REPAIRS:\n{history_context}"
        )

    def _build_clarification_prompt(self, manual_id: str, step_text: str, text_context: str, image_references: list, missing_manual: bool = False) -> str:
        """
        MODE 3: Step Clarification Prompt optimized for simple pointwise English and images.
        """
        disclaimer = self._get_disclaimer() if missing_manual else ""

        return (
            f"You are a Technical Mentor for: {manual_id}.\n\n"
            f"{disclaimer}"
            f"YOUR TASK: Provide a detailed, very simple, and clear explanation for the following maintenance task: '{step_text}'\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. YOUR RESPONSE MUST BE A LIST OF BULLET POINTS. \n"
            "2. ABSOLUTELY NO PARAGRAPHS. DO NOT combine your steps into a block of text.\n"
            "3. DO NOT include introductory filler (e.g., 'Let's break this down' or 'I can help with that'). Jump straight to the bullet points.\n"
            "4. Use SIMPLE ENGLISH (ELI5). No jargon. \n"
            "5. MATERIAL: interleave the tags listed for you below (see AVAILABLE MATERIAL) at the correct logical step. If diagrams or tables are available, you are FORBIDDEN from finishing without placing them.\n"
            "6. For each sub-point, explain: What to do, Why it matters, and How to do it correctly.\n\n"
            f"MANUAL SOURCE OF TRUTH:\n{text_context}"
        )

    def _build_learning_prompt(self, manual_id: str, text_context: str, image_references: list,
                               history: str = "", missing_manual: bool = False) -> str:
        """
        LEARNING MODE: the technician wants to understand the equipment, not fix a
        fault right now. Deliberately free of the diagnostic prompt machinery --
        no repair-wizard suggestion tag, no lockout/tagout preamble -- because
        answering "how does the hydraulic circuit work?" with a safety checklist
        and a repair procedure is the wrong shape of answer.
        """
        disclaimer = self._get_disclaimer() if missing_manual else ""
        return (
            f"""You are a patient technical instructor for: {manual_id}.

{disclaimer}YOUR TASK: Teach the technician how this equipment works, using the manual
as the source of truth.

RULES:
1. EXPLAIN, do not instruct. Describe how and why it works, not a repair procedure.
2. Structure the answer: the short version first, then the detail.
3. Build from fundamentals: assume solid trade knowledge but no familiarity with THIS machine.
4. Name the real components and their function, and how they interact.
5. CITE PAGES: when a fact comes from the manual, cite it as (page N) using the page numbers
   given in the context below.
6. MATERIAL: interleave the tags listed for you below (see AVAILABLE MATERIAL) at the exact
   point in the explanation where each one is what the reader should be looking at.
7. Where genuinely useful, note what typically goes wrong with a component, but do NOT turn
   this into a repair procedure.
8. Do NOT emit any [SUGGESTION: ...] tag. This is a learning answer, not a fault report.

CONVERSATION SO FAR:
{history}

MANUAL SOURCE OF TRUTH:
{text_context}
"""
        )

    def _build_learning_walkthrough_prompt(self, manual_id: str, user_query: str,
                                           text_context: str, image_references: list,
                                           history: str = "", missing_manual: bool = False) -> str:
        """
        Step-by-step TEACHING, as opposed to the repair wizard.

        The repair wizard opens with lockout/tagout because someone is about to put
        hands on a faulty machine. Nothing is broken here, so that preamble is noise;
        what this needs is one concept at a time, checked before moving on.
        """
        disclaimer = self._get_disclaimer() if missing_manual else ""
        return (
            f"""You are a patient technical instructor for: {manual_id}.

{disclaimer}OBJECTIVE: Teach this machine one step at a time, conversationally.

CONVERSATION SO FAR:
{history}

LEARNER INPUT: {user_query}

RULES:
1. ONE concept per reply. Do not dump the whole system at once.
2. READ THE HISTORY: if they said they understood, move to the NEXT part. If they asked
   for it again, re-explain the SAME point differently, do not advance.
3. START WIDE: the first reply covers what the assembly is for and the full diagram,
   before any individual component.
4. Then work through the components in a sensible order, one per reply.
5. End each reply with a short check, e.g. "Make sense, or shall I go over that again?"
6. CITE PAGES as (page N) using the page numbers in the context below.
7. This is teaching, NOT a repair. No lockout/tagout preamble, no PPE checklist, no
   [SUGGESTION: ...] tag.
8. Start your reply with [PHASE: Learning].

MANUAL SOURCE OF TRUTH:
{text_context}
"""
        )

    def _build_evaluation_prompt(self, manual_id: str, user_feedback: str, text_context: str, image_references: list) -> str:
        return (
            f"You are a Quality Assurance Supervisor for: {manual_id}.\n\n"
            f"Technician feedback: '{user_feedback}'\n"
            "YOUR JOB: Evaluate if the task is complete. Start with [STEP_COMPLETE] or [STEP_NEED_HELP] based on their tone and evidence.\n\n"
            f"MANUAL SOURCE OF TRUTH:\n{text_context}"
        )

    def _build_conversational_wizard_prompt(self, manual_id: str, user_query: str, text_context: str, history_context: str, image_references: list, history: str, missing_manual: bool = False) -> str:
        """
        MODE 5: Conversational Wizard Prompt.
        """
        disclaimer = self._get_disclaimer() if missing_manual else ""

        return (
            f"You are an Intelligent Maintenance Mentor for: {manual_id}.\n\n"
            f"{disclaimer}"
            "YOUR OBJECTIVE:\n"
            "Guide the operator through the repair process conversationally but with HIGH STRUCTURE.\n\n"
            "CURRENT CONTEXT:\n"
            f"1. CHAT HISTORY:\n{history}\n"
            f"2. OPERATOR INPUT: '{user_query}'\n"
            f"3. MANUAL SOURCE OF TRUTH:\n{text_context}\n"
            f"4. PAST INTERACTIONS / HISTORICAL KNOWLEDGE:\n{history_context}\n\n"
            "YOUR MISSION CRITICAL TASK:\n"
            "1. READ HISTORY: Determine if we are in Safety (start here!) or Repair phase.\n"
            "2. STRUCTURE: Use a clear, structured format. Use bullet points for steps. No long paragraphs.\n"
            "3. MATERIAL (MANDATORY): interleave the tags listed for you below (see AVAILABLE "
            "MATERIAL) directly into your steps at the EXACT point where they are most helpful. "
            "THIS IS A CRITICAL REQUIREMENT for operator safety. Do not just list them at the end. "
            "If a diagram shows a specific tool or component, match it to that step.\n"
            "4. ADAPT: If they completed a step, provide the EXACT next step from the manual. Do not ask them what to do.\n"
            "5. FIELD WISDOM: If 'PAST INTERACTIONS' shows a successful previous fix for a similar issue on this machine, MENTION IT.\n"
            "6. RESPONSE STYLE: Professional, highly structured, and technical. Use bolding for emphasis.\n\n"
            "7. CITE PAGES: when a step comes from the manual, cite it as (page N) using the "
            "page numbers in the context above, so the technician can check the original.\n"
            "8. USE THE CHECK RESULTS: if the history contains a diagnosis and the technician has "
            "reported what a check showed, start by saying in one or two sentences which cause "
            "that result now points to and why. If it rules a cause out, say so and move to the "
            "next check instead of repairing the wrong thing.\n"
            "9. EXPLAIN EVERY STEP: give 1-2 steps per reply, never the whole repair at once. "
            "Write each step as:\n"
            "   **Step N - <short title>**\n"
            "   - **Do:** the exact action\n"
            "   - **Why:** what this achieves / what it rules in or out\n"
            "   - **How:** tool, location, torque/spec/limit from the manual (page N)\n"
            "   - **You should see:** the result that means the step worked\n"
            "10. ASK AT THE END: finish with EXACTLY ONE question on its own line in this format:\n"
            "   [ASK: <question> | <answer> | <answer> | <answer>]\n"
            "   When you need a reading or observation, ask for it with realistic answer options. "
            "After a step that should clear the fault, ask: "
            "[ASK: Did that fix it? | Problem solved | Done - still faulty | I'm stuck]\n"
            "11. Never invent a value, limit or part number that is not in the manual context. "
            "Do not narrate your reasoning.\n"
            "OUTPUT FORMAT: Start with '[PHASE: <Name>]' where Name is Safety, Repair or "
            "Verification - the diagnosis is already done, so never 'Diagnosis'.\n\n"
            "SAFETY MANDATE:\n"
            "- If the CHAT HISTORY already contains a 'Safety first' section, do NOT repeat it: "
            "open with one line confirming the machine is still stopped and isolated, then continue.\n"
            "- Otherwise, IF CHAT HISTORY IS EMPTY OR HAS NO SAFETY STEPS: You MUST exclusively provide Safety and Preparation steps (PPE, LOTO, etc.) as the very first instruction. DO NOT skip to the actual repair task.\n"
        )
