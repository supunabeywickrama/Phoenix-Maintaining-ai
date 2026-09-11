from enum import Enum
from unified_rag.retrieval.retriever import RetrievalEngine
from unified_rag.db.database import SessionLocal
from unified_rag.ai_client import get_client, chat_text, MODEL_CHAT
from services.image_relevance import verify_images

class RAGMode(Enum):
    SUMMARY = "summary"
    PROCEDURE = "procedure"
    CLARIFICATION = "clarification"
    EVALUATION = "evaluation"
    CONVERSATIONAL_WIZARD = "conversational_wizard"
    LEARNING = "learning"
    LEARNING_WALKTHROUGH = "learning_walkthrough"

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
        
    def generate_response(self, query: str, manual_id: str, machine_id: str, mode: RAGMode = RAGMode.SUMMARY, chat_history: str = "") -> dict:
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

        try:
            # STAGE 1: SEMANTIC RETRIEVAL (Fault Tolerant)
            try:
                db = SessionLocal()
                # Use the CLEAN query for search
                retrieved_data = self.retriever.retrieve(db, query, manual_id, machine_id)
            except Exception as e:
                print(f"RAG_DB_ERROR: {e}. Falling back to manual-only context.")
            finally:
                if db:
                    db.close()
            
            # STAGE 1b: IMAGE RELEVANCE GATE
            # Vector search returns its top-k regardless of fit; showing a
            # technician a diagram of the wrong assembly is worse than none.
            if retrieved_data.get("images"):
                try:
                    retrieved_data["images"] = verify_images(query, retrieved_data["images"])
                except Exception as e:
                    print(f"Image relevance check skipped: {e}")

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

            history_context = ""
            for i, fix in enumerate(retrieved_data.get("historical_fixes", [])):
                history_context += f"--- PREVIOUS FIX {i+1} ({fix.timestamp}) ---\nSummary: {fix.summary}\nOperator Actions: {fix.operator_fix}\n\n"

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
                if not markdown or chunk.id in seen_tables:
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
            if mode == RAGMode.PROCEDURE:
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
                    "at it. Never dump tags at the end.\n"
                    "6. Use each tag EXACTLY ONCE. Referring to the same table three times "
                    "renders it three times; refer back to it in words instead.\n"
                    "7. Name the material in the sentence that introduces it, e.g. 'the wiring "
                    "schematic below' or 'the torque table below', so it is clear what is coming.\n"
                    "8. Never present a component crop as though it were the whole assembly.\n"
                    "9. Use ONLY the tags listed above. If no table is listed, do not write a "
                    "[TABLE_n] tag at all - state the values in your own words instead.\n"
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
                else:
                    user_content = f"Technician query: '{query}'"
                
                answer = chat_text(
                    MODEL_CHAT, user_content, system=system_prompt,
                    max_tokens=3000, temperature=0.1,
                )
            except Exception as e:
                print(f"LLM API call failed: {e}")
                answer = "Error generating response from LLM."
            
            return {
                "answer": answer,
                "images": image_references,
                "attachments": attachments,
                "pages": sorted(list(pages)),
                "cross_manual": cross_refs,
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
        image_tags = "\n".join([f"  - [IMAGE_{i}] for image reference {i}" for i in range(len(image_references))])
        return (
            f"You are an Industrial Maintenance Procedure Generator for: {manual_id}.\n\n"
            "YOUR TASK: Generate a COMPLETE structured repair procedure in JSON format.\n\n"
            "ABSOLUTE RULES:\n"
            "1. Output ONLY the JSON wrapped between [PROCEDURE_START] and [PROCEDURE_END] tags.\n"
            "2. Do NOT write any text before or after the JSON block. No introductions. No summaries.\n"
            "3. The FIRST phase MUST be type 'safety' with lockout/tagout and PPE steps.\n"
            "4. Mark safety/critical tasks with '\"critical\": true'.\n"
            "5. Include image references like [IMAGE_0], [IMAGE_1] in task text where relevant.\n\n"
            f"AVAILABLE IMAGE TAGS:\n{image_tags}\n\n"
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
        image_tags = "\n".join([f"  - [IMAGE_{i}] for image reference {i}" for i in range(len(image_references))])
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
            "5. IMAGE INCLUSION (MANDATORY): You MUST interleave relevant [IMAGE_N] tags within your bullet points. If diagrams are available, you are FORBIDDEN from finishing the response without placing them at the correct logical step.\n"
            "6. For each sub-point, explain: What to do, Why it matters, and How to do it correctly.\n\n"
            f"AVAILABLE IMAGE TAGS:\n{image_tags}\n\n"
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
        image_tags = chr(10).join(
            [f"  - [IMAGE_{i}] for image reference {i}" for i in range(len(image_references))]
        )
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
6. IMAGES: interleave [IMAGE_N] tags at the exact point in the explanation where each diagram
   is what the reader should be looking at.
7. Where genuinely useful, note what typically goes wrong with a component, but do NOT turn
   this into a repair procedure.
8. Do NOT emit any [SUGGESTION: ...] tag. This is a learning answer, not a fault report.

CONVERSATION SO FAR:
{history}

AVAILABLE IMAGE TAGS:
{image_tags}

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
        image_tags = "\n".join([f"  - [IMAGE_{i}] for image reference {i}" for i in range(len(image_references))])
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
            "3. IMAGES (MANDATORY): You MUST interleave [IMAGE_N] tags (e.g. [IMAGE_0], [IMAGE_1]) directly into your steps at the EXACT point where the visual reference is most helpful. THIS IS A CRITICAL REQUIREMENT for operator safety. Do not just list them at the end. If an image shows a specific tool or component, match it to that step.\n"
            "4. ADAPT: If they completed a step, provide the EXACT next step from the manual. Do not ask them what to do.\n"
            "5. FIELD WISDOM: If 'PAST INTERACTIONS' shows a successful previous fix for a similar issue on this machine, MENTION IT.\n"
            "6. RESPONSE STYLE: Professional, highly structured, and technical. Use bolding for emphasis.\n\n"
            "7. CITE PAGES: when a step comes from the manual, cite it as (page N) using the "
            "page numbers in the context above, so the technician can check the original.\n"
            "OUTPUT FORMAT: Start with '[PHASE: <Name>]'.\n\n"
            "SAFETY MANDATE:\n"
            "- IF CHAT HISTORY IS EMPTY: You MUST exclusively provide Safety and Preparation steps (PPE, LOTO, etc.) as the very first instruction. DO NOT skip to the actual repair task.\n\n"
            f"AVAILABLE IMAGES:\n{image_tags}"
        )
