from enum import Enum
from unified_rag.retrieval.retriever import RetrievalEngine
from unified_rag.db.database import SessionLocal
from unified_rag.ai_client import get_client, MODEL_CHAT

class RAGMode(Enum):
    SUMMARY = "summary"
    PROCEDURE = "procedure"
    CLARIFICATION = "clarification"
    EVALUATION = "evaluation"
    CONVERSATIONAL_WIZARD = "conversational_wizard"

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
            
            # STAGE 2: CONTEXT BUILDER
            text_context = ""
            pages = set()
            image_references = []
            
            for i, chunk in enumerate(retrieved_data["text_chunks"]):
                text_context += f"--- Manual Context {i+1} (Page {chunk.page}) ---\n{chunk.content}\n\n"
                if chunk.page is not None: pages.add(chunk.page)
            
            history_context = ""
            for i, fix in enumerate(retrieved_data.get("historical_fixes", [])):
                history_context += f"--- PREVIOUS FIX {i+1} ({fix.timestamp}) ---\nSummary: {fix.summary}\nOperator Actions: {fix.operator_fix}\n\n"

            for i, img in enumerate(retrieved_data["images"]):
                if img.path: image_references.append(img.path)
                text_context += f"--- Image Description {i+1} (Page {img.page}) ---\n{img.content}\n\n"
                
            # STAGE 3: MODE SELECTION (Prompt Construction)
            if mode == RAGMode.PROCEDURE:
                system_prompt = self._build_procedure_prompt(manual_id, text_context, history_context, image_references)
            elif mode == RAGMode.CLARIFICATION:
                system_prompt = self._build_clarification_prompt(manual_id, query, text_context, image_references, manual_missing)
            elif mode == RAGMode.EVALUATION:
                system_prompt = self._build_evaluation_prompt(manual_id, query, text_context, image_references)
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
            
            # STAGE 4: LLM INFERENCE
            try:
                # Differentiate user directive by mode for better steerability
                if mode == RAGMode.CLARIFICATION:
                    user_content = f"Please explain the following task in very simple, pointwise English using bullet points: '{query}'"
                elif mode == RAGMode.EVALUATION:
                    user_content = f"Please evaluate the technician's progress: '{query}'"
                else:
                    user_content = f"Technician query: '{query}'"
                
                response = get_client().chat.completions.create(
                    model=MODEL_CHAT,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=3000,
                    temperature=0.1
                )
                answer = response.choices[0].message.content
            except Exception as e:
                print(f"LLM API call failed: {e}")
                answer = "Error generating response from LLM."
            
            return {
                "answer": answer,
                "images": image_references,
                "pages": sorted(list(pages))
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
            "OUTPUT FORMAT: Start with '[PHASE: <Name>]'.\n\n"
            "SAFETY MANDATE:\n"
            "- IF CHAT HISTORY IS EMPTY: You MUST exclusively provide Safety and Preparation steps (PPE, LOTO, etc.) as the very first instruction. DO NOT skip to the actual repair task.\n\n"
            f"AVAILABLE IMAGES:\n{image_tags}"
        )
