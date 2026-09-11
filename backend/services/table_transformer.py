import json
from unified_rag.ai_client import chat_text, MODEL_CHAT_LIGHT

class TableTransformer:

    def summarize_table(self, table_json: str, context: str = "") -> str:
        """
        Uses an LLM to transform raw technical table JSON into a dense, searchable summary.
        """
        prompt = (
            "You are a Technical Data Specialist. Convert this raw table JSON into a concise, searchable summary.\n"
            f"Context: {context}\n"
            "Format your response as a clear description. Focus on key specifications, ranges, and part numbers.\n"
            "Include every unique column name and its meaning in the context of the technical manual.\n"
            "If it's a troubleshooting table, list the Problem-Cause-Solution pairs in a dense format.\n\n"
            "GROUNDING RULES - this data has already been screened, but stay strict anyway:\n"
            "- Describe ONLY values that literally appear in the JSON below. Never invent a figure "
            "number, title, part name, or value that is not there.\n"
            "- Do not guess at column meanings you are not confident of; describe what the value "
            "looks like instead (e.g. 'a torque value in Nm') rather than assigning a false label.\n"
            "- If the data is too sparse or garbled to summarise meaningfully, say exactly that in "
            "one sentence rather than filling the gap with a plausible-sounding invention."
        )
        
        try:
            # 400 tokens through the OpenAI-compatible endpoint is not enough for a
            # local reasoning model to get past its own thinking; chat_text() uses
            # the native endpoint where thinking is actually suppressed.
            summary = (chat_text(
                MODEL_CHAT_LIGHT,
                f"{prompt}\n\nRAW TABLE DATA:\n{table_json}",
                system="You convert structured technical data into dense, searchable text summaries.",
                max_tokens=700,
                temperature=0.0,
            ) or "").strip()
            if not summary:
                # Never store an empty chunk: the raw grid is still searchable text.
                return f"Table data: {table_json[:800]}"
            return summary
        except Exception as e:
            print(f"❌ [TableTransformer] Error summarizing table: {e}")
            return f"Table Data: {table_json[:500]}..."
