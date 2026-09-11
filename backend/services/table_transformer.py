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
            "If it's a troubleshooting table, list the Problem-Cause-Solution pairs in a dense format."
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
