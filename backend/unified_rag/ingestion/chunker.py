from langchain_text_splitters import RecursiveCharacterTextSplitter

class ContextualChunker:
    def __init__(self, chunk_size=800, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Use LangChain for semantic boundary awareness
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def chunk_data(self, parsed_data: list, manual_id: str):
        """
        Implements semantic chunking for text and tables.
        Retains metadata and handles FigureSplitting artifacts.
        """
        chunks = []
        for item in parsed_data:
            # 1. Handle Text
            if item["type"] == "text":
                text_content = item["content"]
                texts = self.splitter.split_text(text_content)
                
                for t in texts:
                    chunks.append({
                        "manual_id": manual_id,
                        "type": "text",
                        "content": t,
                        "page": item["page"],
                        "metadata": item.get("metadata", {})
                    })
            
            # 2. Handle Tables (treated as discrete units if small, or split if massive)
            elif item["type"] == "table":
                # Tables are often best kept whole unless they exceed the window
                chunks.append({
                    "manual_id": manual_id,
                    "type": "table",
                    "content": item["content"],
                    "page": item["page"],
                    "metadata": item.get("metadata", {})
                })
                
            # 3. Handle Images (Single or Sub-Figures)
            elif item["type"] == "image":
                chunks.append({
                    "manual_id": manual_id,
                    "type": "image",
                    "path": item["path"],
                    "content": item.get("content", ""), # Vision Caption will be filled later
                    "page": item["page"],
                    "metadata": item.get("metadata", {})
                })
                
        return chunks
