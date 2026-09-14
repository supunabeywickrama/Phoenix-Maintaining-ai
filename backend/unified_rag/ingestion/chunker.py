from langchain_text_splitters import RecursiveCharacterTextSplitter

# A table longer than this is split by ROWS. Chosen well under the embedding
# model's window: the pipeline appends an LLM summary to these rows before
# embedding, so the stored chunk ends up roughly double this.
MAX_TABLE_CHARS = 2500


def _split_table_rows(content: str, render_markdown: str, caption_line: bool):
    """Split one oversized table into row-aligned parts.

    Returns [(content, render_markdown), ...]. Content and markdown are cut at
    the SAME row boundaries so each part renders as a self-contained grid with
    its own header, rather than a caption stranded from its data.
    """
    lines = (content or "").splitlines()
    head, rows = (lines[:1], lines[1:]) if caption_line and lines else ([], lines)

    md_lines = (render_markdown or "").splitlines()
    # A markdown table is header + separator + body.
    md_head, md_rows = (md_lines[:2], md_lines[2:]) if len(md_lines) > 2 else ([], [])
    # Only keep the two in lockstep when they genuinely describe the same rows;
    # otherwise the grid goes with the first part and the rest are text-only,
    # which is still correct, just less pretty.
    aligned = len(md_rows) == len(rows) and bool(md_head)

    groups, current, size = [], [], 0
    for i, row in enumerate(rows):
        if current and size + len(row) > MAX_TABLE_CHARS:
            groups.append(current)
            current, size = [], 0
        current.append(i)
        size += len(row) + 1
    if current:
        groups.append(current)

    out = []
    for n, idxs in enumerate(groups):
        part_rows = [rows[i] for i in idxs]
        note = f"(part {n + 1} of {len(groups)})" if len(groups) > 1 else ""
        body = "\n".join(head + ([note] if note else []) + part_rows)
        if aligned:
            md = "\n".join(md_head + [md_rows[i] for i in idxs])
        else:
            md = render_markdown if n == 0 else None
        out.append((body, md))
    return out


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
            
            # 2. Handle Tables (kept whole when small, split by rows when not)
            elif item["type"] == "table":
                meta = item.get("metadata", {})
                content = item.get("content") or ""
                if len(content) <= MAX_TABLE_CHARS:
                    parts = [(content, item.get("render_markdown"))]
                else:
                    # Previously this branch did not exist: a 200-row parts list
                    # went to the embedder whole and either came back truncated
                    # or failed the call outright, and pipeline.py's per-chunk
                    # except just printed and moved on - so the table silently
                    # vanished from the index.
                    parts = _split_table_rows(
                        content,
                        item.get("render_markdown"),
                        caption_line=bool(meta.get("title")),
                    )
                for body, md in parts:
                    chunks.append({
                        "manual_id": manual_id,
                        "type": "table",
                        "content": body,
                        "page": item["page"],
                        "kind": item.get("kind", "table"),
                        "render_markdown": md,
                        "metadata": meta,
                    })
                
            # 2b. Handle parts-list entries: one per ref, already the right size,
            #     never split - each is meant to be exactly one searchable part.
            elif item["type"] == "part":
                chunks.append({
                    "manual_id": manual_id,
                    "type": "part",
                    "content": item["content"],
                    "page": item["page"],
                    "kind": "part",
                    "parent_path": item.get("parent_path"),
                    "metadata": item.get("metadata", {}),
                })

            # 3. Handle Images (Single or Sub-Figures)
            elif item["type"] == "image":
                chunks.append({
                    "manual_id": manual_id,
                    "type": "image",
                    "path": item["path"],
                    "content": item.get("content", ""), # Vision Caption will be filled later
                    "page": item["page"],
                    "kind": item.get("kind"),
                    "figure_role": item.get("figure_role", "full"),
                    "parent_path": item.get("parent_path"),
                    "width": item.get("width"),
                    "height": item.get("height"),
                    "metadata": item.get("metadata", {})
                })
                
        return chunks
