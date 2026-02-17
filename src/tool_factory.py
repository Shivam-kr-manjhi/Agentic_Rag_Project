"""
Tool Factory — creates vector and summary tools for each processed document.

For each DocumentInfo, produces two Tool objects:
  - Vector Tool: fine-grained semantic search over document chunks (ChromaDB)
  - Summary Tool: returns the pre-computed document summary

Tool descriptions are embedded and stored in a `tool_descriptions` ChromaDB
collection so the Agent Worker can perform semantic tool selection.
"""

from dataclasses import dataclass
from typing import Callable, List

import chromadb

from src.config import (
    embed_model,
    CHROMA_DB_DIR,
    TOP_K_CHUNKS,
)
from src.document_processor import DocumentInfo


# ── Data Classes ────────────────────────────────────────────────────

@dataclass
class Tool:
    """A callable tool with semantic metadata."""
    name: str
    description: str
    tool_type: str                   # "vector" or "summary"
    document_name: str               # source document
    function: Callable[[str], str]   # accepts query, returns text


# ── Tool Factory ────────────────────────────────────────────────────

class ToolFactory:
    """Build vector + summary tools for a set of documents."""

    def __init__(self):
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    def build_tools(self, doc_infos: List[DocumentInfo]) -> List[Tool]:
        """Create two tools per document and persist tool descriptions."""
        tools: List[Tool] = []

        for info in doc_infos:
            # ── Vector Tool ─────────────────────────────────────────
            vector_tool = self._make_vector_tool(info)
            tools.append(vector_tool)

            # ── Summary Tool ────────────────────────────────────────
            summary_tool = self._make_summary_tool(info)
            tools.append(summary_tool)

        # ── Persist tool descriptions in ChromaDB ───────────────────
        self._persist_tool_descriptions(tools)

        print(f"[ToolFactory] Built {len(tools)} tools for {len(doc_infos)} document(s)\n")
        return tools

    # ── Vector Tool Builder ─────────────────────────────────────────

    def _make_vector_tool(self, info: DocumentInfo) -> Tool:
        collection_name = info.collection_name
        summary_snippet = info.summary[:200].replace("\n", " ")

        name = f"vector_search_{info.slug}"
        description = (
            f"Search for specific facts, figures, names, dates, definitions, "
            f"or detailed information within the document '{info.name}'. "
            f"This document covers: {summary_snippet}..."
        )

        def vector_fn(query: str, _cname=collection_name) -> str:
            """Query ChromaDB collection for top-K similar chunks."""
            collection = self.chroma_client.get_collection(name=_cname)
            query_embedding = embed_model.encode([query]).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=TOP_K_CHUNKS,
            )
            if results["documents"] and results["documents"][0]:
                chunks = results["documents"][0]
                return "\n\n---\n\n".join(
                    f"[Chunk {i+1}] {chunk}"
                    for i, chunk in enumerate(chunks)
                )
            return "(No relevant chunks found)"

        return Tool(
            name=name,
            description=description,
            tool_type="vector",
            document_name=info.name,
            function=vector_fn,
        )

    # ── Summary Tool Builder ────────────────────────────────────────

    def _make_summary_tool(self, info: DocumentInfo) -> Tool:
        summary_text = info.summary
        summary_snippet = info.summary[:200].replace("\n", " ")

        name = f"summary_{info.slug}"
        description = (
            f"Get a high-level overview, themes, main topics, or general "
            f"understanding of the document '{info.name}'. "
            f"This document covers: {summary_snippet}..."
        )

        def summary_fn(query: str, _summary=summary_text) -> str:
            return _summary

        return Tool(
            name=name,
            description=description,
            tool_type="summary",
            document_name=info.name,
            function=summary_fn,
        )

    # ── Tool Description Persistence ────────────────────────────────

    def _persist_tool_descriptions(self, tools: List[Tool]) -> None:
        """Embed and store all tool descriptions in ChromaDB for semantic lookup."""
        # Recreate the collection fresh each time tools are built
        try:
            self.chroma_client.delete_collection(name="tool_descriptions")
        except Exception:
            pass
        td_collection = self.chroma_client.create_collection(name="tool_descriptions")

        names = [t.name for t in tools]
        descriptions = [t.description for t in tools]
        embeddings = embed_model.encode(descriptions, show_progress_bar=False).tolist()
        metadatas = [
            {"tool_type": t.tool_type, "document_name": t.document_name}
            for t in tools
        ]

        td_collection.add(
            ids=names,
            documents=descriptions,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"[ToolFactory] Persisted {len(tools)} tool descriptions to ChromaDB")
